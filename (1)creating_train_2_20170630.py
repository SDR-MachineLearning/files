# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 04:31:18 2017

@author: Soomin
"""

# Imports
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

import h2o
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator 
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch 
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

from h2o.estimators.glm import H2OGeneralizedLinearEstimator # import GLM models
from h2o.grid.grid_search import H2OGridSearch   

#import xgboost as xgb
h2o.init(max_mem_size='6G') # give h2o as much memory as possible
h2o.no_progress() # turn off h2o progress bars

# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#%matplotlib inline
#njobs = 4




def get_type_lists(frame, rejects):

    """Creates lists of numeric and categorical variables.
    :param frame: The frame from which to determine types.
    :param rejects: Variable names not to be included in returned lists.
    :return: Tuple of lists for numeric and categorical variables in the frame.
    """
    
    nums, cats = [], []
    for key, val in frame.types.items():
        if key not in rejects:
            if val == 'enum':
                cats.append(key)
            else: 
                nums.append(key)
                
    print('Numeric =', nums)                
    print()
    print('Categorical =', cats)
    
    return nums, cats


def get_type_lists2(pandas_frame, rejects):
    
    categorical_features = pandas_frame.select_dtypes(include = ["object"]).columns
    numerical_features = pandas_frame.select_dtypes(exclude = ["object"]).columns
    #numerical_features = numerical_features.drop('SalePrice')
    numerical_features = numerical_features.drop(rejects)
                
    print('Numeric =', numerical_features)                
    print()
    print('Categorical =', categorical_features)
    
    return numerical_features, categorical_features


def glm_grid(X, y, train, valid):
    
    """ Wrapper function for penalized GLM with alpha and lambda search.
    
    :param X: List of inputs.
    :param y: Name of target variable.
    :param train: Name of training H2OFrame.
    :param valid: Name of validation H2OFrame.
    :return: Best H2Omodel from H2OGeneralizedLinearEstimator

    """
    
    alpha_opts = [0.01, 0.25, 0.5, 0.99] # always keep some L2
    hyper_parameters = {'alpha': alpha_opts}

    # initialize grid search
    grid = H2OGridSearch(
        H2OGeneralizedLinearEstimator(
            family="gaussian",
            standardize=True,
            lambda_search=True,
            seed=12345),
        hyper_params=hyper_parameters)
    
    # train grid
    grid.train(y=y,
               x=X, 
               training_frame=train,
               validation_frame=valid)

    # show grid search results
    print(grid.show())

    best = grid.get_grid()[0]
    print(best)
    
    # plot top frame values
    yhat_frame = valid.cbind(best.predict(valid))
    print(yhat_frame[0:10, [y, 'predict']])

    # plot sorted predictions
    yhat_frame_df = yhat_frame[[y, 'predict']].as_data_frame()
    yhat_frame_df.sort_values(by='predict', inplace=True)
    yhat_frame_df.reset_index(inplace=True, drop=True)
   # _ = yhat_frame_df.plot(title='Ranked Predictions Plot')
    
    # select best model
    return best


def target_encoder(training_frame, test_frame, x, y, lambda_=0.15, threshold=150, test=False):

    """ Applies simple target encoding to categorical variables.

    :param training_frame: Training frame which to create target means and to be encoded.
    :param test_frame: Test frame to be encoded using information from training frame.
    :param x: Name of input variable to be encoded.
    :param y: Name of target variable to use for encoding.
    :param lambda_: Balance between level mean and overall mean for small groups.
    :param threshold: Number below which a level is considered small enough to be shrunken.
    :param test: Whether or not to print the row_val_dict for testing purposes.
    :return: Tuple of encoded variable from train and test set as H2OFrames.

    """

    # convert to pandas
    trdf = training_frame.as_data_frame().loc[:, [x,y]] # df
    tss = test_frame.as_data_frame().loc[:, x]          # series


    # create dictionary of level:encode val

    encode_name = x + '_Tencode'
    overall_mean = trdf[y].mean()
    row_val_dict = {}

    for level in trdf[x].unique():
        level_df = trdf[trdf[x] == level][y]
        level_n = level_df.shape[0]
        level_mean = level_df.mean()
        if level_n >= threshold:
            row_val_dict[level] = level_mean
        else:
            row_val_dict[level] = ((1 - lambda_) * level_mean) +\
                                  (lambda_ * overall_mean)

    row_val_dict[np.nan] = overall_mean # handle missing values

    if test:
        print(row_val_dict)

    # apply the transform to training data
    trdf[encode_name] = trdf[x].apply(lambda i: row_val_dict[i])

    # apply the transform to test data
    tsdf = pd.DataFrame(columns=[x, encode_name])
    tsdf[x] = tss
    tsdf.loc[:, encode_name] = overall_mean # handle previously unseen values
    # handle values that are seen in tsdf but not row_val_dict
    for i, col_i in enumerate(tsdf[x]):
        try:
            row_val_dict[col_i]
        except:
            # a value that appeared in tsdf isn't in the row_val_dict so just
            # make it the overall_mean
            row_val_dict[col_i] = overall_mean
    tsdf[encode_name] = tsdf[x].apply(lambda i: row_val_dict[i])


    # convert back to H2O

    trdf = h2o.H2OFrame(trdf[encode_name].as_matrix())
    trdf.columns = [encode_name]

    tsdf = h2o.H2OFrame(tsdf[encode_name].as_matrix())
    tsdf.columns = [encode_name]

    return (trdf, tsdf)


def feature_combiner(pandas_frame, nums):
    
    """ Combines numeric features using simple arithmatic operations.
    
    :param pandas_frame: Training frame from which to generate features and onto which generated 
                           feeatures will be cbound.
    :param test_frame: Test frame from which to generate features and onto which generated 
                       feeatures will be cbound.
    :param nums: List of original numeric features from which to generate combined features.
    
    """
    total = len(nums)
    # convert to pandas
    #train_df = pandas_frame.as_data_frame()
    train_df = pandas_frame
    for i, col_i in enumerate(nums):
        print('Combining: ' + col_i + ' (' + str(i+1) + '/' + str(total) + ') ...')        
        for j, col_j in enumerate(nums):
            
            # don't repeat (i*j = j*i)
            if i < j:
                
                # convert to pandas
                col_i_train_df = train_df[col_i]
                col_j_train_df = train_df[col_j]
 
                # multiply, convert back to h2o
                train_df[str(col_i + '|' + col_j)] = col_i_train_df.values*col_j_train_df.values
    print('Done.')

    # convert back to h2o
    print('Converting to H2OFrame ...')
    #pandas_frame = h2o.H2OFrame(train_df)
    #pandas_frame.columns = list(train_df)
    print('Done.')
    print()
    
    # conserve memory 
    #del train_df
    #return pandas_frame
    return train_df



# Get data
pandas_train = pd.read_csv('C:/Users/Soomin/Google Drive/01. MSBA/03. Summer 2017/Machine Learning/Project/House Prices/train.csv')
pandas_train.shape


# Log transform the target for official scoring
pandas_train.SalePrice = np.log1p(pandas_train.SalePrice)
y = pandas_train.SalePrice

# << Preprocessing >>

# Lotfrontage
temp = pandas_train.groupby('Neighborhood', as_index=False)['LotFrontage'].median()
temp = temp.rename(columns={"LotFrontage":"LotFrontage2"})
pandas_train = pd.merge(pandas_train, temp, how='left', on='Neighborhood')
pandas_train['LotFrontage'][pandas_train['LotFrontage'].isnull()] = pandas_train['LotFrontage2'][pandas_train['LotFrontage'].isnull()]
pandas_train = pandas_train.drop('LotFrontage2', axis=1)


# Alley
pandas_train["Alley"].fillna("None", inplace=True)


# MasVnrType, MasVnrArea
pandas_train['MasVnrType'].fillna(pandas_train['MasVnrType'].value_counts().index[0],inplace=True)
pandas_train['MasVnrArea'].fillna(pandas_train['MasVnrArea'].mode()[0],inplace=True)

# Basement related
basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']

pandas_train["BsmtQual"].fillna("None", inplace=True)
pandas_train["BsmtCond"].fillna("None", inplace=True)
pandas_train["BsmtExposure"].fillna("None", inplace=True)
pandas_train["BsmtFinType1"].fillna("None", inplace=True)
pandas_train["BsmtFinSF1"].fillna(0, inplace=True)
pandas_train["BsmtFinType2"].fillna("None", inplace=True)
pandas_train["BsmtFinSF2"].fillna(0, inplace=True)
pandas_train["BsmtUnfSF"].fillna(0, inplace=True)


# Electrical
pandas_train["Electrical"].fillna("SBrkr", inplace=True)


# FireplaceQu
pandas_train["FireplaceQu"].fillna("None", inplace=True)


# Garage related
garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']

pandas_train["GarageType"].fillna("None", inplace=True)
pandas_train["GarageQual"].fillna("None", inplace=True)
pandas_train["GarageCond"].fillna("None", inplace=True)
pandas_train["GarageFinish"].fillna("None", inplace=True)
pandas_train["GarageCars"].fillna(0, inplace=True)
pandas_train["GarageArea"].fillna(0, inplace=True)


# GarageYrBlt Binning
minval = pandas_train['GarageYrBlt'].min()
maxval = pandas_train['GarageYrBlt'].max()+1
binlist=[0,minval,1920,1940,1960,1980,2000,maxval]
pandas_train['GarageYrBlt'].fillna(0,inplace=True)
#pandas_train['GarageYrBltBins'] = pd.cut(pandas_train['GarageYrBlt'],binlist,include_lowest=True,right=False)


# PoolQC
pandas_train["PoolQC"].fillna("None", inplace=True)


# Fence, MiscFeature
pandas_train["Fence"].fillna("None", inplace=True)
pandas_train["MiscFeature"].fillna("None", inplace=True)


# ------------------------------------------------------
def show_missing(pandas_frame):
    missing = pandas_frame.columns[pandas_frame.isnull().any()].tolist()
    return missing

show_missing(pandas_train)

temp_train = pandas_train


### Set categorical vars-----------------------------------------
pandas_train = pandas_train.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })



# Encode some categorical features as ordered numbers when there is information in the order
pandas_train = pandas_train.replace({
                        "Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"None" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       # "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoneSeWa" : 2, "NoneSewr" : 3, "AllPub" : 4}}
                     )

pandas_train.head(5)

# Create new features
# 1* Simplifications of existing features
#pandas_train["SimplOverallQual"] = pandas_train.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
#                                                       4 : 2, 5 : 2, 6 : 2, # average
#                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
#                                                      })
#pandas_train["SimplOverallCond"] = pandas_train.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
#                                                       4 : 2, 5 : 2, 6 : 2, # average
#                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
#                                                      })
#pandas_train["SimplPoolQC"] = pandas_train.PoolQC.replace({1 : 1, 2 : 1, # average
#                                             3 : 2, 4 : 2 # good
#                                            })
#pandas_train["SimplGarageCond"] = pandas_train.GarageCond.replace({1 : 1, # bad
#                                                     2 : 1, 3 : 1, # average
#                                                     4 : 2, 5 : 2 # good
#                                                    })
#pandas_train["SimplGarageQual"] = pandas_train.GarageQual.replace({1 : 1, # bad
#                                                     2 : 1, 3 : 1, # average
#                                                     4 : 2, 5 : 2 # good
#                                                    })
#pandas_train["SimplFireplaceQu"] = pandas_train.FireplaceQu.replace({1 : 1, # bad
#                                                       2 : 1, 3 : 1, # average
#                                                       4 : 2, 5 : 2 # good
#                                                      })
#pandas_train["SimplFireplaceQu"] = pandas_train.FireplaceQu.replace({1 : 1, # bad
#                                                       2 : 1, 3 : 1, # average
#                                                       4 : 2, 5 : 2 # good
#                                                      })
#pandas_train["SimplFunctional"] = pandas_train.Functional.replace({1 : 1, 2 : 1, # bad
#                                                     3 : 2, 4 : 2, # major
#                                                     5 : 3, 6 : 3, 7 : 3, # minor
#                                                     8 : 4 # typical
#                                                    })
#pandas_train["SimplKitchenQual"] = pandas_train.KitchenQual.replace({1 : 1, # bad
#                                                       2 : 1, 3 : 1, # average
#                                                       4 : 2, 5 : 2 # good
#                                                      })
#pandas_train["SimplHeatingQC"] = pandas_train.HeatingQC.replace({1 : 1, # bad
#                                                   2 : 1, 3 : 1, # average
#                                                   4 : 2, 5 : 2 # good
#                                                  })
#pandas_train["SimplBsmtFinType1"] = pandas_train.BsmtFinType1.replace({1 : 1, # unfinished
#                                                         2 : 1, 3 : 1, # rec room
#                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
#                                                        })
#pandas_train["SimplBsmtFinType2"] = pandas_train.BsmtFinType2.replace({1 : 1, # unfinished
#                                                         2 : 1, 3 : 1, # rec room
#                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
#                                                        })
#pandas_train["SimplBsmtCond"] = pandas_train.BsmtCond.replace({1 : 1, # bad
#                                                 2 : 1, 3 : 1, # average
#                                                 4 : 2, 5 : 2 # good
#                                                })
#pandas_train["SimplBsmtQual"] = pandas_train.BsmtQual.replace({1 : 1, # bad
#                                                 2 : 1, 3 : 1, # average
#                                                 4 : 2, 5 : 2 # good
#                                                })
#pandas_train["SimplExterCond"] = pandas_train.ExterCond.replace({1 : 1, # bad
#                                                   2 : 1, 3 : 1, # average
#                                                   4 : 2, 5 : 2 # good
#                                                  })
#pandas_train["SimplExterQual"] = pandas_train.ExterQual.replace({1 : 1, # bad
#                                                   2 : 1, 3 : 1, # average
#                                                   4 : 2, 5 : 2 # good
#                                                  })

# 2* Combinations of existing features
   

# Overall quality of the house
#pandas_train["OverallGrade"] = pandas_train["OverallQual"] * pandas_train["OverallCond"]
## Overall quality of the garage
#pandas_train["GarageGrade"] = pandas_train["GarageQual"] * pandas_train["GarageCond"]
## Overall quality of the exterior
#pandas_train["ExterGrade"] = pandas_train["ExterQual"] * pandas_train["ExterCond"]
## Overall kitchen score
#pandas_train["KitchenScore"] = pandas_train["KitchenAbvGr"] * pandas_train["KitchenQual"]
## Overall fireplace score
#pandas_train["FireplaceScore"] = pandas_train["Fireplaces"] * pandas_train["FireplaceQu"]
## Overall garage score
#pandas_train["GarageScore"] = pandas_train["GarageArea"] * pandas_train["GarageQual"]
## Overall pool score
#pandas_train["PoolScore"] = pandas_train["PoolArea"] * pandas_train["PoolQC"]
## Simplified overall quality of the house
#pandas_train["SimplOverallGrade"] = pandas_train["SimplOverallQual"] * pandas_train["SimplOverallCond"]
## Simplified overall quality of the exterior
#pandas_train["SimplExterGrade"] = pandas_train["SimplExterQual"] * pandas_train["SimplExterCond"]
## Simplified overall pool score
#pandas_train["SimplPoolScore"] = pandas_train["PoolArea"] * pandas_train["SimplPoolQC"]
## Simplified overall garage score
#pandas_train["SimplGarageScore"] = pandas_train["GarageArea"] * pandas_train["SimplGarageQual"]
## Simplified overall fireplace score
#pandas_train["SimplFireplaceScore"] = pandas_train["Fireplaces"] * pandas_train["SimplFireplaceQu"]
## Simplified overall kitchen score
#pandas_train["SimplKitchenScore"] = pandas_train["KitchenAbvGr"] * pandas_train["SimplKitchenQual"]
## Total number of bathrooms
#pandas_train["TotalBath"] = pandas_train["BsmtFullBath"] + (0.5 * pandas_train["BsmtHalfBath"]) + pandas_train["FullBath"] + (0.5 * pandas_train["HalfBath"])
## Total SF for house (incl. basement)
#pandas_train["AllSF"] = pandas_train["GrLivArea"] + pandas_train["TotalBsmtSF"]
## Total SF for 1st + 2nd floors
#pandas_train["AllFlrsSF"] = pandas_train["1stFlrSF"] + pandas_train["2ndFlrSF"]
## Total SF for porch
#pandas_train["AllPorchSF"] = pandas_train["OpenPorchSF"] + pandas_train["EnclosedPorch"] + pandas_train["3SsnPorch"] + pandas_train["ScreenPorch"]
## Has masonry veneer or not
#pandas_train["HasMasVnr"] = pandas_train.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
#                                               "Stone" : 1, "None" : 0})
## House completed before sale or not
#pandas_train["BoughtOffPlan"] = pandas_train.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
#                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})
    
    

    
    



# Create new features
# Polynomials on the top 10 existing features
#pandas_train["OverallQual|YearBuilt-2"] = pandas_train["OverallQual|YearBuilt"] ** 2
#pandas_train["OverallQual|YearBuilt-3"] = pandas_train["OverallQual|YearBuilt"] ** 3
#pandas_train["OverallQual|YearBuilt-Sq"] = np.sqrt(pandas_train["OverallQual|YearBuilt"])
#pandas_train["OverallQual|YearRemodAdd-2"] = pandas_train["OverallQual|YearRemodAdd"] ** 2
#pandas_train["OverallQual|YearRemodAdd-3"] = pandas_train["OverallQual|YearRemodAdd"] ** 3
#pandas_train["OverallQual|YearRemodAdd-Sq"] = np.sqrt(pandas_train["OverallQual|YearRemodAdd"])
#pandas_train["OverallQual-s2"] = pandas_train["OverallQual"] ** 2
#pandas_train["OverallQual-s3"] = pandas_train["OverallQual"] ** 3
#pandas_train["OverallQual-Sq"] = np.sqrt(pandas_train["OverallQual"])
#pandas_train["OverallQual|YrSold-2"] = pandas_train["OverallQual|YrSold"] ** 2
#pandas_train["OverallQual|YrSold-3"] = pandas_train["OverallQual|YrSold"] ** 3
#pandas_train["OverallQual|YrSold-Sq"] = np.sqrt(pandas_train["OverallQual|YrSold"])
#pandas_train["OverallQual|GarageCars-2"] = pandas_train["OverallQual|GarageCars"] ** 2
#pandas_train["OverallQual|GarageCars-3"] = pandas_train["OverallQual|GarageCars"] ** 3
#pandas_train["OverallQual|GarageCars-Sq"] = np.sqrt(pandas_train["OverallQual|GarageCars"])
#pandas_train["OverallQual|KitchenQual-2"] = pandas_train["OverallQual|KitchenQual"] ** 2
#pandas_train["OverallQual|KitchenQual-3"] = pandas_train["OverallQual|KitchenQual"] ** 3
#pandas_train["OverallQual|KitchenQual-Sq"] = np.sqrt(pandas_train["OverallQual|KitchenQual"])
#pandas_train["Street|OverallQual-2"] = pandas_train["Street|OverallQual"] ** 2
#pandas_train["Street|OverallQual-3"] = pandas_train["Street|OverallQual"] ** 3
#pandas_train["Street|OverallQual-Sq"] = np.sqrt(pandas_train["Street|OverallQual"])
#pandas_train["OverallQual|ExterQual-s2"] = pandas_train["OverallQual|ExterQual"] ** 2
#pandas_train["OverallQual|ExterQual-s3"] = pandas_train["OverallQual|ExterQual"] ** 3
#pandas_train["OverallQual|ExterQual-Sq"] = np.sqrt(pandas_train["OverallQual|ExterQual"])
#pandas_train["OverallQual|BsmtQual-s2"] = pandas_train["OverallQual|BsmtQual"] ** 2
#pandas_train["OverallQual|BsmtQual-s3"] = pandas_train["OverallQual|BsmtQual"] ** 3
#pandas_train["OverallQual|BsmtQual-Sq"] = np.sqrt(pandas_train["OverallQual|BsmtQual"])
#pandas_train["OverallQual|GrLivArea-sq"] = pandas_train["OverallQual|GrLivArea"] ** 2
#pandas_train["OverallQual|GrLivArea-2"] = pandas_train["OverallQual|GrLivArea"] ** 3
#pandas_train["OverallQual|GrLivArea-3"] = np.sqrt(pandas_train["OverallQual|GrLivArea"])



#
#
#pandas_train["BsmtQual|GrLivArea-2"] = pandas_train["BsmtQual|GrLivArea"] ** 2
#pandas_train["BsmtQual|GrLivArea-3"] = pandas_train["BsmtQual|GrLivArea"] ** 3
#pandas_train["BsmtQual|GrLivArea-Sq"] = np.sqrt(pandas_train["BsmtQual|GrLivArea"])
#
#pandas_train["GrLivArea|KitchenQual-2"] = pandas_train["GrLivArea|KitchenQual"] ** 2
#pandas_train["GrLivArea|KitchenQual-3"] = pandas_train["GrLivArea|KitchenQual"] ** 3
#pandas_train["GrLivArea|KitchenQual-Sq"] = np.sqrt(pandas_train["GrLivArea|KitchenQual"])
#
#pandas_train["ExterQual|GrLivArea-2"] = pandas_train["ExterQual|GrLivArea"] ** 2
#pandas_train["ExterQual|GrLivArea-3"] = pandas_train["ExterQual|GrLivArea"] ** 3
#pandas_train["ExterQual|GrLivArea-Sq"] = np.sqrt(pandas_train["ExterQual|GrLivArea"])
#
#pandas_train["GrLivArea|GarageCars-2"] = pandas_train["GrLivArea|GarageCars"] ** 2
#pandas_train["GrLivArea|GarageCars-3"] = pandas_train["GrLivArea|GarageCars"] ** 3
#pandas_train["GrLivArea|GarageCars-Sq"] = np.sqrt(pandas_train["GrLivArea|GarageCars"])
#pandas_train["OverallQual|TotRmsAbvGrd-2"] = pandas_train["OverallQual|TotRmsAbvGrd"] ** 2
#pandas_train["OverallQual|TotRmsAbvGrd-3"] = pandas_train["OverallQual|TotRmsAbvGrd"] ** 3
#pandas_train["OverallQual|TotRmsAbvGrd-Sq"] = np.sqrt(pandas_train["OverallQual|TotRmsAbvGrd"])



#pandas_train["OverallQual-s2"] = pandas_train["OverallQual"] ** 2
#pandas_train["OverallQual-s3"] = pandas_train["OverallQual"] ** 3
#pandas_train["OverallQual-Sq"] = np.sqrt(pandas_train["OverallQual"])
#pandas_train["AllSF-2"] = pandas_train["AllSF"] ** 2
#pandas_train["AllSF-3"] = pandas_train["AllSF"] ** 3
#pandas_train["AllSF-Sq"] = np.sqrt(pandas_train["AllSF"])
#pandas_train["AllFlrsSF-2"] = pandas_train["AllFlrsSF"] ** 2
#pandas_train["AllFlrsSF-3"] = pandas_train["AllFlrsSF"] ** 3
#pandas_train["AllFlrsSF-Sq"] = np.sqrt(pandas_train["AllFlrsSF"])
#pandas_train["GrLivArea-2"] = pandas_train["GrLivArea"] ** 2
#pandas_train["GrLivArea-3"] = pandas_train["GrLivArea"] ** 3
#pandas_train["GrLivArea-Sq"] = np.sqrt(pandas_train["GrLivArea"])
#pandas_train["SimplOverallQual-s2"] = pandas_train["SimplOverallQual"] ** 2
#pandas_train["SimplOverallQual-s3"] = pandas_train["SimplOverallQual"] ** 3
#pandas_train["SimplOverallQual-Sq"] = np.sqrt(pandas_train["SimplOverallQual"])
#pandas_train["ExterQual-2"] = pandas_train["ExterQual"] ** 2
#pandas_train["ExterQual-3"] = pandas_train["ExterQual"] ** 3
#pandas_train["ExterQual-Sq"] = np.sqrt(pandas_train["ExterQual"])
#pandas_train["GarageCars-2"] = pandas_train["GarageCars"] ** 2
#pandas_train["GarageCars-3"] = pandas_train["GarageCars"] ** 3
#pandas_train["GarageCars-Sq"] = np.sqrt(pandas_train["GarageCars"])
#pandas_train["TotalBath-2"] = pandas_train["TotalBath"] ** 2
#pandas_train["TotalBath-3"] = pandas_train["TotalBath"] ** 3
#pandas_train["TotalBath-Sq"] = np.sqrt(pandas_train["TotalBath"])
#pandas_train["KitchenQual-2"] = pandas_train["KitchenQual"] ** 2
#pandas_train["KitchenQual-3"] = pandas_train["KitchenQual"] ** 3
#pandas_train["KitchenQual-Sq"] = np.sqrt(pandas_train["KitchenQual"])
#pandas_train["GarageScore-2"] = pandas_train["GarageScore"] ** 2
#pandas_train["GarageScore-3"] = pandas_train["GarageScore"] ** 3
#pandas_train["GarageScore-Sq"] = np.sqrt(pandas_train["GarageScore"])

# Differentiate numerical features (minus the target) and categorical features
exclude = ['Id','SalePrice']
nums, cats =  get_type_lists2(pandas_train, exclude)




#numerical_features = numerical_features.drop("Id")
#numerical_features = numerical_features.drop("GarageYrBltBins")


print("Numerical features : " + str(len(nums)))
print("Categorical features : " + str(len(cats)))
train_num = pandas_train[nums]
train_cat = pandas_train[cats]

# Handle remaining missing values for numerical features by using median as replacement
print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
train_num = train_num.fillna(train_num.median())
print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))


## Log transform of the skewed numerical features to lessen impact of outliers
## Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
## As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
skewness = train_num.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index
train_num[skewed_features] = np.log1p(train_num[skewed_features])


# Create dummy features for categorical values via one-hot encoding
print("NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))
train_cat = pd.get_dummies(train_cat)
print("Remaining NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))



# Join categorical and numerical features
train = pd.concat([train_num, train_cat, pandas_train[['Id','SalePrice']]], axis = 1)
print("New number of features : " + str(train.shape[1]))


# Feature combine here
train = feature_combiner(train, nums)
train.shape 


nums, cats =  get_type_lists2(train, exclude)


# << Run the models >>
trainset, validset = h2o.H2OFrame(train).split_frame([0.7], seed=12345) #Convert back to H2O frame and split the frames
print(trainset.shape)
print(validset.shape)
finalnums, finalcats =  get_type_lists(trainset, exclude)

half_train, other_half_train = trainset.split_frame([0.5], seed=12345)
half_valid, other_half_valid = validset.split_frame([0.5], seed=12345)
print(half_train.shape)
print(half_valid.shape)
print(other_half_train.shape)
print(other_half_valid.shape)

# Standardize numerical features
#stdSc = StandardScaler()
#trainset.loc[:, finalnums] = stdSc.fit_transform(trainset.loc[:, finalnums])
#validset.loc[:, finalnums] = stdSc.transform(validset.loc[:, finalnums])


variables = finalnums+finalcats


glm0_0 = glm_grid(variables, 'SalePrice',  half_train, half_valid)
glm0_1 = glm_grid(variables, 'SalePrice',  other_half_train, other_half_valid)
glm0_2 = glm_grid(variables, 'SalePrice',  trainset, validset)






def glm_grid(X, y, train, valid):
    
    """ Wrapper function for penalized GLM with alpha and lambda search.
    
    :param X: List of inputs.
    :param y: Name of target variable.
    :param train: Name of training H2OFrame.
    :param valid: Name of validation H2OFrame.
    :return: Best H2Omodel from H2OGeneralizedLinearEstimator

    """
    
    alpha_opts = [0.01, 0.25, 0.5, 0.99] # always keep some L2
    hyper_parameters = {'alpha': alpha_opts}

    # initialize grid search
    grid = H2OGridSearch(
        H2OGeneralizedLinearEstimator(
            family="gaussian",
            standardize=True,
            lambda_search=True,
            seed=12345),
        hyper_params=hyper_parameters)
    
    # train grid
    grid.train(y=y,
               x=X, 
               training_frame=train,
               validation_frame=valid)

    # show grid search results
    print(grid.show())

    best = grid.get_grid()[0]
    print(best)
    
    # plot top frame values
    yhat_frame = valid.cbind(best.predict(valid))
    print(yhat_frame[0:10, [y, 'predict']])

    # plot sorted predictions
    yhat_frame_df = yhat_frame[[y, 'predict']].as_data_frame()
    yhat_frame_df.sort_values(by='predict', inplace=True)
    yhat_frame_df.reset_index(inplace=True, drop=True)
   # _ = yhat_frame_df.plot(title='Ranked Predictions Plot')
    
    # select best model
    return best
