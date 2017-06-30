# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:20:13 2017

@author: Soomin
"""

import time
import os
import re

def gen_submission(preds, test):

    """ Generates submission file for Kaggle House Prices contest.
    
    :param preds: Column vector of predictions.
    :param test: Test data.
    
    """
    
    # create time stamp
    time_stamp = re.sub('[: ]', '_', time.asctime())

    # create predictions column
    sub = test['Id'].cbind(preds.exp())
    sub.columns = ['Id', 'SalePrice']
    
    # save file for submission
    sub_fname = str(time_stamp) + '.csv'
    h2o.download_csv(sub, sub_fname)
    
    
    
def gen_submission_glm(model, test):

    """ Generates submission file for Kaggle House Prices contest.
    
    :param model: Model with which to score test data.
    :param test: Test data.
    
    """
    
    # create time stamp
    time_stamp = re.sub('[: ]', '_', time.asctime())

    # create predictions column
    sub = test['Id'].cbind(model.predict(test).exp())
    sub.columns = ['Id', 'SalePrice']
    
    # save file for submission
    sub_fname = str(time_stamp) + '.csv'
    h2o.download_csv(sub, sub_fname)


def pred_blender(dir_, files):
    
    """ Performs simple blending of prediction files. 
    
    :param dir_: Directory in which files to be read are stored.
    :param files: List of prediction files to be blended.
    
    """
    
    # read predictions in files list and cbind
    for i, file in enumerate(files):
        if i == 0:
            df = pd.read_csv(dir_ + os.sep + file).drop('SalePrice', axis=1)
        col = pd.read_csv(dir_ + os.sep + file).drop('Id', axis=1)
        col.columns = ['SalePrice' + str(i)]
        df = pd.concat([df, col], axis=1)
        
    # create mean prediction    
    df['mean'] = df.iloc[:, 1:].mean(axis=1)
    print(df.head())
        
    # create time stamp
    time_stamp = re.sub('[: ]', '_', time.asctime())        
        
    # write new submission file    
    df = df[['Id', 'mean']]
    df.columns = ['Id', 'SalePrice']
    
    # save file for submission
    sub_fname = str(time_stamp) + '.csv'
    df.to_csv(sub_fname, index=False)
    


# Get data
pandas_test = pd.read_csv('C:/Users/Soomin/Google Drive/01. MSBA/03. Summer 2017/Machine Learning/Project/House Prices/test.csv')
pandas_test.shape


# << Preprocessing >>

# Lotfrontage
temp = pandas_test.groupby('Neighborhood', as_index=False)['LotFrontage'].median()
temp = temp.rename(columns={"LotFrontage":"LotFrontage2"})
pandas_test = pd.merge(pandas_test, temp, how='left', on='Neighborhood')
pandas_test['LotFrontage'][pandas_test['LotFrontage'].isnull()] = pandas_test['LotFrontage2'][pandas_test['LotFrontage'].isnull()]
pandas_test = pandas_test.drop('LotFrontage2', axis=1)


# Alley
pandas_test["Alley"].fillna("None", inplace=True)


# MasVnrType, MasVnrArea
pandas_test['MasVnrType'].fillna(pandas_test['MasVnrType'].value_counts().index[0],inplace=True)
pandas_test['MasVnrArea'].fillna(pandas_test['MasVnrArea'].mode()[0],inplace=True)

# Basement related
basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']

pandas_test["BsmtQual"].fillna("None", inplace=True)
pandas_test["BsmtCond"].fillna("None", inplace=True)
pandas_test["BsmtExposure"].fillna("None", inplace=True)
pandas_test["BsmtFinType1"].fillna("None", inplace=True)
pandas_test["BsmtFinSF1"].fillna(0, inplace=True)
pandas_test["BsmtFinType2"].fillna("None", inplace=True)
pandas_test["BsmtFinSF2"].fillna(0, inplace=True)
pandas_test["BsmtUnfSF"].fillna(0, inplace=True)


# Electrical
pandas_test["Electrical"].fillna("SBrkr", inplace=True)


# FireplaceQu
pandas_test["FireplaceQu"].fillna("None", inplace=True)


# Garage related
garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']

pandas_test["GarageType"].fillna("None", inplace=True)
pandas_test["GarageQual"].fillna("None", inplace=True)
pandas_test["GarageCond"].fillna("None", inplace=True)
pandas_test["GarageFinish"].fillna("None", inplace=True)
pandas_test["GarageCars"].fillna(0, inplace=True)
pandas_test["GarageArea"].fillna(0, inplace=True)


# GarageYrBlt Binning
minval = pandas_test['GarageYrBlt'].min()
maxval = pandas_test['GarageYrBlt'].max()+1
binlist=[0,minval,1920,1940,1960,1980,2000,maxval]
pandas_test['GarageYrBlt'].fillna(0,inplace=True)
#pandas_test['GarageYrBltBins'] = pd.cut(pandas_test['GarageYrBlt'],binlist,include_lowest=True,right=False)


# PoolQC
pandas_test["PoolQC"].fillna("None", inplace=True)


# Fence, MiscFeature
pandas_test["Fence"].fillna("None", inplace=True)
pandas_test["MiscFeature"].fillna("None", inplace=True)


# ------------------------------------------------------
def show_missing(pandas_frame):
    missing = pandas_frame.columns[pandas_frame.isnull().any()].tolist()
    return missing

show_missing(pandas_test)


### Set categorical vars-----------------------------------------
pandas_test = pandas_test.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })



# Encode some categorical features as ordered numbers when there is information in the order
pandas_test = pandas_test.replace({
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

pandas_test.head(5)


# Differentiate numerical features (minus the target) and categorical features
exclude = ['Id']
nums, cats =  get_type_lists2(pandas_test, exclude)




#numerical_features = numerical_features.drop("Id")
#numerical_features = numerical_features.drop("GarageYrBltBins")


print("Numerical features : " + str(len(nums)))
print("Categorical features : " + str(len(cats)))
test_num = pandas_test[nums]
test_cat = pandas_test[cats]

# Handle remaining missing values for numerical features by using median as replacement
print("NAs for numerical features in train : " + str(test_num.isnull().values.sum()))
test_num = test_num.fillna(test_num.median())
print("Remaining NAs for numerical features in train : " + str(test_num.isnull().values.sum()))


## Log transform of the skewed numerical features to lessen impact of outliers
## Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
## As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
#skewness = test_num.apply(lambda x: skew(x))
#skewness = skewness[abs(skewness) > 0.5]
#print(str(skewness.shape[0]) + " skewed numerical features to log transform")
#skewed_features = skewness.index
#test_num[skewed_features] = np.log1p(test_num[skewed_features])


## Encoding
# Create dummy features for categorical values via one-hot encoding
print("NAs for categorical features in train : " + str(test_cat.isnull().values.sum()))
test_cat = pd.get_dummies(test_cat)
# Join categorical and numerical features
test = pd.concat([test_num, test_cat, pandas_test[['Id']]], axis = 1)
print("New number of features : " + str(test.shape[1]))


# Feature combine here
test = feature_combiner(test, nums)
test.shape 



# now remove columns in encoded test not in encoded train and valid
# (they different b/c of different levels in variables)
train_diff_cols = list(set(train.columns) - set(test.columns))
train_diff_cols.remove('SalePrice')
train_diff_cols2 = list(set(test.columns) - set(train.columns))
train_diff_cols
len(train_diff_cols2)

train.drop(train_diff_cols, axis=1, inplace=True)
test.drop(train_diff_cols2, axis=1, inplace=True)


### << Modeling >>


## Standardize numerical features
#stdSc = StandardScaler()
#X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])
#X_test.loc[:, numerical_features] = stdSc.transform(X_test.loc[:, numerical_features])

trainset, validset = h2o.H2OFrame(train).split_frame([0.7], seed=12345) #Convert back to H2O frame and split the frames
print(trainset.shape)
print(validset.shape)
finalnums, finalcats =  get_type_lists(trainset, exclude)
variables = finalnums+finalcats

half_train, other_half_train = trainset.split_frame([0.5], seed=12345)
half_valid, other_half_valid = validset.split_frame([0.5], seed=12345)
print(half_train.shape)
print(half_valid.shape)
print(other_half_train.shape)
print(other_half_valid.shape)



############# GLM #######################
glm0_0 = glm_grid(variables, 'SalePrice',  half_train, half_valid)
glm0_1 = glm_grid(variables, 'SalePrice',  other_half_train, other_half_valid)
glm0_2 = glm_grid(variables, 'SalePrice',  trainset, validset)


############# Random Forest ####################### (Valid RMSE: 0.12620064722709262)

# initialize rf model
rf_model1 = H2ORandomForestEstimator(
    ntrees=10000,                    
    max_depth=20, 
    col_sample_rate_per_tree=0.1,
    sample_rate=0.8,
    stopping_rounds=50,
    score_each_iteration=True,
    nfolds=3,
    keep_cross_validation_predictions=True,
    seed=12345)           

# train rf model
rf_model1.train(
    x=variables,
    y='SalePrice',
    training_frame=trainset,
    validation_frame=validset)

# print model information
print(rf_model1)

rf_preds1_val = rf_model1.predict(validset)
#ranked_preds_plot('SalePrice', validset, rf_preds1_val) # valid RMSE not so hot ...
rf_preds1_test = rf_model1.predict(testset)
#gen_submission(rf_preds1_test) # 0.14574 public leaderboard


#### Extremely random trees model ((Valid RMSE: 0.12527112506898552)
# initialize extra trees model
ert_model1 = H2ORandomForestEstimator(
    ntrees=10000,                    
    max_depth=50, 
    col_sample_rate_per_tree=0.1,
    sample_rate=0.8,
    stopping_rounds=50,
    score_each_iteration=True,
    nfolds=3,
    keep_cross_validation_predictions=True,
    seed=12345,
    histogram_type='random') # <- this is what makes it ERT instead of RF

# train ert model
ert_model1.train(
    x=variables,
    y='SalePrice',
    training_frame=trainset,
    validation_frame=validset)

# print model information/create submission
print(ert_model1)
ert_preds1_val = ert_model1.predict(valid)
ranked_preds_plot('SalePrice', valid, ert_preds1_val) # valid RMSE not so hot ...
ert_preds1_test = ert_model1.predict(test)
gen_submission(ert_preds1_test) # 0.14855 public leaderboard



#### GBM (Valid RMSE: 0.12028460858594642) , Training : RMSE: 0.08066981974592213
# initialize H2O GBM
h2o_gbm_model = H2OGradientBoostingEstimator(
    ntrees = 10000,
    learn_rate = 0.005,
    sample_rate = 0.1, 
    col_sample_rate = 0.8,
    max_depth = 5,
    nfolds = 3,
    keep_cross_validation_predictions=True,
    stopping_rounds = 10,
    seed = 12345)

# execute training
h2o_gbm_model.train(x=variables,
                    y='SalePrice',
                    training_frame=trainset,
                    validation_frame=validset)

# print model information/create submission
print(h2o_gbm_model)
h2o_gbm_preds1_val = h2o_gbm_model.predict(valid)
ranked_preds_plot('SalePrice', valid, h2o_gbm_preds1_val) # better validation error
h2o_gbm_preds1_test = h2o_gbm_model.predict(test)
gen_submission(h2o_gbm_preds1_test) # 0.15062 public leaderboard



# initialize XGB GBM
h2o_xgb_model = H2OXGBoostEstimator(
    ntrees = 10000,
    learn_rate = 0.005,
    sample_rate = 0.1, 
    col_sample_rate = 0.8,
    max_depth = 5,
    nfolds = 3,
    keep_cross_validation_predictions=True,
    stopping_rounds = 10,
    seed = 12345)

# execute training 
h2o_xgb_model.train(x=variables,
                    y='SalePrice',
                    training_frame=trainset,
                    validation_frame=validset)

# print model information/create submission
print(h2o_xgb_model)
h2o_xgb_preds1_val = h2o_xgb_model.predict(valid)
ranked_preds_plot('SalePrice', valid, h2o_xgb_preds1_val) 
h2o_xgb_preds1_test = h2o_xgb_model.predict(test)
gen_submission(h2o_xgb_preds1_test) # 0.16494 on public leaderboard



#### Stacked Ensembles (Valid RMSE: 0.11681957554638625, Train RMSE: 0.053761172832335766)
stack = H2OStackedEnsembleEstimator(training_frame=trainset, 
                                    validation_frame=validset, 
                                    base_models=[rf_model1, ert_model1, 
                                                 h2o_gbm_model])
stack.train(x=variables,
            y='SalePrice',
            training_frame=trainset,
            validation_frame=validset)

# print model information/create submission
print(stack)
stack_preds1_val = stack.predict(validset)
ranked_preds_plot('SalePrice', validset, stack_preds1_val) 
stack_preds1_test = stack.predict(testset)
gen_submission(stack_preds1_test)
# 0.14630 on public leaderboard










### Create testset and predict for the testset and generate submission file
dummy_col = np.random.rand(testset.shape[0])
testset = testset.cbind(h2o.H2OFrame(dummy_col))
cols = testset.columns
cols[-1] = 'SalePrice'
testset.columns = cols

print(testset.shape)
print(trainset.shape)




### Running models
# ----------------- Running GLM --------------------------------
gen_submission_glm(glm0_0,testset) # Valid RMSE: 0.1325 #0.1216
gen_submission_glm(glm0_1,testset) # Valid RMSE: 0.1325 #0.1216
gen_submission_glm(glm0_2,testset) # Valid RMSE: 0.1325 #0.1216



pred_blender('C:\\Users\\Soomin', 
             ['Fri_Jun_30_15_12_19_2017.csv',
              'Fri_Jun_30_15_12_17_2017.csv',
              'Fri_Jun_30_15_12_18_2017.csv'])
