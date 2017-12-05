# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 10:38:37 2017

@author: Sindhuja
"""

import lightgbm as lgb 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000 # Jupyter notebook backend restricts number of points in plot
import pandas as pd
import scipy as scp
import csv
import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.graph_objs import *
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
import category_encoders as encoders
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators=10)

train_master = pd.read_csv('C:/Users/Sindhuja/Desktop/ML Project/train.csv',na_values='-1')
test_master = pd.read_csv('C:/Users/Sindhuja/Desktop/ML Project/test_porto.csv',na_values='-1')
test_result = pd.read_csv('C:/Users/Sindhuja/Desktop/ML Project/test_porto_result.csv',na_values='-1')
binary_columns = [s for s in list(train_master.columns.values) if '_bin' in s]
categorical_columns = [s for s in list(train_master.columns.values) if '_cat' in s]
non_continuous_feature_subs = ['_cat', '_bin', 'target', 'id']
continuous_columns = [s for s in list(train_master.columns.values) if all(x not in s for x in non_continuous_feature_subs)]
target_column = 'target'

ind_columns = [s for s in list(train_master.columns.values) if '_ind' in s]
car_columns = [s for s in list(train_master.columns.values) if '_car' in s]
calc_columns = [s for s in list(train_master.columns.values) if '_calc' in s]
reg_columns = [s for s in list(train_master.columns.values) if '_reg' in s]

na_count = train_master.isnull().sum()
na_columns = list(na_count[na_count>0].index.values)
print("columns with missing values:")
print(na_columns)
na_count.plot(kind = "bar")


#Find duplicates
train_master.drop_duplicates()
train_master.shape

#Define metadata
data = []
for f in train_master.columns:
    # Defining the role
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'
         
    # Defining the level
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif train_master[f].dtype == np.int64:
        level = 'ordinal'
    elif train_master[f].dtype == float:
        level = 'interval'

        
    # Initialize keep to True for all variables except for id
    keep = True
    if f == 'id':
        keep = False
    
    # Defining the data type 
    dtype = train_master[f].dtype
    
    # Creating a Dict that contains all the metadata for the variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)
    
meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)
#Descriptive statistics
pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()
#Interval variables
v = meta[(meta.level == 'interval') & (meta.keep)].index
train_master[v].describe()

#Ordinal variables
v1 = meta[(meta.level == 'ordinal') & (meta.keep)].index
train_master[v1].describe()

#Binary variables
v2 = meta[(meta.level == 'binary') & (meta.keep)].index
train_master[v2].describe()

#Checking missing values
vars_with_missing = []

for f in train_master.columns:
    missings = train_master[f].isnull().sum()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings/train_master.shape[0]
        
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
        
print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))


# Dropping the variables with too many missing values

vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train_master.drop(vars_to_drop, inplace=True, axis=1)
test_master.drop(vars_to_drop, inplace=True, axis=1)
meta.loc[(vars_to_drop),'keep'] = False  # Updating the meta

#test_master_1 = test_master[0:int(len(test_master)/5)][0:]
#test_master_2 = test_master[int(len(test_master)/2):int(len(test_master))][0:]


mean_imp = Imputer(missing_values=np.NaN, strategy='mean', axis=0)
#train_master = train_master.drop(train_master.index[train_master.ps_ind_04_cat.isnull()])
#train_master = train_master.drop(train_master.index[train_master.ps_ind_05_cat.isnull()])
#train_master = train_master.drop(train_master.index[train_master.ps_car_01_cat.isnull()])
#train_master = train_master.drop(train_master.index[train_master.ps_car_02_cat.isnull()])
#train_master = train_master.drop(train_master.index[train_master.ps_car_07_cat.isnull()])
#train_master = train_master.drop(train_master.index[train_master.ps_car_09_cat.isnull()])
#train_master = train_master.drop(train_master.index[train_master.ps_car_11.isnull()])
#train_master = train_master.drop(train_master.index[train_master.ps_car_12.isnull()])
#train_master = train_master.drop(train_master.index[train_master.ps_ind_02_cat.isnull()])
train_master['ps_reg_03'] = mean_imp.fit_transform(train_master[['ps_reg_03']]).ravel()
train_master['ps_car_14'] = mean_imp.fit_transform(train_master[['ps_car_14']]).ravel()
train_master['ps_ind_04_cat'] = mean_imp.fit_transform(train_master[['ps_ind_04_cat']]).ravel()
train_master['ps_ind_05_cat'] = mean_imp.fit_transform(train_master[['ps_ind_05_cat']]).ravel()
train_master['ps_car_01_cat'] = mean_imp.fit_transform(train_master[['ps_car_01_cat']]).ravel()
train_master['ps_car_02_cat'] = mean_imp.fit_transform(train_master[['ps_car_02_cat']]).ravel()
train_master['ps_car_07_cat'] = mean_imp.fit_transform(train_master[['ps_car_07_cat']]).ravel()
train_master['ps_car_09_cat'] = mean_imp.fit_transform(train_master[['ps_car_09_cat']]).ravel()
train_master['ps_car_11'] = mean_imp.fit_transform(train_master[['ps_car_11']]).ravel()
train_master['ps_car_12'] = mean_imp.fit_transform(train_master[['ps_car_12']]).ravel()
train_master['ps_ind_02_cat'] = mean_imp.fit_transform(train_master[['ps_ind_02_cat']]).ravel()

test_master['ps_reg_03'] = mean_imp.fit_transform(test_master[['ps_reg_03']]).ravel()
test_master['ps_car_14'] = mean_imp.fit_transform(test_master[['ps_car_14']]).ravel()
test_master['ps_ind_04_cat'] = mean_imp.fit_transform(test_master[['ps_ind_04_cat']]).ravel()
test_master['ps_ind_05_cat'] = mean_imp.fit_transform(test_master[['ps_ind_05_cat']]).ravel()
test_master['ps_car_01_cat'] = mean_imp.fit_transform(test_master[['ps_car_01_cat']]).ravel()
test_master['ps_car_02_cat'] = mean_imp.fit_transform(test_master[['ps_car_02_cat']]).ravel()
test_master['ps_car_07_cat'] = mean_imp.fit_transform(test_master[['ps_car_07_cat']]).ravel()
test_master['ps_car_09_cat'] = mean_imp.fit_transform(test_master[['ps_car_09_cat']]).ravel()
test_master['ps_car_11'] = mean_imp.fit_transform(test_master[['ps_car_11']]).ravel()
test_master['ps_car_12'] = mean_imp.fit_transform(test_master[['ps_car_12']]).ravel()
test_master['ps_ind_02_cat'] = mean_imp.fit_transform(test_master[['ps_ind_02_cat']]).ravel()

#test_master_2['ps_reg_03'] = mean_imp.fit_transform(test_master_2[['ps_reg_03']]).ravel()
#test_master_2['ps_car_14'] = mean_imp.fit_transform(test_master_2[['ps_car_14']]).ravel()
#test_master_2['ps_ind_04_cat'] = mean_imp.fit_transform(test_master_2[['ps_ind_04_cat']]).ravel()
#test_master_2['ps_ind_05_cat'] = mean_imp.fit_transform(test_master_2[['ps_ind_05_cat']]).ravel()
#test_master_2['ps_car_01_cat'] = mean_imp.fit_transform(test_master_2[['ps_car_01_cat']]).ravel()
#test_master_2['ps_car_02_cat'] = mean_imp.fit_transform(test_master_2[['ps_car_02_cat']]).ravel()
#test_master_2['ps_car_07_cat'] = mean_imp.fit_transform(test_master_2[['ps_car_07_cat']]).ravel()
#test_master_2['ps_car_09_cat'] = mean_imp.fit_transform(test_master_2[['ps_car_09_cat']]).ravel()
#test_master_2['ps_car_11'] = mean_imp.fit_transform(test_master_2[['ps_car_11']]).ravel()
#test_master_2['ps_car_12'] = mean_imp.fit_transform(test_master_2[['ps_car_12']]).ravel()
#test_master_2['ps_ind_02_cat'] = mean_imp.fit_transform(test_master_2[['ps_ind_02_cat']]).ravel()

train = train_master.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin'],axis=1)
test1 = test_master.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin'],axis=1)
#test2 = test_master_2.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin'],axis=1)

vars_to_drop_bin = ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin']
meta.loc[(vars_to_drop_bin),'keep'] = False

train = train.drop(calc_columns, axis=1)  
test1 = test1.drop(calc_columns, axis=1)
#test2 = test2.drop(calc_columns, axis=1)
meta.loc[(calc_columns),'keep'] = False


for f in train.columns:   
	print(f,'  ',np.ptp(train[f]))
    


desired_apriori=0.10
# Get the indices per target value
idx_0 = train[train['target'] == 0].index
idx_1 = train[train['target'] == 1].index
# Get original number of records per target value
nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])



# Calculate the undersampling rate and resulting number of records with target=0
undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
undersampled_nb_0 = int(undersampling_rate*nb_0)
print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))


# Randomly select records with target=0 to get at the desired a priori
undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

# Construct list with remaining indices
idx_list = list(undersampled_idx) + list(idx_1)

# Return undersample data frame
train = train.loc[idx_list].reset_index(drop=True)

x_val_fold = np.empty(len(train));

#Simple K-Fold cross validation. 10 folds.
n_splits=4
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
#cv = cross_validation.KFold(len(train), n_folds=4, indices=False)
results = []

#for traincv, testcv in cv:


#for fold_number, (train_ids, val_ids) in enumerate(
#    folds.split(train.drop(['id',target_column], axis=1), 
#                train[target_column])):
train = train.apply(lambda x: pd.to_numeric(x,errors='ignore'))
test1 = test1.apply(lambda x: pd.to_numeric(x,errors='ignore'))
#test2 = test2.apply(lambda x: pd.to_numeric(x,errors='ignore'))
yy=train['target']

#train = train.drop(['target','id'],axis=1);
#test1 = test1.drop(['id'],axis=1);

#test2 = test2.drop(['id'],axis=1);
def encode_cat_features(train_df, test_df, cat_cols, target_col_name, smoothing=1):
    prior = train_df[target_col_name].mean()
    probs_dict = {}
    for c in cat_cols:
        probs = train_df.groupby(c, as_index=False)[target_col_name].mean()
        probs['counts'] = train_df.groupby(c, as_index=False)[target_col_name].count()[[target_col_name]]
        probs['smoothing'] = 1 / (1 + np.exp(-(probs['counts'] - 1) / smoothing))
        probs['enc'] = prior * (1 - probs['smoothing']) + probs['target'] * probs['smoothing']
        probs_dict[c] = probs[[c,'enc']]
    return probs_dict


for train_ids, val_ids in folds.split(train.drop(['id',target_column], axis=1),train[target_column]):
    
    X = train.iloc[train_ids]
    y=yy.iloc[train_ids]
    
    X_val = train.iloc[val_ids]
    y_val = yy.iloc[val_ids]
    
    X_test = test1
    categorical_columns = [s for s in list(X.columns.values) if '_cat' in s]
    
    
#    enc = encoders.BinaryEncoder(verbose=1,cols=categorical_columns,return_df=True)
#    enc.fit(X, None)
#    df_X_bin = enc.transform(X)
#    df_X_bin = df_X_bin.apply(lambda x: pd.to_numeric(x,errors='ignore'))
#    
#    enc.fit(X_test, None)
#    df_X_test_bin = enc.transform(X_test)
#    df_X_test_bin = df_X_test_bin.apply(lambda x: pd.to_numeric(x,errors='ignore'))
#    
#    enc.fit(test1, None)
#    df_test_bin_1 = enc.transform(test1)
#    df_test_bin_1 = df_test_bin_1.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    
    encoding_dict = encode_cat_features(X, X_val, categorical_columns, target_column)
    
    for c, encoding in encoding_dict.items():
         X = pd.merge(X, encoding[[c,'enc']], how='left', on=c, sort=False,suffixes=('', '_'+c))
         X = X.drop(c, axis = 1)
         X = X.rename(columns = {'enc':'enc_'+c})
        
         X_val = pd.merge(X_val, encoding[[c,'enc']], how='left', on=c, sort=False,suffixes=('', '_'+c))
         X_val = X_val.drop(c, axis = 1)
         X_val = X_val.rename(columns = {'enc':'enc_'+c})
        
         X_test = pd.merge(X_test, encoding[[c,'enc']], how='left', on=c, sort=False,suffixes=('', '_'+c))
         X_test = X_test.drop(c, axis = 1)
         X_test = X_test.rename(columns = {'enc':'enc_'+c})
 
    X_test = X_test.replace(-1, np.nan);
    X_test=X_test.fillna(X_test.mean());       
#    enc.fit(test2, None)
#    df_test_bin_2 = enc.transform(test2)
#    df_test_bin_2 = df_test_bin_2.apply(lambda x: pd.to_numeric(x,errors='ignore'))
         
    X = X.drop(['id',target_column], axis=1)             
    X_val = X_val.drop(['id',target_column], axis=1)
    X_test = X_test.drop('id', axis=1)
    
    
    min_max=MinMaxScaler()
    X_train_minmax=min_max.fit_transform(X)
    X_minmax_df=pd.DataFrame(X_train_minmax)
    X_train = X_minmax_df
    

    #frame_minmax=[df_X_bin,X_minmax_df]
    #X_train=pd.concat(frame_minmax)
    #X_train = X_train.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    
    X_val_minmax=min_max.fit_transform(X_val)               
    X_val_minmax_df=pd.DataFrame(X_val_minmax)
    X_val_std = X_val_minmax_df
    
    
    X_test_minmax = min_max.fit_transform(X_test)               
    X_test_minmax_df=pd.DataFrame(X_test_minmax)
    X_test_std = X_test_minmax_df
    
#    test_minmax_2 = min_max.fit_transform(df_test_bin_2)               
#    test_minmax_df_2=pd.DataFrame(test_minmax_2)
#    test_test_2 = test_minmax_df_2
    #frame_minmax_test=[df_X_test_bin,X_test_minmax_df]
    #X_test_test=pd.concat(frame_minmax_test)
    #X_test_test = X_test_test.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    
    RF_model_cat1= RandomForestClassifier(100,oob_score=True,random_state=13)
    RF_model_cat2= RandomForestClassifier(50,oob_score=True,random_state=13,n_jobs = -1)
    RF_model_cat3= RandomForestClassifier(20,oob_score=True,random_state=13,n_jobs = -1, min_samples_leaf = 100)
    
    y = y.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    
    RF_model_cat1.fit(X_train, y)
    RF_model_cat2.fit(X_train, y)
    RF_model_cat3.fit(X_train, y)
    
    
    test_pred_class1 = RF_model_cat1.predict(X_test_std)
    test_pred_class2 = RF_model_cat2.predict(X_test_std)
    test_pred_class3 = RF_model_cat3.predict(X_test_std)
    
#    test_pred_class1_2 = RF_model_cat1.predict(test_test_2)
#    test_pred_class2_2 = RF_model_cat2.predict(test_test_2)
#    test_pred_class3_2 = RF_model_cat3.predict(test_test_2)
    
    #test_pred = (test_pred_class1 + test_pred_class2 + test_pred_class3)/3;
    
    
    #Obtain Class predictions
    y_pred_RF_class1 = RF_model_cat1.predict(X_val_std)
    print('Predicted classes clf 1: \n', y_pred_RF_class1)
    
    y_pred_RF_class2 = RF_model_cat2.predict(X_val_std)
    print('Predicted classes clf 2: \n', y_pred_RF_class2)
    
    y_pred_RF_class3 = RF_model_cat3.predict(X_val_std)
    print('Predicted classes clf 3: \n', y_pred_RF_class3)
    
    #Obtain probablity predictions
    y_pred_RF_prob1 = RF_model_cat1.predict_proba(X_val_std)
    print('Predicted probabilities clf 1: \n', y_pred_RF_prob1)
    
    y_pred_RF_prob2 = RF_model_cat2.predict_proba(X_val_std)
    print('Predicted probabilities clf 2: \n', y_pred_RF_prob2)
    
    y_pred_RF_prob3 = RF_model_cat3.predict_proba(X_val_std)
    print('Predicted probabilities clf 3: \n', y_pred_RF_prob3)
    
    x_val_fold[val_ids] = (RF_model_cat1.predict(X_val_std) + RF_model_cat2.predict(X_val_std) + RF_model_cat3.predict(X_val_std))/3;
    X_val_std
    
    print('RF Score clf 1 Validation: ', metrics.accuracy_score(y_val, y_pred_RF_class1))
    print('RF Score clf 2 Validation: ', metrics.accuracy_score(y_val, y_pred_RF_class2))
    print('RF Score clf 3 Validation: ', metrics.accuracy_score(y_val, y_pred_RF_class3))
    
    print('RF Score clf 1 Test: ', metrics.accuracy_score(test_result['target'], test_pred_class1))
    print('RF Score clf 2 Test: ', metrics.accuracy_score(test_result['target'], test_pred_class2))
    print('RF Score clf 3 Test: ', metrics.accuracy_score(test_result['target'], test_pred_class3))
    
    



#print(gini(train_master[target_column], x_val_fold));
#fpr, tpr, thr = metrics.roc_curve(train_master[target_column], x_val_fold, pos_label=1)
#plt.plot(thr, tpr)
#plt.plot(thr, 1-fpr)#tnr
#plt.show()
    
#xg_out_y_test=np.array(test_master);
#[count_Ytest,pp]=xg_out_y_test.shape;
#for i in range(0,count_Ytest):
#    if test_pred[i]>=thr:       # setting threshold to .5 
#       test_pred[i]=1 
#    else: 
#       test_pred[i]=0   
#tnr=1-fpr;
#fnr=1-tpr;       
#xg_accuracy_final= (tpr+tnr)/(tpr + tnr + fpr +(fnr));

#print ('XGB GBM',xg_accuracy_final); 

    