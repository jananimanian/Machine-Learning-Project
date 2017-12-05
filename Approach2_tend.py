# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:48:25 2017

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
from sklearn.metrics import accuracy_score
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.graph_objs import *
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics



train_master = pd.read_csv('C:/Users/Sindhuja/Desktop/ML Project/train.csv')
test_master = pd.read_csv('C:/Users/Sindhuja/Desktop/ML Project/test.csv')
train_master.describe()

binary_columns = [s for s in list(train_master.columns.values) if '_bin' in s]
categorical_columns = [s for s in list(train_master.columns.values) if '_cat' in s]
non_continuous_feature_subs = ['_cat', '_bin', 'target', 'id']
continuous_columns = [s for s in list(train_master.columns.values) if all(x not in s for x in non_continuous_feature_subs)]
target_column = 'target'

ind_columns = [s for s in list(train_master.columns.values) if '_ind' in s]
car_columns = [s for s in list(train_master.columns.values) if '_car' in s]
calc_columns = [s for s in list(train_master.columns.values) if '_calc' in s]
reg_columns = [s for s in list(train_master.columns.values) if '_reg' in s]

#Find duolicates
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

#Display meta
meta

#Descriptive statistics

#Interval variables
v = meta[(meta.level == 'interval') & (meta.keep)].index
train_master[v].describe()

#Ordinal variables
v = meta[(meta.level == 'ordinal') & (meta.keep)].index
train_master[v].describe()

#Binary variables
v = meta[(meta.level == 'binary') & (meta.keep)].index
train_master[v].describe()

#Checking missing values
vars_with_missing = []

for f in train_master.columns:
    missings = train_master[train_master[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings/train_master.shape[0]
        
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
        
print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))




##  *********************************************************************************************************************



init_notebook_mode()


labels = ['1','0']
values = [(train_master[target_column]==1).sum(),(train_master[target_column]==0).sum()]
colors = ['#FEBFB3', '#E1396C']

trace = Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
data=[trace]
layout = Layout(
    height=600,
    width=800,
)

fig = dict( data=data, layout=layout )

plot(fig) 


zero_list = []
one_list = []
for col in binary_columns:
    zero_list.append((train_master[col]==0).sum())
    one_list.append((train_master[col]==1).sum())
    



trace0 = Bar(
    x=binary_columns,
    y=zero_list
)
trace1 = Bar(
    x=binary_columns,
    y=one_list
)
data1 = [trace0,trace1]
layout1 = Layout(
    showlegend=False,
    height=600,
    width=800,
    barmode='stack'
)

fig1 = dict( data=data1, layout=layout1 )

plot(fig1,filename='temp1.html') 
            
binary_corr_data = []
r = 0
for i in binary_columns:
    binary_corr_data.append([])
    for j in binary_columns:
        s = sum(train_master[i]^train_master[j])/float(len(train_master[i]))
        binary_corr_data[r].append(s)
    r+=1
            
trace2 = Heatmap(z=binary_corr_data, x=binary_columns, y=binary_columns, colorscale='Greys')
data2=[trace2]

layout2 = Layout(
    height=600,
    width=800,
)

fig2 = dict( data=data2, layout=layout2 )

plot(fig2,filename='temp_bin_corr.html') 
            

binary_target_corr_data = []
for i in binary_columns:
    s = sum(train_master[i]^train_master[target_column])/float(len(train_master[i]))
    binary_target_corr_data.append(s)

binary_target_corr_chart = [Bar(
    x=binary_columns,
    y=binary_target_corr_data
)]
    


fig3 = dict( data=binary_target_corr_chart, layout=layout2 )

plot(fig3,filename='temp_bin_corr_targe.html') 


value_list = []
missing_list = []
for col in continuous_columns:
    value_list.append((train_master[col]!=-1).sum())
    missing_list.append((train_master[col]==-1).sum())

trace4 = Bar(
    x=continuous_columns,
    y=value_list ,
    name='Actual Values'
)
trace5 = Bar(
    x=continuous_columns,
    y=missing_list,
    name='Missing Values'
)

data4 = [trace4, trace5]
layout4 = Layout(
    barmode='stack',
    title='Count of missing values in continuous variables',
    height=600,
    width=800,
)

fig4 = dict(data=data4, layout=layout4)
plot(fig4, filename='temp_cont_corr_target.html')

            
# Dropping the variables with too many missing values
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train_master.drop(vars_to_drop, axis=1)
meta.loc[(vars_to_drop),'keep'] = False  # Updating the meta


# Imputing with the mean or mode
mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
train_master['ps_reg_03'] = mean_imp.fit_transform(train_master[['ps_reg_03']]).ravel()
train_master['ps_car_12'] = mean_imp.fit_transform(train_master[['ps_car_12']]).ravel()
train_master['ps_car_14'] = mean_imp.fit_transform(train_master[['ps_car_14']]).ravel()
train_master['ps_car_11'] = mode_imp.fit_transform(train_master[['ps_car_11']]).ravel()

minfo_target_to_continuous_features = mutual_info_classif(
    train_master[continuous_columns],train_master[target_column])

minfo_target_to_continuous_chart = [Bar(
    x=continuous_columns,
    y=minfo_target_to_continuous_features
)]
    


fig5 = dict(data=minfo_target_to_continuous_chart, layout=layout2)
plot(fig5, filename='temp_cont_corr_target.html')

continuous_corr_data = train_master[continuous_columns].corr(method='pearson').as_matrix()

trace6 = Heatmap(z=continuous_corr_data, x=continuous_columns, 
                   y=continuous_columns, colorscale='Greys')
data6=[trace6]


fig6 = dict(data=data6, layout=layout2)
plot(fig6, filename='temp_cont_corr.html')

n_splits = 2
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)


np.random.seed(3)
model_scores = {}


train = train_master.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin'],axis=1)
test = test_master.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin'],axis=1)
vars_to_drop_bin = ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_13_bin']
meta.loc[(vars_to_drop_bin),'keep'] = False

train = train_master.drop(calc_columns, axis=1)  
test = test_master.drop(calc_columns, axis=1)
meta.loc[(calc_columns),'keep'] = False

train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)

y_val_fold = np.empty(len(train))
y_val_fold_1 = np.empty(len(train))
y_val_fold_2 = np.empty(len(train))
y_val_fold_3 = np.empty(len(train))

# Initialize DS to store test predictions with aggregate model and individual models
y_test = np.zeros(len(test))
y_test_model_1 = np.zeros(len(test))
y_test_model_2 = np.zeros(len(test))
y_test_model_3 = np.zeros(len(test))


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
    
def gini(y, pred):
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) -1
    return g

counter=1
for fold_number, (train_ids, val_ids) in enumerate(
    folds.split(train.drop(['id',target_column], axis=1), 
                train[target_column])):
    
    X = train.iloc[train_ids]
    X_val = train.iloc[val_ids]
    X_test = test

    encoding_dict = encode_cat_features(X, X_val, categorical_columns, target_column)

    for c, encoding in encoding_dict.items():
         X = pd.merge(X, encoding[[c,'enc']], how='left', on=c, sort=False,suffixes=('', '_'+c))
         X = X.drop(c, axis = 1)
         X = X.rename(columns = {'enc':'enc_'+c})
        
         X_test = pd.merge(X_test, encoding[[c,'enc']], how='left', on=c, sort=False,suffixes=('', '_'+c))
         X_test = X_test.drop(c, axis = 1)
         X_test = X_test.rename(columns = {'enc':'enc_'+c})
        
         X_val = pd.merge(X_val, encoding[[c,'enc']], how='left', on=c, sort=False,suffixes=('', '_'+c))
         X_val = X_val.drop(c, axis = 1)
         X_val = X_val.rename(columns = {'enc':'enc_'+c})
        

    # Seperate target column and remove id column from all
    y = X[target_column]
    X = X.drop(['id',target_column], axis=1)
    X_test = X_test.drop('id', axis=1)
    y_val = X_val[target_column]
    X_val = X_val.drop(['id',target_column], axis=1)
         
         
    # Upsample data in training folds
    ids_to_duplicate = pd.Series(y == 1)
    X = pd.concat([X, X.loc[ids_to_duplicate]], axis=0)
    y = pd.concat([y, y.loc[ids_to_duplicate]], axis=0)
    # Again Upsample (total increase becomes 4 times)
    X = pd.concat([X, X.loc[ids_to_duplicate]], axis=0)
    y = pd.concat([y, y.loc[ids_to_duplicate]], axis=0)
        
    
    
    shuffled_ids = np.arange(len(X))
    np.random.shuffle(shuffled_ids)
    X = X.iloc[shuffled_ids]
    y = y.iloc[shuffled_ids]
    



     # Feature Selection goes here
    # TODO
    
    # Define parameters of GBM as explained before for 3 trees
    params_1 = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'max_depth': 3,
        'learning_rate': 0.05,
        'feature_fraction': 1,
        'bagging_fraction': 1,
        'bagging_freq': 10,
        'verbose': 0,
        'min_split_gain': 0.5
    }
    params_2 = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 2,
        'verbose': 0,
        'min_split_gain': 0.1
    }
    params_3 = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 5,
        'learning_rate': 0.05,
        'feature_fraction': 0.3,
        'bagging_fraction': 0.7,
        'bagging_freq': 10,
        'verbose': 0
    }
    
     # Create appropriate format for training and evaluation data
    lgb_train = lgb.Dataset(X, y)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    # Create the 3 classifiers with 1000 rounds and a window of 100 for early stopping
    clf_1 = lgb.train(params_1,lgb_train, num_boost_round=1000,
                      valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=50)
    clf_2 = lgb.train(params_2,lgb_train, num_boost_round=1000,
                      valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=50)
    clf_3 = lgb.train(params_3,lgb_train, num_boost_round=1000,
                      valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=50)
    
    # Predict raw scores for validation ids
    # At each fold, 1/10th of the training data get scores
    y_val_fold_1[val_ids] = (clf_1.predict(X_val))
    y_val_fold_2[val_ids] = (clf_2.predict(X_val))
    y_val_fold_3[val_ids] = (clf_3.predict(X_val))
    
    
    y_val_fold[val_ids] = (clf_1.predict(X_val)+clf_2.predict(X_val)+clf_3.predict(X_val))/3;

    # Predict and average over folds, raw scores for test data
    y_test += (clf_1.predict(X_test, raw_score=True)+
               clf_2.predict(X_test, raw_score=True)+
               clf_3.predict(X_test, raw_score=True)) / (3*n_splits)
    y_test_model_1 += clf_1.predict(X_test, raw_score=True) / n_splits
    y_test_model_2 += clf_2.predict(X_test, raw_score=True) / n_splits
    y_test_model_3 += clf_3.predict(X_test, raw_score=True) / n_splits
    
    # Display fold predictions
    # Gini requires only order and therefore raw scores need not be scaled
    print("Fold %2d : %.9f" % (fold_number + 1, gini(y_val, y_val_fold[val_ids])))
    
    
    
# Display aggregate predictions
# Gini requires only order and therefore raw scores need not be scaled
print("Average score over all folds: %.9f" % gini(train_master[target_column], y_val_fold_1))

print("Accuracy with LGB is: %.3f \n" %((accuracy_score(train_master[target_column],y_val_fold_1))))   
print("Accuracy with LGB is: %.3f \n" %((accuracy_score(train_master[target_column],y_val_fold_2))))   
print("Accuracy with LGB is: %.3f \n" %((accuracy_score(train_master[target_column],y_val_fold_3))))   

temp = y_test
# Scale the raw scores to range [0.0, 1.0]
temp = np.add(temp,abs(min(temp)))/max(np.add(temp,abs(min(temp))))

df = pd.DataFrame(columns=['id', 'target'])
df['id']=test_master['id']
df['target']=temp
df.to_csv('benchmark__0_283.csv', index=False, float_format="%.9f")
df.shape












            