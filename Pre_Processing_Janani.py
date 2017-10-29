import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 100)

train = pd.read_csv('F:/Fall_2017/ML/train.csv')
#test = pd.read_csv('../input/test.csv')

print(train.head());

print(train.tail());
print(train.shape);

train.drop_duplicates()
print("after removing duplicate rows", train.shape);

print(train.info());
data = []
for f in train.columns:
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
    elif train[f].dtype == float:
        print("datatype name",f);
        level = 'interval'
    elif train[f].dtype == np.int64:   
        print("datatype name2222",f);
        level = 'ordinal'
        
    # Initialize keep to True for all variables except for id
    keep = True
    if f == 'id':
        keep = False
    
    # Defining the data type 
    #dtype = train[f].dtype
    
    # Creating a Dict that contains all the metadata for the variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': np.dtype
    }
    data.append(f_dict)
    
meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)
print(meta);

print(meta[(meta.level == 'nominal') & (meta.keep)].index);

print(pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index());
#print(pd.DataFrame({'count' : meta.groupby(['varname'])}).reset_index());

v = meta[(meta.level == 'interval') & (meta.keep)].index;
print("\n interval data");
print(train[v].describe())

v = meta[(meta.level == 'ordinal') & (meta.keep)].index
print("\n ordinal data");
print(train[v].describe())

v = meta[(meta.level == 'nominal') & (meta.keep)].index
print("\n nominal data");
print(train[v].describe())

v = meta[(meta.level == 'binary') & (meta.keep)].index
print("\n binary data");
print(train[v].describe())
