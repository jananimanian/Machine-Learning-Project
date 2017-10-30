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
test = pd.read_csv('F:/Fall_2017/ML/test.csv')

print(train.isnull().sum());
print((train==-1).sum());

### Now let's prepare lists of numeric, categorical and binary columns
all_features = train.columns.tolist()
all_features.remove('target')
numeric_features = [x for x in all_features if x[-3:] not in ['bin', 'cat']]
categorical_features = [x for x in all_features if x[-3:]=='cat']
binary_features = [x for x in all_features if x[-3:]=='bin']

print(numeric_features);
print(categorical_features);
print(binary_features);

train['target_name'] = train['target'].map({0: 'Not Filed', 1: 'Filed'})
