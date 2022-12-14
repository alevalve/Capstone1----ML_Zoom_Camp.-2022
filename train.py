## LIBRARIES

## Librerias 
import pandas as pd
from numpy import mean
import pickle
import numpy as np 
import bentoml
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn import *
import sklearn as skl
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from numpy import std
from sklearn.preprocessing import scale 
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn import model_selection
from scipy import stats
from scipy.stats import boxcox 
import pylab as pl
from sklearn import linear_model
import pyreadr 
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns 
import matplotlib as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import matplotlib.pyplot as plt
import scipy
from scipy.stats import skew
import xgboost as xgb

## Import datasets

por = pd.read_csv("whisky.csv")
por
df = pd.DataFrame(por)


## Clean data 

def clean (data1):
    data1= data1.applymap(lambda s:s.lower().replace(' ', '_') if type(s) == str else s)
    data1.columns = [x.lower().replace(' ', '_') for x in data1.columns]
    data1.columns = [x.lower().replace(':', '') for x in data1.columns]
    data1.columns = [x.lower().replace('*', '') for x in data1.columns]
    data1.columns = [x.lower().replace('.', '') for x in data1.columns] 
    data1.competition = [x.lower().replace('_', '') for x in data1.competition]
    return data1

df1 = clean(df)

## Take out columns

def drop(data):
    
    return data.drop(columns=['natural_foot','position','team_against','gk'])

df2 = drop(df1)
df2


##Create a dataframe with the predictor variables

## Transformar data
df2['direction']=df['direction'].map({'left':0, 'right':1,'middle':2})
categorical = ['team_for', 'time', 'scoreline', 'venue', 'history', 'competition']


## Split data set 

df_train, df_test =train_test_split(df2, test_size=0.20,random_state=123)

y = df2.direction

## Extracr Y variable
y_train = (df_train.direction).values
y_test = (df_test.direction).values

del df_train['direction']
del df_test['direction']

## To dictionary
dict_train = df_train.to_dict(orient='records')
dict_test = df_test.to_dict(orient='records')

## Vectorize

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(dict_train)
X_test = dv.transform(dict_test)


## Xgboost

model = XGBClassifier(
colsample_bytree = 0.7,
learning_rate = 0.1,
max_depth = 7,
min_child_weight = 1,
n_estimators = 100,
num_class = 3,
objective = 'multi:softmax',
subsample = 0.5)

model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_train, y_train)], verbose=False)

y_pred = model.predict(X_train)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_train, y_pred)
print("Accuracy: %f" % (score))


## Save model with BentoML

import bentoml
bentoml.xgboost.save_model("messi_model",model, custom_objects={"DictVectorizer":dv},
signatures={"predict": {"batchable":True,"batch_dim":0,}})


