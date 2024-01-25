#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Normalizer
import xgboost as xgb

df = pd.read_csv('KAG_energydata_complete.csv')

df.drop(columns=['rv1', 'rv2'], inplace=True)
df.drop(columns=['date'], inplace=True)

df_full, df_test = train_test_split(df, test_size=0.2, random_state=123)
df_train, df_val = train_test_split(df_full, test_size=0.25, random_state=123)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full = df_full['Appliances'].values
y_train = df_train['Appliances'].values
y_val = df_val['Appliances'].values
y_test = df_test['Appliances'].values

del df_full['Appliances']
del df_train['Appliances']
del df_val['Appliances']
del df_test['Appliances']

dv = DictVectorizer(sparse=False)
train_dict = df_train.to_dict(orient="records")
X_train = dv.fit_transform(train_dict)
val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

train_norm = normalize(X_train)
val_norm = normalize(X_val)

dtrain = xgb.DMatrix(train_norm, label=y_train)
dval = xgb.DMatrix(val_norm, label=y_val)

def train(df_train, y_train):
    dv = DictVectorizer(sparse=False)
    df_dict_train = df_train.to_dict(orient="records")
    X_train = dv.fit_transform(df_dict_train)
    transformer = Normalizer()
    X_train_norm = Normalizer().fit_transform(X_train)
    
    regressor=xgb.XGBRegressor(learning_rate = 0.015,
                           n_estimators  = 700,
                           max_depth     = 6,
                           eval_metric='rmsle')
    
    regressor.fit(X_train_norm, y_train)
    
    return dv, transformer, regressor

dv, transformer, regressor = train(df_train, y_train)

def predict(df, dv, transformer, regressor):
    
    df_dict = df.to_dict(orient="records")

    X = dv.transform(df_dict)
    X_norm = transformer.transform(X)
    y_pred = regressor.predict(X_norm)

    return y_pred

y_pred = predict(df_val, dv, transformer, regressor)

dv, transformer, regressor = train(df_full, y_full)
y_pred = predict(df_test, dv, transformer, regressor)


mse_val = mean_squared_error(y_test, y_pred)
print(mse_val)


mae_val = mean_absolute_error(y_test, y_pred)
print(mae_val)


import pickle

output_file = 'model_regressor.bin'

f_out = open(output_file, 'wb')
pickle.dump((dv, transformer, regressor), f_out)
f_out.close()


