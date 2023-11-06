import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

from tqdm.notebook import tqdm

np.random.seed(123)


df = pd.read_csv('weatherAUS.csv')
df.columns = df.columns.str.lower()

#Drop evaporation, sunshine, cloud9am, cloud3pm columns due to big amount of NAs

df.drop(columns=['evaporation', 'sunshine', 'cloud9am', 'cloud3pm'], inplace=True)

#Splitting Date column on year, month, and day

split = df.date.str.split('-', n=-1, expand=True)

df['year'] = split[0]
df['month'] = split[1]
df['day'] = split[2]
df.drop(columns=['date'], inplace=True)
df = df.astype({'year': 'int32', 'month': 'int32', 'day': 'int32'})

#Drop NA values in target column 

df.dropna(subset=['raintomorrow'], inplace=True)
df.reset_index(inplace=True, drop=True)

#Filling NA with propagating last valid observation forward to next valid

df = df.fillna(method='ffill')

df.raintomorrow = (df.raintomorrow == 'Yes').astype(int)

df_full, df_test = train_test_split(df, test_size=0.2, random_state=123)
df_train, df_val = train_test_split(df_full, test_size=0.25, random_state=123)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train['raintomorrow'].values
y_val = df_val['raintomorrow'].values
y_test = df_test['raintomorrow'].values

del df_train['raintomorrow']
del df_val['raintomorrow']
del df_test['raintomorrow']

dv = DictVectorizer(sparse=False)
train_dict = df_train.to_dict(orient="records")
X_train = dv.fit_transform(train_dict)
val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)




df_full = df_full.reset_index(drop=True)
y_full = df_full['raintomorrow'].values

del df_full['raintomorrow']

dv = DictVectorizer(sparse=False)
full_dict = df_full.to_dict(orient="records")
X_full = dv.fit_transform(full_dict)
test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(test_dicts)


model_final = RandomForestClassifier(random_state=123, n_estimators=190, max_depth=10)
model_final.fit(X_full, y_full)

y_pred_final = model_final.predict_proba(X_full)[:, 1]
auc = roc_auc_score(y_full, y_pred_final)
#print('train:', auc)

y_pred_final = model_final.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_final)
#print('val:', auc)


def train(df_train, y_train):
    dv = DictVectorizer(sparse=False)
    df_dict_train = df_train.to_dict(orient="records")
    X_train = dv.fit_transform(df_dict_train)
    
    model = RandomForestClassifier(random_state=123, n_estimators=190, max_depth=10)
    model.fit(X_train, y_train)
    
    return dv, model

dv, model = train(df_train, y_train)


def predict(df, dv, model):
    """df.drop(columns=['evaporation', 'sunshine', 'cloud9am', 'cloud3pm'], inplace=True)
    split = df.date.str.split('-', n=-1, expand=True)

    df['year'] = split[0]
    df['month'] = split[1]
    df['day'] = split[2]

    df.drop(columns=['date'], inplace=True)
    
    df = df.astype({'year': 'int32', 'month': 'int32', 'day': 'int32'})"""
    
    df_dict = df.to_dict(orient="records")

    X = dv.transform(df_dict)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


y_pred = predict(df_val, dv, model)


dv, model = train(df_full, y_full)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
print('auc of the final model', auc)


import pickle

output_file = 'model_forest.bin'
f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()

print('the model is saved to', output_file)



