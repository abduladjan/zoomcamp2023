```python
pip install wget
```

    Requirement already satisfied: wget in c:\users\abdul\anaconda3\lib\site-packages (3.2)
    Note: you may need to restart the kernel to use updated packages.
    


```python
!python -m wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv
```

    
    Saved under housing (9).csv
    


```python
import pandas as pd
import numpy as np
```


```python
df = pd.read_csv('housing (5).csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB
    


```python
proximity = ['<1H OCEAN', 'INLAND']
```


```python
df_full = df.loc[df.ocean_proximity.isin(proximity)]
```


```python
df_full.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 15687 entries, 701 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           15687 non-null  float64
     1   latitude            15687 non-null  float64
     2   housing_median_age  15687 non-null  float64
     3   total_rooms         15687 non-null  float64
     4   total_bedrooms      15530 non-null  float64
     5   population          15687 non-null  float64
     6   households          15687 non-null  float64
     7   median_income       15687 non-null  float64
     8   median_house_value  15687 non-null  float64
     9   ocean_proximity     15687 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.3+ MB
    


```python
df[(df.ocean_proximity == '<1H OCEAN') | (df.ocean_proximity == 'INLAND')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>701</th>
      <td>-121.97</td>
      <td>37.64</td>
      <td>32.0</td>
      <td>1283.0</td>
      <td>194.0</td>
      <td>485.0</td>
      <td>171.0</td>
      <td>6.0574</td>
      <td>431000.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>830</th>
      <td>-121.99</td>
      <td>37.61</td>
      <td>9.0</td>
      <td>3666.0</td>
      <td>711.0</td>
      <td>2341.0</td>
      <td>703.0</td>
      <td>4.6458</td>
      <td>217000.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>859</th>
      <td>-121.97</td>
      <td>37.57</td>
      <td>21.0</td>
      <td>4342.0</td>
      <td>783.0</td>
      <td>2172.0</td>
      <td>789.0</td>
      <td>4.6146</td>
      <td>247600.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>860</th>
      <td>-121.96</td>
      <td>37.58</td>
      <td>15.0</td>
      <td>3575.0</td>
      <td>597.0</td>
      <td>1777.0</td>
      <td>559.0</td>
      <td>5.7192</td>
      <td>283500.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>861</th>
      <td>-121.98</td>
      <td>37.58</td>
      <td>20.0</td>
      <td>4126.0</td>
      <td>1031.0</td>
      <td>2079.0</td>
      <td>975.0</td>
      <td>3.6832</td>
      <td>216900.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20635</th>
      <td>-121.09</td>
      <td>39.48</td>
      <td>25.0</td>
      <td>1665.0</td>
      <td>374.0</td>
      <td>845.0</td>
      <td>330.0</td>
      <td>1.5603</td>
      <td>78100.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20636</th>
      <td>-121.21</td>
      <td>39.49</td>
      <td>18.0</td>
      <td>697.0</td>
      <td>150.0</td>
      <td>356.0</td>
      <td>114.0</td>
      <td>2.5568</td>
      <td>77100.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20637</th>
      <td>-121.22</td>
      <td>39.43</td>
      <td>17.0</td>
      <td>2254.0</td>
      <td>485.0</td>
      <td>1007.0</td>
      <td>433.0</td>
      <td>1.7000</td>
      <td>92300.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>-121.32</td>
      <td>39.43</td>
      <td>18.0</td>
      <td>1860.0</td>
      <td>409.0</td>
      <td>741.0</td>
      <td>349.0</td>
      <td>1.8672</td>
      <td>84700.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20639</th>
      <td>-121.24</td>
      <td>39.37</td>
      <td>16.0</td>
      <td>2785.0</td>
      <td>616.0</td>
      <td>1387.0</td>
      <td>530.0</td>
      <td>2.3886</td>
      <td>89400.0</td>
      <td>INLAND</td>
    </tr>
  </tbody>
</table>
<p>15687 rows × 10 columns</p>
</div>




```python
fetures = ['latitude',
'longitude',
'housing_median_age',
'total_rooms',
'total_bedrooms',
'population',
'households',
'median_income',
'median_house_value']
```


```python
df_full = df_full[fetures]
```


```python
df_full.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 15687 entries, 701 to 20639
    Data columns (total 9 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   latitude            15687 non-null  float64
     1   longitude           15687 non-null  float64
     2   housing_median_age  15687 non-null  float64
     3   total_rooms         15687 non-null  float64
     4   total_bedrooms      15530 non-null  float64
     5   population          15687 non-null  float64
     6   households          15687 non-null  float64
     7   median_income       15687 non-null  float64
     8   median_house_value  15687 non-null  float64
    dtypes: float64(9)
    memory usage: 1.2 MB
    

Question 1


```python
df_full.isnull().sum()
```




    latitude                0
    longitude               0
    housing_median_age      0
    total_rooms             0
    total_bedrooms        157
    population              0
    households              0
    median_income           0
    median_house_value      0
    dtype: int64



Question 2


```python
df_full.population.median()
```




    1195.0



Question 3


```python
np.random.seed(42)
```


```python
shuffle_list = np.arange(0,df_full.shape[0])
```


```python
shuffle_list
```




    array([    0,     1,     2, ..., 15684, 15685, 15686])




```python
np.random.shuffle(shuffle_list)
```


```python
shuffle_list
```




    array([15183,  4469,  9316, ...,  5390,   860,  7270])




```python
df_shuffle_42 = df_full.copy()
```


```python
n = df_full.shape[0]
```


```python
len_val = int(0.2*n)
len_test = int(0.2*n)
len_train = n - (len_val + len_test)
```


```python
idx_train = shuffle_list[:len_train]
idx_val = shuffle_list[len_train:(len_train+len_val)]
idx_test = shuffle_list[(len_train+len_val):]
```


```python
df_train = df_shuffle_42.iloc[idx_train]
df_val = df_shuffle_42.iloc[idx_val]
df_test = df_shuffle_42.iloc[idx_test]
```


```python
df_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>latitude</th>
      <th>longitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19963</th>
      <td>36.23</td>
      <td>-119.14</td>
      <td>22.0</td>
      <td>2935.0</td>
      <td>523.0</td>
      <td>1927.0</td>
      <td>530.0</td>
      <td>2.5875</td>
      <td>70400.0</td>
    </tr>
    <tr>
      <th>5929</th>
      <td>34.12</td>
      <td>-117.79</td>
      <td>16.0</td>
      <td>2426.0</td>
      <td>426.0</td>
      <td>1319.0</td>
      <td>446.0</td>
      <td>4.8125</td>
      <td>224500.0</td>
    </tr>
    <tr>
      <th>11377</th>
      <td>33.68</td>
      <td>-117.97</td>
      <td>26.0</td>
      <td>3653.0</td>
      <td>568.0</td>
      <td>1930.0</td>
      <td>585.0</td>
      <td>5.7301</td>
      <td>260900.0</td>
    </tr>
    <tr>
      <th>6443</th>
      <td>34.10</td>
      <td>-118.03</td>
      <td>32.0</td>
      <td>2668.0</td>
      <td>609.0</td>
      <td>1512.0</td>
      <td>541.0</td>
      <td>2.9422</td>
      <td>233100.0</td>
    </tr>
    <tr>
      <th>17546</th>
      <td>37.34</td>
      <td>-121.87</td>
      <td>39.0</td>
      <td>2479.0</td>
      <td>541.0</td>
      <td>1990.0</td>
      <td>506.0</td>
      <td>2.4306</td>
      <td>289100.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_train = df_train['median_house_value'].values
y_val = df_val['median_house_value'].values
y_test = df_test['median_house_value'].values

y_train = np.log1p(y_train)
y_val = np.log1p(y_val)
y_test = np.log1p(y_test)

del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']
```


```python
def fill_with_0(df):
    df = df.copy()
    df = df.fillna(0)
    X = df.values
    return X
```


```python
def fill_with_mean(df, mean):
    df = df.copy()
    df = df.fillna(mean)
    X = df.values
    return X
```


```python
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]
```


```python
X_v0 = fill_with_0(df_train)
```


```python
w_0, w = train_linear_regression(X_v0, y_train)
```


```python
def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)
```


```python
X_val = fill_with_0(df_val)
y_pred = w_0 + X_val.dot(w)
```


```python
round(rmse(y_val, y_pred), 2)
```




    0.34




```python
X_vm = fill_with_mean(df_train, df_train.total_bedrooms.mean())
```


```python
w_0, w = train_linear_regression(X_vm, y_train)
```


```python
X_val = fill_with_mean(df_val, df_train.total_bedrooms.mean())
y_pred = w_0 + X_val.dot(w)
```


```python
round(rmse(y_val, y_pred), 2)
```




    0.34



Question 4


```python
def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]
```


```python
X_train = fill_with_0(df_train)
X_val = fill_with_0(df_val)
```


```python
for reg in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    w_0, w = train_linear_regression_reg(X_train, y_train, r=reg)
    y_pred = w_0 + X_val.dot(w)
    
    print(f"{reg}, {round(rmse(y_val, y_pred), 10)}")
```

    0, 0.3408479034
    1e-06, 0.3408479062
    0.0001, 0.3408481801
    0.001, 0.3408506922
    0.01, 0.34087793
    0.1, 0.3412862042
    1, 0.3448958328
    5, 0.347739807
    10, 0.3483149834
    

Question 5


```python
rmse_with_seeds = []

for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    np.random.seed(seed)
    shuffle_list = np.arange(0,df_full.shape[0])
    np.random.shuffle(shuffle_list)
    df_shuffle = df_full.copy()
    
    len_val = int(0.2*n)
    len_test = int(0.2*n)
    len_train = n - (len_val + len_test)
    idx_train = shuffle_list[:len_train]
    idx_val = shuffle_list[len_train:(len_train+len_val)]
    idx_test = shuffle_list[(len_train+len_val):]
    df_train = df_shuffle.iloc[idx_train]
    df_val = df_shuffle.iloc[idx_val]
    df_test = df_shuffle.iloc[idx_test]
    
    X_train = fill_with_0(df_train)
    X_val = fill_with_0(df_val)
    
    w_0, w = train_linear_regression(X_train, y_train)
    
    y_pred = w_0 + X_val.dot(w)
    
    rmse_with_seeds.append(round(rmse(y_val, y_pred),3))
```


```python
np.std(rmse_with_seeds)
```




    0.0005000000000000004



Question 6


```python
np.random.seed(9)
```


```python
shuffle_list = np.arange(0,df_full.shape[0])
np.random.shuffle(shuffle_list)
df_shuffle = df_full.copy()
    
len_val = int(0.2*n)
len_test = int(0.2*n)
len_train = n - (len_val + len_test)

idx_train = shuffle_list[:len_train+len_val]
idx_test = shuffle_list[(len_train+len_val):]
df_train = df_shuffle.iloc[idx_train]
df_test = df_shuffle.iloc[idx_test]
```


```python
X_train = fill_with_0(df_train)
X_test = fill_with_0(df_test)

y_train = np.concatenate((y_train, y_val))
```


```python
w_0, w = train_linear_regression_reg(X_train, y_train, r=0.001)
y_pred = w_0 + X_test.dot(w)
    
print(rmse(y_test, y_pred))
```

    0.558176112916656
    


```python

```
