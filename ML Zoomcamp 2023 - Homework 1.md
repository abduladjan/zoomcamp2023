```python
import pandas as pd
```


```python
import numpy as np
```


```python
pd.__version__
```




    '1.4.4'




```python
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv
```


      File "C:\Users\abdul\AppData\Local\Temp\ipykernel_25544\1945767389.py", line 1
        wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv
             ^
    SyntaxError: invalid syntax
    



```python
df = pd.read_csv('raw.githubusercontent.com_alexeygrigorev_datasets_master_housing.csv')
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
df.columns
```




    Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms',
           'total_bedrooms', 'population', 'households', 'median_income',
           'median_house_value', 'ocean_proximity'],
          dtype='object')



Question 2


```python
len(df.columns)
```




    10



Question 3


```python
df.isnull().sum()
```




    longitude               0
    latitude                0
    housing_median_age      0
    total_rooms             0
    total_bedrooms        207
    population              0
    households              0
    median_income           0
    median_house_value      0
    ocean_proximity         0
    dtype: int64



Question 4


```python
df.ocean_proximity.nunique()
```




    5



Question 5


```python
df[df['ocean_proximity'] == 'NEAR BAY'].median_house_value.mean()
```




    259212.31179039303



Question 6


```python
total_bedrooms_mean = df.total_bedrooms.mean()
```


```python
total_bedrooms_mean
```




    537.8705525375618




```python
df.total_bedrooms = df.total_bedrooms.fillna(total_bedrooms_mean)
```


```python
df.isnull().sum()
```




    longitude             0
    latitude              0
    housing_median_age    0
    total_rooms           0
    total_bedrooms        0
    population            0
    households            0
    median_income         0
    median_house_value    0
    ocean_proximity       0
    dtype: int64




```python
df.total_bedrooms.mean()
```




    537.8705525375639



Question 7


```python
df_island = df[df.ocean_proximity == 'ISLAND']
```


```python
X = np.array(df_island[['housing_median_age', 'total_rooms', 'total_bedrooms']])
```


```python
X
```




    array([[  27., 1675.,  521.],
           [  52., 2359.,  591.],
           [  52., 2127.,  512.],
           [  52.,  996.,  264.],
           [  29.,  716.,  214.]])




```python
XT = X.transpose()
```


```python
XT
```




    array([[  27.,   52.,   52.,   52.,   29.],
           [1675., 2359., 2127.,  996.,  716.],
           [ 521.,  591.,  512.,  264.,  214.]])




```python
XTX = XT.dot(X)
```


```python
XTX
```




    array([[9.6820000e+03, 3.5105300e+05, 9.1357000e+04],
           [3.5105300e+05, 1.4399307e+07, 3.7720360e+06],
           [9.1357000e+04, 3.7720360e+06, 9.9835800e+05]])




```python
y = np.array([950, 1300, 800, 1000, 1300])
```


```python
XTXi = np.linalg.inv(XTX)
```


```python
XTXi
```




    array([[ 9.19403586e-04, -3.66412216e-05,  5.43072261e-05],
           [-3.66412216e-05,  8.23303633e-06, -2.77534485e-05],
           [ 5.43072261e-05, -2.77534485e-05,  1.00891325e-04]])




```python
XTXiXT = XTXi.dot(XT)
```


```python
XTXiXT
```




    array([[-0.00825608, -0.00653208, -0.00232159,  0.02565144,  0.01204934],
           [-0.00165852,  0.0011141 ,  0.00139656, -0.00103215, -0.00110698],
           [ 0.00754365, -0.00301964, -0.00455125,  0.00181685,  0.00329418]])




```python
w = XTXiXT.dot(y)
```


```python
w
```




    array([23.12330961, -1.48124183,  5.69922946])




```python

```
