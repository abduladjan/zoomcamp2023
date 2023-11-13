```python
%pip install wget
```

    Requirement already satisfied: wget in c:\users\abdul\anaconda3\lib\site-packages (3.2)
    Note: you may need to restart the kernel to use updated packages.
    


```python
!python -m wget https://github.com/SVizor42/ML_Zoomcamp/releases/download/bee-wasp-data/data.zip
```

    
    Saved under data.zip
    


```python
import zipfile
```


```python
with zipfile.ZipFile('data.zip', 'r') as zip_ref:
    zip_ref.extractall()
```


```python
pip install tensorflow
```

    Collecting tensorflow
      Downloading tensorflow-2.14.0-cp39-cp39-win_amd64.whl (2.1 kB)
    Collecting tensorflow-intel==2.14.0
      Downloading tensorflow_intel-2.14.0-cp39-cp39-win_amd64.whl (284.1 MB)
         -------------------------------------- 284.1/284.1 MB 6.2 MB/s eta 0:00:00
    Requirement already satisfied: typing-extensions>=3.6.6 in c:\users\abdul\anaconda3\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (4.3.0)
    Requirement already satisfied: wrapt<1.15,>=1.11.0 in c:\users\abdul\anaconda3\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.14.1)
    Collecting ml-dtypes==0.2.0
      Downloading ml_dtypes-0.2.0-cp39-cp39-win_amd64.whl (938 kB)
         -------------------------------------- 938.4/938.4 kB 9.9 MB/s eta 0:00:00
    Collecting absl-py>=1.0.0
      Downloading absl_py-2.0.0-py3-none-any.whl (130 kB)
         -------------------------------------- 130.2/130.2 kB 8.0 MB/s eta 0:00:00
    Collecting google-pasta>=0.1.1
      Downloading google_pasta-0.2.0-py3-none-any.whl (57 kB)
         ---------------------------------------- 57.5/57.5 kB 3.0 MB/s eta 0:00:00
    Collecting tensorflow-io-gcs-filesystem>=0.23.1
      Downloading tensorflow_io_gcs_filesystem-0.31.0-cp39-cp39-win_amd64.whl (1.5 MB)
         ---------------------------------------- 1.5/1.5 MB 8.6 MB/s eta 0:00:00
    Collecting opt-einsum>=2.3.2
      Downloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
         ---------------------------------------- 65.5/65.5 kB 3.5 MB/s eta 0:00:00
    Collecting astunparse>=1.6.0
      Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Requirement already satisfied: setuptools in c:\users\abdul\anaconda3\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (68.2.2)
    Collecting flatbuffers>=23.5.26
      Downloading flatbuffers-23.5.26-py2.py3-none-any.whl (26 kB)
    Requirement already satisfied: six>=1.12.0 in c:\users\abdul\anaconda3\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (1.16.0)
    Collecting tensorboard<2.15,>=2.14
      Downloading tensorboard-2.14.1-py3-none-any.whl (5.5 MB)
         ---------------------------------------- 5.5/5.5 MB 8.6 MB/s eta 0:00:00
    Collecting libclang>=13.0.0
      Downloading libclang-16.0.6-py2.py3-none-win_amd64.whl (24.4 MB)
         ---------------------------------------- 24.4/24.4 MB 4.9 MB/s eta 0:00:00
    Requirement already satisfied: packaging in c:\users\abdul\anaconda3\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (21.3)
    Collecting protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3
      Downloading protobuf-4.25.0-cp39-cp39-win_amd64.whl (413 kB)
         -------------------------------------- 413.2/413.2 kB 3.2 MB/s eta 0:00:00
    Collecting grpcio<2.0,>=1.24.3
      Downloading grpcio-1.59.2-cp39-cp39-win_amd64.whl (3.7 MB)
         ---------------------------------------- 3.7/3.7 MB 8.7 MB/s eta 0:00:00
    Collecting tensorflow-estimator<2.15,>=2.14.0
      Downloading tensorflow_estimator-2.14.0-py2.py3-none-any.whl (440 kB)
         -------------------------------------- 440.7/440.7 kB 6.9 MB/s eta 0:00:00
    Collecting gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1
      Downloading gast-0.5.4-py3-none-any.whl (19 kB)
    Collecting numpy>=1.23.5
      Downloading numpy-1.26.2-cp39-cp39-win_amd64.whl (15.8 MB)
         ---------------------------------------- 15.8/15.8 MB 7.1 MB/s eta 0:00:00
    Collecting keras<2.15,>=2.14.0
      Downloading keras-2.14.0-py3-none-any.whl (1.7 MB)
         ---------------------------------------- 1.7/1.7 MB 10.9 MB/s eta 0:00:00
    Collecting termcolor>=1.1.0
      Downloading termcolor-2.3.0-py3-none-any.whl (6.9 kB)
    Requirement already satisfied: h5py>=2.9.0 in c:\users\abdul\anaconda3\lib\site-packages (from tensorflow-intel==2.14.0->tensorflow) (3.7.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\users\abdul\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.14.0->tensorflow) (0.37.1)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\users\abdul\anaconda3\lib\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.28.1)
    Collecting google-auth<3,>=1.6.3
      Downloading google_auth-2.23.4-py2.py3-none-any.whl (183 kB)
         -------------------------------------- 183.3/183.3 kB 2.7 MB/s eta 0:00:00
    Requirement already satisfied: werkzeug>=1.0.1 in c:\users\abdul\anaconda3\lib\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.0.3)
    Requirement already satisfied: markdown>=2.6.8 in c:\users\abdul\anaconda3\lib\site-packages (from tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (3.3.4)
    Collecting google-auth-oauthlib<1.1,>=0.5
      Downloading google_auth_oauthlib-1.0.0-py2.py3-none-any.whl (18 kB)
    Collecting tensorboard-data-server<0.8.0,>=0.7.0
      Downloading tensorboard_data_server-0.7.2-py3-none-any.whl (2.4 kB)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in c:\users\abdul\anaconda3\lib\site-packages (from packaging->tensorflow-intel==2.14.0->tensorflow) (3.0.9)
    Collecting cachetools<6.0,>=2.0.0
      Downloading cachetools-5.3.2-py3-none-any.whl (9.3 kB)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\abdul\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (0.2.8)
    Collecting rsa<5,>=3.1.4
      Downloading rsa-4.9-py3-none-any.whl (34 kB)
    Collecting requests-oauthlib>=0.7.0
      Downloading requests_oauthlib-1.3.1-py2.py3-none-any.whl (23 kB)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\abdul\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (1.26.11)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\abdul\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2022.9.14)
    Requirement already satisfied: charset-normalizer<3,>=2 in c:\users\abdul\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\abdul\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (3.3)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\users\abdul\anaconda3\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.15,>=2.14->tensorflow-intel==2.14.0->tensorflow) (0.4.8)
    Collecting oauthlib>=3.0.0
      Downloading oauthlib-3.2.2-py3-none-any.whl (151 kB)
         -------------------------------------- 151.7/151.7 kB 9.4 MB/s eta 0:00:00
    Installing collected packages: libclang, flatbuffers, termcolor, tensorflow-io-gcs-filesystem, tensorflow-estimator, tensorboard-data-server, rsa, protobuf, oauthlib, numpy, keras, grpcio, google-pasta, gast, cachetools, astunparse, absl-py, requests-oauthlib, opt-einsum, ml-dtypes, google-auth, google-auth-oauthlib, tensorboard, tensorflow-intel, tensorflow
      Attempting uninstall: numpy
        Found existing installation: numpy 1.21.5
        Uninstalling numpy-1.21.5:
          Successfully uninstalled numpy-1.21.5
    Successfully installed absl-py-2.0.0 astunparse-1.6.3 cachetools-5.3.2 flatbuffers-23.5.26 gast-0.5.4 google-auth-2.23.4 google-auth-oauthlib-1.0.0 google-pasta-0.2.0 grpcio-1.59.2 keras-2.14.0 libclang-16.0.6 ml-dtypes-0.2.0 numpy-1.26.2 oauthlib-3.2.2 opt-einsum-3.3.0 protobuf-4.25.0 requests-oauthlib-1.3.1 rsa-4.9 tensorboard-2.14.1 tensorboard-data-server-0.7.2 tensorflow-2.14.0 tensorflow-estimator-2.14.0 tensorflow-intel-2.14.0 tensorflow-io-gcs-filesystem-0.31.0 termcolor-2.3.0
    Note: you may need to restart the kernel to use updated packages.
    

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    daal4py 2021.6.0 requires daal==2021.4.0, which is not installed.
    scipy 1.9.1 requires numpy<1.25.0,>=1.18.5, but you have numpy 1.26.2 which is incompatible.
    numba 0.55.1 requires numpy<1.22,>=1.18, but you have numpy 1.26.2 which is incompatible.
    


```python
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
```


```python
train_datagen = ImageDataGenerator(rescale=1/255,
                                  rotation_range=50,
width_shift_range=0.1,
height_shift_range=0.1,
zoom_range=0.1,
horizontal_flip=True,
fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1/255)

batch_size = 20

train_generator = train_datagen.flow_from_directory('./data/train', batch_size=batch_size, class_mode='binary', target_size=(150,150), shuffle=True)
test_generator = val_datagen.flow_from_directory('./data/test', batch_size=batch_size, class_mode='binary', target_size=(150,150), shuffle=True)
```

    Found 3677 images belonging to 2 classes.
    Found 918 images belonging to 2 classes.
    


```python
def create_model():
    model = tf.keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        keras.layers.MaxPooling2D(2,2),
        
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.002, momentum=0.8)
    loss=tf.keras.losses.BinaryCrossentropy()
    
    model.compile(optimizer = optimizer,
                  loss = loss,
                  metrics = ['accuracy'])
    
    return model
```


```python
model = create_model()
```


```python
model.summary()
```

    Model: "sequential_6"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_6 (Conv2D)           (None, 148, 148, 32)      896       
                                                                     
     max_pooling2d_6 (MaxPoolin  (None, 74, 74, 32)        0         
     g2D)                                                            
                                                                     
     flatten_6 (Flatten)         (None, 175232)            0         
                                                                     
     dense_12 (Dense)            (None, 64)                11214912  
                                                                     
     dense_13 (Dense)            (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 11215873 (42.79 MB)
    Trainable params: 11215873 (42.79 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    


```python
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)
```

    Epoch 1/10
    184/184 [==============================] - 36s 194ms/step - loss: 0.4898 - accuracy: 0.7683 - val_loss: 0.4989 - val_accuracy: 0.7484
    Epoch 2/10
    184/184 [==============================] - 45s 245ms/step - loss: 0.4996 - accuracy: 0.7626 - val_loss: 0.5036 - val_accuracy: 0.7734
    Epoch 3/10
    184/184 [==============================] - 41s 224ms/step - loss: 0.4837 - accuracy: 0.7778 - val_loss: 0.4984 - val_accuracy: 0.7680
    Epoch 4/10
    184/184 [==============================] - 39s 214ms/step - loss: 0.4763 - accuracy: 0.7696 - val_loss: 0.4975 - val_accuracy: 0.7789
    Epoch 5/10
    184/184 [==============================] - 49s 269ms/step - loss: 0.4660 - accuracy: 0.7775 - val_loss: 0.4805 - val_accuracy: 0.7778
    Epoch 6/10
    184/184 [==============================] - 40s 219ms/step - loss: 0.4716 - accuracy: 0.7805 - val_loss: 0.4645 - val_accuracy: 0.7930
    Epoch 7/10
    184/184 [==============================] - 41s 220ms/step - loss: 0.4721 - accuracy: 0.7852 - val_loss: 0.4784 - val_accuracy: 0.7908
    Epoch 8/10
    184/184 [==============================] - 43s 231ms/step - loss: 0.4657 - accuracy: 0.7917 - val_loss: 0.4521 - val_accuracy: 0.7919
    Epoch 9/10
    184/184 [==============================] - 40s 219ms/step - loss: 0.4611 - accuracy: 0.7890 - val_loss: 0.5052 - val_accuracy: 0.7756
    Epoch 10/10
    184/184 [==============================] - 41s 225ms/step - loss: 0.4521 - accuracy: 0.7925 - val_loss: 0.4717 - val_accuracy: 0.7963
    


```python
import numpy as np

np.median(history.history['accuracy'])
```




    0.7786238789558411




```python
np.std(history.history['loss'])
```




    0.09095241348313114




```python
np.mean(history.history['val_loss'])
```




    0.48507038354873655




```python
np.average(history.history['val_accuracy'][5:])
```




    0.7895424842834473




```python

```
