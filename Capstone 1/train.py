import os
import zipfile
import urllib.request
import tensorflow as tf
from tensorflow import keras

if 'archive.zip' in os.listdir():
    print('Data already downloaded')
else:
    url = 'https://www.kaggle.com/datasets/puneet6060/intel-image-classification/download?datasetVersionNumber=2'
    filename = 'archive.zip'
    urllib.request.urlretrieve(url, filename)

if 'data' in os.listdir():
    print('The directory exist')
else:
    os.makedirs('data')
    path = os.getcwd()
    with zipfile.ZipFile('archive.zip', 'r') as zf:
        zf.extractall(os.path.join(path, 'data'))

train_path = 'data\seg_train\seg_train'
test_path = 'data\seg_test\seg_test'

train_batchsize = 128
height = 150
width = 150

epochs = 40

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    label_mode='categorical',
    batch_size=train_batchsize,
    image_size=(height, width)
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    label_mode='categorical',
    batch_size=train_batchsize,
    image_size=(height, width)
)

final_model = tf.keras.Sequential([
    keras.layers.Rescaling(1. / 255, input_shape=(150, 150, 3)),
    keras.layers.Conv2D(16, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])

loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.001)

final_model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy'])


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_accuracy') > 0.81):
            print("\nReached 81% validation accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

history = final_model.fit(train_ds, epochs=epochs, validation_data=(val_ds), callbacks=callbacks)


import pickle
output_file = 'model_final.bin'
f_out = open(output_file, 'wb')
pickle.dump((final_model), f_out)
f_out.close()

final_model.save('model_finale')
