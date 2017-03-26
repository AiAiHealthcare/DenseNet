from __future__ import print_function

import os.path

import densenet
import numpy as np
import sklearn.metrics as metrics

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K

batch_size = 64
nb_classes = 3
nb_epoch = 11

img_rows, img_cols = 2048, 2048
img_channels = 1


img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 7
nb_dense_block = 1
growth_rate = 12
nb_filter = 16
dropout_rate = 0.0 # 0.0 for data augmentation

model = densenet.create_dense_net(nb_classes, img_dim, depth, nb_dense_block, growth_rate, nb_filter,
                                  dropout_rate=dropout_rate)
print("Model created")

model.summary()
optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")


## Chollet code
# dimensions of our images.
img_width, img_height = 2048, 2048

train_data_dir = '/data/baseline/P/train'
validation_data_dir = '/data/baseline/P/val'
nb_train_samples = 285
nb_validation_samples = 100
epochs = nb_epoch

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
#    rescale=1. / 255,
#    shear_range=0.2,
#    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

weights_file = "first_try.h5"
lr_reducer      = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6)
early_stopper   = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
model_checkpoint= ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,save_weights_only=True,mode='auto')

callbacks=[lr_reducer,early_stopper,model_checkpoint]

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose=2)

