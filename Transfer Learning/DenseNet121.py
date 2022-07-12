# -*- coding: utf-8 -*-
"""

Created on Mon Feb  8 14:15:30 2021

@author: unaiza.sajid
"""

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import os
import pickle
import tensorflow as tf
from collections import defaultdict
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.optimizers import Adam, SGD, Adagrad
from keras.models import Model, load_model
from keras.layers import *
from sklearn.model_selection import train_test_split
from keras.callbacks import *

N_CLASSES=2
def ClsModel(n_classes=2, input_shape=(224,224,3)):
    base_model = DenseNet121(weights=None, include_top=False, input_shape=input_shape)
    x = AveragePooling2D(pool_size=(3,3), name='avg_pool')(base_model.output)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='dense_post_pool')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, output=output)
    return model

model = ClsModel(N_CLASSES)
model.summary()


model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

model_checkpoint = ModelCheckpoint(('./densenet.{epoch:02d}.hdf5'),
                                   monitor='val_loss',
                                   verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0, verbose=1)

callbacks = [model_checkpoint, reduce_learning_rate]



train_datagen = ImageDataGenerator()


train_batches = train_datagen.flow_from_directory( directory=r'D:\Unaiza\PNG IMAGE\Train', target_size=[224,224],
classes=['Benign','Malignant'], batch_size=10, class_mode = 'categorical')




test_datagen = ImageDataGenerator()


test_batches = test_datagen.flow_from_directory( directory=r'D:\Unaiza\PNG IMAGE\Test', target_size=[224,224],
classes=['Benign','Malignant'], batch_size=10, class_mode = 'categorical')



history = model.fit_generator(train_batches, steps_per_epoch=len(train_batches)/10, validation_data=test_batches,
                   validation_steps=len(test_batches)/10, epochs=100,verbose=1, callbacks=callbacks)
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
