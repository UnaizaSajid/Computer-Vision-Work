# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:33:16 2021

@author: unaiza.sajid
"""


import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras import layers
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import matplotlib.pyplot as plt
from keras.applications import vgg16, inception_v3
from keras.optimizers import Adam, SGD, Adamax
from keras.metrics import categorical_crossentropy, binary_crossentropy
#from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.resnet50 import ResNet50
from keras.applications import InceptionResNetV2
from keras.callbacks import *

model = Sequential()

# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
model.add(InceptionResNetV2(include_top=False, input_shape=(150,150,3)))


model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation = 'softmax'))


# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False
model.summary()

model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model_checkpoint = ModelCheckpoint(('./densenet.{epoch:02d}.hdf5'),
                                   monitor='val_loss',
                                   verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                         verbose=1)

callbacks = [model_checkpoint, reduce_learning_rate]

train_datagen = ImageDataGenerator()


train_batches = train_datagen.flow_from_directory( directory=r'D:\Unaiza\PNG IMAGE\Train', target_size=[150,150],
classes=['Benign','Malignant'], batch_size=10, class_mode = 'categorical')




test_datagen = ImageDataGenerator()


test_batches = test_datagen.flow_from_directory( directory=r'D:\Unaiza\PNG IMAGE\Test', target_size=[150,150],
classes=['Benign','Malignant'], batch_size=10, class_mode = 'categorical')

history = model.fit_generator(train_batches, steps_per_epoch=len(train_batches)/10, validation_data=test_batches,
                   validation_steps=len(test_batches)/10, epochs=100,verbose=1,callbacks=callbacks)
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