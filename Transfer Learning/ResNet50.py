# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:10:35 2021

@author: unaiza.sajid
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
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

model = Sequential()

# 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
model.add(ResNet50(include_top = False, pooling = 'avg'))

# 2nd layer as Dense for 2-class classification, 
model.add(Dense(2, activation = 'softmax'))

# Say not to train first layer (ResNet) model as it is already trained
model.layers[0].trainable = False
model.summary()

model.compile(Adam(lr=0.002), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator()


train_batches = train_datagen.flow_from_directory( directory=r'D:\Unaiza\PNG IMAGE\Train', target_size=[224,224],
classes=['Benign','Malignant'], batch_size=10, class_mode = 'categorical')

test_datagen = ImageDataGenerator()

test_batches = test_datagen.flow_from_directory( directory=r'D:\Unaiza\PNG IMAGE\Test', target_size=[224,224],
classes=['Benign','Malignant'], batch_size=10, class_mode = 'categorical')

history = model.fit_generator(train_batches, steps_per_epoch=len(train_batches)/10, validation_data=test_batches,
                   validation_steps=len(test_batches)/10, epochs=25,verbose=1)
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
