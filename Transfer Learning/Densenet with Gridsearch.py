# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 08:40:19 2021

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
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

N_CLASSES=2
def ClsModel(n_classes=2, input_shape=(224,224,3), learn_rate=0.01):
    base_model = DenseNet121(weights=None, include_top=False, input_shape=input_shape)
    x = AveragePooling2D(pool_size=(3,3), name='avg_pool')(base_model.output)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='dense_post_pool')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, output=output)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = ClsModel(N_CLASSES)
model.summary()


train_datagen = ImageDataGenerator()


train_batches = train_datagen.flow_from_directory( directory=r'D:\Unaiza\PNG Image\Train', target_size=[224,224],
classes=['Benign','Malignant'], batch_size=10, class_mode = 'categorical')




test_datagen = ImageDataGenerator()
test_batches = test_datagen.flow_from_directory( directory=r'D:\Unaiza\PNG Image\Test', target_size=[224,224],
classes=['Benign','Malignant'], batch_size=10, class_mode = 'categorical')


X, Y = train_batches.next()

model = KerasClassifier(build_fn=ClsModel, verbose=0, epochs=100)

batch_size=[10]
learn_rate = [0.001, 0.01, 0.1]
optimizer = ['SGD', 'Adagrad',  'Adam', 'Adamax']
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

param_grid = dict(batch_size=batch_size, learn_rate=learn_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5)
grid_result = grid.fit(X, Y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

