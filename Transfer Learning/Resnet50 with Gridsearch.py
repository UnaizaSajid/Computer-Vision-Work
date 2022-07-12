# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:18:52 2021

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
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifier


def create_model(learn_rate=0.01):
    model = Sequential()
    model.add(ResNet50(include_top = False, pooling = 'avg'))
    # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
    model.add(Dense(2, activation = 'softmax'))
    
    # Say not to train first layer (ResNet) model as it is already trained
    model.layers[0].trainable = False
    model.summary()
    model.compile(Adam(lr=learn_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model



train_datagen = ImageDataGenerator()

train_batches = train_datagen.flow_from_directory( directory=r'D:\Unaiza\PNG Image\Train', target_size=[224,224],
classes=['Benign','Malignant'], batch_size=10, class_mode = 'categorical')

test_datagen = ImageDataGenerator()

test_batches = test_datagen.flow_from_directory( directory=r'D:\Unaiza\PNG Image\Test', target_size=[224,224],
classes=['Benign','Malignant'], batch_size=10, class_mode = 'categorical')


X, Y = train_batches.next()

model = KerasClassifier(build_fn=create_model, verbose=0, epochs=100)

batch_size=[10]
learn_rate = [0.001, 0.01, 0.1, 0.2]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

param_grid = dict(batch_size=batch_size, learn_rate=learn_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5)
grid_result = grid.fit(X, Y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    
