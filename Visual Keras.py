# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:17:10 2022

@author: Home
"""
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, Adamax
from keras.metrics import categorical_crossentropy, binary_crossentropy
#from keras.preprocessing import image
from keras import backend as K
# Import helper functions
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
import numpy as np
import visualkeras
from tensorflow.keras import layers
from collections import defaultdict

num_classes = 2
# input image dimensions
img_rows, img_cols = 255,255
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(255,255,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


#--------------VISUALIZATION OF THE ARCHITECTURE ---------------------------
color_map = defaultdict(dict)
color_map[layers.Conv2D]['fill'] = '#94CBDF'
color_map[layers.MaxPooling2D]['fill'] = '#75F3A2'
color_map[layers.Dense]['fill'] = '#fb5607'
color_map[layers.Flatten]['fill'] = '#EEF107'
visualkeras.layered_view(model, legend=True, color_map=color_map,draw_volume=False)

img=visualkeras.layered_view(model, draw_volume=False)
img