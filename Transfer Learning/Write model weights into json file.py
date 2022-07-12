# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:08:38 2022

@author: Home
"""

from keras.models import model_from_json
import numpy
import os


model_json = model.to_json()
with open("model_resnet50_afteraug_lr002_relu.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_resnet50_afteraug_lr002_relu.h5")
print("Saved model to disk")