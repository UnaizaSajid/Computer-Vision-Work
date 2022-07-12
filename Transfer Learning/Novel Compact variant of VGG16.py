

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

train_datagen = ImageDataGenerator()
train_batches = train_datagen.flow_from_directory( directory=r'D:\Unaiza\PNG Image\Train', target_size=[255,255],
classes=['Benign','Malignant'], batch_size=20, class_mode = 'categorical')

test_datagen = ImageDataGenerator()
test_batches = test_datagen.flow_from_directory( directory=r'D:\Unaiza\PNG Image\Test', target_size=[255,255],
classes=['Benign','Malignant'], batch_size=20, class_mode = 'categorical')

model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_batches, validation_data=test_batches, epochs=100,verbose=1, steps_per_epoch=50,shuffle=False)


#  "Accuracy"
import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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


#Writing the activations of second last layer into a dataframe

from keras.models import Model
import pandas as pd

inputs = model.input
outputs = [model.layers[i].output for i in range(len(model.layers))]
vis = Model(inputs, outputs)

activations = vis.predict(train_batches)
sec_last_layer_activation = activations[-2]
print(sec_last_layer_activation.shape)

activations_cnn=pd.DataFrame(sec_last_layer_activation)
