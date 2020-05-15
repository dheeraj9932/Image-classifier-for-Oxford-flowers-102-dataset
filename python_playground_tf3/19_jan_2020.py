from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.keras
import keras
from keras.optimizers import SGD, Adam
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Sequential
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from keras.layers import *
from keras.models import Sequential
from keras import optimizers
import os
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D

print(tensorflow.keras.__version__)

training_path = '/media/dheeraj/HDD/crnt/OVGU/sem 2/computer vision and deep learning/assignments/flower project/102flowers/training'
validation_path = '/media/dheeraj/HDD/crnt/OVGU/sem 2/computer vision and deep learning/assignments/flower project/102flowers/validation'
test_path = '/media/dheeraj/HDD/crnt/OVGU/sem 2/computer vision and deep learning/assignments/flower project/102flowers/testing'

data_dir_training = pathlib.Path(training_path)
data_dir_validation = pathlib.Path(validation_path)
data_dir_testing = pathlib.Path(test_path)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,width_shift_range=0.2, height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, channel_shift_range=10, horizontal_flip=True, rescale=1./255)
image_generator_2 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
image_generator_1 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

import numpy as np
BATCH_SIZE = 128
IMG_HEIGHT = 125
IMG_WIDTH = 160
epochs = 20
# STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
CLASS_NAMES_TRAIN = np.array([item.name for item in data_dir_training.glob('*') if item.name != ".DS_Store"])
CLASS_NAMES_VALID = np.array([item.name for item in data_dir_validation.glob('*') if item.name != ".DS_Store"])
print(CLASS_NAMES_VALID)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir_training),
                                                     batch_size=64,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES_TRAIN),class_mode='categorical')
valid_data_gen = image_generator_2.flow_from_directory(directory=str(data_dir_validation),
                                                     batch_size=64,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')
test_data_gen = image_generator_1.flow_from_directory(directory=str(data_dir_testing),
                                                     batch_size=64,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical')
train_data_gen
valid_data_gen

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))
model.add(Dense(102, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=Adam(lr=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit_generator(generator=train_data_gen,
                   validation_data= valid_data_gen,
    steps_per_epoch=4874//BATCH_SIZE,
    epochs=epochs, validation_steps=1633//BATCH_SIZE
)

#model.evaluate_generator(train_data_gen)
#model.evaluate_generator(valid_data_gen)
#model.evaluate_generator(test_data_gen)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print('got till here')
STEP_SIZE_TEST = test_data_gen.n // test_data_gen.batch_size
results = model.evaluate_generator(generator=test_data_gen, steps=STEP_SIZE_TEST)
# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('test loss, test acc:', results)