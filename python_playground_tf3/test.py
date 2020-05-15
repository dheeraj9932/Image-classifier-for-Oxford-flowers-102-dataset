from tensorflow.keras.models import Sequential
from tensorflow import math
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D,MaxPool2D,GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

total = 8189
tr = 4874
val = 1633
tst = 1682
num_classes = 102
image_height = 60
image_width = 80
batch_size = 32
epochs = 1

# ###########################################################################################################
training_path = '/media/dheeraj/HDD/crnt/OVGU/sem 2/computer vision and deep learning/assignments/flower project/102flowers/training'
validation_path = '/media/dheeraj/HDD/crnt/OVGU/sem 2/computer vision and deep learning/assignments/flower project/102flowers/validation'
test_path = '/media/dheeraj/HDD/crnt/OVGU/sem 2/computer vision and deep learning/assignments/flower project/102flowers/testing'
# ###########################################################################################################
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                             rescale=1. / 255,
                                             horizontal_flip=True,
                                             width_shift_range=0.1,
                                             height_shift_range=0.1)
data_generator_without_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                                rescale=1. / 255)
# ############################################################################################################
train_generator = data_generator_with_aug.flow_from_directory(
    directory=training_path,
    shuffle=False,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = data_generator_without_aug.flow_from_directory(
    directory=validation_path,
    shuffle=False,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = data_generator_without_aug.flow_from_directory(
    directory=test_path,
    shuffle=False,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

train_imgs, train_labels = train_generator.next()
valid_imgs, valid_labels = validation_generator.next()
test_imgs, test_labels = test_generator.next()

# ###########################################################################################################

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(x, bins=n_bins)
axs[1].hist(y, bins=n_bins)

# ############################################################################################################
model = Sequential()

model.add(Conv2D(12, kernel_size=3, activation='relu', input_shape=(image_height, image_width, 3)))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(20, kernel_size=3, strides=1, activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(40, kernel_size=3, strides=1, activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("image_ht="+str(image_height)+" "+"image_wd="+str(image_width)+" "+"batch_size="+str(batch_size)+" "+"epochs="+str(epochs))

model.summary()

# #######################################################################################################
#
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=tr//batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=val//batch_size)
#
# # ########################################################################################################
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(epochs)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
#
# STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
# results = model.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST)
#
# test_loss, test_acc = model.evaluate(test_imgs, test_labels, verbose=2)
#
# print('test loss, test acc:', results)
# print('test loss:'+str(test_loss)+' test acc:'+str(test_acc))
#
# predictions = model.predict(test_imgs)
#
# Y_pred = model.predict_generator(validation_generator, tst // batch_size+1)
# y_pred = np.argmax(Y_pred, axis=1)
#
# print('Confusion Matrix')
# conf_mat = confusion_matrix(validation_generator.classes, y_pred)
# print(conf_mat)
#
# print('Classification Report')
# class_rep = classification_report(validation_generator.classes, y_pred, output_dict=True)
# print(class_rep)
#
#
# df = pandas.DataFrame(conf_mat).transpose()
# df.to_csv(r'./dftocsv.csv', sep='\t', encoding='utf-8', header='true')
#
# df2 = pandas.DataFrame(class_rep).transpose()
# df2.to_csv(r'./df2tocsv.csv', sep='\t', encoding='utf-8', header='true')
#
