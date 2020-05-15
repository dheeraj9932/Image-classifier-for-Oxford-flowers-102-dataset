from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import os

total = 8189
tr = 4874
val = 1633
tst = 1682
num_classes = 102
image_height = 60
image_width = 50
batch_size = 12
epochs = 5

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
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = data_generator_without_aug.flow_from_directory(
    directory=validation_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = data_generator_without_aug.flow_from_directory(
    directory=test_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

print(test_generator)
# for file in test_generator.filenames:
#     print(file)

print('ran')