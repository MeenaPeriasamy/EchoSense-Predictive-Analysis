from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Directory setup
base_dir = 'your directory'

# Define image data generators with a split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2)  # using 20% of the data for validation

# Setup generators
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    subset='training')  # set as training data

validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    subset='validation')  # set as validation data