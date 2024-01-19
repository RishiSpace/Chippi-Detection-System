# Importing Libraries
import os
#!python3 -m pip install tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = 'Chippi-Images'
train_dataset_path = os.path.join(path, 'Train')
test_dataset_path = os.path.join(path, 'Test')

# Define parameters
input_shape = (350, 350, 3)  # Adjust the input shape based on your images
batch_size = 32
epochs = 40

# Create a CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (cat or non-cat)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dataset_path,
    target_size=(350,350),
    batch_size=32,
    class_mode='binary'
)

# No Data Augmentation for Testing
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dataset_path,
    target_size=(350,350),
    batch_size=32,
    class_mode='binary'
)

# Train the model
model.fit(train_generator, epochs=epochs, validation_data=test_generator)


# Saving the Model
model.save('kidney_stone_detection_model.h5')

