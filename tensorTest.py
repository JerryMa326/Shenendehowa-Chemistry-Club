import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# Define paths and parameters
image_folder = 'path_to_your_image_folder'  # Change to your image folder path
batch_size = 32
img_height = 224
img_width = 224
epochs = 10
train_test_split = 0.8  # 80% train, 20% validation

# 1. Load and preprocess the image data
# Using ImageDataGenerator to load and preprocess images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=1 - train_test_split  # split for validation
)

train_generator = train_datagen.flow_from_directory(
    image_folder,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # For multi-class classification
    subset='training'  # Set as 'training' for training data
)

validation_generator = train_datagen.flow_from_directory(
    image_folder,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # For multi-class classification
    subset='validation'  # Set as 'validation' for validation data
)

# 2. Build the MobileNetV2 model
# Load MobileNetV2 with weights pre-trained on ImageNet and exclude the top classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model to retain its pre-trained weights
base_model.trainable = False

# Add custom top layers for your classification task
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer
])

# 3. Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# 5. Evaluate and create a chart of the training/validation accuracy and loss
# Plot accuracy
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

# Show plots
plt.tight_layout()
plt.show()

# Save the model for future use
model.save('mobilenetv2_model.h5')
