import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set dataset path
base_dir = "C:/Users/nihal/PycharmProjects/ML Glob/dog-vs-cat"

# Set model save path
model_path = "C:/Users/nihal/PycharmProjects/ML Glob/dog_vs_cat_model.h5"

if os.path.exists(model_path):
    # Load the model if it exists
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully from:", model_path)
else:
    # Create datasets
    # train dataset
    train_dataset = image_dataset_from_directory(
        base_dir,
        image_size=(200, 200),
        subset="training",
        seed=1,
        validation_split=0.1,
        batch_size=32
    )
    # validation dataset
    val_dataset = image_dataset_from_directory(
        base_dir,
        image_size=(200, 200),
        subset="validation",
        seed=1,
        validation_split=0.1,
        batch_size=32
    )

    # Define the layers in the CNN model
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset
    )

    # Save the trained model
    model.save(model_path)
    print(f"Model saved to {model_path}")

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

# Path to the test image
test_image_path = "C:/Users/nihal/PycharmProjects/ML Glob/2.jpg"

# Load the image with the target size
test_image = image.load_img(test_image_path, target_size=(200, 200))

# Convert the image to an array and preprocess
test_image_array = image.img_to_array(test_image)
test_image_array = np.expand_dims(test_image_array, axis=0)

# Predict the class
result = model.predict(test_image_array)[0][0]  # Extract the predicted probability

# Interpret the result
predicted_label = "Dog" if result >= 0.5 else "Cat"
confidence = result if result >= 0.5 else 1 - result  # Confidence is based on the prediction

# Display the image with the prediction and confidence
plt.imshow(test_image)
plt.axis("off")  # Turn off axes for a cleaner display
plt.title(f"Prediction: {predicted_label} ({confidence:.2f})", fontsize=14, color="blue")
plt.show()


