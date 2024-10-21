import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Define the directory paths for training and validation datasets
train_dir = 'D:/opencv/dataset/train'  # Use forward slashes
validation_dir = 'D:/opencv/dataset/validation'  # Use forward slashes

# Image dimensions
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
num_classes = 8  # Adjust based on the number of disease categories

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generator for validation set
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Creating training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Build a custom CNN architecture from scratch
model = models.Sequential()

# First Convolution + Pooling layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Second Convolution + Pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Third Convolution + Pooling layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Fourth Convolution + Pooling layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Flatten the results to feed into a fully connected layer
model.add(layers.Flatten())

# Fully connected (Dense) layer
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))

# Output layer with softmax activation (multi-class classification)
model.add(layers.Dense(num_classes, activation='softmax'))

# Print the model summary to understand its structure
model.summary()

# Compile the CNN model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting and save the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Save the trained model
model.save('custom_cnn_model.h5')

# Load the model (if you want to use it later)
model = tf.keras.models.load_model('custom_cnn_model.h5')

# Define the class labels based on your dataset
class_labels = [
    'Anthracnose',
    'Bacterial Canker',
    'Cutting Weevil',
    'Die Back',
    'Gall Midge',
    'Healthy',
    'Powdery Mildew',
    'Sooty Mould'
]

# Function to predict disease from an image
def predict_image(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open image at {image_path}")
        return

    # Resize image to the input size expected by the model
    image_resized = cv2.resize(image, (150, 150))  # Adjust the size as needed
    image_array = img_to_array(image_resized) / 255.0  # Normalize the pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(image_array)

    # Get the predicted class and its confidence
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index] * 100

    # Display the result
    print(f"Predicted Disease: {class_labels[predicted_class_index]} with confidence {confidence:.2f}%")

# Video Processing Function (Real-time Detection)
def process_video():
    cap = cv2.VideoCapture(0)  # Open default camera
    if not cap.isOpened():
        print("Error: Cannot access the camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize the frame for consistent processing
        frame_resized = cv2.resize(frame, (150, 150))  # Adjust to model input size
        frame_array = img_to_array(frame_resized) / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)

        # Make predictions
        predictions = model.predict(frame_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index] * 100

        # Check if there is enough "green" in the frame (indicating leaves)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([36, 25, 25])
        upper_green = np.array([86, 255, 255])
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        green_percentage = np.sum(green_mask) / (green_mask.shape[0] * green_mask.shape[1])

        if green_percentage > 0.1:  # If more than 10% of the frame is green
            # Display the prediction on the frame
            cv2.putText(frame, f"{class_labels[predicted_class_index]}: {confidence:.2f}%", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No tree detected", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the video feed
        cv2.imshow('Video Feed', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Photo Processing Function
def process_photo(image_path):
    print("Starting Photo Mode...")
    predict_image(image_path)

# Main Function with Mode Selection
def main():
    while True:
        print("\nChoose an option:")
        print("0: Video Mode")
        print("1: Photo Mode")
        print("q: Quit")

        user_input = input("Enter your choice: ")

        if user_input == '0':
            print("Starting Video Mode...")
            process_video()
        elif user_input == '1':
            image_path = input("Enter the full path to the image file: ")
            process_photo(image_path)
        elif user_input == 'q':
            print("Quitting...")
            break
        else:
            print("Invalid input. Please choose '0', '1', or 'q'.")

if __name__ == "__main__":
    main()
