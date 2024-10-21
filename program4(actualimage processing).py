import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras import layers, models# Define the directory paths for training and validation datasets
train_dir = 'D:/opencv/dataset/train'  # Use forward slashes
validation_dir = 'D:/opencv/dataset/validation'  # Use forward slashes

# Image dimensions
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

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

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Output layer with number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # You can change the number of epochs as needed
    validation_data=validation_generator
)

# Save the model
model.save('path/to/your/model.h5')  # Adjust the path for saving the model

# Load your trained model (adjust the path accordingly)
model = tf.keras.models.load_model('path/to/your/model.h5')

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
    predicted_class_index = np.argmax(predictions[0])  # Get index of the highest probability
    confidence = predictions[0][predicted_class_index] * 100  # Convert to percentage

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

        # Display the prediction on the frame
        cv2.putText(frame, f"{class_labels[predicted_class_index]}: {confidence:.2f}%", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
