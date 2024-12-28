import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import serial
import time


# Define the directory paths for training and validation datasets
train_dir = 'D:/opencv/dataset/train'  # Use forward slashes
validation_dir = 'D:/opencv/dataset/validation'  # Use forward slashes

# Image dimensions
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generator for validation set
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Initialize serial communication with Arduino
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)  # Adjust '/dev/ttyUSB0' as needed

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

# Check if the model exists and load it if found
model_path = 'D:/opencv/saved_model/mango_disease_model.h5'
if os.path.exists(model_path):
    print("Loading the saved model...")
    model = load_model(model_path)
else:
    print("No saved model found. Building a new model...")
    
    # Load pre-trained MobileNetV2 as the base model
    base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    # Build the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(train_generator.class_indices), activation='softmax')  # Output layer with number of classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        epochs=10,  # Adjust based on your needs
        validation_data=validation_generator
    )

    # Fine-tune by unfreezing the base model and training again
    base_model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

    # Save the model after training
    print("Saving the model...")
    model.save(model_path)

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
    if confidence >= 70:  # Disease confidence threshold
        print("High confidence disease detected! Sending signal to Arduino...")
        arduino.write(b'RUN')  # Send command to Arduino to run the motor
        time.sleep(2)  # Optional: Allow Arduino to process the command
    else:
        print("No significant disease detected.")
        # After making predictions in predict_image
    graph_path, result_path = save_prediction_results(image_path, predictions, class_labels)
    print(f"Graph saved at {graph_path}, Results saved at {result_path}")


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
            cv2.putText(frame, "No tree leafs detected", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if confidence >= 70:  # Disease confidence threshold
            detection_counter += 1
            if detection_counter >= 15:  # Detected for 15 consecutive seconds
                print("Persistent disease detected! Sending signal to Arduino...")
                arduino.write(b'RUN')  # Send command to Arduino
                time.sleep(2)
                detection_counter = 0  # Reset counter after sending the signal
        else:
            detection_counter = 0  # Reset counter if no disease is detected

        # Show the video feed
        cv2.imshow('Video Feed', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example: Confidence values for each class
classes = ['Anthracnose', 'Bacterial Canker', 'Die Back', 'Healthy']
confidences = [93.25, 3.12, 2.18, 1.45]  # Replace with your model's predictions

# Create bar graph
plt.figure(figsize=(10, 5))
plt.bar(classes, confidences, color='skyblue')
plt.xlabel('Disease Types')
plt.ylabel('Confidence Level (%)')
plt.title('Confidence Levels of Disease Prediction')
plt.savefig('confidence_graph.png')
plt.close()

def save_prediction_results(image_path, predictions, class_labels, output_dir="D:/opencv/results"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate bar chart for confidence levels
    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, predictions[0] * 100, color='skyblue')
    plt.xlabel("Diseases")
    plt.ylabel("Confidence (%)")
    plt.title("Prediction Confidence Levels")
    plt.xticks(rotation=45)
    
    # Save the chart
    graph_path = os.path.join(output_dir, "confidence_graph.png")
    plt.savefig(graph_path)
    plt.close()
    
    # Write results to a text file
    result_path = os.path.join(output_dir, "results.txt")
    with open(result_path, "w") as f:
        f.write(f"Image: {image_path}\n")
        for i, label in enumerate(class_labels):
            f.write(f"{label}: {predictions[0][i] * 100:.2f}%\n")
        predicted_class_index = np.argmax(predictions[0])
        f.write(f"Predicted Disease: {class_labels[predicted_class_index]} with confidence {predictions[0][predicted_class_index] * 100:.2f}%\n")

    print(f"Results saved at {output_dir}")
    return graph_path, result_path
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
