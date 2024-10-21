import cv2
import numpy as np
from gpiozero import Servo
from time import sleep

# Simulate Servo control (GPIO mock)
class MockServo:
    def __init__(self):
        self.angle = 0
    
    def set_position(self, position):
        self.angle = position
        print(f"Servo moved to position {position}")  # Simulated action

# Disease Detection Function
def detect_disease(image, disease_type):
    # Resize image for easier processing
    image = cv2.resize(image, (600, 400))  

    # Convert image to HSV color space for better color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define disease-specific HSV ranges (adjust based on disease type)
    if disease_type == "Anthracnose":
        lower_bound = np.array([15, 50, 50])
        upper_bound = np.array([35, 255, 255])
    elif disease_type == "Bacterial Canker":
        lower_bound = np.array([20, 40, 40])
        upper_bound = np.array([40, 255, 255])
    elif disease_type == "Cutting Weevil":
        lower_bound = np.array([10, 100, 50])
        upper_bound = np.array([30, 255, 255])
    elif disease_type == "Die Back":
        lower_bound = np.array([5, 40, 40])
        upper_bound = np.array([25, 255, 255])
    elif disease_type == "Gall Midge":
        lower_bound = np.array([12, 60, 60])
        upper_bound = np.array([32, 255, 255])
    elif disease_type == "Powdery Mildew":
        lower_bound = np.array([0, 0, 200])  # White
        upper_bound = np.array([180, 30, 255])
    elif disease_type == "Sooty Mould":
        lower_bound = np.array([0, 0, 0])  # Black
        upper_bound = np.array([180, 255, 50])
    elif disease_type == "Healthy":
        lower_bound = np.array([35, 100, 100])  # Green
        upper_bound = np.array([85, 255, 255])
    else:
        raise ValueError(f"Unknown disease type: {disease_type}")

    # Create mask for diseased parts
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Use bitwise operation to extract diseased regions
    result = cv2.bitwise_and(image, image, mask=mask)

    # Count non-zero pixels in the mask to determine disease presence
    disease_pixels = cv2.countNonZero(mask)
    total_pixels = mask.size
    disease_percentage = (disease_pixels / total_pixels) * 100

    # Display the original and result images
    cv2.imshow(f'Disease Detection Result - {disease_type}', result)

    # Threshold to trigger servo if disease is detected
    if disease_percentage > 5:  # 5% of the leaf is diseased
        print(f"{disease_type} detected: {disease_percentage:.2f}% of leaf affected.")
        return True
    else:
        print(f"No significant {disease_type} detected.")
        return False

# Servo Control Function (Mocked)
def activate_servo(servo, duration=3):
    print("Activating Servo for spraying medicine...")
    servo.set_position(1)  # Move to position (spraying)
    sleep(duration)  # Simulated spraying time
    servo.set_position(-1)  # Reset position
    print("Servo reset to initial position.")

# Video Processing Function
def process_video(servo_motor):
    # Open the camera (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Loop to continuously get frames from the camera
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break

        # Display the live video feed from the camera
        cv2.imshow('Live Camera Feed', frame)

        # Process the frame for disease detection
        disease_detected = False
        for disease in ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
                        'Gall Midge', 'Powdery Mildew', 'Sooty Mould', 'Healthy']:
            if detect_disease(frame, disease):
                disease_detected = True

        # If any disease is detected, activate the servo motor
        if disease_detected:
            activate_servo(servo_motor)

        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Photo Processing Function
def process_photo(servo_motor, image_path):
    # Load image from the provided path
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open image at {image_path}")
        return

    # Process the image for disease detection
    disease_detected = False
    for disease in ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
                    'Gall Midge', 'Powdery Mildew', 'Sooty Mould', 'Healthy']:
        if detect_disease(image, disease):
            disease_detected = True

    # If any disease is detected, activate the servo motor
    if disease_detected:
        activate_servo(servo_motor)

# Main Function
def main():
    # Initialize the mock servo
    servo_motor = MockServo()

    while True:
        # Ask the user for input to either capture video or photo
        user_input = input("Press 0 for Video mode, 1 for Photo mode, or 'q' to quit: ")

        if user_input == '0':
            # Video mode
            print("Starting Video Mode...")
            process_video(servo_motor)
        elif user_input == '1':
            # Photo mode - ask for image path
            image_path = input("Enter the full path to the image file: ")
            print("Starting Photo Mode...")
            process_photo(servo_motor, image_path)
        elif user_input == 'q':
            # Quit the program
            print("Exiting...")
            break
        else:
            print("Invalid input, please enter 0, 1, or 'q'.")

if __name__ == "__main__":
    main()
