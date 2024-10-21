import cv2
import numpy as np

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

    # Display the result images
    cv2.imshow(f'Disease Detection Result - {disease_type}', result)

    # Threshold to decide if disease is present
    if disease_percentage > 5:  # 5% threshold
        print(f"{disease_type} detected: {disease_percentage:.2f}% of leaf affected.")
        return mask, disease_percentage
    else:
        print(f"No significant {disease_type} detected.")
        return np.zeros_like(mask), 0  # Return empty mask if no disease detected

# Photo Processing Function
def process_photo(image_path):
    # Load image from the provided path
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open image at {image_path}")
        return

    # Resize the image once for consistent dimensions
    resized_image = cv2.resize(image, (600, 400))

    # Initialize combined disease mask
    combined_disease_mask = np.zeros(resized_image.shape[:2], dtype=np.uint8)
    
    diseases = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
                'Gall Midge', 'Powdery Mildew', 'Sooty Mould']

    # Detect each disease and update the combined disease mask
    for disease in diseases:
        disease_mask, _ = detect_disease(resized_image, disease)
        combined_disease_mask = cv2.bitwise_or(combined_disease_mask, disease_mask)
    
    # Detect healthy areas not covered by diseases
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    # Adjusted HSV range for healthy green color
    lower_bound_healthy = np.array([30, 80, 80])
    upper_bound_healthy = np.array([90, 255, 255])

    # Create mask for healthy areas
    healthy_mask = cv2.inRange(hsv_image, lower_bound_healthy, upper_bound_healthy)

    # Visualize the healthy mask before excluding diseased areas
    cv2.imshow('Healthy Mask Before Excluding Diseased Areas', healthy_mask)

    # Exclude diseased areas from healthy mask
    healthy_mask = cv2.bitwise_and(healthy_mask, cv2.bitwise_not(combined_disease_mask))

    # Calculate healthy percentage
    healthy_pixels = cv2.countNonZero(healthy_mask)
    total_pixels = resized_image.shape[0] * resized_image.shape[1]
    healthy_percentage = (healthy_pixels / total_pixels) * 100

    print(f"Healthy detected: {healthy_percentage:.2f}% of leaf unaffected by diseases.")

    # Display the healthy areas
    healthy_result = cv2.bitwise_and(resized_image, resized_image, mask=healthy_mask)
    cv2.imshow('Healthy Areas', healthy_result)

    # Wait for key press to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        frame_resized = cv2.resize(frame, (600, 400))

        # Initialize combined disease mask
        combined_disease_mask = np.zeros(frame_resized.shape[:2], dtype=np.uint8)

        # Detect each disease and update the combined disease mask
        diseases = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
                    'Gall Midge', 'Powdery Mildew', 'Sooty Mould']

        for disease in diseases:
            disease_mask, _ = detect_disease(frame_resized, disease)
            combined_disease_mask = cv2.bitwise_or(combined_disease_mask, disease_mask)

        # Detect healthy areas not covered by diseases
        hsv_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        lower_bound_healthy = np.array([30, 80, 80])
        upper_bound_healthy = np.array([90, 255, 255])

        healthy_mask = cv2.inRange(hsv_image, lower_bound_healthy, upper_bound_healthy)
        healthy_mask = cv2.bitwise_and(healthy_mask, cv2.bitwise_not(combined_disease_mask))

        healthy_pixels = cv2.countNonZero(healthy_mask)
        total_pixels = frame_resized.shape[0] * frame_resized.shape[1]
        healthy_percentage = (healthy_pixels / total_pixels) * 100

        print(f"Healthy detected: {healthy_percentage:.2f}% of leaf unaffected by diseases.")

        # Display the video feed with healthy areas highlighted
        healthy_result = cv2.bitwise_and(frame_resized, frame_resized, mask=healthy_mask)
        cv2.imshow('Video Feed - Healthy Areas', healthy_result)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

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
            print("Starting Photo Mode...")
            process_photo(image_path)
        elif user_input == 'q':
            print("Quitting...")
            break
        else:
            print("Invalid input. Please choose '0', '1', or 'q'.")

if __name__ == "__main__":
    main()
