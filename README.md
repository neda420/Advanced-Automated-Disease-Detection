# Mango Tree Disease Detection and Irrigation System

This project implements a real-time mango tree disease detection and targeted irrigation system. The solution leverages image processing, machine learning, and hardware automation to detect diseases in mango leaves and provide precise irrigation to affected areas.

## Features

- **Real-Time Disease Detection**: Uses a CNN model for detecting diseases in mango leaves with high confidence.
- **Automated Irrigation**: Activates a targeted sprinkler system upon detecting diseased leaves.
- **Energy Efficiency**: Designed to minimize energy and water usage.
- **Multi-Crop Support**: Can be extended to support other crops with dataset updates.
- **Data Logging**: Maintains records of detected diseases and irrigation activity.
- **Hardware Integration**: Combines Raspberry Pi, Arduino, ESP camera, and water pump for system functionality.

## Components

1. **Hardware**:
   - Raspberry Pi 4
   - Arduino Uno
   - ESP Camera
   - Logitech C922 Pro Webcam
   - MINI DC12V Brushless Pump
   - Relay Module
   - Rotating Garden Sprinkler
   - LiPo Battery (2200mAh)

2. **Software**:
   - TensorFlow/Keras for model training
   - OpenCV for image processing
   - Python for system control and integration
   - Arduino IDE for microcontroller programming

## System Architecture

1. **Image Processing**:
   - Frames captured using a webcam are resized and passed to a pre-trained CNN model.
   - The system filters frames based on green area detection to avoid false positives.

2. **Disease Detection**:
   - The CNN model predicts the presence and type of disease with a confidence threshold of 70%.
   - Continuous detection for 15 seconds triggers the irrigation system.

3. **Irrigation System**:
   - Raspberry Pi sends a command to Arduino to activate the water pump.
   - The system ensures precise watering only to affected areas.

## Setup Instructions

### Hardware Setup

1. Connect the Logitech webcam to the Raspberry Pi.
2. Interface the ESP camera with Raspberry Pi for additional image capture.
3. Link the Raspberry Pi and Arduino via serial communication.
4. Connect the relay module to the Arduino to control the water pump.
5. Assemble the sprinkler system and ensure the pump is functional.

### Software Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Install required Python libraries:
   ```bash
   pip install tensorflow keras opencv-python pyserial
   ```
3. Load the pre-trained CNN model into the project.
4. Upload the Arduino sketch to the Arduino Uno using the Arduino IDE.
5. Run the Python script to start the system:
   ```bash
   python main.py
   ```

## How It Works

1. The webcam captures video frames of mango leaves.
2. Frames are processed and analyzed using the CNN model to detect diseases.
3. If a disease is detected, and the confidence exceeds the threshold, the system sends a signal to activate the water pump.
4. Irrigation is targeted to the affected area, conserving water and reducing resource wastage.

## Future Enhancements

- **Mobile Application**: Provide a user-friendly interface for monitoring and controlling the system.
- **Cloud Integration**: Store and analyze disease data for long-term insights.
- **Weather Adaptability**: Adjust irrigation based on real-time weather data.
- **Multi-Crop Support**: Expand the system for detecting diseases in other crops.

## Acknowledgments

This project integrates concepts from precision agriculture, machine learning, and IoT, aimed at advancing sustainable farming practices.

