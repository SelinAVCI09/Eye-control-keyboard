# Eye-Tracking Virtual Keyboard

## Overview
This project is an eye-tracking virtual keyboard that allows users to type using only their eye movements. It utilizes OpenCV and MediaPipe FaceMesh to detect eye positions and map them to a virtual keyboard. The user can calibrate the system by looking at reference points, after which the application tracks eye movements to determine which key the user is focusing on.

## Features
- **Eye-tracking technology**: Uses MediaPipe FaceMesh to detect eye positions in real-time.
- **Calibration system**: Users calibrate the system by looking at predefined points.
- **Virtual keyboard**: A full on-screen keyboard for text input.
- **Key selection**: Users select keys by gazing at them for a few seconds.
- **Smoothing algorithm**: Reduces noise and improves accuracy in gaze tracking.
- **Basic text editing**: Includes space and delete functionality.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/eye-tracking-keyboard.git
   cd eye-tracking-keyboard
   ```
2. Install dependencies:
   ```sh
   pip install opencv-python mediapipe numpy
   ```
3. Run the application:
   ```sh
   python eye_tracking_keyboard.py
   ```

## How It Works
1. **Calibration**: The program asks the user to look at four reference points displayed on the screen. This helps map eye movements to screen coordinates.
2. **Eye tracking**: The system continuously tracks the position of the user's pupils.
3. **Key selection**: The program determines which key the user is focusing on and selects it after a short delay.
4. **Typing**: Selected keys are displayed on the screen, forming words and sentences.

## Usage Instructions
- Start the program, and follow the calibration instructions.
- Once calibration is complete, look at the key you want to select.
- Hold your gaze on a key for about 3 seconds to register the selection.
- Use the "Spc" key for space and "Del" to delete characters.
- Press `q` to exit the application.

## Future Improvements
- Enhancing gaze-tracking accuracy with machine learning techniques.
- Adding support for multiple languages and layouts.
- Implementing an adaptive dwell-time system for faster typing.


