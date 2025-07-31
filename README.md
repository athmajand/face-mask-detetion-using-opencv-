# Face Mask Detection

A computer vision project using OpenCV and Deep Learning to detect whether people are wearing face masks.

## Setup Instructions

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up the models (downloads face detector and trains mask detector):
   ```
   python setup_models.py
   ```
   This script will:
   - Download the face detector model files
   - Download a dataset of faces with and without masks
   - Train a mask detection model using transfer learning with MobileNetV2

3. Run the face mask detection application:
   ```
   python detect_mask_video.py
   ```

## Features

- Real-time face mask detection using your webcam
- User-friendly GUI with camera on/off controls
- Detection statistics display
- Visual indicators for mask detection (green for mask, red for no mask)

## Requirements

- Python 3.6+
- Webcam
- Dependencies listed in requirements.txt

## Project Structure

- `detect_mask_video.py`: Main application with GUI
- `setup_models.py`: Script to download and train required models
- `face_detector/`: Directory containing face detection model
- `mask_detector.model`: Trained model for mask detection


