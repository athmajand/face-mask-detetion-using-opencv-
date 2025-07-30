# Face Mask Detection - Project Documentation

## Project Summary

This project implements a real-time face mask detection system using computer vision and deep learning techniques. The application can detect faces in a video stream from a webcam and determine whether each person is wearing a face mask or not. The system uses a two-stage detection approach:

1. Face detection using OpenCV's DNN module with a pre-trained SSD model
2. Mask classification using a custom-trained MobileNetV2-based model

The application features a user-friendly GUI built with Tkinter that displays the video feed with bounding boxes around detected faces, color-coded to indicate mask status (green for mask, red for no mask), and provides real-time statistics on detection results.

## Installation and Setup

### Prerequisites

- Python 3.6 or higher
- Webcam
- Internet connection (for initial setup to download models)

### Installation Steps

1. Clone the repository or download the project files.

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up the models by running:
   ```
   python setup_models.py
   ```
   This script will:
   - Download the face detector model files
   - Download a dataset of faces with and without masks
   - Train a mask detection model using transfer learning with MobileNetV2

## Running the Application

### Main Application (with Mask Detection)

To run the full face mask detection application:

```
python detect_mask_video.py
```

This will launch the GUI application with the following features:
- Real-time video feed from your webcam
- Face detection with bounding boxes
- Mask classification with color-coded indicators
- Detection statistics display

### Simplified Face Detection (without Mask Detection)

For a simpler version that only performs face detection without mask classification:

```
python simplified_face_detection.py
```

This version requires less computational resources and can be useful for testing or on less powerful hardware.

## Project Structure and Files

### Main Application Files

| File | Description |
|------|-------------|
| `detect_mask_video.py` | Main application with GUI for real-time face mask detection |
| `setup_models.py` | Script to download face detector model and train mask detector model |
| `simplified_face_detection.py` | Simplified version with face detection only (no mask detection) |
| `simplified_setup.py` | Simplified setup script that only downloads face detector model |

### Support Files

| File | Description |
|------|-------------|
| `requirements.txt` | List of Python dependencies required by the project |
| `utils.py` | Utility functions for image processing (placeholder) |
| `test_gui.py` | Simple script to test if Tkinter is working properly |
| `detect_mask_image.py` | Placeholder for static image detection (not fully implemented) |

### Directories

| Directory | Description |
|-----------|-------------|
| `face_detector/` | Contains the face detection model files (downloaded during setup) |
| `dataset/` | Contains the face mask dataset (downloaded during setup) |
| `training/` | Directory for training data and models |

### Models

| Model | Description | Location |
|-------|-------------|----------|
| Face Detector | Pre-trained SSD model for face detection | `face_detector/deploy.prototxt` and `face_detector/res10_300x300_ssd_iter_140000.caffemodel` |
| Mask Detector | Custom-trained model based on MobileNetV2 | `mask_detector.h5` |

## Detailed File Descriptions

### detect_mask_video.py

The main application file that implements the face mask detection system with a GUI. Key features:

- Real-time video capture from webcam
- Face detection using OpenCV DNN
- Mask classification using a trained TensorFlow/Keras model
- Tkinter-based GUI with video display and controls
- Threading for smooth video processing
- Error handling and user feedback

### setup_models.py

Script responsible for setting up all required models and datasets:

- Downloads the face detector model files from OpenCV repositories
- Downloads a dataset of faces with and without masks
- Trains a mask detection model using transfer learning with MobileNetV2
- Saves the trained model for later use

### simplified_face_detection.py

A simplified version of the application that only performs face detection:

- Uses the same face detector as the main application
- Provides a similar GUI interface
- Does not include mask detection functionality
- Useful for testing or on systems with limited resources

### simplified_setup.py

A minimal setup script that only downloads the face detector model:

- Does not download the mask dataset
- Does not train the mask detection model
- Useful for quickly setting up the simplified face detection application

## Technical Implementation Details

### Face Detection

The project uses OpenCV's DNN module with a pre-trained Single Shot Multibox Detector (SSD) with a ResNet base network. This model is efficient for real-time face detection and provides good accuracy.

Key implementation details:
- Input images are converted to 300x300 blobs
- Detection confidence threshold is set to 0.5
- Bounding boxes are drawn around detected faces

### Mask Detection

The mask detection is implemented using a MobileNetV2-based model fine-tuned on a dataset of faces with and without masks:

- Uses transfer learning with MobileNetV2 as the base model
- Adds custom classification layers on top of the base model
- Trained with data augmentation for better generalization
- Outputs probability scores for "mask" and "no mask" classes

### GUI Implementation

The GUI is built using Tkinter and includes:

- Video display canvas
- Start/Stop camera button
- Status indicators
- Results display area
- Proper threading to keep the UI responsive during video processing

## Troubleshooting

### Common Issues

1. **Camera not working**
   - Ensure your webcam is properly connected
   - Check if other applications are using the camera
   - Try restarting the application

2. **Models not found**
   - Run `setup_models.py` to download and train the required models
   - Check if the model files exist in the expected locations

3. **Slow performance**
   - The application requires significant computational resources
   - Close other resource-intensive applications
   - Consider using the simplified version (`simplified_face_detection.py`)

4. **Installation errors**
   - Ensure you have Python 3.6 or higher installed
   - Try creating a virtual environment before installing dependencies
   - Check for any specific platform requirements in the error messages

## Future Improvements

Potential enhancements for the project:

1. Improved mask detection accuracy with a larger and more diverse dataset
2. Support for detecting improper mask wearing (nose exposed, etc.)
3. Multiple camera support
4. Recording and playback functionality
5. Integration with other systems (e.g., access control)
6. Optimization for edge devices and mobile platforms
7. Support for processing video files in addition to webcam feed
8. Batch processing of image directories

## Credits and References

- Face detection model: OpenCV's pre-trained models
- MobileNetV2 architecture: Google/TensorFlow
- Face mask dataset: Based on publicly available datasets
