# FACE MASK DETECTION

## ABSTRACT
The Face Mask Detection project is an advanced computer vision application that utilizes deep learning techniques to identify whether individuals are wearing face masks in real-time video streams. In response to the global COVID-19 pandemic, this technology offers an automated solution for monitoring mask compliance in public spaces, educational institutions, and workplaces. The system employs a two-stage detection approach combining OpenCV's DNN module for face detection and a custom-trained MobileNetV2-based model for mask classification. With a user-friendly GUI built using Tkinter, the application provides real-time feedback with visual indicators and detection statistics, making it accessible for users with minimal technical expertise.

## LIST OF TABLES
1. Table 2.1: Hardware Requirements
2. Table 2.2: Software Requirements
3. Table 2.3: System Requirements
4. Table 3.1: Comparison of Existing and Proposed Systems
5. Table 4.1: Database Schema
6. Table 5.1: Test Cases and Results

## LIST OF FIGURES
1. Figure 1.1: Project Architecture Overview
2. Figure 2.1: System Configuration Diagram
3. Figure 3.1: System Workflow
4. Figure 4.1: Context Level DFD
5. Figure 4.2: Level 1 DFD
6. Figure 4.3: ER Diagram
7. Figure 5.1: Application GUI
8. Figure 5.2: Mask Detection Results
9. Figure 5.3: Test Case Results

## INTRODUCTION
### 1.1 PROJECT OVERVIEW
The Face Mask Detection project is a computer vision application developed to automatically detect whether individuals in a video stream are wearing face masks. The COVID-19 pandemic has made face masks an essential part of public health safety measures worldwide. This system provides an automated solution for monitoring mask compliance in various settings such as schools, offices, public transportation, and other crowded areas.

The project utilizes deep learning techniques and computer vision algorithms to detect faces in real-time video feeds and classify whether each detected face is wearing a mask or not. The system provides visual feedback by drawing bounding boxes around detected faces, color-coded to indicate mask status (green for mask, red for no mask), and displays real-time statistics on detection results.

### 1.2 FEATURES
1. **Real-time Face Detection**: Identifies human faces in video streams using OpenCV's DNN module with a pre-trained SSD model.
2. **Mask Classification**: Determines whether detected faces are wearing masks using a custom-trained MobileNetV2-based model.
3. **User-friendly GUI**: Provides an intuitive interface with camera controls and real-time video display.
4. **Visual Indicators**: Color-coded bounding boxes (green for mask, red for no mask) for easy interpretation.
5. **Detection Statistics**: Real-time display of detection counts and percentages.
6. **Webcam Integration**: Works with standard webcam hardware for accessibility.
7. **Cross-platform Compatibility**: Functions on Windows, macOS, and Linux operating systems.
8. **Lightweight Implementation**: Optimized for performance on standard consumer hardware.

## SYSTEM CONFIGURATION
### 2.1 HARDWARE SPECIFICATION
- **Processor**: Intel Core i3 or equivalent (Intel Core i5 or higher recommended)
- **RAM**: 4GB minimum (8GB or higher recommended)
- **Storage**: 1GB free disk space for application and models
- **Camera**: Standard webcam (720p or higher resolution recommended)
- **Graphics**: Integrated graphics (dedicated GPU recommended for better performance)
- **Display**: 1366x768 resolution or higher

### 2.2 SOFTWARE SPECIFICATION
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.6 or higher
- **Libraries**:
  - OpenCV (Computer Vision)
  - TensorFlow (Deep Learning)
  - NumPy (Numerical Computing)
  - Pillow (Image Processing)
  - Tkinter (GUI Framework)
  - Matplotlib (Data Visualization)
  - Scikit-learn (Machine Learning Utilities)

### 2.3 SYSTEM SPECIFICATION
- **Python Environment**: Virtual environment recommended for dependency management
- **Model Storage**: Local storage for pre-trained and custom models
- **Camera Access**: System permissions for webcam access
- **User Interface**: Tkinter-based GUI for interaction
- **Processing**: Real-time video processing capabilities
- **Network**: Internet connection required only for initial setup (model download)

## HTML, CSS, JAVASCRIPT
The Face Mask Detection project is primarily a Python-based application and does not directly utilize web technologies like HTML, CSS, and JavaScript in its core functionality. However, these technologies could be implemented in future extensions of the project for web-based deployment:

- **HTML**: Could be used to create a web interface for the application
- **CSS**: Would style the web interface for better user experience
- **JavaScript**: Could handle client-side interactions and potentially integrate with the Python backend via frameworks like Flask or Django

## MYSQL SERVER, PHP
The current implementation does not include database integration or web server components. Future versions could incorporate:

- **MySQL Database**: For storing detection logs, user accounts, and system configuration
- **PHP Backend**: For creating a web-based management interface and API endpoints

## SYSTEM ANALYSIS
### 3.1 PRELIMINARY INVESTIGATION
The preliminary investigation for this project involved:

1. **Problem Identification**: The need for automated mask compliance monitoring during the COVID-19 pandemic
2. **Technology Assessment**: Evaluation of computer vision and deep learning techniques for face and mask detection
3. **Resource Requirements**: Determination of hardware, software, and development resources needed
4. **Feasibility Study**: Analysis of technical, operational, and economic feasibility
5. **Stakeholder Consultation**: Gathering requirements from potential users and administrators

### 3.2 EXISTING SYSTEM
Existing approaches to mask compliance monitoring typically involve:

1. **Manual Monitoring**: Security personnel or staff visually checking for mask compliance
2. **Basic Temperature Screening**: Systems that only check body temperature without mask verification
3. **Simple Image Processing**: Basic systems using rule-based image processing without deep learning
4. **Commercial Solutions**: Expensive proprietary systems with limited customization options

Limitations of existing systems include:
- Labor-intensive manual monitoring
- Inconsistent enforcement
- Limited accuracy in varied lighting conditions
- High cost of commercial solutions
- Lack of real-time statistics and reporting

### 3.3 PROPOSED SYSTEM
The proposed Face Mask Detection system addresses the limitations of existing approaches by:

1. **Automated Detection**: Eliminates the need for constant manual monitoring
2. **Deep Learning Approach**: Provides higher accuracy across various conditions
3. **Real-time Processing**: Offers immediate feedback on mask compliance
4. **User-friendly Interface**: Makes the system accessible to non-technical users
5. **Open-source Foundation**: Allows for customization and extension
6. **Cost-effective Implementation**: Utilizes standard hardware and open-source libraries
7. **Statistical Reporting**: Provides real-time metrics on compliance rates

### 3.4 FEASIBILITY ANALYSIS
#### Technical Feasibility
- The project utilizes established technologies (OpenCV, TensorFlow) with proven capabilities
- Required hardware is readily available and affordable
- Development team has the necessary expertise in computer vision and deep learning
- Implementation is achievable within the project timeframe

#### Operational Feasibility
- The system addresses a clear need for automated mask compliance monitoring
- User interface is designed for ease of use by non-technical personnel
- Minimal training required for system operation
- Can be integrated into existing security protocols

#### Economic Feasibility
- Development costs are reasonable due to use of open-source technologies
- Hardware requirements are modest and widely available
- Return on investment through reduced manual monitoring needs
- Potential reduction in health-related costs through improved compliance

### 3.5 FEASIBILITY CONSTRAINTS
- **Processing Power**: Real-time detection requires adequate computational resources
- **Lighting Conditions**: System performance may vary in extreme lighting conditions
- **Occlusion**: Partial face occlusion (beyond masks) may affect detection accuracy
- **Distance Limitations**: Detection accuracy decreases with increasing distance
- **Privacy Concerns**: Face detection raises potential privacy issues requiring proper handling
- **Deployment Environment**: Physical installation constraints in varied environments

## SYSTEM DESIGN
### 4.1 INTRODUCTION
The system design follows a modular architecture that separates concerns between face detection, mask classification, and user interface components. This design enables maintainability, extensibility, and performance optimization.

The core components include:
1. **Video Capture Module**: Interfaces with the webcam to acquire video frames
2. **Face Detection Module**: Processes frames to identify and locate faces
3. **Mask Classification Module**: Analyzes detected faces to determine mask presence
4. **User Interface Module**: Presents the processed video and results to the user
5. **Statistics Module**: Tracks and calculates detection metrics

### 4.2 DATA FLOW DIAGRAM
#### Context Level DFD (Figure 4.1)
The system interacts with the User and Camera as external entities, with the Face Mask Detection System as the central process.

#### Level 1 DFD (Figure 4.2)
1. **Video Capture Process**: Receives raw video input from the camera
2. **Face Detection Process**: Processes video frames to identify faces
3. **Mask Classification Process**: Analyzes face regions to detect masks
4. **Results Display Process**: Presents processed video and statistics to the user
5. **Data Stores**: Model files for face detection and mask classification

### 4.3 DATABASE DESIGN
The current implementation does not include a database component. A potential database schema for future extensions could include:

#### Users Table
- UserID (Primary Key)
- Username
- Password (Hashed)
- Role (Admin/User)
- LastLogin

#### DetectionLogs Table
- LogID (Primary Key)
- Timestamp
- LocationID (Foreign Key)
- TotalDetections
- MaskedCount
- UnmaskedCount
- ImageReference

#### Locations Table
- LocationID (Primary Key)
- LocationName
- Description
- CameraID

## CODING AND TESTING PHASE
### 5.1 CODING
The application is implemented in Python, leveraging several key libraries:
- **OpenCV**: For video capture and image processing
- **TensorFlow/Keras**: For deep learning model implementation
- **Tkinter**: For GUI development
- **NumPy**: For numerical operations
- **PIL (Pillow)**: For image manipulation

The code follows object-oriented principles with clear separation of concerns between different functional components.

### 5.2 SAMPLE CODE
Key code snippets from the implementation:

#### Face Detection Function
```python
def detect_faces(frame, faceNet, minConfidence=0.5):
    # Get frame dimensions
    (h, w) = frame.shape[:2]
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # Pass the blob through the network
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    # Initialize list for face locations
    faces = []
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence
        confidence = detections[0, 0, i, 2]
        
        # Filter weak detections
        if confidence > minConfidence:
            # Compute bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding box falls within the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # Extract the face ROI
            face = frame[startY:endY, startX:endX]
            
            # Add to faces list if valid
            if face.shape[0] > 0 and face.shape[1] > 0:
                faces.append((face, (startX, startY, endX, endY)))
    
    return faces
```

#### Mask Detection Function
```python
def detect_mask(face, maskNet):
    # Preprocess the face for the mask detector
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    
    # Pass the face through the mask detector model
    (mask, withoutMask) = maskNet.predict(face)[0]
    
    # Determine the class label and color
    label = "Mask" if mask > withoutMask else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
    return (label, color, max(mask, withoutMask) * 100)
```

### 5.3 FORM LAYOUT
The application GUI includes:
1. **Video Display Area**: Shows the processed video feed with detection overlays
2. **Control Panel**: Contains buttons for starting/stopping the camera
3. **Results Panel**: Displays detection statistics and counts
4. **Status Bar**: Shows system status and messages

### 5.4 TESTING
The testing phase included:
1. **Unit Testing**: Individual components tested in isolation
2. **Integration Testing**: Combined components tested for proper interaction
3. **System Testing**: Complete system tested for functionality
4. **Performance Testing**: System evaluated for real-time processing capabilities
5. **Usability Testing**: Interface tested with potential users

### 5.5 TEST CASES
1. **Face Detection Accuracy**: Testing the system's ability to detect faces in various conditions
2. **Mask Classification Accuracy**: Evaluating correct classification of masked vs. unmasked faces
3. **Performance Under Load**: Testing system performance with multiple faces in frame
4. **Lighting Condition Handling**: Testing in various lighting environments
5. **GUI Responsiveness**: Evaluating interface responsiveness during operation
6. **Error Handling**: Testing system response to unexpected conditions
7. **Resource Utilization**: Monitoring CPU, memory, and GPU usage

## CONCLUSION AND FUTURE SCOPE
### 6.1 CONCLUSION
The Face Mask Detection system successfully demonstrates the application of computer vision and deep learning techniques to address a real-world problem. By automating the detection of face mask usage, the system provides an efficient and reliable solution for monitoring compliance with health safety protocols.

Key achievements of the project include:
1. Development of a real-time face mask detection system with high accuracy
2. Implementation of a user-friendly interface accessible to non-technical users
3. Integration of advanced deep learning models in a practical application
4. Creation of a system that can be deployed in various environments
5. Demonstration of how technology can contribute to public health initiatives

The project showcases the potential of artificial intelligence in addressing contemporary challenges and provides a foundation for further development in related applications.

### 6.2 FUTURE SCOPE
The Face Mask Detection system can be extended and enhanced in several directions:

1. **Multi-camera Support**: Integration with multiple camera feeds for broader coverage
2. **Cloud Integration**: Remote storage and processing capabilities
3. **Mobile Application**: Development of companion mobile apps for remote monitoring
4. **Advanced Analytics**: Enhanced statistical analysis and reporting features
5. **Improper Mask Detection**: Identifying incorrectly worn masks (e.g., nose exposed)
6. **Integration with Access Control**: Linking with door systems for automated entry management
7. **Crowd Density Analysis**: Adding capabilities to monitor social distancing
8. **Edge Device Optimization**: Adapting the system for deployment on edge computing devices
9. **Alert System**: Implementing notification mechanisms for compliance violations
10. **Web Dashboard**: Creating a web-based monitoring interface for administrators

## BIBLIOGRAPHY
1. Rosebrock, A. (2020). "COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning." PyImageSearch.
2. Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." arXiv preprint arXiv:1704.04861.
3. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
4. Redmon, J., & Farhadi, A. (2018). "YOLOv3: An Incremental Improvement." arXiv preprint arXiv:1804.02767.
5. Bradski, G. (2000). "The OpenCV Library." Dr. Dobb's Journal of Software Tools.
6. Abadi, M., et al. (2016). "TensorFlow: A System for Large-Scale Machine Learning." 12th USENIX Symposium on Operating Systems Design and Implementation.
7. World Health Organization. (2020). "Advice on the use of masks in the context of COVID-19."
8. Centers for Disease Control and Prevention. (2020). "Use of Cloth Face Coverings to Help Slow the Spread of COVID-19."
