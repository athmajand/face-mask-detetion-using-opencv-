import os
import urllib.request

def download_face_detector():
    print("[INFO] Downloading face detector model...")
    
    # Create face_detector directory if it doesn't exist
    if not os.path.exists("face_detector"):
        os.makedirs("face_detector")
    
    # Download the face detector files
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    prototxt_path = "face_detector/deploy.prototxt"
    caffemodel_path = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    
    if not os.path.exists(prototxt_path):
        urllib.request.urlretrieve(prototxt_url, prototxt_path)
        print("[INFO] Downloaded deploy.prototxt")
    else:
        print("[INFO] deploy.prototxt already exists")
    
    if not os.path.exists(caffemodel_path):
        urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
        print("[INFO] Downloaded res10_300x300_ssd_iter_140000.caffemodel")
    else:
        print("[INFO] res10_300x300_ssd_iter_140000.caffemodel already exists")

if __name__ == "__main__":
    print("[INFO] Setting up models for face detection...")
    
    # Download face detector model
    download_face_detector()
    
    print("[INFO] Setup complete! You can now run simplified_face_detection.py")