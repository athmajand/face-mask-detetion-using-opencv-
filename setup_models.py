import os
import urllib.request
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import sys

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

def download_dataset():
    print("[INFO] Downloading mask dataset...")

    # Create dataset directory if it doesn't exist
    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    # URL for the dataset
    dataset_url = "https://github.com/prajnasb/observations/archive/master.zip"
    dataset_zip = "dataset/master.zip"

    if not os.path.exists(dataset_zip):
        print("[INFO] Downloading dataset zip file...")
        urllib.request.urlretrieve(dataset_url, dataset_zip)

        # Extract the dataset
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall("dataset")
        print("[INFO] Dataset extracted")
    else:
        print("[INFO] Dataset zip already exists")

def train_mask_detector():
    print("[INFO] Training mask detector model...")

    # Check if the dataset exists
    dataset_path = "dataset/observations-master/experiements/data"
    if not os.path.exists(dataset_path):
        print("[ERROR] Dataset not found. Please run download_dataset() first.")
        return

    # Initialize hyperparameters
    INIT_LR = 1e-4
    EPOCHS = 20
    BS = 32

    # Create data generators for training and validation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2
    )

    # Load and preprocess the dataset
    print("[INFO] Loading images...")
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=BS,
        class_mode="categorical",
        subset="training"
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=BS,
        class_mode="categorical",
        subset="validation"
    )

    # Load the MobileNetV2 network
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                           input_tensor=Input(shape=(224, 224, 3)))

    # Construct the head of the model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    # Place the head FC model on top of the base model
    model = Model(inputs=baseModel.input, outputs=headModel)

    # Freeze the base model layers
    for layer in baseModel.layers:
        layer.trainable = False

    # Compile the model
    print("[INFO] Compiling model...")
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Train the model
    print("[INFO] Training head...")
    H = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BS,
        epochs=EPOCHS
    )

    # Save the model
    print("[INFO] Saving mask detector model...")
    model.save("mask_detector.h5")
    print("[INFO] Model saved successfully!")

if __name__ == "__main__":
    print("[INFO] Setting up models for face mask detection...")

    # Download face detector model
    download_face_detector()

    # Download and prepare dataset
    download_dataset()

    # Train mask detector model
    train_mask_detector()

    print("[INFO] Setup complete! You can now run detect_mask_video.py")