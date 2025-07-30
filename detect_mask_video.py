import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import os
import sys

# Add debug print statements
print("Starting application...")
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"TensorFlow version: {tf.__version__}")

try:
    class MaskDetectionApp:
        def __init__(self, window, window_title):
            self.window = window
            self.window.title(window_title)
            self.window.geometry("900x800")  # Increased height for better visibility
            self.window.minsize(900, 800)    # Set minimum size

            print("Loading face detector model...")
            # Check if face detector exists
            prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
            weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])

            if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
                messagebox.showerror("Error", "Face detector model files not found. Please run setup_models.py first.")
                self.window.destroy()
                return

            # Load face detector model
            try:
                self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
                print("Face detector model loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load face detector model: {str(e)}")
                self.window.destroy()
                return

            print("Loading mask detector model...")
            # Check if mask detector exists
            if not os.path.exists("mask_detector.h5"):
                messagebox.showerror("Error", "Mask detector model not found. Please run setup_models.py first.")
                self.window.destroy()
                return

            # Load mask detector model
            try:
                self.maskNet = load_model("mask_detector.h5")
                print("Mask detector model loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load mask detector model: {str(e)}")
                self.window.destroy()
                return

            # Initialize video source
            self.vid = None
            self.is_running = False
            self.thread = None

            # Create UI elements
            print("Creating UI elements...")
            self.create_ui()

            # Set a callback to handle window close
            self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
            print("GUI initialization complete")

        def create_ui(self):
            # Create a frame for controls
            control_frame = ttk.Frame(self.window)
            control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

            # Start/Stop button
            self.btn_start_stop = ttk.Button(control_frame, text="Start Camera", command=self.toggle_camera)
            self.btn_start_stop.pack(side=tk.LEFT, padx=5)

            # Status label
            self.lbl_status = ttk.Label(control_frame, text="Camera: OFF")
            self.lbl_status.pack(side=tk.LEFT, padx=20)

            # Video frame
            self.canvas = tk.Canvas(self.window, width=640, height=480)
            self.canvas.pack(padx=10, pady=10)

            # Results frame - moved below video for better visibility
            results_frame = ttk.LabelFrame(self.window, text="Detection Results", relief=tk.RAISED, borderwidth=3)
            results_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

            # Results text - increased height and added a background color
            self.txt_results = tk.Text(results_frame, height=8, width=80, bg="lightyellow", font=("Arial", 12))
            self.txt_results.pack(padx=5, pady=5)

        def toggle_camera(self):
            if self.is_running:
                # Stop the camera
                self.is_running = False
                if self.thread is not None:
                    self.thread.join()
                if self.vid is not None:
                    self.vid.release()
                    self.vid = None
                self.btn_start_stop.config(text="Start Camera")
                self.lbl_status.config(text="Camera: OFF")
                # Clear the canvas
                self.canvas.delete("all")
            else:
                # Start the camera
                try:
                    self.vid = cv2.VideoCapture(0)
                    if not self.vid.isOpened():
                        messagebox.showerror("Error", "Could not open webcam. Please check your camera connection.")
                        self.update_results("Error: Could not open webcam")
                        return

                    self.is_running = True
                    self.btn_start_stop.config(text="Stop Camera")
                    self.lbl_status.config(text="Camera: ON")

                    # Start video processing in a separate thread
                    self.thread = threading.Thread(target=self.process_video)
                    self.thread.daemon = True
                    self.thread.start()
                except Exception as e:
                    messagebox.showerror("Error", f"Camera error: {str(e)}")
                    self.update_results(f"Error: {str(e)}")

        def process_video(self):
            while self.is_running:
                try:
                    ret, frame = self.vid.read()
                    if not ret:
                        self.update_results("Error: Failed to capture frame")
                        break

                    # Detect faces and masks
                    results, processed_frame = self.detect_mask(frame)

                    # Update the UI with results
                    self.update_results(results)

                    # Convert to RGB for tkinter
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)
                    imgtk = ImageTk.PhotoImage(image=img)

                    # Update the canvas with the new frame
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                    self.canvas.imgtk = imgtk  # Keep a reference
                except Exception as e:
                    self.update_results(f"Error in processing: {str(e)}")
                    break

                time.sleep(0.03)  # Limit to ~30 FPS

        def detect_mask(self, frame):
            # Grab frame dimensions and convert to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

            # Pass the blob through the network and get detections
            self.faceNet.setInput(blob)
            detections = self.faceNet.forward()

            # Initialize result variables
            faces = 0
            with_mask = 0
            without_mask = 0

            # Loop over the detections
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections
                if confidence > 0.5:
                    # Compute coordinates of bounding box
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Ensure the bounding boxes are within the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # Extract the face ROI
                    face = frame[startY:endY, startX:endX]
                    if face.size == 0:
                        continue

                    faces += 1

                    # Preprocess the face for the mask detector
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)

                    # Pass the face through the mask detector model
                    (mask, withoutMask) = self.maskNet.predict(face)[0]

                    # Determine the class label and color
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                    if label == "Mask":
                        with_mask += 1
                    else:
                        without_mask += 1

                    # Include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    # Display the label and bounding box on the output frame
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Prepare results text
            results = f"Total faces detected: {faces}\n"
            results += f"With mask: {with_mask}\n"
            results += f"Without mask: {without_mask}\n"
            results += f"Detection time: {time.strftime('%Y-%m-%d %H:%M:%S')}"

            return results, frame

        def update_results(self, text):
            # Update the results text widget from the main thread
            self.window.after(0, self._update_results_text, text)

        def _update_results_text(self, text):
            self.txt_results.delete(1.0, tk.END)
            self.txt_results.insert(tk.END, "MASK DETECTION RESULTS:\n\n")
            self.txt_results.insert(tk.END, text)

        def on_closing(self):
            # Stop the camera when closing the window
            self.is_running = False
            if self.thread is not None:
                self.thread.join()
            if self.vid is not None:
                self.vid.release()
            self.window.destroy()

    # Run the application
    if __name__ == "__main__":
        print("Creating main window...")
        root = tk.Tk()
        print("Initializing application...")
        app = MaskDetectionApp(root, "Face Mask Detection")
        print("Starting main loop...")
        root.mainloop()
        print("Application closed")

except Exception as e:
    print(f"Critical error: {str(e)}")
    import traceback
    traceback.print_exc()


