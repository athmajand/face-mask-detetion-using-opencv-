import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import os
import sys

class FaceDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("900x700")
        
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
        
        # Results frame
        results_frame = ttk.LabelFrame(self.window, text="Detection Results")
        results_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Results text
        self.txt_results = tk.Text(results_frame, height=5, width=80)
        self.txt_results.pack(padx=5, pady=5)
        
        # Video frame
        self.canvas = tk.Canvas(self.window, width=640, height=480)
        self.canvas.pack(padx=10, pady=10)
    
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
                
                # Detect faces
                results, processed_frame = self.detect_faces(frame)
                
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
    
    def detect_faces(self, frame):
        # Grab frame dimensions and convert to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # Pass the blob through the network and get detections
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        
        # Initialize result variables
        faces = 0
        
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
                
                faces += 1
                
                # Display the bounding box on the output frame
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, f"Face #{faces}", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        
        # Prepare results text
        results = f"Total faces detected: {faces}\n"
        results += f"Detection time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        return results, frame
    
    def update_results(self, text):
        # Update the results text widget from the main thread
        self.window.after(0, self._update_results_text, text)
    
    def _update_results_text(self, text):
        self.txt_results.delete(1.0, tk.END)
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
    app = FaceDetectionApp(root, "Face Detection")
    print("Starting main loop...")
    root.mainloop()
    print("Application closed")