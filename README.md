Face Mesh ROI Detection

This project uses OpenCV and MediaPipe to detect facial landmarks and highlight regions of interest (ROIs) on a live webcam feed. Specifically, it draws bounding boxes around the left eye, right eye, and the center of the forehead.

Features

Real-time webcam capture using OpenCV.

Face landmark detection powered by MediaPipe Face Mesh.

Bounding boxes drawn for:

Left eye (green)

Right eye (blue)

Forehead center (red)

Requirements

Python 3.7+

OpenCV

MediaPipe

Install dependencies:

pip install opencv-python mediapipe

Usage

Clone the repository:

git clone https://github.com/jk2006-sudo/PrismPy.git
cd PrismPy

Run the script:

python face_mesh_roi.py

Press ESC to exit the webcam window.

Code Overview

cv2.VideoCapture(0): Opens the webcam.

mediapipe.solutions.face_mesh.FaceMesh: Initializes the face mesh detector.

draw_roi(): Helper function to compute bounding boxes for eye regions.

Forehead ROI: Computed around landmark 10 with an expanded fixed-size rectangle.

Example Output

Green rectangle: Left eye

Blue rectangle: Right eye

Red rectangle: Center forehead

Notes

Ensure your webcam is accessible and not used by another application.

You can adjust the forehead box size by modifying box_w and box_h in the script.

This README provides setup instructions and explains how the script works for detecting and visualizing facial ROIs using MediaPipe and OpenCV.
