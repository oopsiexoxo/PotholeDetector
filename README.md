# Pothole Detector

A real-time pothole detection system using **YOLOv8** and **OpenCV**. This project detects potholes in video feeds or images to help identify road damage.

## Features
- **Real-time Detection**: Works on images or video files.
- **Custom Trained Model**: Includes a trained pothole model.

## Installation
1. Clone the repo
2. Run pip install -r requirements.txt 

## Usage (Preferably use on images)
1) Add images/video of potholes into the folder

2) Run pothole detector: 
Command: 
Images - `py main.py --source "YourImage.png"`
Video - `py main.py --source "YourVideo.mp4"`

3) Result:
You should see the detector correctly classify the pothole.

Set-Content README.md -Value 
