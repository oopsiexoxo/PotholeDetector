import cv2
from ultralytics import YOLO
import numpy as np

class PotholeDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the Pothole Detector with a YOLO model.
        
        Args:
            model_path (str): Path to the YOLO weights file (.pt). 
                              Defaults to 'yolov8n.pt' (standard COCO model).
                              For actual potholes, you need a trained pothole model.
        """
        print(f"Loading model from {model_path}...")
        try:
            self.model = YOLO(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def detect(self, frame, conf_threshold=0.25):
        """
        Detect objects in the frame.

        Args:
            frame (numpy.ndarray): The input image/frame.
            conf_threshold (float): Confidence threshold for detection.

        Returns:
            list: List of detections, where each detection is a dict or object
                  containing bbox, confidence, and class_id.
            results: The raw results object from YOLO.
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)
        return results

    def annotate_frame(self, frame, results):
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame (numpy.ndarray): The original frame.
            results: The results object from self.detect().

        Returns:
            numpy.ndarray: The annotated frame.
        """
        # YOLOv8 results object has a plot() method that returns the annotated frame (BGR)
        annotated_frame = results[0].plot()
        return annotated_frame
