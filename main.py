import cv2
import argparse
from detector import PotholeDetector

def main():
    parser = argparse.ArgumentParser(description="Pothole Detection System")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam or path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model weights (e.g., pothole_model.pt)")
    args = parser.parse_args()

    # Initialize source
    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Initialize Detector
    try:
        detector = PotholeDetector(model_path=args.model)
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return

    print("Starting detection... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or cannot read frame.")
            break

        # Run detection
        results = detector.detect(frame)

        # Annotate frame
        annotated_frame = detector.annotate_frame(frame, results)

        # Display
        cv2.imshow("Pothole Detector", annotated_frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
