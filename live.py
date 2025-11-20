import argparse
from ultralytics import YOLO
import cv2
import os

def main():
    parser = argparse.ArgumentParser(description="Real-time object detection with YOLOv8.")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/mdoel.pt", help="Path to the trained model weights file")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detection (default: 0.5)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")

    args = parser.parse_args()

    # Check if the model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return

    try:
        # Load the trained YOLO model
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)

        # Open the camera
        cap = cv2.VideoCapture(args.camera)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Starting real-time detection. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Perform object detection with the specified confidence threshold
            results = model(frame, imgsz=640, conf=args.conf, stream=True)

            # Plot the results on the frame
            for r in results:
                frame = r.plot()

            # Display the frame with detections
            cv2.imshow("YOLOv8 Live - Rock Paper Scissors", frame)

            # Check for the 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the camera and close the window
        cap.release()
        cv2.destroyAllWindows()
        print("Real-time detection ended.")

    except Exception as e:
        print(f"An error occurred during real-time detection: {e}")

if __name__ == "__main__":
    main()
