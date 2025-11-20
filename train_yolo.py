import argparse
from ultralytics import YOLO
import os

def main():
    """
    Main function to train the YOLO model.
    """
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to the pre-trained model file (e.g., 'yolov8n.pt', 'yolov8s.pt')")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to the data configuration file (e.g., 'data.yaml')")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (default: 640)")

    args = parser.parse_args()


    # Check if the model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        return

    # Check if the data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data configuration file '{args.data}' not found.")
        return

    try:
        # Load YOLO model
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)

        # Train the model
        print(f"Starting training with {args.epochs} epochs...")
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz
        )
        print("Training completed successfully!")

    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    main()
