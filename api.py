from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import io
import os

app = FastAPI(title="Rock Paper Scissors Object Detection API")

# Configure CORS so the Next.js frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], # Specific origins instead of *
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model (falling back to default if the fine_tuned_5 doesn't exist)
MODEL_PATH = 'runs/detect/train/weights/model.pt'
if not os.path.exists(MODEL_PATH):
    print(f"Warning: Model not found at {MODEL_PATH}. Attempting to load default yolov8n.pt")
    # If the user hasn't trained it, this will download the base model, 
    # though it won't accurately detect rock/paper/scissors out of the box.
    model = YOLO('yolov8n.pt') 
else:
    print(f"Loading custom model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)


@app.get("/")
def read_root():
    return {"message": "Rock Paper Scissors YOLOv8 API is running!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image, runs inference using YOLO, and returns detection results.
    """
    try:
        # Read the uploaded file into memory
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image file format"}

        # Perform inference
        results = model(img)
        
        # Parse results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract coordinates, confidence, and class id
                b = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                
                detections.append({
                    "box": {
                        "x1": b[0],
                        "y1": b[1],
                        "x2": b[2],
                        "y2": b[3]
                    },
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": name
                })

        return {"detections": detections}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Make sure we're binding to localhost specifically instead of 0.0.0.0 to avoid Windows socket issues
    uvicorn.run(app, host="127.0.0.1", port=8001)
