# Rock Paper Scissors Real-Time Object Detection

A computer vision application powered by YOLOv8 and Next.js that detects "Rock", "Paper", and "Scissors" hand gestures in real-time directly from a webcam feed.

## Features
* **Custom Trained YOLOv8 Model**: Accurately recognizes rock, paper, and scissors hand gestures.
* **FastAPI Backend**: A lightweight, high-performance Python API to serve the ML model.
* **Next.js Frontend**: A modern, interactive dark-themed UI that captures webcam frames and draws bounding boxes in real-time.

## Tech Stack
* **Machine Learning**: Ultralytics YOLOv8, OpenCV
* **Backend**: Python, FastAPI, Uvicorn
* **Frontend**: Next.js, React, Tailwind CSS

---

## Getting Started

You need to run both the Python backend and the Next.js frontend simultaneously to use the application.

### 1. Start the API Backend
The backend runs the YOLOv8 model inference.

```bash
# 1. Navigate to the project directory
cd Rock-Paper-Scissor-Object-Detection

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# (Optional) Install dependencies if you haven't already
pip install -r requirements-web.txt

# 3. Start the FastAPI server (Runs on port 8001)
python api.py
```

### 2. Start the Next.js Frontend
The frontend provides the webcam interface. Open a *new* terminal window.

```bash
# 1. Navigate to the frontend directory
cd Rock-Paper-Scissor-Object-Detection/frontend

# (Optional) Install Node dependencies if you haven't already
npm install

# 2. Start the Next development server
npm run dev
```

### 3. Use the App
Open your web browser and navigate to:
**[http://localhost:3000](http://localhost:3000)**

Click **"Start Detection"**, grant camera permissions, and hold your hand up to the camera!
