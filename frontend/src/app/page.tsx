"use client";

import { useRef, useState, useEffect, useCallback } from "react";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [fps, setFps] = useState(0);

  // Stats
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "user" } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsCameraOn(true);
        setErrorMsg("");
      }
    } catch (err: any) {
      console.error("Error accessing webcam:", err);
      setErrorMsg("Error accessing webcam. Please check permissions.");
    }
  };

  const stopCamera = () => {
    const stream = videoRef.current?.srcObject as MediaStream;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setIsCameraOn(false);
    }
  };

  const processFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isCameraOn) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    if (!context || video.videoWidth === 0 || video.videoHeight === 0) return;

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to blob to send to FastAPI
    canvas.toBlob(async (blob) => {
      if (!blob) return;

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      try {
        const response = await fetch("http://127.0.0.1:8001/predict/", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          drawDetections(data.detections);
          updateFps();
        }
      } catch (err) {
        console.error("Error calling API:", err);
      }
    }, "image/jpeg", 0.7); // Compress a bit for speed
  }, [isCameraOn]);

  const updateFps = () => {
    frameCountRef.current++;
    const now = Date.now();
    const elapsed = now - lastTimeRef.current;
    if (elapsed >= 1000) {
      setFps(Math.round((frameCountRef.current * 1000) / elapsed));
      frameCountRef.current = 0;
      lastTimeRef.current = now;
    }
  };

  const drawDetections = (detections: any[]) => {
    if (!canvasRef.current || !videoRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear previous drawings (but leave the video feed transparent underneath!)
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach((det) => {
      const { box, confidence, class_name } = det;
      const { x1, y1, x2, y2 } = box;
      
      const width = x2 - x1;
      const height = y2 - y1;

      // Draw bounding box
      ctx.strokeStyle = "#4ade80"; // Bright green (tailwind green-400)
      ctx.lineWidth = 4;
      ctx.strokeRect(x1, y1, width, height);

      // Draw label background
      ctx.fillStyle = "#4ade80";
      const text = `${class_name} ${(confidence * 100).toFixed(0)}%`;
      const textWidth = ctx.measureText(text).width;
      ctx.fillRect(x1, y1 - 30, textWidth + 10, 30);

      // Draw text
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 16px Inter, sans-serif";
      ctx.fillText(text, x1 + 5, y1 - 10);
    });
  };

  useEffect(() => {
    let animationFrameId: number;

    const loop = async () => {
      if (isCameraOn) {
        await processFrame();
      }
      // Request next frame, throttling slightly with setTimeout is an option,
      // but requestAnimationFrame is smoother.
      animationFrameId = requestAnimationFrame(loop);
    };

    if (isCameraOn) {
      loop();
    }

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isCameraOn, processFrame]);


  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100 font-sans flex flex-col items-center py-10">
      
      {/* Header */}
      <div className="text-center mb-10 max-w-2xl px-4">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4 bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-cyan-500">
          AI Hand Gesture Recognition
        </h1>
        <p className="text-neutral-400 text-lg">
          Real-time object detection for Rock, Paper, Scissors powered by YOLOv8 and Next.js.
        </p>
      </div>

      {/* Main Content Area */}
      <div className="w-full max-w-4xl px-4 flex flex-col items-center">
        
        {/* Error State */}
        {errorMsg && (
          <div className="mb-4 p-4 rounded-lg bg-red-900/50 border border-red-500 text-red-200 w-full text-center">
            {errorMsg}
          </div>
        )}

        {/* Video feed container */}
        <div className="relative w-full aspect-video rounded-2xl overflow-hidden bg-neutral-900 border border-neutral-800 shadow-2xl flex items-center justify-center">
          
          {/* Default state */}
          {!isCameraOn && (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-neutral-500">
              <svg xmlns="http://www.w3.org/.svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className="mb-4 opacity-50"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>
              <p className="text-xl font-medium">Camera is offline</p>
            </div>
          )}

          {/* Actual Video Element */}
          <video
            ref={videoRef}
            className="absolute inset-0 w-full h-full object-cover"
            playsInline
            muted
            style={{ display: isCameraOn ? 'block' : 'none' }}
          />
          
          {/* Overlay Canvas for YOLO bounds */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full object-cover pointer-events-none"
            style={{ display: isCameraOn ? 'block' : 'none' }}
          />

          {/* FPS Badge */}
          {isCameraOn && (
            <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-sm text-emerald-400 font-mono text-sm px-3 py-1 rounded-full border border-emerald-900/50 flex items-center gap-2 shadow">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              {fps} FPS
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="mt-8 flex gap-4">
          {!isCameraOn ? (
            <button
              onClick={startCamera}
              className="px-8 py-3.5 rounded-full bg-emerald-500 hover:bg-emerald-400 text-black font-semibold transition-all duration-300 shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:shadow-[0_0_30px_rgba(16,185,129,0.5)] flex items-center gap-2 transform hover:-translate-y-1"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
              Start Detection
            </button>
          ) : (
             <button
              onClick={stopCamera}
              className="px-8 py-3.5 rounded-full bg-red-500 hover:bg-red-400 text-white font-semibold transition-all duration-300 shadow-[0_0_20px_rgba(239,68,68,0.3)] hover:shadow-[0_0_30px_rgba(239,68,68,0.5)] flex items-center gap-2 transform hover:-translate-y-1"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>
              Stop Camera
            </button>
          )}
        </div>
      </div>
    </main>
  );
}
