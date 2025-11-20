from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train/weights/fine_tuned_5.pt')
results = model('scissor2.jpg')

for r in results:
    detected_img = r.plot()
    cv2.imshow("YOLOv8 Detection", detected_img)
    cv2.waitKey(0)

# Close the window when done
cv2.destroyAllWindows()