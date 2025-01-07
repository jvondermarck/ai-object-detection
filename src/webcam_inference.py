import logging

import cv2
from picsellia import Client
from ultralytics import YOLO

from config import (
    PICSELLIA_API_TOKEN,
    PICSELLIA_ORGANIZATION_NAME,
)

# Configure Ultralytics logs to avoid displaying unnecessary information
logging.getLogger("ultralytics").setLevel(logging.WARNING)


client = Client(
    api_token=PICSELLIA_API_TOKEN, organization_name=PICSELLIA_ORGANIZATION_NAME
)

model_version_id = "01943c53-956c-775b-afd1-38e9f87ed22e"
model_version = client.get_model_version_by_id(model_version_id)
model_version.get_file("model").download()

model = YOLO("best.pt")

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Inference loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference with YOLO
    results = model(frame, show=False, conf=0.5)

    # Check and display information if detection
    if results[0].boxes:
        print("\nDetected objects:")
        for box in results[0].boxes:
            # Get coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence level
            class_id = int(box.cls[0])  # Class ID
            class_name = model.names[class_id]  # Class name

            # Display details of the detected object
            print(f"- Object: {class_name}")
            print(f"  Coordinates: ({x1:.0f}, {y1:.0f}), ({x2:.0f}, {y2:.0f})")
            print(f"  Confidence: {confidence:.2f}")

    # Annotate the image with the detections
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Live Inference", annotated_frame)

    # Exit with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
