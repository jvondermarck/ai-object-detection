import logging
import os
from typing import List, Union

import cv2
from picsellia import Client
from ultralytics import YOLO

from src.config import (
    PICSELLIA_API_TOKEN,
    PICSELLIA_ORGANIZATION_NAME,
)


def initialize_model(model_version_id: str) -> YOLO:
    """Initialize the YOLO model with Picsellia integration."""
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    client = Client(
        api_token=PICSELLIA_API_TOKEN, organization_name=PICSELLIA_ORGANIZATION_NAME
    )
    model_version = client.get_model_version_by_id(model_version_id)
    model_version.get_file("model").download()

    model = YOLO("best.pt")
    return model


def save_video(
    frames: List, output_path: str, fps: int = 30, frame_size: tuple = (640, 480)
) -> None:
    """Save a video from a list of frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for frame in frames:
        out.write(frame)
    out.release()


def display_bounding_box_info(model: YOLO, results) -> None:
    """Display bounding box information for detected objects."""
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


def infer_on_image(
    model: YOLO, source: str, output: Union[str, None], conf: float, iou: float
) -> None:
    """Perform inference on a single image."""
    frame = cv2.imread(source)
    if frame is None:
        print("Error: Unable to load image.")
        return

    results = model(frame, show=False, conf=conf, iou=iou)
    display_bounding_box_info(model, results)
    annotated_frame = results[0].plot()
    output_path = (
        os.path.join(output, "annotated_image.jpg") if output else "annotated_image.jpg"
    )
    cv2.imwrite(output_path, annotated_frame)
    print(f"Annotated image saved to {output_path}")


def infer_on_video(
    model: YOLO, source: str, output: Union[str, None], conf: float, iou: float
) -> None:
    """Perform inference on a video."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    processed_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, show=False, conf=conf, iou=iou)
        display_bounding_box_info(model, results)
        processed_frames.append(results[0].plot())

    cap.release()
    output_path = (
        os.path.join(output, "annotated_video.mp4") if output else "annotated_video.mp4"
    )
    save_video(processed_frames, output_path, fps, frame_size)
    print(f"Annotated video saved to {output_path}")


def infer_on_webcam(
    model: YOLO, output: Union[str, None], conf: float, iou: float
) -> None:
    """Perform inference using a webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, show=False, conf=conf, iou=iou)
        display_bounding_box_info(model, results)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Webcam Inference", annotated_frame)

        if output:
            frame_path = os.path.join(
                output, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg"
            )
            cv2.imwrite(frame_path, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def infer(
    model_version_id: str,
    source: Union[str, int] = 0,
    output: Union[str, None] = None,
    conf: float = 0.5,
    iou: float = 0.45,
) -> None:
    """Perform inference using YOLOv8 with Picsellia integration."""
    # Validate output directory
    if output and not os.path.exists(output):
        os.makedirs(output)

    # Initialize YOLO model
    model = initialize_model(model_version_id)

    # Determine source type and perform inference
    if isinstance(source, str):
        if source.lower().endswith((".jpg", ".jpeg", ".png")):
            print("Running inference on image...")
            infer_on_image(model, source, output, conf, iou)  # Passage de conf et iou
        elif source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            print("Running inference on video...")
            infer_on_video(model, source, output, conf, iou)  # Passage de conf et iou
        else:
            print("Error: Unsupported file type for source.")
    elif isinstance(source, int):
        print("Running inference on webcam...")
        infer_on_webcam(model, output, conf, iou)  # Passage de conf et iou
    else:
        print(
            "Error: Invalid source type. Please provide a valid image, video, or webcam source."
        )
