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


def save_video(
    frames: List, output_path: str, fps: int = 30, frame_size: tuple = (640, 480)
) -> None:
    """Saves a video from a list of frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for frame in frames:
        out.write(frame)
    out.release()


def infer(
    model_version_id: str,
    source: Union[str, int] = 0,
    output: Union[str, None] = None,
    conf: float = 0.5,
    iou: float = 0.45,
) -> None:
    """
    Perform inference using YOLOv8 with Picsellia integration.

    Args:
        model_version_id (str): ID of the Picsellia model version.
        source (Union[str, int]): Input source (image, video path, or webcam index).
        output (Union[str, None]): Directory to save annotated results.
        conf (float): Confidence threshold for predictions.
        iou (float): Intersection over Union threshold.
    """
    # Configure logging
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # Initialize Picsellia client and fetch model
    client = Client(
        api_token=PICSELLIA_API_TOKEN, organization_name=PICSELLIA_ORGANIZATION_NAME
    )
    model_version = client.get_model_version_by_id(model_version_id)
    model_version.get_file("model").download()

    # Load the YOLO model
    model = YOLO("best.pt")
    model.overrides["conf"] = conf  # Set confidence threshold
    model.overrides["iou"] = iou  # Set IoU threshold

    # Validate output directory
    if output and not os.path.exists(output):
        os.makedirs(output)

    # IMAGE mode
    if isinstance(source, str) and os.path.isfile(source):
        if source.lower().endswith((".jpg", ".png", ".jpeg")):
            print("Running inference on image...")
            frame = cv2.imread(source)
            results = model(frame, show=False)

            # Annotate and save image
            annotated_frame = results[0].plot()
            output_path = (
                os.path.join(output, "annotated_image.jpg")
                if output
                else "annotated_image.jpg"
            )
            cv2.imwrite(output_path, annotated_frame)
            print(f"Annotated image saved to {output_path}")
        else:
            print("Invalid image file format.")

    # VIDEO mode
    elif isinstance(source, str) and os.path.isfile(source):
        print("Running inference on video...")
        cap = cv2.VideoCapture(source)
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

            # Perform inference and annotate
            results = model(frame, show=False)
            processed_frames.append(results[0].plot())

        cap.release()

        # Save video
        output_path = (
            os.path.join(output, "annotated_video.mp4")
            if output
            else "annotated_video.mp4"
        )
        save_video(processed_frames, output_path, fps, frame_size)
        print(f"Annotated video saved to {output_path}")

    # WEBCAM mode
    elif isinstance(source, int):
        print("Running inference on webcam...")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Unable to access the webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference and annotate
            results = model(frame, show=False)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Webcam Inference", annotated_frame)

            # Save frame if output is specified
            if output:
                frame_path = os.path.join(
                    output, f"frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg"
                )
                cv2.imwrite(frame_path, annotated_frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        print(
            "Error: Invalid source. Please provide a valid image, video, or webcam source."
        )
