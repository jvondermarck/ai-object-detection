import torch
from ultralytics import YOLO


class YOLOManager:
    """Manages the YOLO model and its training process.

    Methods:
        configure_hardware: Configures the model to use either GPU or CPU.
        train: Trains the YOLO model with custom parameters.
    """

    def __init__(self, model_path: str) -> None:
        """Initializes the YOLO manager.

        Args:
            model_path (str): Path to the pre-trained or custom YOLO model.
        """

        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to("cuda")
            return

        if torch.backends.mps.is_available():
            self.model.to("mps")

    def train(self, config_path: str, hyperparameters: dict, project_path: str) -> None:
        """Trains the YOLO model.

        Args:
            config_path (str): Path to the dataset configuration file.
            hyperparameters (dict): Custom hyperparameters for training.
            project_path (str): Path to save training results.
        """
        self.add_callbacks()
        self.model.train(data=config_path, project=project_path, **hyperparameters)

    def add_callbacks(self) -> None:
        """Adds custom callbacks to the model."""

        # define callbacks

        def on_train_epoch_end(trainer):
            metrics = trainer.metrics
            # Log all metrics
            # Print all metrics
            print(f"Metrics: {metrics}")
            # Precision (Proportion of correct positive predictions)
            print(f"metrics/precision(B): {metrics['metrics/precision(B)']}")
            # Recall (Proportion of true positives that are correctly identified)
            print(f"metrics/recall(B): {metrics['metrics/recall(B)']}")
            # mAP50 (Mean Average Precision with an IoU threshold of 50%)
            print(f"metrics/mAP50(B): {metrics['metrics/mAP50(B)']}")
            # mAP50-95 (Mean Average Precision for IoU thresholds from 50% to 95%)
            print(f"metrics/mAP50-95(B): {metrics['metrics/mAP50-95(B)']}")
            # Bounding box loss
            print(f"val/box_loss: {metrics['val/box_loss']}")
            # Classification loss
            print(f"val/cls_loss: {metrics['val/cls_loss']}")
            # Distribution Focal Loss
            print(f"val/dfl_loss: {metrics['val/dfl_loss']}")

        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)
