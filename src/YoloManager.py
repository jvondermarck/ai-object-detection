from picsellia import Experiment
from picsellia.types.enums import LogType
import torch
from ultralytics import YOLO


class YOLOManager:
    """Manages the YOLO model and its training process.

    Methods:
        configure_hardware: Configures the model to use either GPU or CPU.
        train: Trains the YOLO model with custom parameters.
    """

    def __init__(self, model_path: str, experiment: Experiment) -> None:
        """Initializes the YOLO manager.

        Args:
            model_path (str): Path to the pre-trained or custom YOLO model.
            experiment: Picsellia experiment object.
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

        def on_train_epoch_end(trainer):
            metrics = trainer.metrics
            for metric_name, metric_value in metrics.items():
                self.experiment.log(metric_name, [metric_value], LogType.LINE)
                print(f"{metric_name} logged to Picsellia: {metric_value:.3f}")

        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)
