import platform

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

    def configure_hardware(self) -> None:
        """Configures hardware for running the model (GPU or CPU).

        Raises:
            ValueError: If the operating system is not supported.
        """
        os_name = platform.system()
        if os_name in ["Windows", "Linux"]:
            self.model.to("cuda")
        elif os_name == "Darwin":
            self.model.to("mps")
        else:
            raise ValueError(f"Unrecognized operating system: {os_name}")

    def train(self, config_path: str, hyperparameters: dict, project_path: str) -> None:
        """Trains the YOLO model.

        Args:
            config_path (str): Path to the dataset configuration file.
            hyperparameters (dict): Custom hyperparameters for training.
            project_path (str): Path to save training results.
        """
        self.model.train(data=config_path, project=project_path, **hyperparameters)
