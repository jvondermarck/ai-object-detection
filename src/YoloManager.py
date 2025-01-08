import os

import torch
import yaml
from picsellia import Experiment, Model
from picsellia.types.enums import AddEvaluationType, InferenceType, LogType
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator


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
        self.experiment = experiment

        if torch.cuda.is_available():
            self.model.to("cuda")
            return

        if torch.backends.mps.is_available():
            self.model.to("mps")

        self.add_callbacks()

    def train(self, config_path: str, hyperparameters: dict, project_path: str) -> None:
        """Trains the YOLO model.

        Args:
            config_path (str): Path to the dataset configuration file.
            hyperparameters (dict): Custom hyperparameters for training.
            project_path (str): Path to save training results.
        """
        self.model.train(data=config_path, project=project_path, **hyperparameters)

    def evaluate_metrics(self, config_path: str) -> None:
        """Evaluates the YOLO model and logs the metrics to Picsellia.

        Args:
            config_path (str): Path to the dataset configuration file.
        """
        print("Evaluating the model...")
        print("Config path:", config_path)
        self.model.val(data=config_path)

    def evaluate_model(self, config_path: str) -> None:
        """Evaluates the YOLO model and adds the evaluation to the Picsellia experiment.

        Args:
            config_path (str): Path to the dataset configuration file.
        """

        testing_dataset = self.experiment.get_dataset(name="⭐️ cnam_product_2024")

        # Retrieve the labels from the Picsellia dataset
        picsellia_labels_name = testing_dataset.list_labels()
        label_matching = {k.name: k for k in picsellia_labels_name}
        model = YOLO("best.pt")

        # Load the configuration file
        with open(config_path, "r") as file:
            config_yaml = yaml.safe_load(file)

        test_images_dir = config_yaml["test"]

        # Iterate over the test images
        for image_name in os.listdir(test_images_dir):
            image_path = os.path.join(test_images_dir, image_name)

            # Perform prediction with YOLO
            results = model.predict(image_path)

            # Retrieve the corresponding asset in Picsellia
            asset = testing_dataset.find_asset(id=image_name.split(".")[0])

            rectangles = []
            for result in results:
                for box, cls, conf in zip(
                    result.boxes.xywh.tolist(),
                    result.boxes.cls.tolist(),
                    result.boxes.conf.tolist(),
                ):
                    x_center, y_center, w, h = box
                    x = x_center - w / 2
                    y = y_center - h / 2
                    class_id = int(cls)
                    label_name = result.names[class_id]
                    label = label_matching[label_name]
                    rectangles.append(
                        (int(x), int(y), int(w), int(h), label, float(conf))
                    )

            # Add the evaluation to the experiment
            self.experiment.add_evaluation(
                asset=asset,
                add_type=AddEvaluationType.REPLACE,
                rectangles=rectangles,
            )

        # Calculate evaluation metrics
        self.experiment.compute_evaluations_metrics(InferenceType.OBJECT_DETECTION)

        print("Evaluations completed and logged in Picsellia.")

    def add_callbacks(self) -> None:
        """Adds custom callbacks to the model."""

        def on_train_epoch_end(trainer):
            metrics = trainer.metrics
            for metric_name, metric_value in metrics.items():
                self.experiment.log(metric_name, [metric_value], LogType.LINE)
                print(f"{metric_name} logged to Picsellia: {metric_value:.3f}")

        def on_val_end(trainer: DetectionValidator):
            metrics = trainer.metrics
            print("Metrics:", metrics)
            for metric_name, metric_value in metrics.items():
                metric_value = float(metric_value)
                self.experiment.log(metric_name, [metric_value], LogType.LINE)
                print(f"{metric_name} logged to Picsellia: {metric_value:.3f}")

        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)
        self.model.add_callback("on_val_end", on_val_end)

    def export_model_version(self, base_model: Model) -> None:
        """Exports the model in the model registry."""
        new_model = self.experiment.export_in_existing_model(base_model)
        new_model.store("config", "args.yaml")
        new_model.store("model", "best.pt")
