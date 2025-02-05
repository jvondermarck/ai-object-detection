import hashlib

from picsellia import Client, DatasetVersion, Experiment, Project
from picsellia.exceptions import ResourceNotFoundError

from src.config import (
    PICSELLIA_API_TOKEN,
    PICSELLIA_ORGANIZATION_NAME,
)
from src.DatasetManager import DatasetManager
from src.YoloManager import YOLOManager


def generate_experiment_name(
    hyperparameters: dict, dataset_version: DatasetVersion
) -> str:
    """Generates a unique experiment name based on the given hyperparameters and dataset version."""
    hash_input = f"{hyperparameters}{dataset_version.id}"
    return hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest()


def get_or_create_experiment(
    project: Project,
    dataset_version: DatasetVersion,
    experiment_name: str,
    hyperparameters: dict,
) -> Experiment:
    """Gets a Picsellia experiment or creates a new one if it does not exist.

    Args:
        project: Picsellia project object.
        dataset_version: Picsellia dataset version object.
        experiment_name: Name of the experiment.
        hyperparameters: Dictionary of hyperparameters for the experiment.

    Returns:
        The existing or newly created Picsellia experiment.
    """
    try:
        experiment = project.get_experiment(name=experiment_name)
        print(f"Using existing experiment: {experiment_name}")
    except ResourceNotFoundError:
        description = f"""
        This training experiment has been created {hyperparameters['epochs']} epochs, with a batch size of {hyperparameters['batch']} and an image size of {hyperparameters['imgsz']}.
        Learning rate is set to {hyperparameters['lr0']} with an optimizer of {hyperparameters['optimizer']}.
        """
        experiment = project.create_experiment(
            name=experiment_name, description=description
        )
        experiment.attach_dataset(
            name=dataset_version.name, dataset_version=dataset_version
        )
        experiment.log_parameters(hyperparameters)
    return experiment


def train(dataset_version_id: str, project_id: str):
    """Trains a YOLO model using the specified dataset version and project."""
    hyperparameters = {
        "epochs": 20,
        "batch": 32,
        "imgsz": 640,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "seed": 42,
        "augment": True,
        "cache": "ram",
        "close_mosaic": 0,
    }

    client = Client(
        api_token=PICSELLIA_API_TOKEN, organization_name=PICSELLIA_ORGANIZATION_NAME
    )
    dataset_version = client.get_dataset_version_by_id(dataset_version_id)
    project = client.get_project_by_id(project_id)

    experiment_name = generate_experiment_name(hyperparameters, dataset_version)
    experiment = get_or_create_experiment(
        project, dataset_version, experiment_name, hyperparameters
    )
    base_model = client.get_model("Groupe_7")

    dataset_manager = DatasetManager(
        client, base_dir="./datasets", id_version=dataset_version_id
    )
    yolo_manager = YOLOManager(model_path="yolo11n.pt", experiment=experiment)

    dataset_manager.prepare_dataset()

    yolo_manager.train(dataset_manager.config_path, hyperparameters)
    yolo_manager.evaluate_metrics(dataset_manager.config_path)
    yolo_manager.evaluate_model(dataset_manager.config_path)
    yolo_manager.export_model_version(base_model)
