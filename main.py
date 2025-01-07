import hashlib
import os

from dotenv import load_dotenv
from picsellia import Client, DatasetVersion, Experiment, Project
from picsellia.exceptions import ResourceNotFoundError
from picsellia.types.enums import AnnotationFileType

from src.DatasetManager import DatasetManager
from src.YamlConfig import YAMLConfig
from src.YoloManager import YOLOManager


def generate_experiment_name(
    hyperparameters: dict, dataset_version: DatasetVersion
) -> str:
    """Generates a unique experiment name based on the given hyperparameters and dataset version."""
    hash_input = f"{hyperparameters}{dataset_version.id}"
    return hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest()


def get_or_create_experiment(
    project: Project, dataset_version: DatasetVersion, experiment_name: str
) -> Experiment:
    try:
        experiment = project.get_experiment(name=experiment_name)
        print(f"Using existing experiment: {experiment_name}")
    except ResourceNotFoundError:
        experiment = project.create_experiment(name=experiment_name)
        experiment.attach_dataset(
            name=dataset_version.name, dataset_version=dataset_version
        )
    return experiment


def main():
    load_dotenv()

    # Environment variables
    PICSELLIA_API_TOKEN = os.getenv("PICSELLIA_API_TOKEN")
    if not PICSELLIA_API_TOKEN:
        raise ValueError("Missing 'PICSELLIA_API_TOKEN' environment variable.")

    PICSELLIA_ORGANIZATION_NAME = os.getenv("PICSELLIA_ORGANIZATION_NAME")
    if not PICSELLIA_ORGANIZATION_NAME:
        raise ValueError("Missing 'PICSELLIA_ORGANIZATION_NAME' environment variable.")

    PICSELLIA_PROJECT_ID = os.getenv("PICSELLIA_PROJECT_ID")
    if not PICSELLIA_PROJECT_ID:
        raise ValueError("Missing 'PICSELLIA_PROJECT_ID' environment variable.")

    PICSELLIA_DATASET_VERSION = os.getenv("PICSELLIA_DATASET_VERSION")
    if not PICSELLIA_DATASET_VERSION:
        raise ValueError("Missing 'PICSELLIA_DATASET_VERSION' environment variable.")

    # Hyperparameters
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
    }

    # Initialize the client, get project and dataset, create an experiment, attach the dataset to the experiment
    client = Client(
        api_token=PICSELLIA_API_TOKEN, organization_name=PICSELLIA_ORGANIZATION_NAME
    )
    dataset_version = client.get_dataset_version_by_id(PICSELLIA_DATASET_VERSION)
    project = client.get_project_by_id(PICSELLIA_PROJECT_ID)

    experiment_name = generate_experiment_name(hyperparameters, dataset_version)
    experiment = get_or_create_experiment(project, dataset_version, experiment_name)
    print(experiment)

    dataset_manager = DatasetManager(
        base_dir="./datasets", id_version="0193688e-aa8f-7cbe-9396-bec740a262d0"
    )
    yolo_manager = YOLOManager(model_path="yolo11n.pt", experiment=experiment)

    # Download dataset
    dataset_manager.download_dataset(client, dataset_manager.id_version)

    # Structure and export data
    dataset_manager.export_annotations(
        client.get_dataset_version_by_id(dataset_manager.id_version),
        AnnotationFileType.YOLO,
    )
    dataset_manager.extract_zip()

    split_ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    images_dir, labels_dir = dataset_manager.structure_data_for_yolo(split_ratios)

    # Generate the config.yaml file
    data_yaml = YAMLConfig.load_yaml(
        os.path.join(dataset_manager.annotations_dir, "data.yaml")
    )
    config_data = {
        "train": os.path.abspath(f"{images_dir.get('train')}"),
        "val": os.path.abspath(f"{images_dir.get('val')}"),
        "test": os.path.abspath(f"{images_dir.get('test')}"),
        "nc": data_yaml.get("nc", 10),
        "names": data_yaml.get(
            "names", [f"class{i}" for i in range(data_yaml.get("nc", 10))]
        ),
    }
    config_path = os.path.join(dataset_manager.structured_dir, "config.yaml")
    YAMLConfig.save_yaml(config_data, config_path)

    yolo_manager.train(config_path, hyperparameters, project_path="./results")
    yolo_manager.evaluate_metrics(config_path)
    yolo_manager.evaluate_model(config_path)


if __name__ == "__main__":
    main()
