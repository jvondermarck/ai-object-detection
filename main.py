import os

from picsellia import Client
from picsellia.types.enums import AnnotationFileType

from src.ConfigManager import ConfigManager
from src.DatasetManager import DatasetManager
from src.YamlConfig import YAMLConfig
from src.YoloManager import YOLOManager


def main():
    ConfigManager.load_environment()

    # Initialize clients and paths
    api_token = ConfigManager.get_env_variable("PICSELLIA_API_TOKEN")
    client = Client(api_token=api_token, organization_name="Picsalex-MLOps")
    dataset_manager = DatasetManager(
        base_dir="./datasets", id_version="0193688e-aa8f-7cbe-9396-bec740a262d0"
    )
    yolo_manager = YOLOManager(model_path="yolo11n.pt")

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

    # Train the model
    yolo_manager.configure_hardware()
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
    yolo_manager.train(config_path, hyperparameters, project_path="./results")


if __name__ == "__main__":
    main()
