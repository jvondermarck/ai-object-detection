import os
import shutil
import zipfile

from picsellia import Client
from picsellia.types.enums import AnnotationFileType

from src.YamlConfig import YAMLConfig


class DatasetManager:
    """Manages the downloading, extraction, and structuring of datasets.

    Attributes:
        client (Client): Picsellia client.
        base_dir (str): Root directory for dataset files.
        dataset: Picsellia dataset object.
        images_dir (str): Directory for image files.
        labels_dir (str): Directory for label files.
        config_path (str): Path to the configuration file.

    Methods:
        prepare_dataset: Prepares the dataset for training.
        _download_assets: Downloads assets from Picsellia.
        _export_annotations: Exports annotations in a given format.
        _export_annotation_file: Exports the annotation file from the dataset.
        _extract_annotation_file: Extracts the annotation file to the labels directory.
        _structure_annotations: Structures annotations into split directories.
        _generate_config_file: Generates a configuration file for YOLO.
    """

    RANDOM_SEED = 42
    SPLIT_RATIOS = [0.6, 0.2, 0.2]

    def __init__(self, client: Client, base_dir: str, id_version: str) -> None:
        """Initializes a dataset manager.

        Args:
            client (Client): Picsellia client.
            base_dir (str): Root directory for the dataset.
            id_version (str): Dataset version identifier.
        """
        self.client = client
        self.dataset = client.get_dataset_version_by_id(id_version)
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "images")
        self.labels_dir = os.path.join(base_dir, "labels")
        self.config_path = os.path.join(base_dir, "config.yaml")

    def prepare_dataset(self) -> None:
        """Prepares the dataset for training."""
        if os.path.exists(self.config_path):
            print("Dataset has already been downloaded. Skipping...")
            return

        self._download_assets()
        self._export_annotations()
        self._generate_config_file()

    def _download_assets(self) -> None:
        """Downloads assets from Picsellia.

        Dataset is split into train, test, and validation sets.
        Each split is downloaded to a separate directory.
        """
        (
            train_assets,
            test_assets,
            val_assets,
            count_train,
            count_test,
            count_val,
            labels,
        ) = self.dataset.train_test_val_split(
            ratios=self.SPLIT_RATIOS, random_seed=self.RANDOM_SEED
        )

        train_assets.download(os.path.join(self.images_dir, "train"), use_id=True)
        test_assets.download(os.path.join(self.images_dir, "test"), use_id=True)
        val_assets.download(os.path.join(self.images_dir, "val"), use_id=True)

    def _export_annotations(self) -> None:
        """Exports annotations in a given format, structured for YOLO."""
        annotation_file = self._export_annotation_file()
        self._extract_annotation_file(annotation_file)
        self._structure_annotations()

    def _export_annotation_file(self) -> str:
        """Exports the annotation file from the dataset.

        Returns:
            str: Path to the exported annotation file.
        """
        return self.dataset.export_annotation_file(
            AnnotationFileType.YOLO, self.base_dir, use_id=True
        )

    def _extract_annotation_file(self, annotation_file: str) -> None:
        """Extracts the annotation file to the labels directory.

        Args:
            annotation_file (str): Path to the annotation file.
        """
        with zipfile.ZipFile(annotation_file, "r") as zip_ref:
            zip_ref.extractall(self.labels_dir)
        annotation_file_grandparent = os.path.dirname(os.path.dirname(annotation_file))
        shutil.rmtree(annotation_file_grandparent)

    def _structure_annotations(self) -> None:
        """Structures annotations into split directories."""
        for split in ["train", "test", "val"]:
            split_dir = os.path.join(self.labels_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            for file in os.listdir(os.path.join(self.images_dir, split)):
                file_id = file.split(".")[0]
                shutil.move(
                    os.path.join(self.labels_dir, f"{file_id}.txt"),
                    os.path.join(self.labels_dir, split, f"{file_id}.txt"),
                )

    def _generate_config_file(self) -> None:
        """Generates a configuration file for YOLO."""
        data_yaml = YAMLConfig.load_yaml(os.path.join(self.labels_dir, "data.yaml"))
        config_data = {
            "train": os.path.abspath(f"{self.images_dir}/train"),
            "val": os.path.abspath(f"{self.images_dir}/val"),
            "test": os.path.abspath(f"{self.images_dir}/test"),
            "nc": data_yaml.get("nc", 10),
            "names": data_yaml.get(
                "names", [f"class{i}" for i in range(data_yaml.get("nc", 10))]
            ),
        }

        YAMLConfig.save_yaml(config_data, self.config_path)
