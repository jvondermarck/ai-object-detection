import os
import random
import shutil
import zipfile

from picsellia import Client
from picsellia.types.enums import AnnotationFileType


class DatasetManager:
    """Manages the downloading, extraction, and structuring of datasets.

    Attributes:
        base_dir (str): Root directory for dataset files.
        id_version (str): Dataset version identifier.
        annotations_dir (str): Directory for extracted annotations.
        structured_dir (str): Structured directory for YOLO.

    Methods:
        download_dataset: Downloads a dataset via Picsellia.
        export_annotations: Exports annotations in a specific format.
        extract_zip: Extracts ZIP files found in the base directory.
        structure_data_for_yolo: Structures data for YOLO based on given ratios.
    """

    def __init__(self, base_dir: str, id_version: str):
        """Initializes a dataset manager.

        Args:
            base_dir (str): Root directory for the dataset.
            id_version (str): Dataset version identifier.
        """
        self.base_dir = base_dir
        self.id_version = id_version
        self.annotations_dir = os.path.join(base_dir, "annotations")
        self.structured_dir = os.path.join(base_dir, "structured")

    def download_dataset(self, client: Client, dataset_id: str):
        """Downloads a dataset from Picsellia.

        Args:
            client (Client): Picsellia client.
            dataset_id (str): Dataset identifier.
        """
        dataset = client.get_dataset_version_by_id(dataset_id)
        dataset.list_assets().download(self.base_dir)

    def export_annotations(self, dataset, export_format: AnnotationFileType):
        """Exports annotations in a given format.

        Args:
            dataset: Picsellia dataset object.
            export_format (AnnotationFileType): Format for annotation export.
        """
        dataset.export_annotation_file(
            export_format, os.path.join(self.base_dir, "annotations.zip")
        )

    def extract_zip(self):
        """Extracts ZIP files found in the base directory.

        Raises:
            FileNotFoundError: If no ZIP file is found.
        """
        zip_file = self._find_zip_file()
        if zip_file:
            os.makedirs(self.annotations_dir, exist_ok=True)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(self.annotations_dir)
            os.remove(zip_file)
        else:
            raise FileNotFoundError("No ZIP archive found.")

    def structure_data_for_yolo(self, split_ratios: dict):
        """Structures data for the YOLO model.

        Args:
            split_ratios (dict): Ratios for splitting data (train, val, test).

        Returns:
            tuple: Paths to the image and label directories for each split.
        """
        os.makedirs(self.structured_dir, exist_ok=True)
        images_dir, labels_dir = self._prepare_directories()
        paired_files = self._pair_images_and_labels()

        splits = self._split_data(paired_files, split_ratios)
        for split, files in splits.items():
            for img, lbl in files:
                shutil.move(
                    os.path.join(self.base_dir, img),
                    os.path.join(images_dir[split], img),
                )
                shutil.move(
                    os.path.join(self.annotations_dir, lbl),
                    os.path.join(labels_dir[split], lbl),
                )

        return images_dir, labels_dir

    def _find_zip_file(self):
        """Searches for a ZIP file in the base directory.

        Returns:
            str: Path to the ZIP file, or None if no file is found.
        """
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".zip"):
                    return os.path.join(root, file)
        return None

    def _prepare_directories(self):
        """Prepares directories for images and labels.

        Returns:
            tuple: Dictionaries containing paths to image and label directories.
        """
        images_dir = {
            split: os.path.join(self.structured_dir, "images", split)
            for split in ["train", "val", "test"]
        }
        labels_dir = {
            split: os.path.join(self.structured_dir, "labels", split)
            for split in ["train", "val", "test"]
        }

        for directory in images_dir.values():
            os.makedirs(directory, exist_ok=True)
        for directory in labels_dir.values():
            os.makedirs(directory, exist_ok=True)

        return images_dir, labels_dir

    def _pair_images_and_labels(self):
        """Pairs images with their corresponding label files.

        Returns:
            list: List of tuples containing image and label file names.
        """
        image_files = [
            f
            for f in os.listdir(self.base_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        label_files = [
            f for f in os.listdir(self.annotations_dir) if f.endswith(".txt")
        ]

        image_to_label = {img: img.rsplit(".", 1)[0] + ".txt" for img in image_files}
        return [(img, lbl) for img, lbl in image_to_label.items() if lbl in label_files]

    def _split_data(self, paired_files, split_ratios):
        """Splits data into splits (train, val, test).

        Args:
            paired_files (list): List of (image, label) tuples.
            split_ratios (dict): Ratios for each split.

        Returns:
            dict: Dictionary containing splits and associated files.
        """
        random.shuffle(paired_files)
        n_total = len(paired_files)
        n_train = int(n_total * split_ratios["train"])
        n_val = int(n_total * split_ratios["val"])

        return {
            "train": paired_files[:n_train],
            "val": paired_files[n_train : n_train + n_val],
            "test": paired_files[n_train + n_val :],
        }
