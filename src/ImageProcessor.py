import os
from datetime import datetime
from typing import Final

from PIL import Image

from src.image_utils import add_padding_to_image, get_resized_dimensions


class ImageProcessor:
    PADDING_COLOR: Final[tuple[int, int, int]] = (114, 114, 114)
    OUTPUT_FOLDER: Final[str] = "dataset"
    DATETIME_FORMAT: Final[str] = "%Y%m%d%H%M%S"

    def __init__(self, relative_path_image_folder: str) -> None:
        """
        Initialize the ImageProcessor with a folder path containing images.

        :param relative_path_image_folder: Path to the folder containing images to be processed
        :raises Exception: If the folder does not exist
        """
        if not os.path.exists(relative_path_image_folder):
            raise Exception("Folder does not exist...")

        self.relative_path_image_folder = relative_path_image_folder
        self.output_folder_path = self._generate_output_folder(
            self.output_dataset_folder_name
        )

    @property
    def output_dataset_folder_name(self) -> str:
        return datetime.now().strftime(self.DATETIME_FORMAT)

    def _generate_output_folder(self, folder_name: str) -> str:
        folder_path = os.path.join(self.OUTPUT_FOLDER, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def process_image(self, file_path: str, filename: str, new_size: int) -> None:
        """
        Resize and add padding to an image, then save it in the output folder.

        :param file_path: Path to the input image
        :param filename: The name for the output image
        :param new_size: The desired size for the final image
        :raises Exception: If an error occurs while processing the image file
        :return: None
        """
        try:
            with Image.open(file_path) as img:
                new_width, new_height = get_resized_dimensions(img, new_size)
                img = img.resize((new_width, new_height))
                img = add_padding_to_image(img, new_size, self.PADDING_COLOR)
                output_path = os.path.join(self.output_folder_path, filename)
                img.save(output_path)
        except Exception as e:
            print(f"An unexpected error occurred while processing '{filename}': {e}")

    def process_folder(self, new_size: int) -> None:
        """
        Process all images in the specified folder, resizing and adding padding as needed.

        :param new_size: The desired size for the images
        :return: None
        """
        for filename in os.listdir(self.relative_path_image_folder):
            file_path = os.path.join(self.relative_path_image_folder, filename)
            if os.path.isfile(file_path):
                self.process_image(file_path, filename, new_size)
