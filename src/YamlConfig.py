import yaml


class YAMLConfig:
    """Manages YAML configurations.

    Methods:
        load_yaml: Loads a YAML file.
        save_yaml: Saves a dictionary to a YAML file.
    """

    @staticmethod
    def load_yaml(file_path: str) -> dict:
        """Loads a YAML file.

        Args:
            file_path (str): Path to the YAML file.

        Returns:
            dict: Content of the YAML file.
        """
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def save_yaml(data: dict, file_path: str) -> None:
        """Saves data to a YAML file.

        Args:
            data (dict): Data to save.
            file_path (str): Path to the YAML file.
        """
        with open(file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)
