import os

from dotenv import load_dotenv


class ConfigManager:
    """Configuration manager for environment variables.

    Methods:
        load_environment: Loads environment variables from a .env file.
        get_env_variable: Retrieves an environment variable by its name.
    """

    @staticmethod
    def load_environment() -> None:
        """Loads environment variables from a .env file."""
        load_dotenv()

    @staticmethod
    def get_env_variable(key: str) -> str:
        """Retrieves the value of an environment variable.

        Args:
            key (str): The name of the environment variable.

        Returns:
            str: The value of the environment variable.

        Raises:
            ValueError: If the environment variable is missing or empty.
        """
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Environment variable '{key}' is missing.")
        return value
