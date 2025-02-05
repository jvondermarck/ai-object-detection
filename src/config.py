import os

from dotenv import load_dotenv

load_dotenv()


def get_required_env(name: str) -> str:
    """Get an environment variable, raise an error if it is missing."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing '{name}' environment variable.")
    return value


PICSELLIA_API_TOKEN = get_required_env("PICSELLIA_API_TOKEN")
PICSELLIA_ORGANIZATION_NAME = get_required_env("PICSELLIA_ORGANIZATION_NAME")
