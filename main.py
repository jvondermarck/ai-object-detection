import os

from dotenv import load_dotenv
from picsellia import Client

load_dotenv()


client = Client(
    api_token=os.getenv("PICSELLIA_API_TOKEN"), organization_name="Picsalex-MLOps"
)
dataset = client.get_dataset_version_by_id("0193688e-aa8f-7cbe-9396-bec740a262d0")
dataset.list_assets().download("./datasets")
