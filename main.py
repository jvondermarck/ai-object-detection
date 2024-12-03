import os

from dotenv import load_dotenv
from picsellia import Client

load_dotenv()


client = Client(
    api_token=os.getenv("PICSELLIA_API_TOKEN"), organization_name="Picsalex-MLOps"
)
dataset = client.get_dataset_version_by_id("0193688e-aa8f-7cbe-9396-bec740a262d0")
dataset.list_assets().download("./datasets")

project = client.get_project(project_name="Groupe_7")

experiment = project.create_experiment(
    name="experiment-0",
    description="base experiment",
)

experiment.attach_dataset(
    name="⭐️ cnam_product_2024",
    dataset_version=dataset,
)
