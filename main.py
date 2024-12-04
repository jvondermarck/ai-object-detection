import os
import random
import shutil
import zipfile

import yaml
from dotenv import load_dotenv
from picsellia import Client
from picsellia.types.enums import AnnotationFileType
from ultralytics import YOLO

# Charger les variables d'environnement
load_dotenv()

# Initialisation du client Picsellia
client = Client(
    api_token=os.getenv("PICSELLIA_API_TOKEN"), organization_name="Picsalex-MLOps"
)

# Télécharger les données depuis Picsellia
dataset = client.get_dataset_version_by_id("0193688e-aa8f-7cbe-9396-bec740a262d0")
dataset.list_assets().download("./datasets")

project = client.get_project(project_name="Groupe_7")
experiment = project.get_experiment("experiment-0")

datasets = experiment.list_attached_dataset_versions()
print("Datasets attachés à l'expérience :", datasets)

# Exporter les annotations au format YOLO
for dataset in datasets:
    dataset.export_annotation_file(
        AnnotationFileType.YOLO, "./datasets/annotations.zip"
    )

# Étape 1 : Trouver et décompresser le fichier ZIP
base_dir = "./datasets"
annotations_dir = os.path.join(base_dir, "annotations")

# Recherche du premier fichier ZIP dans les sous-dossiers
zip_file = None
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".zip"):
            zip_file = os.path.join(root, file)
            break
    if zip_file:
        break

if zip_file:
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    # Décompresser le fichier ZIP
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(annotations_dir)

    print(f"Archive décompressée dans : {annotations_dir}")

    # Supprimer l'archive ZIP
    os.remove(zip_file)
    print(f"Archive {zip_file} supprimée.")
else:
    raise FileNotFoundError(
        "Aucune archive ZIP trouvée dans le dossier datasets ou ses sous-dossiers."
    )

# Vérification des fichiers extraits dans annotations_dir
extracted_files = os.listdir(annotations_dir)
print(f"Fichiers extraits : {extracted_files}")

# Étape 2 : Structurer les données pour YOLO
output_dir = "./datasets/structured"
images_dir = f"{output_dir}/images"
labels_dir = f"{output_dir}/labels"
train_dir = "train"
val_dir = "val"
test_dir = "test"
random.seed(42)  # Fixer la seed pour la reproductibilité
split_ratios = {"train": 0.6, "val": 0.2, "test": 0.2}

# Créer les répertoires de sortie
os.makedirs(f"{images_dir}/{train_dir}", exist_ok=True)
os.makedirs(f"{images_dir}/{val_dir}", exist_ok=True)
os.makedirs(f"{images_dir}/{test_dir}", exist_ok=True)
os.makedirs(f"{labels_dir}/{train_dir}", exist_ok=True)
os.makedirs(f"{labels_dir}/{val_dir}", exist_ok=True)
os.makedirs(f"{labels_dir}/{test_dir}", exist_ok=True)

# Liste des fichiers d'images dans le dossier datasets (et non annotations)
image_dir = base_dir  # Les images sont dans le dossier datasets
image_files = [
    f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png", ".JPG"))
]
print("Images trouvées :", image_files)

# Liste des fichiers d'annotations dans le dossier annotations
label_files = [f for f in extracted_files if f.endswith(".txt")]
print("Labels trouvés :", label_files)

# Vérifier que chaque image a son fichier d'annotation correspondant
image_to_label = {
    img: img.replace(".JPG", ".txt")
    .replace(".jpg", ".txt")
    .replace(".png", ".txt")
    .replace(".jpeg", ".txt")
    for img in image_files
}

paired_files = [(img, lbl) for img, lbl in image_to_label.items() if lbl in label_files]

# Vérification des fichiers pairs
print(f"Fichiers d'images et annotations correspondants : {paired_files}")

# Mélanger les données et les répartir en train/val/test
random.shuffle(paired_files)
n_total = len(paired_files)
n_train = int(n_total * split_ratios["train"])
n_val = int(n_total * split_ratios["val"])

splits: dict[str, list[tuple[str, str]]] = {
    "train": paired_files[:n_train],
    "val": paired_files[n_train : n_train + n_val],
    "test": paired_files[n_train + n_val :],
}

# Déplacer les fichiers vers les répertoires correspondants
for split in splits.keys():
    for img, lbl in splits[split]:
        image_dest = f"{images_dir}/{split}"
        label_dest = f"{labels_dir}/{split}"
        shutil.move(os.path.join(image_dir, img), os.path.join(image_dest, img))
        shutil.move(os.path.join(annotations_dir, lbl), os.path.join(label_dest, lbl))

# Charger le fichier data.yaml
data_yaml_path = os.path.join(annotations_dir, "data.yaml")

with open(data_yaml_path, "r") as f:
    data = yaml.safe_load(f)

# Extraire les informations nécessaires pour config.yaml
nc = data.get("nc", 10)  # Nombre de classes, par défaut 10 si non spécifié
names = data.get("names", [f"class{i}" for i in range(nc)])  # Liste des noms de classes

# Étape 3 : Générer le fichier config.yaml en utilisant les données du data.yaml
config = {
    "train": os.path.abspath(f"{images_dir}/train"),
    "val": os.path.abspath(f"{images_dir}/val"),
    "test": os.path.abspath(f"{images_dir}/test"),
    "nc": nc,  # Nombre de classes récupéré depuis data.yaml
    "names": names,  # Liste des classes récupérée depuis data.yaml
}

# Sauvegarder le fichier config.yaml
config_path = os.path.abspath(f"{output_dir}/config.yaml")
with open(config_path, "w") as yaml_file:
    yaml.dump(config, yaml_file, default_flow_style=False)

print("Dataset structuré avec succès et fichier config.yaml généré.")
print(f"Chemin absolu du fichier config.yaml : {config_path}")

# Charger le modèle YOLOv11 (assurez-vous que vous avez la bonne version de YOLO)
model = YOLO(
    "yolo11n.pt"
)  # Utilisez yolov11.yaml ou un autre modèle YOLOv11 pré-entrainé

model.to("mps")  # Utiliser le GPU pour l'entraînement
# Définir les hyper-paramètres
hyperparameters = {
    "epochs": 5,  # Augmenter légèrement pour une meilleure convergence
    "batch": 8,  # Ajuster selon la mémoire disponible (tester 8 ou 4 si limite)
    "imgsz": 512,  # Réduire la taille des images pour économiser la mémoire
    "close_mosaic": 0,  # Activer si vous voulez plus de stabilité, sinon désactiver (0)
    "optimizer": "AdamW",  # Utiliser un optimiseur efficace (AdamW recommandé)
    "lr0": 0.001,  # Learning rate initial légèrement réduit pour stabilité
    "momentum": 0.937,  # Momentum standard pour YOLO (reste optimal)
    "weight_decay": 0.0005,  # Décroissance de poids pour éviter le surapprentissage
    "seed": 42,  # Reproductibilité
    "augment": True,  # Augmentation des données pour enrichir l'entraînement
    "cache": "ram",  # Charger les données en mémoire pour accélérer l'entraînement
}

# Charger les données d'entraînement avec le fichier config.yaml
data_yaml_path = "./datasets/structured/config.yaml"
print("Chargement du fichier config.yaml :", data_yaml_path)
model.train(
    data=data_yaml_path,  # Chemin vers votre fichier config.yaml
    **hyperparameters,  # Appliquez les hyper-paramètres
    project="./results",
)

print("Entraînement YOLOv11 lancé avec succès.")
