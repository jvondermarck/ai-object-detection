# AI Project - Image detection

## Introduction

Image detection AI with a neural networks to detect some sodas.

## Installation

1. Install Python 3.11 via Windows Store
2. Clone the project from GitHub and open it via PyCharm
3. Set up the virtual environment and do not forget to activate it before running the project
4. Install the required packages via `pip install -r requirements.txt`
   - On windows, type this command `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` and also do not forget to install Cuda.
   - https://pytorch.org/get-started/locally/#start-locally for more information on which PyTorch version to install
6. Copy `.env.example` to `.env` and setup your environment variables
6. Run the project via `python main.py`

> You can find the API Token of Picsellia in Personal Settings > Tokens

### Usage of pre-commit

1. Install pre-commit via `pip install pre-commit` (if not already installed via the requirements.txt)
2. Run the command `pre-commit install` in the project directory
3. Now, every time you commit, the pre-commit hooks will be executed

> You can manually run the pre-commit hooks via `pre-commit run --all-files`

### Prerequisites

- Python 3.11
- PyCharm
- Cuda for Windows

## Run the project

Run either the training or the inference command ! :)

> If you are lost, run the command `python main.py infer -h` or `python main.py train -h` to get the help message

### To train the model

Here is the usage version
```
usage: main.py train [-h] dataset_version_id project_id

positional arguments:
  dataset_version_id  Version ID of the Picsellia dataset to use for training
  project_id          Picsellia Project ID to use for training

options:
  -h, --help          show this help message and exit
```

### To test your model with inference

Here is the usage version to run the inference with an image, a video or open the webcam to detect the items

```
usage: main.py infer [-h] (--image IMAGE | --video VIDEO | --webcam) [--output OUTPUT] [--conf CONF] [--iou IOU] model_version_id

positional arguments:
  model_version_id  Version ID of the model to use for inference

options:
  -h, --help        show this help message and exit
  --image IMAGE     Path to the image for inference
  --video VIDEO     Path to the video for inference
  --webcam          Use webcam for inference
  --output OUTPUT   Path to save inference results (ex: output directory for annotated images/videos)
  --conf CONF       Confidence threshold for detections (default: 0.5)
  --iou IOU         IoU threshold for NMS (default: 0.45)
```

Examples :
```
python main.py infer --video 'video.mp4' --output output --conf 0.9 --iou <0.5 01943c53-956c-775b-afd1-38e9f87ed22e>
python main.py infer --webcam --conf 0.9 <01943c53-956c-775b-afd1-38e9f87ed22e>
```

> [!NOTE]
> IoU (Intersection over Union) measures how accurately a predicted bounding box overlaps the ground truth, while confidence reflects the model's certainty in its prediction.
