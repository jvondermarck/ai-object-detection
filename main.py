from argparse import ArgumentParser

from src.inference import infer
from src.training import train


def main():
    parser = ArgumentParser(description="Train or run inference with Picsellia")
    add_subparsers(parser)

    args = parser.parse_args()

    if args.command == "train":
        return train(args.dataset_version_id, args.project_id)

    if args.command == "infer":
        if args.image:
            return infer(args.model_version_id, args.image)
        if args.video:
            return infer(args.model_version_id, args.video)
        if args.webcam:
            return infer(args.model_version_id)

    parser.print_help()


def add_subparsers(parser: ArgumentParser):
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Add train subparser
    train_parser = subparsers.add_parser("train", help="Launch the training pipeline")
    train_parser.add_argument(
        "dataset_version_id",
        help="Version ID of the Picsellia dataset to use for training",
    )
    train_parser.add_argument(
        "project_id", help="Picsellia Project ID to use for training"
    )

    # Add infer subparser
    infer_parser = subparsers.add_parser("infer", help="Launch the inference pipeline")
    infer_parser.add_argument(
        "model_version_id",
        help="Version ID of the model to use for inference",
    )

    infer_group = infer_parser.add_mutually_exclusive_group(required=True)
    infer_group.add_argument("--image", help="Path to the image for inference")
    infer_group.add_argument("--video", help="Path to the video for infer")
    infer_group.add_argument(
        "--webcam", action="store_true", help="Use webcam for inference"
    )


if __name__ == "__main__":
    main()
