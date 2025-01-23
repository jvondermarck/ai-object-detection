from argparse import ArgumentParser

from src.inference import infer
from src.training import train


def main() -> None:
    parser = ArgumentParser(description="Train or run inference with Picsellia")
    add_subparsers(parser)

    args = parser.parse_args()

    if args.command == "train":
        return train(args.dataset_version_id, args.project_id)

    if args.command == "infer":
        source = args.image or args.video or (0 if args.webcam else None)

        if not isinstance(source, (str, int)):
            raise ValueError(
                "Invalid source type. Must be a string (image/video path) or an integer (webcam index)."
            )

        return infer(
            model_version_id=args.model_version_id,
            source=source,
            output=args.output,
            conf=args.conf,
            iou=args.iou,
        )

    parser.print_help()


def add_subparsers(parser: ArgumentParser) -> None:
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
    infer_group.add_argument("--video", help="Path to the video for inference")
    infer_group.add_argument(
        "--webcam", action="store_true", help="Use webcam for inference"
    )

    infer_parser.add_argument(
        "--output",
        help="Path to save inference results (ex: output directory for annotated images/videos)",
    )
    infer_parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )
    infer_parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )


if __name__ == "__main__":
    main()
