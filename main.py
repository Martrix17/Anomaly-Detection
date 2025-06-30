import argparse
from src.train_test_infer import run


def parse_args():
    parser = argparse.ArgumentParser(description="Image Anomaly Detection")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "infer"],
        default="train",
        help="Operation mode (train, test, or infer)",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/mvtec-ad/2",
        help="Root directory of the MVTec dataset",
    )
    parser.add_argument(
        "--class_name", type=str, default="bottle", help="Target class to analyze "
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for the encoder and decoder",
    )
    parser.add_argument(
        "--disc_learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for the discriminator",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training and testing"
    )
    parser.add_argument(
        "--epochs", type=int, default=500, help="Number of training epochs"
    )
    parser.add_argument(
        "--backbone", type=str, default="conv", help="Backbone architecture name"
    )
    parser.add_argument(
        "--recon_weights",
        type=float,
        nargs=6,
        default=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        help="Reconstruction loss weights (MSE, SSIM, Perceptual, Edge, Color, Texture)",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="Latent space dimensionality"
    )
    parser.add_argument(
        "--image_resize",
        type=int,
        default=256,
        help="Size of the input images to be resized to",
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Final cropped/resized image size"
    )
    parser.add_argument(
        "--rand_aug",
        action="store_true",
        help="Whether to use random data augmentations",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (required for test/infer, optional for train to resume)",
    )

    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()
    elif isinstance(args, list):
        args = parse_args(args)

    if args.mode in ["test", "infer"] and not args.checkpoint:
        raise ValueError(f"--checkpoint is required for {args.mode} mode")

    run(args)


if __name__ == "__main__":
    main()
