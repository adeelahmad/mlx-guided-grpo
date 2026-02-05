"""MLX-GRPO command line interface."""

import sys


def main():
    """Main entry point for mlx-grpo CLI."""
    from .train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
