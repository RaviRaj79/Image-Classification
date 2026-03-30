import argparse
from pathlib import Path

import tensorflow as tf

from .config import Settings
from .data import load_image_for_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict image class")
    parser.add_argument("--image", required=True, help="Path to an input image")
    return parser.parse_args()


def load_labels(settings: Settings) -> list[str]:
    if not settings.label_map_path.exists():
        raise FileNotFoundError(
            f"Label map not found at '{settings.label_map_path}'. Run training first."
        )
    return settings.label_map_path.read_text(encoding="utf-8").splitlines()


def main() -> None:
    args = parse_args()
    settings = Settings()
    image_path = Path(args.image)

    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{settings.model_path}'. Run training first."
        )

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found at '{image_path}'.")

    model = tf.keras.models.load_model(settings.model_path)
    labels = load_labels(settings)

    image = load_image_for_prediction(image_path, settings.image_size)
    probabilities = model.predict(image, verbose=0)[0]

    best_index = int(tf.argmax(probabilities).numpy())
    print(f"Predicted class: {labels[best_index]}")
    print(f"Confidence: {probabilities[best_index]:.4f}")


if __name__ == "__main__":
    main()
