import tensorflow as tf

from .config import Settings
from .data import build_datasets


def main() -> None:
    settings = Settings()
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{settings.model_path}'. Run training first with "
            "'python -m src.image_classification.train'."
        )

    _, val_ds, class_names = build_datasets(settings)
    model = tf.keras.models.load_model(settings.model_path)

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Classes: {class_names}")
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
