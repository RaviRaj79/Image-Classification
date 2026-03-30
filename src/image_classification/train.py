from pathlib import Path

import pandas as pd
import tensorflow as tf

from .config import Settings
from .data import build_datasets, get_dataset_summary
from .model import build_model


def save_label_map(class_names: list[str], output_path: Path) -> None:
    output_path.write_text("\n".join(class_names), encoding="utf-8")


def save_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    pd.DataFrame(history.history).to_csv(output_path, index=False)


def print_dataset_summary(settings: Settings) -> None:
    summary = get_dataset_summary(settings)
    print("Dataset summary:")
    for class_name, count in summary:
        print(f"- {class_name}: {count} images")


def main() -> None:
    settings = Settings()
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    print_dataset_summary(settings)
    train_ds, val_ds, class_names = build_datasets(settings)
    model = build_model(settings, num_classes=len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=settings.epochs,
        callbacks=callbacks,
    )

    model.save(settings.model_path)
    save_label_map(class_names, settings.label_map_path)
    save_history(history, settings.history_path)

    print(f"Saved model to {settings.model_path}")
    print(f"Saved labels to {settings.label_map_path}")
    print(f"Saved training history to {settings.history_path}")


if __name__ == "__main__":
    main()
