from pathlib import Path

import tensorflow as tf

from .config import Settings


AUTOTUNE = tf.data.AUTOTUNE


def get_class_directories(settings: Settings) -> list[Path]:
    if not settings.data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory '{settings.data_dir}' does not exist."
        )

    return sorted(path for path in settings.data_dir.iterdir() if path.is_dir())


def count_images_in_directory(
    directory: Path, supported_extensions: tuple[str, ...]
) -> int:
    return sum(
        1
        for file_path in directory.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions
    )


def get_dataset_summary(settings: Settings) -> list[tuple[str, int]]:
    class_directories = get_class_directories(settings)
    return [
        (
            class_dir.name,
            count_images_in_directory(class_dir, settings.supported_extensions),
        )
        for class_dir in class_directories
    ]


def validate_dataset_directory(settings: Settings) -> None:
    class_summary = get_dataset_summary(settings)
    if not class_summary:
        raise ValueError(
            "No class folders found in data/raw. Create folders like "
            "'data/raw/cats' and 'data/raw/dogs', then add images inside them."
        )

    empty_classes = [class_name for class_name, count in class_summary if count == 0]
    if empty_classes:
        formatted = ", ".join(empty_classes)
        raise ValueError(
            f"These class folders do not contain any images: {formatted}. "
            "Add .jpg/.jpeg/.png/.bmp files before training."
        )

    if len(class_summary) < 2:
        raise ValueError("At least 2 class folders are required for classification.")


def build_datasets(
    settings: Settings,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    validate_dataset_directory(settings)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        settings.data_dir,
        validation_split=settings.validation_split,
        subset="training",
        seed=settings.seed,
        image_size=settings.image_size,
        batch_size=settings.batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        settings.data_dir,
        validation_split=settings.validation_split,
        subset="validation",
        seed=settings.seed,
        image_size=settings.image_size,
        batch_size=settings.batch_size,
    )

    class_names = train_ds.class_names
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def load_image_for_prediction(
    image_path: str | Path, image_size: tuple[int, int]
) -> tf.Tensor:
    image = tf.keras.utils.load_img(image_path, target_size=image_size)
    array = tf.keras.utils.img_to_array(image)
    return tf.expand_dims(array, axis=0)
