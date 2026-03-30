import shutil
import uuid
from pathlib import Path

from src.image_classification.config import Settings
from src.image_classification.data import count_images_in_directory
from src.image_classification.dataset_tools import format_summary_lines


def test_settings_paths_are_built_from_model_dir() -> None:
    settings = Settings()

    assert settings.model_path.name == "cifar10_classifier.keras"
    assert settings.label_map_path.name == "labels.txt"
    assert settings.history_path.name == "history.csv"


def test_count_images_in_directory_filters_supported_extensions() -> None:
    temp_dir = Path("tests_runtime") / f"images_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        (temp_dir / "cat.jpg").write_text("x", encoding="utf-8")
        (temp_dir / "dog.png").write_text("x", encoding="utf-8")
        (temp_dir / "notes.txt").write_text("x", encoding="utf-8")

        count = count_images_in_directory(temp_dir, (".jpg", ".png"))

        assert count == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_format_summary_lines_handles_empty_summary() -> None:
    assert format_summary_lines([]) == ["No class folders found in data/raw."]
