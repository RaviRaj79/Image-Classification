import shutil
import uuid
from pathlib import Path

from src.image_classification.config import Settings
from src.image_classification.dataset_tools import initialize_dataset_folders


def test_initialize_dataset_folders_creates_class_directories() -> None:
    base_dir = Path("tests_runtime") / f"dataset_{uuid.uuid4().hex}"
    settings = Settings(data_dir=base_dir / "raw")

    try:
        initialize_dataset_folders(settings, ["cats", "dogs"])

        assert (settings.data_dir / "cats").is_dir()
        assert (settings.data_dir / "dogs").is_dir()
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
