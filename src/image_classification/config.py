from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    data_dir: Path = Path("data/raw")
    model_dir: Path = Path("models")
    image_size: tuple[int, int] = (32, 32)
    batch_size: int = 128
    validation_split: float = 0.2
    seed: int = 42
    epochs: int = 10
    learning_rate: float = 1e-3
    use_augmentation: bool = True
    model_name: str = "cifar10_classifier.keras"
    label_map_name: str = "labels.txt"
    history_name: str = "history.csv"
    supported_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_name

    @property
    def label_map_path(self) -> Path:
        return self.model_dir / self.label_map_name

    @property
    def history_path(self) -> Path:
        return self.model_dir / self.history_name

