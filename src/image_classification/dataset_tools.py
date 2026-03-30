import argparse
from pathlib import Path

from .config import Settings
from .data import get_dataset_summary


DEFAULT_CLASSES = ["cats", "dogs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset setup and validation tools")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create class folders under data/raw",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Class names to create when using --init",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the current dataset summary",
    )
    return parser.parse_args()


def initialize_dataset_folders(settings: Settings, class_names: list[str]) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for class_name in class_names:
        (settings.data_dir / class_name).mkdir(parents=True, exist_ok=True)


def format_summary_lines(summary: list[tuple[str, int]]) -> list[str]:
    if not summary:
        return ["No class folders found in data/raw."]

    total_images = sum(count for _, count in summary)
    lines = [f"Found {len(summary)} classes and {total_images} images total:"]
    lines.extend(f"- {class_name}: {count} images" for class_name, count in summary)
    return lines


def main() -> None:
    args = parse_args()
    settings = Settings()

    if args.init:
        initialize_dataset_folders(settings, args.classes)
        created = ", ".join(args.classes)
        print(f"Created dataset folders in {Path(settings.data_dir)}: {created}")

    if args.summary or not args.init:
        for line in format_summary_lines(get_dataset_summary(settings)):
            print(line)


if __name__ == "__main__":
    main()
