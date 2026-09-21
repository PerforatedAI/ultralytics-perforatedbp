# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Download Cityscapes and build the 8-class detection set in YOLO layout.

Run from the repository root:
    python examples/YOLO26-PerforatedAI-Python/prepare_cityscapes_det.py --download
The output directory must match the `path` key in cityscapes-det.yaml.
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from multiprocessing import Pool, cpu_count
from pathlib import Path

import appdirs
import cityscapesscripts.helpers.labels as cs_labels
import cv2
import numpy as np

from ultralytics.utils import LOGGER, TQDM

# Cityscapes package names csDownload understands. gtFine carries the instance id maps, leftImg8bit the images
DOWNLOAD_PACKAGES = ("gtFine_trainvaltest.zip", "leftImg8bit_trainvaltest.zip")
CREDENTIALS_APP = ("cityscapesscripts", "cityscapes")  # csDownload reads its credentials from this appdirs location
CREDENTIALS_FILE_NAME = "credentials.json"
SPLITS = ("train", "val")
IMAGE_DIR_NAME = "leftImg8bit"
LABEL_DIR_NAME = "gtFine"
IMAGE_SUFFIX = "_leftImg8bit.png"
INSTANCE_SUFFIX = "_gtFine_instanceIds.png"
FIRST_INSTANCE_ID = 24  # ids below this are stuff classes
CROWD_ID_CEILING = 1000  # ids below this mark a crowd region, which mmdet tags iscrowd and COCO consumers drop
MIN_BOX_PIXELS = 1  # matches the guard convert_coco applies before writing a YOLO line


def build_class_map() -> tuple[dict[int, int], list[str]]:
    """Map Cityscapes label ids onto contiguous YOLO class indices using mmdet's hasInstances filter.

    Returns:
        class_map (dict[int, int]): Cityscapes label id mapped to contiguous class index.
        class_names (list[str]): Class names in index order.
    """
    selected = [label for label in cs_labels.labels if label.hasInstances and not label.ignoreInEval]
    return {label.id: i for i, label in enumerate(selected)}, [label.name for label in selected]


def build_label_lines(instance_path: Path, class_map: dict[int, int]) -> list[str]:
    """Convert one instance id map into normalized YOLO detection lines.

    Instance ids encode the class as id // 1000 above the crowd ceiling and as the bare label id below it. The box is
    the tight extent of the instance mask, which is what pycocotools toBbox returns for the same mask in mmdet.

    Args:
        instance_path (Path): Path to a gtFine instanceIds png.
        class_map (dict[int, int]): Cityscapes label id mapped to contiguous class index.

    Returns:
        (list[str]): YOLO label lines as "class cx cy w h" with normalized coordinates.
    """
    instances = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
    if instances is None:
        raise RuntimeError(f"Could not read instance map {instance_path}")
    height, width = instances.shape[:2]

    lines = []
    for instance_id in np.unique(instances[instances >= FIRST_INSTANCE_ID]):
        if instance_id < CROWD_ID_CEILING:
            continue
        class_index = class_map.get(int(instance_id) // CROWD_ID_CEILING)
        if class_index is None:
            continue
        rows, cols = np.nonzero(instances == instance_id)
        box_width = cols.max() - cols.min() + 1
        box_height = rows.max() - rows.min() + 1
        if box_width < MIN_BOX_PIXELS or box_height < MIN_BOX_PIXELS:
            continue
        cx = (cols.min() + cols.max() + 1) / 2 / width
        cy = (rows.min() + rows.max() + 1) / 2 / height
        lines.append(f"{class_index} {cx:.6f} {cy:.6f} {box_width / width:.6f} {box_height / height:.6f}")
    return lines


def convert_one_image(job: tuple[Path, Path, Path, dict[int, int]]) -> int:
    """Write one image's label file and symlink its image into the flat split directory.

    Args:
        job (tuple[Path, Path, Path, dict[int, int]]): Instance map path, image path, split image dir, class map.

    Returns:
        (int): Number of boxes written.
    """
    instance_path, image_path, image_dir, class_map = job
    label_dir = image_dir.parent.parent / "labels" / image_dir.name
    lines = build_label_lines(instance_path, class_map)
    (label_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    target = image_dir / image_path.name
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(image_path.resolve())
    return len(lines)


def build_split(
    cityscapes_root: Path, out_root: Path, split: str, class_map: dict[int, int], workers: int
) -> tuple[int, int]:
    """Convert one split into YOLO labels and a flat image directory, dropping the per-city nesting.

    Args:
        cityscapes_root (Path): Directory holding leftImg8bit and gtFine.
        out_root (Path): Destination dataset root.
        split (str): Split name, train or val.
        class_map (dict[int, int]): Cityscapes label id mapped to contiguous class index.
        workers (int): Worker processes used for the conversion.

    Returns:
        images (int): Number of images converted.
        boxes (int): Number of boxes written.
    """
    image_root = cityscapes_root / IMAGE_DIR_NAME / split
    label_root = cityscapes_root / LABEL_DIR_NAME / split
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(
            f"Expected {image_root} and {label_root}. Run with --download, or point --cityscapes-root at the "
            f"directory that holds {IMAGE_DIR_NAME} and {LABEL_DIR_NAME}."
        )
    image_out = out_root / "images" / split
    (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    image_out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for image_path in sorted(image_root.rglob(f"*{IMAGE_SUFFIX}")):
        stem = image_path.name[: -len(IMAGE_SUFFIX)]
        instance_path = label_root / image_path.parent.name / f"{stem}{INSTANCE_SUFFIX}"
        if not instance_path.is_file():
            raise FileNotFoundError(f"Image {image_path} has no instance map at {instance_path}.")
        jobs.append((instance_path, image_path, image_out, class_map))
    if not jobs:
        raise FileNotFoundError(f"Found no {IMAGE_SUFFIX} files under {image_root}.")

    with Pool(workers) as pool:
        counts = list(
            TQDM(
                pool.imap_unordered(convert_one_image, jobs, chunksize=16), total=len(jobs), desc=f"Converting {split}"
            )
        )
    return len(jobs), int(sum(counts))


def download_cityscapes(cityscapes_root: Path) -> None:
    """Fetch and unzip the two Cityscapes packages, leaving an existing extraction alone.

    Args:
        cityscapes_root (Path): Directory the packages are downloaded and unzipped into.
    """
    credentials = Path(appdirs.user_data_dir(*CREDENTIALS_APP)) / CREDENTIALS_FILE_NAME
    if not credentials.is_file():
        raise RuntimeError(
            f"Cityscapes downloads need an account, and csDownload reads it only from {credentials}. Write that file "
            f'as {{"username": ..., "password": ...}} at 600 permissions, or run csDownload once by hand and answer '
            f"yes when it offers to store the credentials."
        )
    cityscapes_root.mkdir(parents=True, exist_ok=True)
    for package in DOWNLOAD_PACKAGES:
        archive = cityscapes_root / package
        if (cityscapes_root / package.split("_")[0]).is_dir():
            LOGGER.info(f"{package} already extracted, skipping")
            continue
        if not archive.is_file():
            LOGGER.info(f"Downloading {package}")
            subprocess.run(["csDownload", "-d", str(cityscapes_root), package], check=True)
        LOGGER.info(f"Unzipping {package}")
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(cityscapes_root)


def main():
    """Parse arguments, optionally download Cityscapes, and convert both splits."""
    parser = argparse.ArgumentParser(description="Build Cityscapes 8-class detection in YOLO layout")
    parser.add_argument(
        "--cityscapes-root", type=str, default="datasets/cityscapes-raw", help="dir holding leftImg8bit and gtFine"
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="datasets/cityscapes-det",
        help="destination root, matches cityscapes-det.yaml path",
    )
    parser.add_argument("--download", action="store_true", help="fetch the Cityscapes packages before converting")
    parser.add_argument(
        "--workers", type=int, default=max(1, cpu_count() - 1), help="worker processes for the conversion"
    )
    args = parser.parse_args()

    cityscapes_root = Path(args.cityscapes_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    if args.download:
        download_cityscapes(cityscapes_root)

    class_map, class_names = build_class_map()
    LOGGER.info(f"Class order: {class_names}")
    for split in SPLITS:
        images, boxes = build_split(cityscapes_root, out_root, split, class_map, args.workers)
        LOGGER.info(f"{split}: {images} images, {boxes} boxes")
    LOGGER.info(
        f"Cityscapes detection set ready at {out_root}. Confirm names in cityscapes-det.yaml match the class order above."
    )


if __name__ == "__main__":
    main()