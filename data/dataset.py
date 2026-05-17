"""
data/dataset.py
───────────────
Dataset loaders for:
  • FPUS23          — 4-class fetal phantom ultrasound (AC, BPD, FL, No Plane)
  • FETAL_PLANES_DB — 6-class real maternal-fetal ultrasound

Both datasets are publicly available:
  FPUS23:          https://github.com/bharathprabakaran/FPUS23
  FETAL_PLANES_DB: https://doi.org/10.5281/zenodo.3904280

Expected directory layout
─────────────────────────
data/
  FPUS23/
    AC_PLANE/   *.png / *.jpg
    BPD_PLANE/
    FL_PLANE/
    NO_PLANE/

  FETAL_PLANES_DB/
    Fetal_Abdomen/
    Fetal_Brain/
    Fetal_Femur/
    Fetal_Thorax/
    Maternal_Cervix/
    Other/
"""

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# ─────────────────────────── label maps ───────────────────────────────────────

FPUS23_CLASSES: Dict[str, int] = {
    "AC_PLANE": 0,
    "BPD_PLANE": 1,
    "FL_PLANE": 2,
    "NO_PLANE": 3,
}

FETAL_PLANES_CLASSES: Dict[str, int] = {
    "Fetal_Abdomen": 0,
    "Fetal_Brain": 1,
    "Fetal_Femur": 2,
    "Fetal_Thorax": 3,
    "Maternal_Cervix": 4,
    "Other": 5,
}

DATASET_INFO = {
    "fpus23": {
        "class_map": FPUS23_CLASSES,
        "num_classes": 4,
        "root_subdir": "FPUS23",
    },
    "fetal_planes_db": {
        "class_map": FETAL_PLANES_CLASSES,
        "num_classes": 6,
        "root_subdir": "FETAL_PLANES_DB",
    },
}

# ─────────────────────────── transforms ───────────────────────────────────────

IMG_SIZE = 224

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────── base dataset ─────────────────────────────────────

class FetalUltrasoundDataset(Dataset):
    """
    Generic fetal ultrasound image dataset.

    Parameters
    ----------
    root      : str | Path — path to the dataset root directory.
    dataset   : str        — 'fpus23' or 'fetal_planes_db'.
    split     : str        — 'train' or 'test'.
    transform : callable   — torchvision transform pipeline.
    seed      : int        — random seed for reproducible train/test split.
    test_ratio: float      — fraction of data reserved for testing (default 0.20).
    """

    def __init__(
        self,
        root: str,
        dataset: str = "fpus23",
        split: str = "train",
        transform: Optional[Callable] = None,
        seed: int = 42,
        test_ratio: float = 0.20,
    ):
        assert dataset in DATASET_INFO, f"dataset must be one of {list(DATASET_INFO)}"
        assert split in ("train", "test"), "split must be 'train' or 'test'"

        self.root = Path(root)
        self.dataset = dataset
        self.split = split
        self.transform = transform or (TRAIN_TRANSFORM if split == "train" else EVAL_TRANSFORM)
        self.seed = seed

        info = DATASET_INFO[dataset]
        self.class_map = info["class_map"]
        self.num_classes = info["num_classes"]
        self.class_names = list(self.class_map.keys())

        # Scan all image paths
        data_dir = self.root / info["root_subdir"]
        self.samples: List[Tuple[Path, int]] = []
        for class_name, label in self.class_map.items():
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found in '{data_dir}'. "
                "Please download the dataset and place it under the expected directory."
            )

        # Stratified train/test split
        self.samples = self._stratified_split(self.samples, test_ratio, seed, split)

    # ── internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _stratified_split(
        samples: List[Tuple[Path, int]],
        test_ratio: float,
        seed: int,
        split: str,
    ) -> List[Tuple[Path, int]]:
        """Return the train or test portion using a stratified split."""
        rng = random.Random(seed)
        by_class: Dict[int, List] = {}
        for item in samples:
            by_class.setdefault(item[1], []).append(item)
        train_items, test_items = [], []
        for label, items in by_class.items():
            rng.shuffle(items)
            n_test = max(1, int(len(items) * test_ratio))
            test_items.extend(items[:n_test])
            train_items.extend(items[n_test:])
        return train_items if split == "train" else test_items

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_labels(self) -> List[int]:
        """Return a flat list of all labels (used by partitioning utilities)."""
        return [label for _, label in self.samples]


# ─────────────────────────── convenience loaders ──────────────────────────────

def get_dataloader(
    root: str,
    dataset: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
) -> DataLoader:
    """Return a DataLoader for the full train or test split."""
    ds = FetalUltrasoundDataset(root=root, dataset=dataset, split=split, seed=seed)
    shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def get_dataset_info(dataset: str) -> Dict:
    """Return class names and num_classes for a dataset string."""
    info = DATASET_INFO[dataset]
    return {
        "num_classes": info["num_classes"],
        "class_names": list(info["class_map"].keys()),
    }
