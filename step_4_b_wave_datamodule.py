"""Lightning data module for the leakage-resistant waviness splits."""

from __future__ import annotations

from pathlib import Path

# Import the dataset module first: it installs a small torchvision compatibility
# definition before Lightning imports torchmetrics/torchvision transitively.
from step_4_a_wave_dataset import WaveDataset, build_evaluation_transform, build_training_transform

import lightning.pytorch as pl
from torch.utils.data import DataLoader


class WaveDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str | Path = ".", batch_size: int = 8, num_workers: int = 0) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_dir = self.data_dir / "step-2-final-water-data"
        self.split_dir = self.data_dir / "step-3-dataset-splits"
        self.train_dataset: WaveDataset | None = None
        self.val_dataset: WaveDataset | None = None
        self.test_dataset: WaveDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = WaveDataset(self.split_dir / "train.csv", self.image_dir, build_training_transform())
            self.val_dataset = WaveDataset(self.split_dir / "validation.csv", self.image_dir, build_evaluation_transform())
        if stage in (None, "test"):
            self.test_dataset = WaveDataset(self.split_dir / "test.csv", self.image_dir, build_evaluation_transform())

    def _loader(self, dataset: WaveDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        return self._loader(self.test_dataset, shuffle=False)
