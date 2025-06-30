import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.transforms import v2
import lightning as L
from PIL import Image


class AnomalyDataset(Dataset):
    def __init__(self, root_dir, class_name, image_resize, image_size, rand_aug, split):
        super().__init__()
        self.root_dir = root_dir
        self.class_name = class_name
        self.image_resize = image_resize
        self.image_size = image_size
        self.rand_aug = rand_aug
        self.split = split

        self.transform = v2.Compose(
            [
                v2.Resize(self.image_resize),
                v2.CenterCrop(self.image_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.mask_transform = v2.Compose(
            [
                v2.Resize(int(self.image_resize)),
                v2.CenterCrop(self.image_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        self.img_paths = []
        self.mask_paths = []
        self.labels = []

        self._get_image_data()

    def _get_image_data(self):
        class_path = os.path.join(self.root_dir, self.class_name, self.split)
        for anomaly in os.listdir(class_path):
            anomaly_path = os.path.join(class_path, anomaly)
            for img_name in os.listdir(anomaly_path):
                img_path = os.path.join(anomaly_path, img_name)
                self.img_paths.append(img_path)
                self.labels.append(anomaly)

                if self.split == "test" and anomaly != "good":
                    mask_path = os.path.join(
                        self.root_dir,
                        self.class_name,
                        "ground_truth",
                        anomaly,
                        img_name.replace(".png", "_mask.png"),
                    )
                    self.mask_paths.append(mask_path)
                else:
                    self.mask_paths.append(None)

    def _rand_transforms(self, index):
        np.random.seed(index)
        aug_list = [
            v2.ColorJitter(contrast=(0.8, 1.2)),
            v2.ColorJitter(brightness=(0.8, 1.2)),
            v2.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            v2.RandomHorizontalFlip(p=1),
            v2.RandomVerticalFlip(p=1),
            v2.RandomGrayscale(p=1),
            v2.RandomAutocontrast(p=1),
            v2.RandomRotation(degrees=45, interpolation=v2.InterpolationMode.BILINEAR),
        ]
        aug_idx = np.random.choice(len(aug_list), 3, replace=False)
        return v2.Compose(
            [
                v2.Resize(int(self.image_resize)),
                aug_list[aug_idx[0]],
                aug_list[aug_idx[1]],
                aug_list[aug_idx[2]],
                v2.CenterCrop(self.image_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert("RGB")
        label = self.labels[index]

        if self.split == "train":
            img = (
                self._rand_transforms(index)(img)
                if self.rand_aug
                else self.transform(img)
            )
            mask = torch.zeros((1, self.image_size, self.image_size))

        if self.split == "test":
            img = self.transform(img)
            mask_path = self.mask_paths[index]
            if mask_path is not None:
                mask = Image.open(mask_path).convert("L")
                mask = self.mask_transform(mask)
            else:
                mask = torch.zeros((1, self.image_size, self.image_size))

        return img, label, mask


class AnomalyDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir,
        batch_size,
        class_name,
        image_size,
        image_resize,
        rand_aug,
    ):
        super().__init__()
        self.root_dir = os.path.join(root_dir)
        self.batch_size = batch_size
        self.class_name = class_name
        self.image_size = image_size
        self.image_resize = image_resize
        self.rand_aug = rand_aug
        self.train_samples_per_epoch = 1000

        self.train_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage=None):
        print(f"Setup called with stage={stage}")

        if stage == "fit" or stage is None:
            self.train_dataset = AnomalyDataset(
                root_dir=self.root_dir,
                class_name=self.class_name,
                image_size=self.image_size,
                image_resize=self.image_resize,
                rand_aug=self.rand_aug,
                split="train",
            )

        if stage == "test" or stage is None:
            self.test_dataset = AnomalyDataset(
                root_dir=self.root_dir,
                class_name=self.class_name,
                image_size=self.image_size,
                image_resize=self.image_resize,
                rand_aug=self.rand_aug,
                split="test",
            )

        if stage == "predict" or stage is None:
            self.predict_dataset = AnomalyDataset(
                root_dir=self.root_dir,
                class_name=self.class_name,
                image_size=self.image_size,
                image_resize=self.image_resize,
                rand_aug=False,
                split="test",
            )

    def train_dataloader(self):
        sampler = RandomSampler(
            self.train_dataset,
            replacement=True,
            num_samples=self.train_samples_per_epoch,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=11,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=11,
            pin_memory=True,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=11,
            pin_memory=True,
            persistent_workers=True,
        )
