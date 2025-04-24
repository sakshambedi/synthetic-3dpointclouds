import os

import numpy as np
import torch
from torch.utils.data import Dataset


class Wheat3dPointCloud(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        split_ratio=(0.8, 0.1, 0.1),  # train, val, test ratios
        n_points=2048,
        transform=None,
        colors: str = False,
    ):
        self.root_dir = root_dir

        all_files = [f for f in os.listdir(self.root_dir) if f.endswith(".npy")]
        if not all_files:
            raise ValueError(f"No .npy files found in {root_dir}.")
        
        total = len(all_files)
        train_end = int(split_ratio[0] * total)
        val_end = train_end + int(split_ratio[1] * total)
        
        # Assign files based on split
        if split == "train":
            self.files = all_files[:train_end]
        elif split == "val":
            self.files = all_files[train_end:val_end]
        elif split == "test":
            self.files = all_files[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'.")
        
        self.n_points = n_points
        self.transform = transform
        self.colors = colors


    def __len__(self):
        return len(self.files)

    def _normalize_old(self, pc):
        # Differnce from the refrence :
        # normalization centers the point cloud just the same
        # scaling ensures a unit sphere but is different from the reference’s standard deviation-based approach.
        # shift also includes RGB padding, which the reference doesn’t do.

        geom = pc[:, :3]  # x, y, z

        if self.colors:
            color = pc[:, 3:]  # r, g, b
            color_normalized = color / 255.0  # normalize colors

        # For Computing the centroid of the geometry and center the points
        center = geom.mean(axis=0)
        geom_centered = geom - center

        max_distance = np.linalg.norm(geom_centered, axis=1).max()

        if max_distance == 0:
            max_distance = 1.0

        geom_normalized = geom_centered / max_distance  # Normalize geometry

        if self.colors:
            pc_normalized = np.concatenate([geom_normalized, color_normalized], axis=1)
        else:
            pc_normalized = geom_normalized

        scale = geom.flatten().std().reshape(1, 1)

        shift = np.concatenate([center, np.ones(3)])
        return pc_normalized, shift, scale

    def _normalize(self, pc):
        geom = pc[:, :3]  # x, y, z

        if self.colors:
            color = pc[:, 3:]  # r, g, b
            color = color / 255.0  # normalize colors

        shift = geom.mean(axis=0).reshape(1, 3)
        scale = geom.flatten().std().reshape(1, 1)

        pc = (geom - shift) / scale
        return pc, shift, scale

    def _resample(self, pc):
        n_actual = pc.shape[0]
        if n_actual > self.n_points:
            idx = np.random.choice(n_actual, self.n_points, replace=False)
            pc = pc[idx]
        elif n_actual < self.n_points:
            idx = np.random.choice(n_actual, self.n_points - n_actual, replace=True)
            pc = np.concatenate([pc, pc[idx]], axis=0)
        return pc

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        pc = np.load(file_path)        
        pc = self._resample(pc)
        pc, shift, scale = self._normalize(pc)

        pc_tensor = torch.tensor(pc, dtype=torch.float32)
        shift_tensor = torch.tensor(shift, dtype=torch.float32)
        scale_tensor = torch.tensor(scale, dtype=torch.float32)

        if self.transform:
            pc_tensor = self.transform(pc_tensor)

        return {"pointcloud": pc_tensor, "shift": shift_tensor, "scale": scale_tensor}
