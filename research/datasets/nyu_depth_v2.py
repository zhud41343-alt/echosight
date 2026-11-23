"""
NYU Depth V2 Automatic Dataset Loaders for EchoDepthNet (SCI/CCF Submission Ready)
Supports two seamless loading paths on Cloud GPUs (AutoDL / Colab / Kaggle):
  1) NYUDepthV2HFDataset: Zero-setup automatic download via HuggingFace `datasets`
     (`sayakpaul/nyu_depth_v2`), with optional max_samples cap for fast prototyping.
  2) NYUDepthV2MatDataset: Official `nyu_depth_v2_labeled.mat` (HDF5, 1,449 indoor RGB-D pairs)
     using standard Eigen crop and deterministic train/val split (795 / 654).
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def metric_depth_to_proximity(depth_m: np.ndarray, min_m: float = 0.5, max_m: float = 10.0) -> np.ndarray:
    """
    Converts metric depth in meters (higher = farther) to normalized inverse-depth proximity in [0.02, 1.0]
    (higher = closer obstacle), matching EchoSight's near-field acoustic hazard convention.
    """
    d = np.clip(depth_m.astype(np.float32), min_m, max_m)
    inv = (1.0 / d - 1.0 / max_m) / (1.0 / min_m - 1.0 / max_m)
    return np.clip(inv, 0.02, 1.0).astype(np.float32)


def derive_acoustic_matrix(prox_t: torch.Tensor) -> torch.Tensor:
    """Derives ground-truth 16x48 acoustic hazard matrix from proximity tensor [1, H, W]."""
    d = prox_t.unsqueeze(0)
    gx = F.pad(torch.abs(d[:, :, :, 1:] - d[:, :, :, :-1]), (0, 1, 0, 0))
    gy = F.pad(torch.abs(d[:, :, 1:, :] - d[:, :, :-1, :]), (0, 0, 0, 1))
    edge = torch.clamp((gx + gy) * 3.0, 0.0, 1.0)
    hazard = torch.clamp(0.65 * (d ** 1.35) + 0.35 * edge, 0.0, 1.0)
    return F.adaptive_avg_pool2d(hazard, (16, 48)).squeeze(0)


class NYUDepthV2MatDataset(Dataset):
    """
    Loads official `nyu_depth_v2_labeled.mat` (HDF5 format).
    Applies standard Eigen crop [45:471, 41:601].
    If `splits_mat_path` (official Nathan Silberman `splits.mat` with `trainNdxs`/`testNdxs`)
    is provided, uses the exact official 795/654 split; otherwise falls back to a deterministic
    disjoint 795/654 partition (explicitly NOT the official scene split).
    """

    def __init__(
        self,
        mat_path: str,
        split: str = "train",
        img_size: int = 256,
        eigen_crop: bool = True,
        splits_mat_path: Optional[str] = None,
    ):
        import h5py

        self.mat_path = Path(mat_path)
        if not self.mat_path.exists():
            raise FileNotFoundError(f"NYU Depth V2 .mat file not found: {self.mat_path}")
        self.split = split
        self.img_size = img_size
        self.eigen_crop = eigen_crop

        with h5py.File(self.mat_path, "r") as f:
            total = f["images"].shape[0]

        if splits_mat_path:
            sp_path = Path(splits_mat_path)
            if not sp_path.exists():
                raise FileNotFoundError(
                    f"Explicitly requested official splits.mat not found: '{sp_path}'. "
                    "Refusing to silently fall back to non-official sequential split."
                )
            import scipy.io as sio
            sp = sio.loadmat(sp_path)
            train_idx = [int(i) - 1 for i in sp["trainNdxs"].reshape(-1)]
            test_idx = [int(i) - 1 for i in sp["testNdxs"].reshape(-1)]
            if len(train_idx) != len(set(train_idx)) or len(test_idx) != len(set(test_idx)):
                raise ValueError("Official splits.mat contains duplicate indices within a split!")
            if set(train_idx) & set(test_idx):
                raise ValueError("Official splits.mat contains overlapping train/test indices!")
            chosen = train_idx if split == "train" else test_idx
            if any(i < 0 or i >= total for i in chosen):
                raise IndexError(f"splits.mat index out of bounds for dataset size {total}")
            self.indices = chosen
            self.split_mode = "official_splits_mat"
        else:
            n_train = min(795, int(total * 0.55)) if total == 1449 else int(total * 0.85)
            self.indices = list(range(0, n_train)) if split == "train" else list(range(n_train, total))
            self.split_mode = "sequential_fallback_non_official"
        self._h5 = None

    def _ensure_open(self):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.mat_path, "r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_open()
        real_idx = self.indices[idx]
        # NYU .mat stores images as [N, 3, W, H] and depths as [N, W, H] -> transpose to [H, W, 3] / [H, W]
        rgb = np.transpose(self._h5["images"][real_idx], (2, 1, 0))  # uint8 [480, 640, 3]
        depth_m = np.transpose(self._h5["depths"][real_idx], (1, 0))  # float32 [480, 640] in meters

        if self.eigen_crop:
            rgb = rgb[45:471, 41:601, :]
            depth_m = depth_m[45:471, 41:601]

        rgb_img = Image.fromarray(rgb).resize((self.img_size, self.img_size), Image.BILINEAR)
        prox = metric_depth_to_proximity(depth_m)
        prox_img = Image.fromarray(prox).resize((self.img_size, self.img_size), Image.BILINEAR)

        rgb_arr = np.asarray(rgb_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        prox_arr = np.clip(np.asarray(prox_img, dtype=np.float32), 0.02, 1.0)

        # Random horizontal flip augmentation on train split
        if self.split == "train" and np.random.rand() < 0.5:
            rgb_arr = np.ascontiguousarray(rgb_arr[:, :, ::-1])
            prox_arr = np.ascontiguousarray(prox_arr[:, ::-1])

        rgb_t = torch.from_numpy(rgb_arr)
        prox_t = torch.from_numpy(prox_arr).unsqueeze(0)
        acoustic_t = derive_acoustic_matrix(prox_t)
        return rgb_t, prox_t, acoustic_t


class NYUDepthV2HFDataset(Dataset):
    """
    Zero-setup automatic loader using HuggingFace `datasets` (`sayakpaul/nyu_depth_v2`).
    """

    def __init__(
        self,
        split: str = "train",
        img_size: int = 256,
        max_samples: Optional[int] = None,
        repo_id: str = "sayakpaul/nyu_depth_v2",
    ):
        from datasets import load_dataset

        self.split = split
        self.img_size = img_size
        ds = load_dataset(repo_id, split=split)
        if max_samples is not None and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.ds[idx]
        rgb_pil = sample["image"].convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        depth_arr = np.asarray(sample["depth_map"], dtype=np.float32)
        prox = metric_depth_to_proximity(depth_arr)
        prox_pil = Image.fromarray(prox).resize((self.img_size, self.img_size), Image.BILINEAR)

        rgb_arr = np.asarray(rgb_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
        prox_arr = np.clip(np.asarray(prox_pil, dtype=np.float32), 0.02, 1.0)

        if self.split == "train" and np.random.rand() < 0.5:
            rgb_arr = np.ascontiguousarray(rgb_arr[:, :, ::-1])
            prox_arr = np.ascontiguousarray(prox_arr[:, ::-1])

        rgb_t = torch.from_numpy(rgb_arr)
        prox_t = torch.from_numpy(prox_arr).unsqueeze(0)
        acoustic_t = derive_acoustic_matrix(prox_t)
        return rgb_t, prox_t, acoustic_t
