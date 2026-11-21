"""
EchoDepthNet Training & Evaluation Pipeline (RTX 4060 FP16 Ready)
Supports:
  1) Real paired RGB-Depth datasets (NYU Depth V2 / SUN RGB-D): --data-dir path/to/dataset
  2) Built-in Synthetic 3D Obstacle Benchmark for instant verification: --synthetic
  3) Ablation Study toggle: --ablation-no-edge (disables FG-EDFM to compare against Baseline)
"""

import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from losses.distillation_loss import EchoSightCompositeLoss
from models.echo_depth_net import EchoDepthNet, count_parameters


def derive_acoustic_target(depth_tensor: torch.Tensor) -> torch.Tensor:
    """
    Generates ground-truth 16x48 acoustic hazard matrix from proximity depth map [1, H, W]:
    Combines near-field proximity weighting + boundary edge emphasis.
    """
    d = depth_tensor.unsqueeze(0)  # [1, 1, H, W]
    gx = torch.abs(d[:, :, :, 1:] - d[:, :, :, :-1])
    gy = torch.abs(d[:, :, 1:, :] - d[:, :, :-1, :])
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    edge = torch.clamp((gx + gy) * 3.0, 0.0, 1.0)
    hazard = torch.clamp(0.65 * (d ** 1.35) + 0.35 * edge, 0.0, 1.0)
    pooled = F.adaptive_avg_pool2d(hazard, (16, 48))
    return pooled.squeeze(0)  # [1, 16, 48]


class SyntheticObstacleDepthDataset(Dataset):
    """
    Procedural 3D indoor corridor & obstacle RGB-Depth dataset for rapid architecture verification
    and ablation benchmarking without downloading multi-GB archives first.
    """

    def __init__(self, num_samples: int = 320, img_size: int = 256, seed: int = 42):
        self.num_samples = num_samples
        self.img_size = img_size
        self.rng = np.random.default_rng(seed)
        self.seeds = self.rng.integers(0, 10_000_000, size=num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = np.random.default_rng(int(self.seeds[idx]))
        s = self.img_size

        # Base corridor perspective depth gradient (floor closer, horizon farther)
        y_grid = np.linspace(0.08, 0.45, s, dtype=np.float32)[:, None]
        depth = np.repeat(y_grid, s, axis=1)

        # Background RGB texture
        bg_color = r.uniform(0.08, 0.25, size=(3, 1, 1)).astype(np.float32)
        rgb = np.ones((3, s, s), dtype=np.float32) * bg_color
        rgb += np.linspace(0.0, 0.12, s, dtype=np.float32)[None, :, None]

        # Place 2~5 3D rectangular/circular obstacles at varying depths
        num_obs = int(r.integers(2, 6))
        for _ in range(num_obs):
            prox = float(r.uniform(0.45, 0.96))  # proximity (higher = closer)
            bw = int(r.integers(s // 10, s // 3))
            bh = int(r.integers(s // 8, s // 2))
            x0 = int(r.integers(0, max(1, s - bw)))
            y0 = int(r.integers(s // 6, max(s // 6 + 1, s - bh)))

            color = r.uniform(0.35, 0.95, size=(3, 1, 1)).astype(np.float32) * (0.5 + 0.5 * prox)
            rgb[:, y0 : y0 + bh, x0 : x0 + bw] = color
            depth[y0 : y0 + bh, x0 : x0 + bw] = np.maximum(depth[y0 : y0 + bh, x0 : x0 + bw], prox)

        # Mild sensor noise
        rgb = np.clip(rgb + r.normal(0.0, 0.02, size=rgb.shape).astype(np.float32), 0.0, 1.0)
        depth_t = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        rgb_t = torch.from_numpy(rgb)                   # [3, H, W]
        acoustic_t = derive_acoustic_target(depth_t)    # [1, 16, 48]
        return rgb_t, depth_t, acoustic_t


class PairedRGBDepthDataset(Dataset):
    """
    Standard folder dataset for NYU Depth V2 / SUN RGB-D:
      data_dir/
        rgb/   (*.png or *.jpg)
        depth/ (matching filename, 16-bit or 8-bit normalized depth PNG)
    """

    def __init__(self, data_dir: str, img_size: int = 256):
        self.rgb_dir = Path(data_dir) / "rgb"
        self.depth_dir = Path(data_dir) / "depth"
        self.img_size = img_size
        self.files = sorted([p.name for p in self.rgb_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not self.files:
            raise FileNotFoundError(f"No RGB images found in {self.rgb_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        rgb_img = Image.open(self.rgb_dir / fname).convert("RGB").resize((self.img_size, self.img_size))
        depth_path = self.depth_dir / fname
        if not depth_path.exists():
            depth_path = self.depth_dir / (Path(fname).stem + ".png")
        depth_img = Image.open(depth_path).resize((self.img_size, self.img_size))

        rgb_arr = np.asarray(rgb_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        d_arr = np.asarray(depth_img, dtype=np.float32)
        if d_arr.ndim == 3:
            d_arr = d_arr[..., 0]
        d_max = max(float(d_arr.max()), 1e-6)
        d_norm = np.clip(d_arr / d_max, 0.02, 1.0)

        rgb_t = torch.from_numpy(rgb_arr)
        depth_t = torch.from_numpy(d_norm).unsqueeze(0)
        acoustic_t = derive_acoustic_target(depth_t)
        return rgb_t, depth_t, acoustic_t


@torch.no_grad()
def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor):
    p = pred.clamp(min=1e-3)
    t = target.clamp(min=1e-3)
    abs_rel = torch.mean(torch.abs(p - t) / t).item()
    rmse = torch.sqrt(torch.mean((p - t) ** 2)).item()
    thresh = torch.maximum(p / t, t / p)
    delta1 = torch.mean((thresh < 1.25).float()).item()
    return abs_rel, rmse, delta1


def main():
    parser = argparse.ArgumentParser(description="Train EchoDepthNet on RTX 4060 / GPU")
    parser.add_argument("--data-dir", type=str, default="", help="Path to paired rgb/ and depth/ folder")
    parser.add_argument("--nyu-hf", action="store_true", help="Auto-download/load NYU Depth V2 via HuggingFace datasets")
    parser.add_argument("--nyu-mat", type=str, default="", help="Path to official nyu_depth_v2_labeled.mat (HDF5)")
    parser.add_argument("--synthetic", action="store_true", help="Use built-in synthetic 3D obstacle dataset")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed for fair ablation")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained MobileNetV3 weights")
    parser.add_argument("--ablation-no-edge", action="store_true", help="Disable FG-EDFM for ablation study")
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EchoDepthNet] Device: {device} | Seed: {args.seed} | Edge Fusion (Residual FG-EDFM): {not args.ablation_no_edge}")

    if args.nyu_mat:
        from datasets.nyu_depth_v2 import NYUDepthV2MatDataset
        train_ds = NYUDepthV2MatDataset(args.nyu_mat, split="train", img_size=args.img_size)
        val_ds = NYUDepthV2MatDataset(args.nyu_mat, split="val", img_size=args.img_size)
    elif args.nyu_hf:
        from datasets.nyu_depth_v2 import NYUDepthV2HFDataset
        train_ds = NYUDepthV2HFDataset(split="train", img_size=args.img_size)
        val_ds = NYUDepthV2HFDataset(split="validation", img_size=args.img_size)
    elif args.data_dir and not args.synthetic:
        full_ds = PairedRGBDepthDataset(args.data_dir, img_size=args.img_size)
        n_val = max(1, int(0.15 * len(full_ds)))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
        )
    else:
        print("[EchoDepthNet] Using built-in Synthetic 3D Obstacle RGB-Depth Benchmark...")
        train_ds = SyntheticObstacleDepthDataset(num_samples=320, img_size=args.img_size, seed=args.seed)
        val_ds = SyntheticObstacleDepthDataset(num_samples=64, img_size=args.img_size, seed=999)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = EchoDepthNet(
        pretrained_backbone=args.pretrained,
        use_edge_fusion=not args.ablation_no_edge,
    ).to(device)
    print(f"[EchoDepthNet] Trainable Parameters: {count_parameters(model) / 1e6:.2f} M")

    criterion = EchoSightCompositeLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_rmse = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for rgb, depth, acoustic in train_loader:
            rgb, depth, acoustic = rgb.to(device), depth.to(device), acoustic.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred_d, pred_a = model(rgb)
                loss, _ = criterion(pred_d, depth, pred_a, acoustic)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * rgb.size(0)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_abs_rel, val_rmse, val_d1 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for rgb, depth, _ in val_loader:
                rgb, depth = rgb.to(device), depth.to(device)
                pred_d, _ = model(rgb)
                ar, rm, d1 = compute_depth_metrics(pred_d, depth)
                val_abs_rel += ar * rgb.size(0)
                val_rmse += rm * rgb.size(0)
                val_d1 += d1 * rgb.size(0)

        val_abs_rel /= len(val_ds)
        val_rmse /= len(val_ds)
        val_d1 /= len(val_ds)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({dt:.1f}s) | "
            f"Loss: {train_loss:.4f} | AbsRel: {val_abs_rel:.4f} | RMSE: {val_rmse:.4f} | d1: {val_d1:.4f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            ckpt_name = "echodepth_no_edge.pt" if args.ablation_no_edge else "echodepth_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "use_edge_fusion": not args.ablation_no_edge,
                    "val_rmse": val_rmse,
                    "val_abs_rel": val_abs_rel,
                    "val_delta1": val_d1,
                },
                out_dir / ckpt_name,
            )

    print(f"[EchoDepthNet] Training complete! Best Val RMSE: {best_rmse:.4f} | Saved in {out_dir}")


if __name__ == "__main__":
    main()
