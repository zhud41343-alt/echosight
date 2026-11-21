"""
Paper Evaluation & Publication Figure Generator:
Compares Full EchoDepthNet (with FG-EDFM) vs. Ablation Baseline (--ablation-no-edge),
benchmarks inference FPS, writes outputs/ablation_report.md, and renders
outputs/paper_figure_preview.png (RGB | GT Depth | Pred Depth | 16x48 Acoustic Matrix).
"""

import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from models.echo_depth_net import EchoDepthNet, FastDepthLite, count_parameters
from train import SyntheticObstacleDepthDataset, compute_depth_metrics


def benchmark_model(ckpt_path: Path, model_kind: str, device: torch.device):
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"ERROR: Required trained checkpoint '{ckpt_path}' not found. "
            "Refusing to silently evaluate untrained random weights."
        )
    if model_kind == "fastdepth":
        model = FastDepthLite(pretrained_backbone=False).to(device)
    else:
        model = EchoDepthNet(pretrained_backbone=False, use_edge_fusion=(model_kind == "full")).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model_state" not in ckpt:
        raise ValueError(f"ERROR: Checkpoint '{ckpt_path}' is missing 'model_state'.")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    ds = SyntheticObstacleDepthDataset(num_samples=64, img_size=256, seed=999)
    abs_rel_sum, rmse_sum, d1_sum = 0.0, 0.0, 0.0

    # Warmup
    dummy = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    n_bench = 50
    with torch.no_grad():
        for _ in range(n_bench):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fps = n_bench / max(1e-6, time.time() - t0)

    sample_vis = None
    acoustic_rmse_sum = 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            rgb, gt_d, gt_a = ds[i]
            pred_d, pred_a = model(rgb.unsqueeze(0).to(device))
            ar, rm, d1 = compute_depth_metrics(pred_d.cpu(), gt_d.unsqueeze(0))
            a_rmse = torch.sqrt(torch.mean((pred_a.cpu() - gt_a.unsqueeze(0)) ** 2)).item()
            abs_rel_sum += ar
            rmse_sum += rm
            d1_sum += d1
            acoustic_rmse_sum += a_rmse
            if i == 0:
                sample_vis = (rgb.numpy(), gt_d.squeeze(0).numpy(), pred_d.squeeze().cpu().numpy(), pred_a.squeeze().cpu().numpy())

    n = len(ds)
    return {
        "params_m": count_parameters(model) / 1e6,
        "abs_rel": abs_rel_sum / n,
        "rmse": rmse_sum / n,
        "delta1": d1_sum / n,
        "acoustic_rmse": acoustic_rmse_sum / n,
        "fps": fps,
        "sample_vis": sample_vis,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    fast_res = benchmark_model(out_dir / "fastdepth_lite.pt", model_kind="fastdepth", device=device)
    base_res = benchmark_model(out_dir / "echodepth_no_edge.pt", model_kind="no_edge", device=device)
    full_res = benchmark_model(out_dir / "echodepth_best.pt", model_kind="full", device=device)

    report = f"""# EchoDepthNet 三模型对比与消融实验自动评测报告（Pretrained + Residual FG-EDFM）

运行设备：`{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}`

| 模型版本 | 参数量 (M) ↓ | AbsRel ↓ | RMSE ↓ | δ₁ (<1.25) ↑ | Acoustic RMSE ↓ | 推理速度 (FPS) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FastDepth-Lite (UNet-MobileNetV3) | {fast_res['params_m']:.2f} | {fast_res['abs_rel']:.4f} | {fast_res['rmse']:.4f} | {fast_res['delta1']:.4f} | {fast_res['acoustic_rmse']:.4f} | {fast_res['fps']:.1f} |
| EchoDepthNet Baseline (FPN w/o FG-EDFM) | {base_res['params_m']:.2f} | {base_res['abs_rel']:.4f} | {base_res['rmse']:.4f} | {base_res['delta1']:.4f} | {base_res['acoustic_rmse']:.4f} | {base_res['fps']:.1f} |
| **EchoDepthNet (Ours, +Residual FG-EDFM)** | **{full_res['params_m']:.2f}** | **{full_res['abs_rel']:.4f}** | **{full_res['rmse']:.4f}** | **{full_res['delta1']:.4f}** | **{full_res['acoustic_rmse']:.4f}** | **{full_res['fps']:.1f}** |
"""
    (out_dir / "ablation_report.md").write_text(report, encoding="utf-8")
    print(report)

    # Render publication-quality 4-panel figure
    rgb, gt_d, pred_d, pred_a = full_res["sample_vis"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), dpi=200)
    axes[0].imshow(rgb.transpose(1, 2, 0))
    axes[0].set_title("Input RGB")
    axes[1].imshow(gt_d, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Ground-Truth Proximity")
    axes[2].imshow(pred_d, cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title("EchoDepthNet Pred")
    axes[3].imshow(pred_a, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axes[3].set_title("16x48 Acoustic Hazard Matrix")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "paper_figure_preview.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved publication figure to {out_dir / 'paper_figure_preview.png'}")


if __name__ == "__main__":
    main()
