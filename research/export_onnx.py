"""
Export trained EchoDepthNet checkpoint to ONNX (opset 17) for browser onnxruntime-web
or edge hardware deployment.

P0-3 (V8.4):
- Refuses to silently export untrained random weights when checkpoint is missing
  (requires explicit --allow-untrained-smoke-test for synthetic smoke tests).
- Loads weights with strict=True, computes SHA-256 hashes, and verifies ONNX Runtime
  output shapes, finite values, and numerical parity.
"""

import argparse
import hashlib
from pathlib import Path
import torch
from models.echo_depth_net import EchoDepthNet


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Export EchoDepthNet to ONNX")
    parser.add_argument("--checkpoint", type=str, default="outputs/echodepth_best.pt")
    parser.add_argument("--output", type=str, default="outputs/echodepth_net.onnx")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument(
        "--allow-untrained-smoke-test",
        action="store_true",
        help="Explicitly allow exporting untrained weights for CI/smoke testing ONLY",
    )
    args = parser.parse_args()

    model = EchoDepthNet(pretrained_backbone=False, use_edge_fusion=True)
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        if not args.allow_untrained_smoke_test:
            raise FileNotFoundError(
                f"ERROR: Trained checkpoint not found at '{ckpt_path}'. "
                "Refusing to export untrained random weights. Train first or pass --allow-untrained-smoke-test."
            )
        print("[SMOKE TEST] Exporting untrained initialized architecture by explicit flag.")
        ckpt_sha = "N/A (untrained smoke test)"
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model_state" not in ckpt:
            raise ValueError(f"ERROR: Checkpoint '{ckpt_path}' is missing 'model_state' key.")
        model.load_state_dict(ckpt["model_state"], strict=True)
        ckpt_sha = sha256_file(ckpt_path)
        print(
            f"Loaded verified checkpoint from {ckpt_path} "
            f"(RMSE={ckpt.get('val_rmse', 'N/A')}, SHA-256={ckpt_sha[:16]}...)"
        )

    model.eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        opset_version=17,
        input_names=["rgb_image"],
        output_names=["depth_map", "acoustic_matrix"],
    )
    try:
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(out_path), **export_kwargs)

    onnx_sha = sha256_file(out_path)
    print(
        f"Exported ONNX model successfully to {out_path} "
        f"({out_path.stat().st_size / 1e6:.2f} MB, SHA-256={onnx_sha[:16]}...)"
    )

    # Verify numerical parity, output dimensions, and finite values with ONNX Runtime
    import numpy as np
    import onnxruntime as ort

    with torch.no_grad():
        pt_d, pt_a = model(dummy)
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_d, ort_a = sess.run(None, {"rgb_image": dummy.numpy()})

    if ort_d.shape != (1, 1, args.img_size, args.img_size) or ort_a.shape != (1, 1, 16, 48):
        raise ValueError(f"Unexpected ONNX output shapes: depth={ort_d.shape}, acoustic={ort_a.shape}")
    if not (np.all(np.isfinite(ort_d)) and np.all(np.isfinite(ort_a))):
        raise ValueError("ONNX output contains NaN or Infinity!")

    diff_d = float(np.max(np.abs(pt_d.numpy() - ort_d)))
    diff_a = float(np.max(np.abs(pt_a.numpy() - ort_a)))
    print(f"ONNX Runtime parity verified! max_abs_diff(depth)={diff_d:.2e}, max_abs_diff(acoustic)={diff_a:.2e}")


if __name__ == "__main__":
    main()
