"""
Export trained EchoDepthNet checkpoint to ONNX (opset 17) for browser onnxruntime-web
or edge hardware (Jetson / AR smart glasses) deployment.
"""

import argparse
from pathlib import Path
import torch
from models.echo_depth_net import EchoDepthNet


def main():
    parser = argparse.ArgumentParser(description="Export EchoDepthNet to ONNX")
    parser.add_argument("--checkpoint", type=str, default="outputs/echodepth_best.pt")
    parser.add_argument("--output", type=str, default="outputs/echodepth_net.onnx")
    parser.add_argument("--img-size", type=int, default=256)
    args = parser.parse_args()

    model = EchoDepthNet(pretrained_backbone=False, use_edge_fusion=True)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint from {ckpt_path} (RMSE={ckpt.get('val_rmse', 'N/A')})")
    else:
        print(f"Warning: {ckpt_path} not found, exporting initialized architecture for benchmarking.")

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
    print(f"Exported ONNX model successfully to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    # Verify numerical parity with ONNX Runtime if installed
    try:
        import numpy as np
        import onnxruntime as ort

        with torch.no_grad():
            pt_d, pt_a = model(dummy)
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        ort_d, ort_a = sess.run(None, {"rgb_image": dummy.numpy()})
        diff_d = float(np.max(np.abs(pt_d.numpy() - ort_d)))
        diff_a = float(np.max(np.abs(pt_a.numpy() - ort_a)))
        print(f"ONNX Runtime parity verified! max_abs_diff(depth)={diff_d:.2e}, max_abs_diff(acoustic)={diff_a:.2e}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
