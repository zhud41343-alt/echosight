"""
P0-3 (V8.4) Integrity Tests for EchoDepthNet Training, Evaluation & ONNX Export Pipeline.
Verifies:
1. export_onnx.py fails with non-zero exit code when checkpoint is missing (no silent random export).
2. evaluate_ablation.py fails with non-zero exit code when checkpoints are missing.
3. Published public/models/echodepth_net.onnx loads in ONNX Runtime with exact output shapes
   (1, 1, 256, 256) and (1, 1, 16, 48) and 100% finite values in [0, 1].
"""

import subprocess
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

RESEARCH_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = RESEARCH_DIR.parent


def test_export_fails_on_missing_checkpoint():
    res = subprocess.run(
        [
            sys.executable,
            str(RESEARCH_DIR / "export_onnx.py"),
            "--checkpoint",
            "outputs/does_not_exist_12345.pt",
            "--output",
            "outputs/should_not_exist.onnx",
        ],
        cwd=str(RESEARCH_DIR),
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0, f"export_onnx.py MUST fail when checkpoint is missing! stdout={res.stdout}"
    assert "FileNotFoundError" in res.stderr and "Refusing to export untrained" in res.stderr, (
        f"Expected FileNotFoundError on missing checkpoint, got stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    )


def test_evaluate_fails_on_missing_checkpoint(tmp_path: Path):
    res = subprocess.run(
        [sys.executable, str(RESEARCH_DIR / "evaluate_ablation.py")],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0, f"evaluate_ablation.py MUST fail when checkpoints are missing! stdout={res.stdout}"
    assert "FileNotFoundError" in res.stderr and "Refusing to silently evaluate" in res.stderr, (
        f"Expected FileNotFoundError on missing checkpoints, got stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    )


def test_published_onnx_shapes_and_finite_outputs():
    onnx_path = REPO_ROOT / "public" / "models" / "echodepth_net.onnx"
    assert onnx_path.exists(), f"Published ONNX model missing at {onnx_path}"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    dummy = rng.random((1, 3, 256, 256), dtype=np.float32)
    depth, acoustic = sess.run(None, {"rgb_image": dummy})

    assert depth.shape == (1, 1, 256, 256), f"Bad depth shape: {depth.shape}"
    assert acoustic.shape == (1, 1, 16, 48), f"Bad acoustic shape: {acoustic.shape}"
    assert np.all(np.isfinite(depth)) and np.all(np.isfinite(acoustic))
    assert float(depth.min()) >= 0.0 and float(depth.max()) <= 1.0
    assert float(acoustic.min()) >= 0.0 and float(acoustic.max()) <= 1.0


if __name__ == "__main__":
    import tempfile
    test_export_fails_on_missing_checkpoint()
    with tempfile.TemporaryDirectory() as td:
        test_evaluate_fails_on_missing_checkpoint(Path(td))
    test_published_onnx_shapes_and_finite_outputs()
    print("ALL 3 RESEARCH PIPELINE INTEGRITY TESTS PASSED!")
