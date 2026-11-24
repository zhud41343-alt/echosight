"""
V8.4–V8.8 Integrity & Provenance Tests for EchoDepthNet Pipeline.
Verifies:
1. export_onnx.py fails fast with FileNotFoundError when checkpoint is missing.
2. evaluate_ablation.py fails fast with FileNotFoundError when checkpoints are missing.
3. Published public/models/echodepth_net.onnx loads in ONNX Runtime with exact output shapes
   (1, 1, 256, 256) and (1, 1, 16, 48) and 100% finite values in [0, 1].
4. Cryptographic SHA-256 of public/models/echodepth_net.onnx matches MODEL_PROVENANCE.md.
5. NYU Depth V2 splits.mat validator rejects empty, non-integer, duplicate, overlapping,
   and out-of-bounds indices while accepting valid disjoint 1-based splits.
"""

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

RESEARCH_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = RESEARCH_DIR.parent

# Load split_validator.py directly without triggering datasets/__init__.py (which imports torch)
_spec = importlib.util.spec_from_file_location(
    "split_validator", RESEARCH_DIR / "datasets" / "split_validator.py"
)
assert _spec is not None and _spec.loader is not None
_split_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_split_mod)
validate_nyu_split_indices = _split_mod.validate_nyu_split_indices


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


def test_published_onnx_sha256_matches_provenance():
    onnx_path = REPO_ROOT / "public" / "models" / "echodepth_net.onnx"
    prov_path = RESEARCH_DIR / "outputs" / "MODEL_PROVENANCE.md"
    assert onnx_path.exists() and prov_path.exists()

    actual_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest().upper()
    prov_text = prov_path.read_text(encoding="utf-8")
    assert actual_sha in prov_text, (
        f"ONNX SHA-256 mismatch! Actual={actual_sha} not found in {prov_path}"
    )


def test_nyu_splits_validation():
    # Valid 1-based MATLAB indices -> 0-based Python indices
    tr, te = validate_nyu_split_indices(np.array([1, 2, 3]), np.array([4, 5]), total=5)
    assert tr == [0, 1, 2] and te == [3, 4]

    # Reject empty split
    try:
        validate_nyu_split_indices(np.array([]), np.array([1, 2]), total=5)
        raise AssertionError("Should reject empty split")
    except ValueError:
        pass

    # Reject non-integer split
    try:
        validate_nyu_split_indices(np.array([1.5, 2.0]), np.array([3, 4]), total=5)
        raise AssertionError("Should reject non-integer split")
    except ValueError:
        pass

    # Reject duplicate within split
    try:
        validate_nyu_split_indices(np.array([1, 1, 2]), np.array([3, 4]), total=5)
        raise AssertionError("Should reject duplicate indices")
    except ValueError:
        pass

    # Reject overlapping train/test split
    try:
        validate_nyu_split_indices(np.array([1, 2, 3]), np.array([3, 4]), total=5)
        raise AssertionError("Should reject overlapping train/test")
    except ValueError:
        pass

    # Reject out-of-bounds index (0 or > total in 1-based MATLAB)
    try:
        validate_nyu_split_indices(np.array([1, 2]), np.array([3, 6]), total=5)
        raise AssertionError("Should reject out-of-bounds index")
    except IndexError:
        pass


if __name__ == "__main__":
    import tempfile
    test_export_fails_on_missing_checkpoint()
    with tempfile.TemporaryDirectory() as td:
        test_evaluate_fails_on_missing_checkpoint(Path(td))
    test_published_onnx_shapes_and_finite_outputs()
    test_published_onnx_sha256_matches_provenance()
    test_nyu_splits_validation()
    print("ALL 5 RESEARCH PIPELINE INTEGRITY TESTS PASSED!")
