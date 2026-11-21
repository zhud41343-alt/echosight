#!/usr/bin/env bash
set -e

# Usage:
#   bash run_cloud_experiments.sh                # Default: NYU Depth V2 via HuggingFace (pretrained MobileNetV3)
#   MODE=nyu_mat MAT_PATH=/data/nyu_depth_v2_labeled.mat bash run_cloud_experiments.sh
#   MODE=synthetic bash run_cloud_experiments.sh

MODE="${MODE:-nyu_hf}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
IMG_SIZE="${IMG_SIZE:-256}"

echo "=================================================="
echo " EchoDepthNet V2 Cloud GPU One-Click Suite        "
echo " Mode: ${MODE} | Epochs: ${EPOCHS} | Batch: ${BATCH_SIZE}"
echo "=================================================="
nvidia-smi || true

echo "[1/4] Installing dependencies..."
pip install -q onnx onnxruntime matplotlib tqdm pillow h5py datasets

DATA_FLAG="--nyu-hf --pretrained"
if [ "$MODE" = "nyu_mat" ]; then
  DATA_FLAG="--nyu-mat ${MAT_PATH:?Please set MAT_PATH} --pretrained"
elif [ "$MODE" = "synthetic" ]; then
  DATA_FLAG="--synthetic"
fi

echo "[2/4] Training Full EchoDepthNet (+Residual FG-EDFM)..."
python train.py ${DATA_FLAG} --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --img-size "${IMG_SIZE}" --seed 42 --out-dir outputs

echo "[3/4] Training Ablation Baseline (w/o FG-EDFM)..."
python train.py ${DATA_FLAG} --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --img-size "${IMG_SIZE}" --seed 42 --ablation-no-edge --out-dir outputs

echo "[4/4] Generating Paper Ablation Table, 4-Panel Figure & ONNX Edge Model..."
python evaluate_ablation.py
python export_onnx.py --checkpoint outputs/echodepth_best.pt --output outputs/echodepth_net.onnx

echo "=================================================="
echo " ALL DONE! Check outputs/ for:"
echo "  - outputs/ablation_report.md       (论文消融实验表)"
echo "  - outputs/paper_figure_preview.png (论文4联高清可视化图)"
echo "  - outputs/echodepth_net.onnx       (端侧 ONNX/WebGPU 部署模型)"
echo "=================================================="
