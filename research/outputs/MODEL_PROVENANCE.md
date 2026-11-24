# EchoDepthNet 模型来源与可追溯清单（MODEL_PROVENANCE）

- **模型名称**：`EchoDepthNet`（MobileNetV3-Small + 可微空间 Sobel 高通边缘先验 + **Residual EG-DFM** 残差边缘引导深度融合模块）
- **可训练参数量**：`1.11 M`
- **训练数据集与划分说明**：
  - 内置程序化 3D 走廊障碍物合成基准（`SyntheticObstacleDepthDataset`，`320 train (seed=42) / 64 val (seed=999)`，`256×256`）。
  - **重要说明**：当前消融表为 `64` 场景**验证集结果**（非独立留出测试集，亦非真实 NYU Depth V2 公开数据集）；NYU Depth V2 训练管道（支持官方 `splits.mat`）已就绪，真实数据集指标待云端 GPU 运行后补充。
- **完整可复现训练命令**（注意：通过命令行显式覆盖默认学习率 `1.5e-3` 为 `2.5e-3`）：
  ```bash
  python train.py --synthetic --pretrained --model fastdepth --epochs 15 --lr 2.5e-3 --seed 42
  python train.py --synthetic --pretrained --model echodepth --ablation-no-edge --epochs 15 --lr 2.5e-3 --seed 42
  python train.py --synthetic --pretrained --model echodepth --epochs 15 --lr 2.5e-3 --seed 42
  python evaluate_ablation.py
  python export_onnx.py --checkpoint outputs/echodepth_best.pt --output ..\public\models\echodepth_net.onnx
  ```
- **训练环境与依赖**：Windows 11 CPU，PyTorch `2.12.0+cpu`，AdamW (`lr=2.5e-3`, `weight_decay=1e-4`)，CosineAnnealingLR，复合损失 `1.8 L1 + 0.45 SILog + 0.6 Grad + 1.0 Acoustic`，最优模型按最低验证 RMSE 保存。
- **验证集实测指标**：`AbsRel = 0.0754` ｜ `RMSE = 0.0824` ｜ `δ₁ (<1.25) = 0.8970` ｜ `Acoustic RMSE = 0.0577` ｜ `CPU 74.2 FPS`
- **训练权重 Checkpoint SHA-256**（本地 `research/outputs/echodepth_best.pt`，按 `.gitignore` 不随 Git 仓库提交，可由上述命令复现）：
  `E2A21BEBB620ABF292EB5605E5C6427C9AF96838568266FFD43E519B46341844`
- **公开部署 ONNX 模型 SHA-256**（`public/models/echodepth_net.onnx`，`4.46 MB`）：
  `95EBF7FAEC842F51D424E7D6730B373D739112025F893BC1121B7279F7B62009`
- **PyTorch vs ONNX Runtime 数值对齐误差**：
  `max_abs_diff(depth) = 4.77e-07`，`max_abs_diff(acoustic) = 7.75e-07`
- **浏览器推理耗时说明**（单次前向推理，非端到端总延迟）：
  - 本地 Vite Dev Server（Windows 11 Chromium WASM，256×256）：实测 `51 ms`
  - 线上 GitHub Pages 生产构建（Windows 11 Chromium WASM，256×256）：实测 `44 ms`
- **管道完整性与哈希校验测试**：`python research/tests/test_pipeline_integrity.py` → **5/5 通过**（含 ONNX SHA-256 密码学一致性与 NYU `splits.mat` 校验）。
