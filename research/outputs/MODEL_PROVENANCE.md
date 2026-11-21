# EchoDepthNet 模型来源与可追溯清单（MODEL_PROVENANCE）

- **模型名称**：`EchoDepthNet`（MobileNetV3-Small + 可微空间 Sobel 先验 + **Residual EG-DFM** 残差边缘引导深度融合）
- **可训练参数量**：`1.11 M`
- **训练数据集**：内置程序化 3D 走廊障碍物合成基准（`SyntheticObstacleDepthDataset`，`320 train / 64 val`，`256×256`）— *注：非真实 NYU Depth V2 公开数据集，真实 RGB-D 评测待云端 GPU 执行*
- **预训练主干**：ImageNet 预训练 `MobileNet_V3_Small_Weights.DEFAULT`
- **训练随机种子 / 轮次 / 学习率 / 损失**：`seed=42` / `15 epochs` / `AdamW lr=2.5e-3` / `1.8 L1 + 0.45 SILog + 0.6 Grad + 1.0 Acoustic`
- **训练与评测设备**：Windows 11 CPU（PyTorch `2.12.0+cpu`）
- **验证集实测指标**：`AbsRel = 0.0754` ｜ `RMSE = 0.0824` ｜ `δ₁ (<1.25) = 0.8970` ｜ `Acoustic RMSE = 0.0577` ｜ `CPU 74.2 FPS`
- **训练权重 Checkpoint SHA-256**（本地 `research/outputs/echodepth_best.pt`）：
  `E2A21BEBB620ABF292EB5605E5C6427C9AF96838568266FFD43E519B46341844`
- **线上部署 ONNX 模型 SHA-256**（`public/models/echodepth_net.onnx`，`4.46 MB`）：
  `95EBF7FAEC842F51D424E7D6730B373D739112025F893BC1121B7279F7B62009`
- **导出命令**：
  `python export_onnx.py --checkpoint outputs/echodepth_best.pt --output ..\public\models\echodepth_net.onnx`
- **PyTorch vs ONNX Runtime 数值对齐误差**：
  `max_abs_diff(depth) = 4.77e-07`，`max_abs_diff(acoustic) = 7.75e-07`
- **线上生产浏览器实测（GitHub Pages Chromium `onnxruntime-web` WASM）**：
  HTTP `200` 加载成功，单帧推理 **`44 ms`**，输出维度 `(1, 1, 256, 256)` 与 `(1, 1, 16, 48)` 全部为有限数值 `[0, 1]`。
- **管道完整性自动化测试**：`python research/tests/test_pipeline_integrity.py` → **3/3 通过**（缺失 checkpoint 导出/评测强制报错非零退出 + 已发布 ONNX 形状与数值域校验）。
