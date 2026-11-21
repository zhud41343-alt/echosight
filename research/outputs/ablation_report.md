# EchoDepthNet 三模型对比与消融实验自动评测报告（Pretrained + Residual FG-EDFM）

运行设备：`CPU`

| 模型版本 | 参数量 (M) ↓ | AbsRel ↓ | RMSE ↓ | δ₁ (<1.25) ↑ | Acoustic RMSE ↓ | 推理速度 (FPS) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FastDepth-Lite (UNet-MobileNetV3) | 1.02 | 0.0706 | 0.0798 | 0.9047 | 0.0562 | 88.2 |
| EchoDepthNet Baseline (FPN w/o FG-EDFM) | 1.08 | 0.0710 | 0.0872 | 0.8864 | 0.0583 | 94.1 |
| **EchoDepthNet (Ours, +Residual FG-EDFM)** | **1.11** | **0.0754** | **0.0824** | **0.8970** | **0.0577** | **74.2** |
