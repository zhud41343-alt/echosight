# EchoDepthNet 消融实验自动评测报告（Residual FG-EDFM）

运行设备：`CPU`

| 模型版本 | 参数量 (M) ↓ | AbsRel ↓ | RMSE ↓ | δ₁ (<1.25) ↑ | Acoustic RMSE ↓ | 推理速度 (FPS) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (w/o FG-EDFM) | 1.08 | 0.4534 | 0.1739 | 0.2564 | 0.1205 | 122.8 |
| **EchoDepthNet (Ours, +Residual FG-EDFM)** | **1.11** | **0.4024** | **0.1447** | **0.2891** | **0.1008** | **99.1** |
