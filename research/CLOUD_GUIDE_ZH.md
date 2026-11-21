# 云端 GPU 一键训练指南（全中文 · 零消耗本地显卡）

这份压缩包 `EchoDepthNet_Cloud_Pack.zip` 可以直接上传到 **阿里云 PAI-DSW（可用天池高校认证送的 300 元算力券）**、**魔搭社区（ModelScope 免费 GPU）** 或 **AutoDL** 一键跑完全部训练、消融实验和论文图表生成。

## 只需 3 步：
1. **打开云端 GPU 终端**（选择任意带 PyTorch + GPU 的镜像，如 T4 / A10 / RTX 4090）。
2. **上传并解压压缩包**：
   把 `EchoDepthNet_Cloud_Pack.zip` 拖进云端文件目录，在终端输入：
   ```bash
   unzip EchoDepthNet_Cloud_Pack.zip -d echosight_research
   cd echosight_research
   ```
3. **一键启动全套实验（训练 + 消融对比 + 出论文图 + 导出 ONNX）**：
   ```bash
   bash run_cloud_experiments.sh
   ```

## 运行完成后，直接下载 `outputs/` 目录里的 3 个核心成果：
* `outputs/ablation_report.md` —— 自动生成的论文消融实验对比表（AbsRel / RMSE / δ₁ / FPS）
* `outputs/paper_figure_preview.png` —— 论文四联高清可视化对比图（原图 / 真值深度 / 预测深度 / 16×48 声学危险度矩阵）
* `outputs/echodepth_net.onnx` —— 导出的轻量级端侧推理模型（约 4–5 MB）

## 进阶：接入真实公开数据集（NYU Depth V2 / SUN RGB-D）
把真实数据集按以下结构放好：
```text
my_dataset/
  ├── rgb/    (*.jpg 或 *.png)
  └── depth/  (同名深度图 .png)
```
然后执行：
```bash
python train.py --data-dir my_dataset --pretrained --epochs 50 --batch-size 32
```
