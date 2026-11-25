# EchoSight 参赛数据包

本目录整理了可用于比赛附件、演示和工程审计的数据。

## 文件

- `model_metrics_synthetic.csv`：基于现有 64 场景确定性合成验证的三模型指标。
- `model_metrics_comparison.png`：RMSE 与 δ₁ 对比图，图中明确标注为合成验证。
- `synthetic_scene_manifest.csv`：64 条可复现的合成演示场景清单。
- `synthetic_scene_manifest.json`：同一场景清单的 JSON 版本。
- `validation_evidence.csv`：已完成工程检查与待完成真实验证的状态表。
- `package_manifest.json`：数据包元信息。

## 重新生成

在项目根目录运行 `python research/prepare_competition_data.py`。模型指标来源于 `research/outputs/ablation_report.md`，不会自动替换真实外场记录。

## 数据边界

合成场景和模拟距离只用于演示、回归和管线说明。它们不能代替真实手机测距、VoiceOver/TalkBack 真机测试或视障用户研究。真实数据完成后，应将原始记录、设备信息、采集日期和计算方法一并归档。
