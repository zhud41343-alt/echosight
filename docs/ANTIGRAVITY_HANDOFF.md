# 听见光影 EchoSight｜反重力交接说明

## 项目目标

使用 EchoSight 参加 **2026 Qoder 高校创造力大赛 · Hack4Good 技术向善赛道**。

项目目录：`D:\听见光影`

GitHub 仓库：`https://github.com/zhud41343-alt/echosight`

备用体验：`https://zhud41343-alt.github.io/echosight/`

## 已完成的优化

- EchoSafe 六状态可靠性状态机和冻结画面失效保护。
- 浏览器端 EchoDepthNet ONNX 实时推理和 3D HRTF 声场路由。
- 随身助行模式和专业实验台六个模块。
- 参赛提交帖补齐官方要求的自我介绍、作品说明、创作原因、5 步使用方法和 Qoder 开发过程。
- 2 分钟演示脚本已更新。
- 合成数据包已整理到 `research/outputs/competition_data/`。
- 旧版测试数字已统一为前端 49/49、Python 管道 5/5。

## 当前验证结果

```text
npm test                                  10 files, 49/49 passed
python research/tests/test_pipeline_integrity.py  5/5 passed
npm run build                             success
```

构建存在大体积 chunk warning，但构建成功，不属于阻塞错误。

## 数据包内容

- `model_metrics_synthetic.csv`：64 场景合成验证中的三模型指标。
- `synthetic_scene_manifest.csv`：64 条可复现合成场景清单。
- `synthetic_scene_manifest.json`：同一清单的 JSON 版本。
- `validation_evidence.csv`：工程验证结果和待完成真实验证。
- `model_metrics_comparison.png`：RMSE 与 δ₁ 对比图。
- `README.md`：数据来源和使用边界。

合成数据只能用于演示、回归和可复现说明，不能写成真实外场测量结果。

## 还需要完成的提交事项

1. 在 Qoder Sites 发布 `dist`，取得公开 `qoder.zone` 链接。
2. 设置为任何人可访问，使用无痕窗口和手机网络核验。
3. 填写参赛人、学校、专业、年级和 Qoder 账号或 UID。
4. 完成天池报名、实名认证和阿里云学生认证。
5. 在天池 Hack4Good 赛题页发布作品帖。

截止时间：**2026 年 11 月 30 日 18:00（北京时间）**。

## 关键文件

- `TIANCHI_SUBMISSION_DRAFT.md`：天池作品帖草稿。
- `docs/COMPETITION_PPT_AND_VIDEO_SCRIPT.md`：12 页路演大纲、2 分钟视频脚本和 Q&A。
- `docs/QODER_SITES_PUBLISH_STEPS.md`：Qoder Sites 发布和核验步骤。
- `research/prepare_competition_data.py`：数据包生成脚本。

## 反重力下一步任务

请优先完成 Qoder Sites 发布和公开访问核验。不要删除项目中的安全边界声明，不要把合成数据、模拟场景或待开展用户研究写成真实实测结果。发布成功后，把正式 `qoder.zone` 链接填入 `TIANCHI_SUBMISSION_DRAFT.md`，再检查所有占位符并生成最终作品帖。
