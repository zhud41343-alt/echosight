# EchoSight 整改前后技术差异审计（AUDIT_BEFORE_AFTER）

- **V1 基线 Commit**：`fd1d425e9e9b3dde4f5efd0f31996e9acc72774a`
- **V2 中间 Commit**：`4531763827c08bb0ce2d6fdf34bb2c4acfebc796`
- **V3 整改分支**：`feat/echosafe-v3`（合并后见最新 `main` HEAD）

| 问题类别 | V1 / V2 历史缺陷 | V3 整改后实现 | 仓库代码与测试证据 | 验证层级 |
| :--- | :--- | :--- | :--- | :--- |
| **P0-1 空时间戳与伪新鲜度** | `lastFrameAt === null` / `lastInferenceAt === null` 未阻断 `ACTIVE`；重复 `drawImage` 伪造帧刷新 | 强制要求真实视频帧到达（`requestVideoFrameCallback` / `video.currentTime` 推进）且至少完成 1 次有效推理后才能进入 `ACTIVE` | `src/engine/echoSafe.ts`, `src/components/BlindCompanionView.tsx`, `tests/echoSafe.test.ts` | 已实现且自动化验证 |
| **P0-2 异步任务失效回写与流泄漏** | 停止/切换模式后，迟到的 `getUserMedia` 流未释放，迟到的推理结果可能回写状态与播报 | 引入 `SessionGuard` 单调会话代次令牌；停止/卸载/中断立即失效，迟到媒体流主动 `.stop()`，迟到推理结果全部丢弃 | `src/engine/echoSafe.ts`, `src/components/BlindCompanionView.tsx`, `tests/echoSafe.test.ts` | 已实现且自动化验证 |
| **P0-3 音频调度器竞态** | 旧语音取消后延迟触发 `onend`/`onerror` 覆盖新任务状态；混用 `performance.now()` 与 `Date.now()` | 引入递增 `playbackId` 回调守卫、统一可注入单调时钟 `clock()`，并增加 `shouldAllowSpatialCue` 统一协调空间音效 | `src/engine/audioScheduler.ts`, `tests/audioScheduler.test.ts` | 已实现且自动化验证 |
| **P0-4 感知数据包有效性校验** | 局部功能直接读取旧数组 `detections.length > 0` | 建立 `PerceptionPacket`（含 `dataSource`/`sessionId`/`capturedAt`/`inferredAt`/`expiresAt`/`safeState`）与 `isPerceptionPacketUsable` 统一校验 | `src/engine/echoSafe.ts`, `src/components/BlindCompanionView.tsx`, `tests/echoSafe.test.ts` | 已实现且自动化验证 |
| **P0-5 语音识别失败处理** | 语音识别不可用时静默返回场景报告 | 对不支持、拒权、无语音、网络异常分别明确反馈并展示可访问替代按钮 | `src/components/BlindCompanionView.tsx` | 已实现且浏览器验证 |
| **P1-1 随身首页减负与坐标系说明** | 主界面展示英文状态码与 FPS；未限定摄像头坐标系 | 中文主状态显示、技术指标折叠进 `<details>`、明确标注“方位均相对当前摄像头画面” | `src/components/BlindCompanionView.tsx` | 已实现且浏览器验证 |
| **P1-2 单目测距与实验模块诚实边界** | 未标定输出小数米数；单麦宣称 360°；RGB 信号灯判通行 | 未标定仅输出保守区间、过滤截断框为未知、提供 MAE/P90 统计工具；单麦降格为非方向性异常提示；红绿灯加 3 帧防抖且 `isCrossingClearance: false` | `src/engine/aiObjectDetector.ts`, `src/engine/acousticThreatDetector.ts`, `src/engine/blindAssistTools.ts`, `tests/distance.test.ts`, `tests/acousticAndTraffic.test.ts` | 仅模拟验证（真机实拍待实测） |
| **P1-3 WAI-ARIA 与高对比命名** | Tab 缺少 ARIA 关联与方向键；未审计即标 WCAG AAA | 补齐 WAI-ARIA Tabs 规范与键盘焦点；高对比模式如实改称「高对比模式」；披露语音识别云端依赖 | `src/App.tsx` | 已实际浏览器验证（真实手机读屏待实测） |
