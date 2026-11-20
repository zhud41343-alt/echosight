# EchoSight 原版与整改版技术差异审计（AUDIT_BEFORE_AFTER）

- **基线版本（Before）**：`main` 分支 `fd1d425e9e9b3dde4f5efd0f31996e9acc72774a`
- **整改分支（After）**：`feat/echosafe-v2`
- **自动化单元测试验证**：`npm test`（Vitest v5.1.1，5 个测试文件，21/21 项通过）
- **生产构建验证**：`npm run build`（`tsc -b && vite build`，0 报错）

| 问题类别 | 原版实现（`fd1d425`） | 整改后实现（`feat/echosafe-v2`） | 代码证据 | 验证状态 |
| :--- | :--- | :--- | :--- | :--- |
| **P0-1 模型加载失败伪装正常** | `loadOnDeviceAIModel` 失败时仍返回 `{ ready: true, fallback: true }` | 拆分为 `modelReady: false` 与 `variant: 'none'`，状态机进入 `DEGRADED`，禁止伪装正常 | [aiObjectDetector.ts](file:///d:/听见光影/src/engine/aiObjectDetector.ts) + [echoSafe.test.ts](file:///d:/听见光影/tests/echoSafe.test.ts) | 已实现且已测试 |
| **P0-2 模拟数据混入实时助行** | `BlindCompanionView` 无摄像头时绘制 `drawPresetScene('crosswalk')` 并推理 | 实时模式 `allowDemoFallback = false`，摄像头拒权/断开/过期立即进入 `UNAVAILABLE` 并清空旧检测框 | [BlindCompanionView.tsx](file:///d:/听见光影/src/components/BlindCompanionView.tsx) + [echoSafe.ts](file:///d:/听见光影/src/engine/echoSafe.ts) | 已实现且已测试 |
| **P0-3 未经验证的安全保证与移动指令** | 空检测播报“前方3-5米开阔、脚下平整”“向左/右偏两步” | 全面清除安全保证与盲目转向指令；空检测明确提示“未检出不代表前方无障碍”；过马路询问明确拒绝给出通行许可 | [xiaoGuangAssistant.ts](file:///d:/听见光影/src/engine/xiaoGuangAssistant.ts) + [visionProcessor.ts](file:///d:/听见光影/src/engine/visionProcessor.ts) + [safetyWording.test.ts](file:///d:/听见光影/tests/safetyWording.test.ts) | 已实现且已测试 |
| **P0-4 缺少可靠性状态机** | 仅依赖 `active` 布尔值，未监听前后台与帧过期，无单并发锁 | 新增 `EchoSafe` 6 状态机（`IDLE`/`INITIALIZING`/`ACTIVE`/`DEGRADED`/`UNAVAILABLE`/`DEMO`）、`SingleFlightRunner`、`RuntimeMetricsTracker` 与 `visibilitychange` 自检 | [echoSafe.ts](file:///d:/听见光影/src/engine/echoSafe.ts) + [echoSafe.test.ts](file:///d:/听见光影/tests/echoSafe.test.ts) | 已实现且已测试 |
| **P0-5 语音识别失败掩盖** | `!SpeechRec` 时静默返回场景报告，误导用户以为听到了提问 | 对不支持、权限拒绝、无语音、网络错误分别明确播报与提示，并提供可访问快捷查询按钮 | [BlindCompanionView.tsx](file:///d:/听见光影/src/components/BlindCompanionView.tsx) | 已实现且已测试 |
| **P1-1 单目测距过度精确宣传** | 未标定相机直接输出小数米数（如 `约1.2米`） | 区分 `CameraCalibration` 标定态与未标定态；未标定时仅输出保守区间（`较近/中距/较远（未标定粗估）`）；拒绝截断/过小框为 `距离未知`；新增 MAE/P90 评测函数 | [aiObjectDetector.ts](file:///d:/听见光影/src/engine/aiObjectDetector.ts) + [distance.test.ts](file:///d:/听见光影/tests/distance.test.ts) | 仅有模拟验证（真机标定待实测） |
| **P1-2 单路麦克风夸大为 360° 定位** | 宣称“360°后方车辆逼近声学警戒” | 降格为实验性「单路环境异常声响提示（无法判断方向）」并增加离线 FPR/FNR 评测接口 | [acousticThreatDetector.ts](file:///d:/听见光影/src/engine/acousticThreatDetector.ts) + [acousticAndTraffic.test.ts](file:///d:/听见光影/tests/acousticAndTraffic.test.ts) | 仅有模拟验证 |
| **P1-3 红绿灯 RGB 直接判通行** | 单帧简单 RGB 阈值输出“绿灯/红灯” | 明确标注为实验性像素颜色识别（`isCrossingClearance: false`），增加过曝/多灯干扰拒识与 3 帧稳定器 `TrafficLightStabilizer` | [blindAssistTools.ts](file:///d:/听见光影/src/engine/blindAssistTools.ts) + [acousticAndTraffic.test.ts](file:///d:/听见光影/tests/acousticAndTraffic.test.ts) | 仅有模拟验证 |
| **P1-4 多模块语音互抢** | 多处直接调用 `speechSynthesis.cancel()` | 新增 `AudioSpeechScheduler` 三级优先级（`EMERGENCY > USER_QUERY > AMBIENT`）、冷却去重与 4 种用户可选听觉策略（默认简洁语音优先） | [audioScheduler.ts](file:///d:/听见光影/src/engine/audioScheduler.ts) + [audioScheduler.test.ts](file:///d:/听见光影/tests/audioScheduler.test.ts) | 已实现且已测试 |
| **P1-5 WAI-ARIA 与无障碍宣称** | Tab 缺少 `aria-controls` / 方向键焦点；未审计即标注 WCAG AAA | 补齐 WAI-ARIA Tabs 规范（`id`/`aria-controls`/`aria-labelledby`/左右方向键/Home/End）；高对比模式如实改称「高对比模式」；披露语音识别云端依赖 | [App.tsx](file:///d:/听见光影/src/App.tsx) | 已实现且已浏览器验证（真实 VoiceOver/TalkBack 待实测） |
