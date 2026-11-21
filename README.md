# 听见光影 EchoSight — 基于 EchoSafe 可靠性状态机与可控 3D 空间声场的视障辅助感知原型

> **2026 Qoder 高校创造力大赛 · Hack4Good 技术向善赛道参赛作品**
> - **线上备用体验（GitHub Pages）**：https://zhud41343-alt.github.io/echosight/
> - **重要定位声明**：EchoSight 是浏览器端跨模态辅助感知与感官替代研究原型，**不可替代白手杖、导盲犬、定向行走（O&M）训练或现场交通安全判断**。

## 核心工程创新
1. **EchoSafe V5 可靠性状态机与失效保护（`src/engine/echoSafe.ts`）**：
   - 六状态精细管理（`IDLE` / `INITIALIZING` / `ACTIVE` / `DEGRADED` / `UNAVAILABLE` / `DEMO`），严格隔离真实摄像头（`REAL_CAMERA`）与模拟演示数据（`SIMULATED_SCENE`）。
   - **单调视频帧标识（`FrameTracker`）**：为每一帧真实摄像头新画面分配单调递增 `frameId`，防止低帧率摄像头下 AI 对同一帧重复推理并伪造两帧连续确认。
   - **连续声场即时熔断（`canEmitContinuousSpatialAudio` + `spatialAudio.stopAllImmediate()`）**：画面冻结、轨道断开、页面切入后台或系统进入 `UNAVAILABLE` 时，立即中止 Sobel/HRTF 连续声场并释放活跃音频节点。
   - **IoU 多目标空间追踪（`SpatialObjectTracker`）**：按 IoU、中心距与独立 `frameId` 追踪多个同类物体，防止不同位置同类目标误确认；未标定相机仅输出保守距离区间，空检测绝不生成安全保证。
2. **低认知负担的可控空间音频交互（`src/engine/audioScheduler.ts`）**：
   - 无竞态三级优先级语音与空间音效协调（`EMERGENCY > USER_QUERY > AMBIENT`、`playbackId` 回调守卫、统一单调时钟）。
   - 提供 4 种听觉提示策略，默认「简洁语音优先」以减少环境听觉干扰；实验性单路环境异常声响提示默认关闭、按需开启。

## 双模式架构
- **随身助行（默认首页）**：中文清晰主状态、全幅原生纵横比摄像头视窗与实时识别框、生命周期受控的语音查询与快捷按钮、折叠式技术参数栏、纯黑高对比显示模式（已标注为「高对比模式」，未宣称已通过 WCAG AAA 认证）。
- **专业实验台（6 大模块）**：
  1. **01 声画扫描**：摄像头/图片/内置演示场景（明确标注数据来源）+ Sobel 边缘 + 48×16 声学矩阵 + 180° 双耳方位雷达 + 点探声场 + FFT 频谱。
  2. **02 场景剧场**：4 个盲人生活动态模拟场景，支持上帝视角与全黑视角切换。
  3. **03 指尖听画**：自由绘制并实时听见 3D 方位与音高，支持音阶切换、撤销、PNG 导出与扫描回放。
  4. **04 黑暗迷宫**：3 个递进关卡，8 方向射线追踪 + 3D HRTF 回声定位演示。
  5. **05 原理与硬件**：技术流水线、数学公式、穿戴硬件演进路线与适用人群说明。
  6. **06 技术验证**：EchoSafe V4 故障注入对比演示与分层能力证据矩阵。

## 隐私与离线边界
- 视觉边缘计算、TensorFlow.js 推理与 Web Audio 音频合成均在浏览器本地完成，不上传视频画面。
- 浏览器原生语音识别（Web Speech Recognition）在多数设备上会依赖浏览器厂商云端转写，已在界面与文档如实披露。
- 浏览器安全策略限制锁屏后台摄像头采集，页面切入后台时 EchoSafe 会自动进入 `UNAVAILABLE` 状态。

## 本地开发、测试与构建
```bash
npm install
npm test        # 运行 Vitest 自动化单元测试套件（30 项用例）
npm run dev     # 本地预览
npm run build   # TypeScript 严格检查并生成 dist/ 静态构建产物
```
