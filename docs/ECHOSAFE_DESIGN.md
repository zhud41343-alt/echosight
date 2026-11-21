# EchoSafe V6 可靠性状态机与全链路失效保护设计（ECHOSAFE_DESIGN）

## 1. 核心设计目标
面向视障辅助感知场景，**误报“安全”或无声失效比“暂不可用”危险得多**。EchoSafe V6 沿实际执行链（摄像头 → 新视频帧 → Canvas → TensorFlow.js → `SpatialObjectTracker` → `PerceptionPacket` → `EchoSafe` → `AccessibleNotificationDispatcher` → ARIA Live / TTS / 触觉 / HRTF）逐节点建立集中式有效性闸门与失效保护（Fail-Safe）机制。

## 2. 六状态定义（`src/engine/echoSafe.ts`）
- `IDLE`：未启动，`dataSource = 'UNAVAILABLE'`，`canTrustLivePerception = false`。
- `INITIALIZING`：正在申请权限、加载模型或等待首帧真实视频画面到达（`lastFrameAt === null` 绝不进入 `ACTIVE`）。
- `ACTIVE`：摄像头权限已授予、视频轨道 `live`、真实新帧未过期（`< 2500ms`）、端侧模型就绪、已完成至少一次有效推理且未过期（`< 4500ms`）、页面在前台、音频上下文运行中。
- `DEGRADED`：摄像头正常但存在部分能力失效（模型加载失败、等待首次推理、连续推理失败 ≥ 3 次、单次推理超时 > 3500ms、推理结果过期、或音频挂起）。
- `UNAVAILABLE`：页面切入后台（`document.hidden`）、摄像头拒权/无设备、视频轨道断开、或视频画面冻结（超过 `frameStaleMs` 无真实新帧推进）。此时立即清空识别框与旧数据包，并立即中止所有连续空间音频。
- `DEMO`：明确隔离的模拟演示模式（仅用于「专业实验台」），绝不混入实时助行通道。

## 3. V5 全链路防伪与熔断机制
1. **单调帧身份（`FrameTracker`）**：仅在 `requestVideoFrameCallback`（或保守回退 `video.currentTime` 变化）收到真实新帧时递增 `frameId`。AI 推理循环必须通过 `claimNextFrameForInference` 获取尚未推理过的新鲜帧，避免低帧率摄像头下重复推理同一帧。
2. **空间多目标追踪（`SpatialObjectTracker`）**：基于检测框 IoU（≥ 0.2）与归一化中心距（≤ 14% 对角线）独立追踪同类物体，且校验 `frameId` 严格递增；中置信度目标（`0.45–0.68`）必须经过两个**独立新视频帧**的空间匹配才确认输出。
3. **会话代数守卫（`SessionGuard`）**：每次启动递增 `sessionId`，停止/卸载/切换时立即失效，先校验会话再允许修改追踪器状态或写回结果，迟到流自动停止轨道。
4. **全链路数据包校验（`PerceptionPacket` + `isPerceptionPacketUsable`）**：统一校验 `dataSource === 'REAL_CAMERA'`、`sessionId`、合法正整数 `frameId`、当前与包内 `ACTIVE` 状态、时间单调序（`capturedAt <= inferredAt <= expiresAt`）、拒收未来时间戳、原始帧新鲜度与包 TTL。
5. **连续声场即时熔断（`canEmitContinuousSpatialAudio` + `spatialAudio.stopAllImmediate()`）**：Sobel 边缘计算与 3D HRTF 连续声场严格校验画面新鲜度与 EchoSafe 状态；一旦冻结、断流、切后台或切换策略，立即停止并释放所有活跃振荡器节点。
6. **单并发锁与双指标监控（`SingleFlightRunner` + `RuntimeMetricsTracker`）**：杜绝推理队列堆积，严格分离视觉帧率（`visionFps`）与 AI 推理帧率（`aiFps`）及推理/端到端 P95 延迟。

## 4. 无障碍通知分发与状态转换反馈（V6 新增：`src/engine/accessibleNotifier.ts`）
- **消息有效性与 TTS 解耦（`AccessibleNotificationDispatcher`）**：统一处理去重、冷却与优先级，开启「读屏器防冲突模式」或浏览器不支持 TTS 时，仍将有效环境与查询更新送达 `role="status"` / `role="alert"` ARIA Live 区域，绝不因 `speechSynthesis` 未接受而丢消息。
- **状态转换多通道协调（`planEchoSafeTransition`）**：覆盖 `IDLE / INITIALIZING / ACTIVE / DEGRADED / UNAVAILABLE` 关键跃迁；进入 `UNAVAILABLE`（如视频冻结）时联动 `aria-live="assertive"`、TTS、触觉报警与 `stopAllImmediate()`，并防止 1 秒健康检查重复播报同一故障。
- **会话绑定 Wake Lock（`src/engine/deviceFeedback.ts`）**：`ScreenWakeLockController` 采用单调代次守卫与单并发去重，若用户在异步申请完成前点击停止，迟到返回的 `WakeLockSentinel` 将立即自释放。

## 5. 无竞态语音与音频协调（`src/engine/audioScheduler.ts`）
- **三级优先级**：`EMERGENCY (3)` > `USER_QUERY (2)` > `AMBIENT (1)`。
- **防竞态 `playbackId` 守卫**：被抢占语音的迟到 `onend`/`onerror` 回调不会覆盖新播报状态；统一使用单调时钟。
- **全空间音互斥与听觉减负**：默认「简洁语音优先」策略；播报 `EMERGENCY` 或 `USER_QUERY` 时自动静音连续声场、寻物信标与低频空间提示。
