# 真实浏览器端到端与多视口测试报告（REAL_BROWSER_TEST_REPORT）

- **测试时间**：2026-11-20
- **测试分支**：`feat/echosafe-v2`（对比基线 `fd1d425e9e9b3dde4f5efd0f31996e9acc72774a`）
- **测试环境**：Windows 11 + Vite Dev Server (`http://localhost:5173/`) + 真实内置 Chromium 浏览器（物理摄像头可用）
- **单元测试命令**：`npm test`（Vitest v5.1.1）→ **5/5 测试文件、21/21 用例通过（164ms）**
- **生产构建命令**：`npm run build`（`tsc -b && vite build`）→ **0 错误**

## 1. 真实浏览器端到端检查矩阵

| 测试项 | 视口 / 条件 | 实际观测结果 | 结论 |
| :--- | :--- | :--- | :--- |
| 随身助行默认首页 | `1440×900` | 进入 `EchoSafe 状态 · IDLE`，明确提示「当前无实时摄像头输入（实时模式严禁混入模拟画面）」，0 模拟车/斑马线数据混入 | 通过 |
| 真实摄像头启动与性能分离测量 | `1440×900` | 初始化帧未就绪前进入 `UNAVAILABLE`（`isActive: false`）；真实摄像头帧到达后平滑转入 `ACTIVE`，实测显示 **`视觉 40fps · AI 1.2fps · 39ms`**（真实分离视觉与 AI 推理帧率） | 通过 |
| WAI-ARIA 键盘 Tab 切换 | `1440×900` | 聚焦 `01 声画扫描` 连续按 5 次 `ArrowRight`，成功聚焦并激活 `06 技术验证`（`aria-selected="true"`） | 通过 |
| EchoSafe 故障对比交互 | `1440×900` | 点击「手机锁屏 / 页面切入后台」，状态机输出 `UNAVAILABLE`（`dataSource: none · canTrustLivePerception: false`） | 通过 |
| 手机标准视口适配 | `390×844` | `scrollWidth <= clientWidth`（`382px <= 390px`），0 横向溢出，按钮与文字清晰可触 | 通过 |
| 小屏手机视口适配 | `360×740` | `scrollWidth <= clientWidth`（`352px <= 360px`），0 横向溢出 | 通过 |
| 控制台异常发现与修复 | 切换经过 `04 黑暗迷宫` | 捕获 1 处真实 `IndexSizeError`（`DarkMaze.tsx` 声呐涟漪 `now - createdAt` 偶发微负导致 `arc` 半径为 `-2.36`）；已用 `Math.max(0, ...)` 修复并回归构建通过 | 已修复 |

## 2. 尚未完成 / 受限项（如实披露，严禁伪造）
- **真实 iOS VoiceOver / Android TalkBack 手机读屏实测**：当前测试环境无真实移动读屏设备，标记为 **「尚未完成」**。
- **真实室外街道 0.5–10m 标定测距数据集与真实街道噪音误报率**：当前仅有合成基准单元测试，真实拍摄评测标记为 **「尚未完成」**。
- **真实视障用户受控环境共创实验**：已完成伦理方案（`docs/USER_RESEARCH_PROTOCOL.md`），尚未开展真人试验，标记为 **「尚未完成」**。
- **Qoder Sites 正式参赛链接**：当前已部署 GitHub Pages 补充链接（`https://zhud41343-alt.github.io/echosight/`），正式参赛提交时需按大赛规则在 Qoder 平台部署 Qoder Sites 链接。
