# EchoSight 发布到 Qoder Sites

本项目的正式参赛链接必须是 Qoder Sites 的公开地址。GitHub Pages 仅作为备用体验链接，不能替代正式 Qoder Sites 链接。

## 1. 发布前检查

在 Qoder 中打开项目目录 `D:\听见光影`，执行：

```powershell
npm install
npm test
npm run build
```

当前基线应为：

- Vitest：10 个测试文件，49/49 通过
- 生产构建：`dist/` 生成成功
- 模型文件：`dist/models/echodepth_net.onnx` 存在

## 2. 在 Qoder Sites 创建并发布

在 Qoder 的 Sites 面板中打开新站点，或在 Qoder 对话中输入：

> 将当前项目 `D:\听见光影` 的生产构建发布为 Qoder Sites 静态站点。使用 `dist` 作为静态目录，保留 React/Vite 构建结果、`models/echodepth_net.onnx`、PWA 文件和所有资源。站点名称设置为「听见光影 EchoSight」，先生成预览。

检查预览中的首页、随身助行、01 声画扫描、02 场景剧场、04 黑暗迷宫和 06 技术验证，然后点击发布。

## 3. 设置公开访问

在站点详情中打开分享设置：

1. 选择「任何人都可访问」。
2. 确认允许公开访问。
3. 复制完整的 `qoder.zone` 地址。

评委需要未登录也能打开页面。不要把站点保留为仅自己可访问或仅受邀用户可访问。

## 4. 发布后核验

用无痕窗口和手机网络打开 `qoder.zone` 地址，逐项确认：

- 页面无需登录即可打开。
- 首页能进入随身助行和专业实验台。
- 「01 声画扫描 → 神经深度」能加载 ONNX 模型并显示状态。
- 「06 技术验证」触发画面冻结后，旧识别框清空，连续声场停止。
- 手机宽度 360–390 像素没有横向滚动。
- 浏览器控制台没有新增报错。

把正式 `qoder.zone` 地址替换到 `TIANCHI_SUBMISSION_DRAFT.md` 的链接位置，并在作品帖中同时附上 GitHub 仓库和 GitHub Pages 备用链接。

## 5. 天池提交

在 2026 Qoder 高校创造力大赛的 Hack4Good 赛题页论坛发布作品帖。截止时间为北京时间 **2026 年 11 月 30 日 18:00**。帖子需要包含个人介绍、作品一句话说明、创作原因、5 步使用方法、Qoder 开发过程和 Qoder Sites 公开链接。

