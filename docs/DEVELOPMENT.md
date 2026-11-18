# 开发说明

## 环境与资源

先按 README 创建虚拟环境、安装运行依赖，并执行 `python download_models.py --check`。通用模型与 `face_models/models.json` 随源码分发；文件缺失或损坏时运行 `python download_models.py`，脚本检查 SHA-256、使用临时文件下载，校验通过后才替换目标文件。`--check` 只验证本地文件，不访问网络。

开发构建依赖单独安装：

```bash
python -m pip install -r requirements-dev.txt
```

Windows 要构建精简 CPU 包时，先安装 `requirements-windows.txt`，再安装 `pyinstaller>=6,<7`，避免通用开发依赖拉取不同的 Torch 版本。Intel Mac 请使用 Python 3.11 x86_64 和对应依赖清单。新平台的验证边界见 [PLATFORMS.md](PLATFORMS.md)。

不需要模型或真实摄像头的自动测试可直接执行。测试依赖项目运行库，不额外引入 pytest。

## 模块划分

| 文件 | 职责 |
| --- | --- |
| `main.py` | 一体窗口、页面切换、初始化流程及退出协调 |
| `chi_camera.py` | 人脸检测、目标及排除范围、视线检查、截图和监控窗口 |
| `latest_camera.py` / `camera.py` | 最新帧采集与标记摄像头 |
| `gaze.py` | 头部与双眼几何规则、镜头和屏幕校准 |
| `identity.py` | SFace 特征提取、阈值和类别间距匹配 |
| `identity_training.py` | 材料快照、身份登记、留出验证及模型发布 |
| `gui.py` / `labeling.py` | 图片与摄像头裁剪标记、训练和验证工作室 |
| `test.py` / `model.py` / `data.py` | CNN 训练、数据划分和模型加载 |
| `tracking.py` | 每人的命中状态与重复提示控制 |
| `menu_bar.py` | macOS 原生 Peek 状态栏 |
| `tray_window.py` | 最小化隐藏、Dock 显隐、原生动画清理与恢复 |
| `runtime_paths.py` / `vigil_app.py` | 源码与独立应用路径、训练子进程、自检入口 |
| `runtime_support.py` | 架构归一化、CPU 线程预算、可用训练设备及 Windows 日志管道 |
| `check_environment.py` | 本机依赖诊断、三种通用视觉模型的离线加载检查 |
| `build_macos.py` / `build_windows.py` / `VigilScreen.spec` | 原生应用构建、资源和许可证收集，默认排除训练数据 |
| `audit_macos.py` / `optimize_macos.py` / `finalize_macos.py` | macOS 动态库审计、精简签名、最低系统版本及 DMG 发布 |
| `initialization.py` | 清理限定数据目录，保留通用视觉资源 |
| `download_models.py` | 下载和核对通用视觉模型 |

## 状态与并发

摄像头由采集线程持有，只保留最新帧。界面通过队列更新，不在工作线程操作 Tk 控件。身份切换先作废旧范围的结果，工作线程在单次人脸任务之间取消旧处理，再应用目标及过滤名单，从新帧恢复检测。

命中截图先在切换锁外编码并写入隐藏临时文件，持锁复查范围后才原子提交；范围过期时删除临时文件，不发 Peek。不能把 JPEG 编码和完整写盘放回 `transition_lock`，否则 Tk 的目标选择回调会等磁盘操作。`test_identity_switch.py` 模拟写盘阻塞，验证切换仍立即返回且旧截图不提交。

目标下拉菜单映射或 macOS 原生菜单跟踪期间，监控轮询暂缓 Tk 视频及状态重绘，保留最新检测帧；后台检测和截图继续。恢复时每轮最多处理 64 条消息、约 6 ms 的事件工作量，并合并性能文本更新，避免消息积压一次占满界面线程。

每张当前人脸重新判断身份与视线，不把旧帧身份或视线迁移到新帧。位置关联仅用于防止持续重复提示。身份过滤优先于指定目标，判断是否过滤仍需先提取身份特征。

macOS 托盘的 Cocoa selector 只能把动作写入队列，不能直接调用 Tk，包括 `after`、`after_idle`。窗口恢复和退出由 Tk 定时轮询执行，并避开原生菜单跟踪模式，防止 Tcl/Python 线程状态重入导致进程崩溃。

恢复窗口时合并重复请求，为隐藏/恢复定时回调记录版本，过期回调不能反转新状态。不能仅用 `root.state() == 'normal'` 或 Dock 图标出现判断成功；需要检查 Tk 映射、内部页面可见以及原生 NSWindow 的可见/最小化状态。对 Tk 认为正常但 Cocoa 已隐藏的窗口，重新映射并调用原生前置接口，再确认恢复稳定。

隐藏阶段只使用应用隐藏和窗口撤映射，不执行 `deminiaturize`、`deiconify` 或强制焦点；解除最小化留到用户明确恢复时执行。应用隐藏会隐藏其最小化窗口，不需要为了清除原生 mini 标志反复显示窗口。延迟焦点回调同样检查版本和当前可见意图，避免最小化后被旧回调重新激活。

训练及验证在独立进程执行。打包应用通过自身 `--identity-worker` 入口运行任务，源码通过当前 Python 启动 `identity_training.py`。初始化等待数据写入者退出后才清理目录，随后重建界面以清空缓存、过滤、校准及验证结果。

标记类别列表合并输出目录的直接类别子目录、`.labels.json` 和已登记身份。两个标记页共用刷新逻辑：目录变化时延迟刷新、展开下拉框和保存后立即刷新；只更新选项，不改写用户正在输入的名称。窗口关闭或初始化重建前应取消相关定时器及变量监听，避免旧页面访问已销毁控件。发现类别不等于完成身份登记，实时监控仍以当前身份模型为准。

## 模型与验证

不要把验证图加入身份登记特征。先划分原图再裁剪，保留材料快照、内容哈希及模型批次标识。验证任务固定当前模型批次，切换模型时清除旧验证缓存；验证报告不能覆盖身份入口或模型配置。

CNN 图表用于观察分类训练；实时身份匹配使用 SFace 登记特征。注视准确率需要人工标注的视频单独评估，不能用身份验证或单类别分类准确率替代。

## 测试

```bash
python -m unittest discover -s tests -v
python check_environment.py --models
python tests/gui_smoke.py
python tests/gui_target_switch.py
python tests/gui_initialization.py
python tests/gui_training_layout.py
python tests/gui_annotation_layout.py
python tests/gui_monitor_layout.py
python tests/gui_tray.py
python -B tests/gui_tray.py --tracking --repeat 8
python -B tests/gui_tray.py --tracking --repeat 8 --burst 25 --late-hide
```

GUI 脚本需要图形桌面，使用模拟摄像头；初始化测试只清理临时目录。托盘测试检查原生最小化、Accessory/Regular 策略、恢复/退出菜单以及隐藏期间持续监控。测试生成的日志、报告、图片及模型不要提交到 Git。

`unittest discover` 不执行 `gui_*.py`；修改布局或原生窗口交互时，需要单独执行对应 GUI 脚本。`check_environment.py --models` 会加载真实通用模型并检查空白图和身份特征输出，不访问摄像头，不评价实际身份或注视准确率。

`gui_monitor_layout.py` 检查监控侧栏在小窗口、长截图名、滚轮操作和窗口放大缩小时的布局。开始 / 暂停和退出位于滚动区域外；滚轮仅绑定侧栏控件，不能滚动改变当前命中目标。

`gui_annotation_layout.py` 使用临时图片检查两个标记页的空画面引导、冻结、裁剪保存、窗口缩放后侧栏可见性、水平分隔条拖动，以及“用于训练”的目录传递和页面跳转。两个页面共用 `annotation_layout`、`annotation_status` 和 `annotation_actions`，标题、预览底色和分组面板参照实时监测；小窗口下仍须完整显示右侧控件。

`--tracking` 使用 Cocoa 原生计时器实际展开托盘菜单，并在事件跟踪循环中选择恢复；`--repeat` 验证连续操作。它覆盖真实菜单回调的路径，不能仅用 Tk 回调中直接执行菜单 action 的测试替代。

`--burst` 在每次菜单选择时连续发送多次显示请求；`--late-hide` 模拟恢复期间迟到的 Cocoa 隐藏事件。恢复后检查真实主窗口及内部页面可见，不只检查 Dock/菜单栏状态。

托盘联调同时观察原生解除最小化和应用重新显示通知，检查整个隐藏过程中没有反复出现主窗口，不能仅检查隐藏结束后的状态。

## 原生构建与 CI

构建脚本、`.spec` 和权限 plist 是开发配置；实际 `build/`、`dist/`、日志、签名结果和诊断报告不属于源码仓库。平台、独立运行自检和安装映像步骤见 [DISTRIBUTION.md](../DISTRIBUTION.md)。

构建脚本默认排除本地 `data/`、`checkpoints/`；只有显式使用 `--include-training-data` 时才复制为应用初始资源。Intel Mac 和 Windows 原生构建、依赖及验证矩阵见 [PLATFORMS.md](PLATFORMS.md)。

`.github/workflows/compatibility.yml` 在 pull request 或手动触发时，分别使用 Intel Mac 和 Windows x64 runner 安装依赖、运行单元测试和通用模型检查。手动触发且选择 `package` 后，才构建并自检原生包；macOS 输出 DMG，Windows 输出包含完整运行目录的 ZIP，不自动发布 Release。CI 数据放在被 Git 忽略的 `.ci-user-data/` 中。

依赖解析、单元测试、模型加载、GUI 联调和目标设备实测分别记录。没有运行的 CI、未生成的安装包、模拟摄像头测试都不能描述为新平台实机通过。提交前检查 `git diff --check`，根据改动运行所需测试；源码提交不要求重新打包。

## 提交范围

提交 Python 源码、测试源码、Markdown 文档、依赖清单、构建脚本及配置，以及 `face_models/` 内的通用模型、校验清单和第三方许可。`.gitignore` 只对该目录的 ONNX / MediaPipe 模型放行；不要提交训练图片、个人身份特征、个人训练权重、截图、其他模型二进制、虚拟环境、缓存、安装包及本地验证/构建报告。

`.gitignore` 同时覆盖整个运行数据目录及常见图片、视频和模型扩展名；提交前仍应检查 `git diff --cached --stat` 与 `git diff --cached --name-only`。
