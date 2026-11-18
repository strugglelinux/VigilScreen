# VigilScreen 独立应用

现有已验证发布包适用平台：Apple Silicon（M1/M2/M3/M4 等）Mac，macOS 15 或更新版本。无需安装 Python、PyTorch、虚拟环境或其他软件。现有 arm64 包不适用于 Intel Mac 或 Windows。

源码现已提供 Intel Mac x86_64 和 Windows Intel / AMD x64 的独立依赖及原生构建入口，须分别在目标系统构建、自检。完整步骤和验证边界见 [平台适配说明](docs/PLATFORMS.md)。这不代表现有 arm64 安装包已转换或新增目标包已经实机验证。

打开 `VigilScreen-macOS-arm64.dmg`，将 `VigilScreen.app` 拖到“应用程序”，再双击运行。也可直接复制 `dist/VigilScreen.app` 到其他目录使用。应用启动时不打开摄像头，点击“开始监控”或工作室“打开摄像头”时才申请权限。

应用包含监控、目标选择、多选过滤、图片与摄像头标记裁剪、模型训练、身份效果验证、Peek 菜单栏提示、红框截图及初始化功能。它内置 Python、Torch、OpenCV、MediaPipe、Tk 与必要的视觉模型；没有通过删除训练功能来缩小体积。

macOS 最小化时隐藏到顶部眼睛托盘图标，Dock 图标与窗口缩略图隐藏，后台任务继续。点击图标选择“显示主窗口”恢复，或选择“退出 VigilScreen”结束任务。系统摄像头隐私图标独立保留。

macOS 用户数据保存于 `~/Library/Application Support/VigilScreen/`，Windows 为 `%LOCALAPPDATA%\VigilScreen\`。构建默认不携带个人材料；只有显式添加 `--include-training-data` 才复制训练材料和身份模型作为初始资源。历史命中截图不打包。之后训练、截图和初始化都操作此用户目录，不改动应用包及源码项目。初始化后不会重新导入内置材料。诊断日志位于该目录的 `logs/application.log`。

体积优化包括只保留目标架构、排除未使用依赖、限制 Matplotlib 为 MediaPipe 所需的 Agg 后端、删除开发文件和未使用的 protobuf 编译器，并在 macOS 裁剪动态库符号、使用压缩磁盘映像发布。Intel Mac 的旧版 MediaPipe 依赖 JAX / SciPy，构建配置会保留；Windows 使用 CPU 版 Torch 避免打入 CUDA 库。PyTorch 的本地训练库是体积的主要组成部分。精确体积和验证结果以对应包实际生成的报告为准，旧 arm64 报告不代表当前源码或其他平台构建结果。

应用使用本机临时签名，未进行 Apple 公证。若首次打开时系统拦截，可在“系统设置 → 隐私与安全性”中允许打开此应用。摄像头权限仍由 macOS 控制。

## 重新构建

在本项目目录的 arm64 虚拟环境下安装依赖，并先下载通用模型：

```bash
python -m pip install -r requirements-dev.txt
python download_models.py
```

然后执行：

```bash
.venv/bin/python build_macos.py
.venv/bin/python audit_macos.py
```

Intel Mac 使用 Python 3.11 x86_64 环境和 `build_macos.py --arch x86_64`；Windows 使用 `build_windows.py`。不能在当前 Apple Silicon 环境直接构建这两个目标。构建默认排除训练数据。

构建脚本创建独立 `.app`，裁剪共享库并重新签名。训练和验证使用应用自身的 `--identity-worker` 入口，不调用系统 Python。路径及子进程处理参考 [PyInstaller 运行时说明](https://pyinstaller.org/en/stable/runtime-information.html) 和 [子进程注意事项](https://pyinstaller.org/en/latest/common-issues-and-pitfalls.html)。

发布前在隔离的数据目录运行内置自检，避免改动真实用户数据：

```bash
VIGILSCREEN_DATA_DIR=/tmp/vigilscreen-check \
  dist/VigilScreen.app/Contents/MacOS/VigilScreen --self-test build/standalone-self-test.json
```

自检创建窗口、加载实际模型、验证图像显示和菜单栏，在有内置身份材料时完成一次 CPU 训练与身份验证；不打开摄像头。

自检报告写入 `build/standalone-self-test.json` 后，执行 `.venv/bin/python finalize_macos.py`，可重新检查动态库与签名，并生成 `VigilScreen-macOS-arm64.dmg` 或 `VigilScreen-macOS-x86_64.dmg` 和构建报告。报告与包架构必须一致；最低 macOS 版本会根据实际本地库审计结果提高。每次重新构建都应重新自检，不能复用旧包报告。

Windows 构建、自检和分发步骤见 [平台适配说明](docs/PLATFORMS.md)。必须分发完整 `dist/VigilScreen/`，不能只复制 `.exe`。Windows 当前使用窗口内命中提示和普通任务栏最小化，没有 macOS 菜单栏托盘。源码修改、Git 提交和重新打包是不同步骤，修改源码不会自动更新已安装应用。
