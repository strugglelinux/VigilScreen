# Intel Mac、Windows 与 CPU 兼容

同一份 Python 源码使用 CPU 完成人脸检测、身份匹配和视线判断；Intel 与 AMD x64 无需不同版本的模型，也不要求独立显卡。训练选择 `auto` 时，无可用加速设备会使用 CPU；Intel Mac 的自动选择默认使用 CPU。显式选择 MPS / CUDA 时仍验证是否可用。GUI 只列出本机可用的训练设备。

| 目标 | Python / 依赖 | 原生包 | 当前验证范围 |
| --- | --- | --- | --- |
| Intel Mac x86_64 | Python 3.11、`requirements-macos-intel.txt` | `.app` / x86_64 DMG | 已核实兼容轮子并增加模拟平台测试；需在 Intel Mac 执行真实库、摄像头及包自检 |
| Windows 10/11 x64，Intel 或 AMD | 推荐 Python 3.11、`requirements-windows.txt` | `VigilScreen.exe` 及同目录运行库 | 已核实依赖轮子并增加 Windows 路径、日志、进程及设备回退测试；需在 Windows 实机验证 |
| Apple Silicon Mac | Python 3.13、`requirements.txt` | arm64 `.app` / DMG | 保留现有支持，可在当前机器验证真实库与界面 |

Intel Mac 依赖组合最低为 macOS 13（OpenCV 轮子的要求）；实际独立包的最低版本还取决于构建机 Python 与本地库。发布脚本根据原生库审计结果提高包的最低系统版本，不承诺在较老系统上运行。Windows 使用 x64 Python，32 位 Python 与 Windows ARM 原生构建不在本次范围内。

## Intel Mac 安装

安装 Python 3.11 x86_64，在项目目录运行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-macos-intel.txt
.venv/bin/python -m pip check
.venv/bin/python download_models.py
.venv/bin/python check_environment.py --models
.venv/bin/python vigil_app.py
```

不能安装旧要求中的 `torch>=2.6`：PyTorch 官方停止在 2.2 系列之后提供 macOS x86_64 轮子。兼容组合为 Torch 2.2.2、Torchvision 0.17.2、NumPy 1.26.4、MediaPipe 0.10.21、OpenCV 4.11 和 PyObjC Cocoa 10.3.2。NumPy 固定在 1.x，避免旧版 Torch 的 NumPy ABI 不兼容。JAX / jaxlib 固定为具有 Intel Mac 轮子的 0.4.30，以满足旧 MediaPipe 的依赖。

## Windows 安装

在项目目录的 PowerShell 运行；无需更改 PowerShell 执行策略：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe download_models.py
.venv\Scripts\python.exe check_environment.py --models
.venv\Scripts\python.exe vigil_app.py
```

后续可双击 `run_vigilscreen.bat`。CPU 版依赖使用 PyTorch 官方 CPU 下载源，Intel / AMD 共用，不安装 CUDA 库。Windows 使用普通任务栏最小化；命中在窗口中提示并保存红框截图。macOS 顶部眼睛图标及 Peek 菜单栏功能属于 macOS 专用界面，本次没有移植 Windows 托盘。

首次使用摄像头需要允许系统权限；Windows 在“设置 → 隐私和安全性 → 摄像头”中允许桌面应用访问摄像头。摄像头仍只在点击开始监控或打开摄像头后开启。

## CPU 与日志

默认 CPU 线程预算根据可用核心数设定，上限 4；OpenCV 实时检测最多使用其中 2 个线程，避免每个组件各自占满核心造成界面卡顿。可通过 `VIGILSCREEN_CPU_THREADS=1/2/4` 调整；已有 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 配置会保留。这是线程预算控制，实际速度需在目标 CPU 上测量。

训练、验证子进程使用 UTF-8 管道；Windows 不弹命令提示符窗口。windowed EXE 恢复继承的标准输出句柄，避免训练完成却没有日志；若没有管道则回退到用户目录 `logs/identity-worker.log`。

独立应用的数据目录：

- macOS：`~/Library/Application Support/VigilScreen/`
- Windows：`%LOCALAPPDATA%\VigilScreen\`
- 源码运行：项目目录；也可统一通过 `VIGILSCREEN_DATA_DIR` 指定。

训练材料路径和 JSON 记录正确处理反斜杠、中文及空格。原有 ONNX / MediaPipe 模型、身份模板和本项目格式的 CNN 权重无需按 CPU 品牌分别训练；跨机器迁移身份项目时应一并复制对应的材料与模型目录，保留相对路径结构。

## 构建和验证

必须在目标系统、目标架构的 Python 环境构建。Apple Silicon 的 arm64 包不能直接在 Intel Mac 或 Windows 运行，Windows 包也不能在 macOS 上交叉生成。

Intel Mac：

```bash
.venv/bin/python -m pip install 'pyinstaller>=6,<7'
.venv/bin/python build_macos.py --arch x86_64
.venv/bin/python audit_macos.py
VIGILSCREEN_DATA_DIR=/tmp/vigilscreen-package-test \
  dist/VigilScreen.app/Contents/MacOS/VigilScreen --self-test build/standalone-self-test.json
.venv/bin/python finalize_macos.py
```

Windows：

```powershell
.venv\Scripts\python.exe -m pip install "pyinstaller>=6,<7"
.venv\Scripts\python.exe build_windows.py
$env:VIGILSCREEN_DATA_DIR = "$env:TEMP\VigilScreen-package-test"
$process = Start-Process -FilePath "$PWD\dist\VigilScreen\VigilScreen.exe" `
  -ArgumentList @('--self-test', 'build/standalone-self-test.json') -WorkingDirectory $PWD -Wait -PassThru
if ($process.ExitCode -ne 0) { throw '独立程序自检失败' }
Get-Content -Raw -Encoding UTF8 build/standalone-self-test.json
```

Windows 交付整个 `dist/VigilScreen/` 目录，可压缩后解压运行；单独复制 `.exe` 会缺少运行库。构建默认不包含本地训练材料；只有显式使用 `--include-training-data` 才加入。自检不访问真实摄像头；发布前仍需在目标机器检查真实摄像头、身份切换、命中截图和训练验证。

仓库提供 `.github/workflows/compatibility.yml`，在 Intel Mac 和 Windows x64 runner 上安装各自依赖、执行测试并加载真实模型。手动触发时可选择构建并自检独立应用；没有触发或没有成功报告时，不能把这些目标写成已实机通过。

本轮本机验证（2026-11-18）：98 项测试通过；Apple Silicon 上三种真实视觉模型检查和 GUI 联调通过；针对 macOS 13 x86_64 / CPython 3.11、Windows AMD64 / CPython 3.11 的依赖解析试运行均成功。试运行只解析和下载轮子，没有安装目标依赖。Windows 的直接依赖按目标平台条件展开，跨平台解析不能替代原生安装及运行；本轮没有触发远程 CI，没有生成或验证 Intel / Windows 安装包。

依据：[PyTorch Intel Mac 停止发布说明](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690)、[MediaPipe 0.10.21 安装文件](https://pypi.org/project/mediapipe/0.10.21/#files)、[OpenCV 4.11 安装文件](https://pypi.org/project/opencv-contrib-python/4.11.0.86/#files)、[PyInstaller 原生构建限制](https://www.pyinstaller.org/en/stable/)。
