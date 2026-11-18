# VigilScreen

本地摄像头视线监控、图片标记裁剪、身份模型训练和验证的一体应用。默认检查所有人，也可指定身份或排除选中的身份。

仓库包含源码、测试、开发文档、配置及 `face_models/` 中的通用视觉模型、来源清单和许可证。不包含训练图片、个人身份特征、个人训练权重、命中截图、构建产物或本地运行报告。下载脚本可校验通用模型，并在文件缺失或损坏时重新下载。

## 安装与启动

支持为 Intel Mac、Windows Intel / AMD x64 选择独立依赖与原生构建配置，保留 Apple Silicon。当前实际开发验证机器为 Apple Silicon macOS；新增目标需运行本机检查与实机摄像头验证。Intel Mac 使用 Python 3.11 和 `requirements-macos-intel.txt`，Windows 推荐 Python 3.11 和 CPU 版 `requirements-windows.txt`，安装、构建与验证步骤见 [平台适配说明](docs/PLATFORMS.md)。macOS 使用原生菜单栏提示，Windows 在窗口中显示命中信息。

以下为 Apple Silicon macOS 示例；Intel Mac 与 Windows 请使用上面的平台说明，分别安装对应依赖。

```bash
git clone https://github.com/strugglelinux/VigilScreen.git
cd VigilScreen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_models.py
python main.py
```

后续可双击 `run_vigilscreen.command` 启动。macOS 首次使用摄像头需要允许系统权限。

首次使用可按 [操作指南](docs/USER_GUIDE.md) 完成“标记 → 训练并登记 → 验证 → 指定身份监控”。本轮变更及验证范围见 [更新记录](CHANGELOG.md)。

程序启动时不打开摄像头，点击 **开始监控** 才加载模型并采集图像。切换到训练工作室会暂停监控并释放设备，返回后需手动开始监控。

macOS 点击黄色最小化按钮后，窗口隐藏到顶部菜单栏的眼睛托盘图标，Dock 中不显示应用图标或窗口缩略图，已有监控和训练继续运行。点击托盘菜单“显示主窗口”恢复窗口与 Dock 图标，也可通过菜单隐藏或退出。隐藏时平常只显示眼睛图标，命中时继续显示 Peek 数字与范围。系统摄像头隐私指示保持显示，不由应用托盘替换。

## 功能

- **全员或指定身份**：默认无需个人身份模型，对画面中的人脸检查头部朝向和双眼视线；选择已登记身份后，仅该身份可触发。
- **过滤目标**：支持多选排除的已登记身份。先进行身份比对，被排除的人跳过视线检查，不触发 Peek 或命中截图。过滤优先于指定目标，全员启用身份过滤时也需要身份模型。
- **切换目标**：按停止检测、应用选择、恢复检测执行；取消旧画面任务，保留实时预览并在同一会话复用模型。可取消切换并暂停。
- **菜单操作响应**：展开目标下拉菜单时暂缓界面重绘，后台检测继续；收起后显示最新帧。截图编码与写盘不占用目标切换锁，切换期间未提交的旧范围截图自动丢弃。
- **命中提示**：macOS 菜单栏显示 `Peek N · 全员` 或 `Peek N · 指定：姓名`，短时间的新命中累计计数；截图使用红框标出符合命中条件的人脸。
- **监控侧栏**：目标、校准、截图及模型设置支持滚轮和右侧滚动条；开始 / 暂停与退出按钮固定在底部，缩小窗口或显示较长截图名时也不会被挤出窗口。
- **防止持续重复提示**：各人分别维护状态，持续注视只提示一次，连续 3 个处理帧不符合条件后可重新触发。
- **标记与训练**：图片及摄像头画面支持鼠标画框裁剪；摄像头画面可点击冻结。支持单身份、多身份训练，提供曲线、日志、停止任务及身份效果验证。
- **标记页面布局**：参照实时监测页，采用左侧大预览、右侧分组操作面板；统一标题、深色画面底色和间距。右侧包含来源、标记身份、裁剪与尺寸，可左右拖动分隔条调整画面宽度；状态与保存信息显示在画面下方。
- **标记操作**：两个标记页统一显示步骤引导和空画面提示；类别输入可伸缩，保存按钮突出显示。“用于训练”会填入当前标记输出目录并切换到可视化训练页。
- **已有类别 / 身份**：图片和摄像头标记共用类别列表，合并当前输出目录的类别文件夹、标记记录及已登记身份。启动、打开下拉框、更改输出目录或保存标记时自动刷新，保留正在输入的新名称。
- **标记预览大小**：默认优先将可用空间分给图片和摄像头画面，支持等比例放大；手动拖动分隔条后保留调整，保存仍按原图坐标裁剪。
- **训练输出布局**：上下拖动曲线与日志之间的分隔条可调整高度；“放大日志”“均分”可快速调整。日志支持横向、纵向滚动，曲线区域过小时显示展开提示。
- **初始化**：从系统顶部菜单栏“操作 → 初始化…”进入；弹出确认后停止任务，清空本地训练图片、模型、身份特征、训练和验证记录及命中截图，重置目标、过滤名单和视线校准。通用视觉模型及项目外文件保留。
- **监测启动提示**：“点击开始监控后才打开摄像头”仅显示在实时监测页，标记页不占用这部分空间。

### 镜头与屏幕校准

两种注视参考可分别校准，任一种通过均可触发。校准时画面中仅保留一张清晰目标脸，睁眼注视对应位置，收集 5 帧；期间不触发命中。切换目标或重新开始监控后，需要重新校准屏幕视线。

程序根据头部关键点、眼睑和虹膜相对位置近似判断方向，并非精确眼动仪。闭眼、双眼太小或虹膜不可确认时不触发。单点屏幕校准不能保证覆盖整个屏幕。

## 准备身份材料

按身份分目录，选择上层目录：

```text
materials/
├── alice/
│   ├── 01.jpg
│   └── ...
└── bob/
    ├── 01.jpg
    └── ...
```

也支持选择平铺图片目录并填写“单人材料名称”，或提供显式 `train/<身份>/`、`val/<身份>/` 划分。建议每人至少 20 张不同光线、角度和距离的清晰图片；每类最低 4 张，划分后至少 3 张训练图能够提取唯一人脸特征。

训练先固定材料快照，再划分原图；身份特征只使用训练图，留出原图用于验证。训练完成后原子更新模型入口，失败或停止保留先前模型。

分类曲线与身份效果是两个指标：单类别 CNN 分类准确率不能用于判断陌生人排除能力。实时身份匹配使用 SFace 与登记特征，默认相似度阈值 0.50、类别间距阈值 0.06。

### 身份效果验证

外部验证材料可按 `alice/`、`bob/`、`unknown/` 分目录；平铺图片需填写预期身份。留空外部目录时使用训练时的留出原图。

验证记录逐图身份结果、相似度、失败原因及与训练材料是否重复。已知身份匹配率以全部已知验证图为分母；陌生人误接受率只使用可读取、唯一人脸且可提取特征的陌生人图，无样本时不报告。验证衡量身份匹配，不衡量注视准确率。

## 数据位置

源码运行时使用本项目的 `data/`、`checkpoints/` 和 `screenshots/`；这些目录由程序按需创建并被 Git 忽略。克隆仓库后没有已登记身份，默认全员模式可在准备通用模型后使用。

独立 macOS 应用使用 `~/Library/Application Support/VigilScreen/`，Windows 使用 `%LOCALAPPDATA%\VigilScreen\`。也可通过 `VIGILSCREEN_DATA_DIR` 指定数据根目录。用户数据与应用资源分开，初始化后不会重新自动导入内置训练材料。

```text
data/labeled/<身份>/                  默认标记图片
data/labeled_captures/                摄像头裁剪结果归档
checkpoints/identities/active_profile.json  当前身份模型入口
checkpoints/identities/runs/<批次>/     特征、材料快照、训练历史及验证报告
screenshots/                         命中截图
face_models/                         随源码分发的通用视觉模型、校验清单与许可证
```

个人材料、身份模型和验证报告请保留在本地，不提交到仓库。

## 命令行

```bash
# 仅启动监控窗口；点击开始监控后打开摄像头
python chi_camera.py

# 有已登记模型后选择身份
python chi_camera.py --target alice --profile checkpoints/identities/active_profile.json

# 训练、登记与验证
python identity_training.py train --dataset folder --data-dir materials \
  --output-dir checkpoints/identities --epochs 20 --batch-size 8 --device cpu

# 单人平铺材料
python identity_training.py train --dataset folder --data-dir alice_photos --person-name alice \
  --output-dir checkpoints/alice --epochs 20 --device cpu

# 重新验证留出材料
python identity_training.py validate --profile checkpoints/identities/active_profile.json \
  --output checkpoints/identities/recheck.json

# 独立外部验证材料
python identity_training.py validate --profile checkpoints/identities/active_profile.json \
  --images-dir validation --output checkpoints/identities/external.json
```

## 开发与测试

```bash
python -m unittest discover -s tests -v
python download_models.py --check
python check_environment.py --models
```

测试使用合成数据或临时目录，GUI 联调使用模拟摄像头，不访问真实摄像头。测试方法和模块划分见 [开发文档](docs/DEVELOPMENT.md)，独立应用构建见 [DISTRIBUTION.md](DISTRIBUTION.md)，通用模型来源与许可见 [face_models/README.md](face_models/README.md)。
