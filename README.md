# Facial Emotion Recognition

本地实时面部表情识别：支持电脑摄像头和视频文件切换，中文界面，暂停／继续、停止、镜像预览及八类分数显示。

## 算法

YuNet 人脸检测 → RGB 224×224 归一化 → EmotiEffNet-B0 ONNX 分类 → IoU 匹配与时间平滑 → 界面显示。使用 ONNX Runtime CPU 推理，不需要 CUDA。

八类：愤怒、轻蔑、厌恶、恐惧、高兴、中性、悲伤、惊讶。低于阈值时显示“不确定”。模型分数不是正确率，也不代表真实内心情绪。没有统一基准证明本算法最优。

## 安装与运行（Windows，Python 3.12）

在仓库目录打开 PowerShell：

```powershell
py -3.12 -m venv D:\fer_experiments\venv
& D:\fer_experiments\venv\Scripts\python.exe -m pip install -r realtime_fer/requirements.txt
& D:\fer_experiments\venv\Scripts\python.exe -m realtime_fer.download_models
& D:\fer_experiments\venv\Scripts\python.exe -m realtime_fer.app
```

安装完成后也可双击 `启动实时表情识别.bat`。默认模型目录为 `D:\fer_experiments\models\realtime`。无 D 盘时自行创建 `.venv`，下载和运行都传入 `--models models/realtime`；双击脚本仅适用于默认 D 盘环境。

摄像头默认“自动”尝试编号 0–3。视频选择文件后开始；停止后可以切换来源。视频逐帧处理，推理较慢时播放会减速，不播放音轨。最多处理四张脸，分数面板显示最大人脸。

## 核心文件

- `realtime_fer/app.py`：中文界面与摄像头／视频读取。
- `realtime_fer/engine.py`：检测、分类和时间平滑。
- `realtime_fer/download_models.py`：下载公开权重并记录 SHA-256。
- `realtime_fer/requirements.txt`：推理依赖。
- `tests/test_realtime.py`：无需模型或摄像头的单元测试。

```powershell
& D:\fer_experiments\venv\Scripts\python.exe -m unittest discover -s tests -p test_realtime.py -v
```

本地七项单元测试通过，真实 ONNX 输出及空白帧检查通过。这些是功能检查，不是识别精度评测。

## 常见问题

- 摄像头打不开：选择“自动”或编号 0，关闭占用设备的软件，检查 Windows 桌面应用相机权限。
- 双击启动失败：检查默认 Python 路径和依赖；BAT 必须使用 CRLF 换行（`.gitattributes` 已配置）。从 PowerShell 启动可查看错误文本。
- 只显示部分类别：关闭旧窗口再启动新版，必要时最大化窗口。
- 模型校验失败：保留旧文件，指定新的 `--models` 目录重新下载；不要手改哈希。

## 来源与范围

模型来自 [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib)，检测器来自 [OpenCV Zoo / YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)。请遵守上游模型、代码和数据适用条款。本仓库暂未指定统一开源许可证。

本次只保存实时识别核心文件。模型权重、数据集、虚拟环境和用户视频不上传。应用本地处理，不录制或上传摄像头画面；错误日志保存在被 Git 忽略的 `realtime_output` 中。

## 工作流与基线实验总结

- [实时识别工作流](docs/WORKFLOW.md)：输入、检测、分类、时间平滑、界面和验证状态。
- [基线对比实验](docs/BASELINE_EXPERIMENTS.md)：五组基线、消融、统一配置、评估指标及结果状态。

基线训练尚未完成，文档没有虚构精度；训练命令针对本地完整项目，当前仓库不包含 `ferlab`。
