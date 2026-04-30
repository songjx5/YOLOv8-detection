# YOLOv8 能效标签检测项目

本项目用于能效标签相关目标检测（`energy_arrow / label / box / stain / fold / brakeage`），提供训练、单图推理、摄像头实时推理和 GUI 检测界面。

## 1. 环境配置

### 1.1 前置要求

- Python 3.11+
- Windows（当前工程已在 Windows 路径和字体环境下适配）
- 建议使用 [uv](https://docs.astral.sh/uv/)

### 1.2 创建并同步环境（推荐）

在项目根目录执行：

```powershell
uv sync --extra gpu
```

如果只用 CPU：

```powershell
uv sync --extra cpu
```

如需 OCR 相关依赖（可选）：

```powershell
uv sync --extra gpu --extra ocr
```

> 当前 `pyproject.toml` 中已固定了与 OCR 兼容的版本（如 `opencv-python==4.6.0.66`、`numpy<2.0`）。

---

## 2. 运行命令

### 2.1 GUI 检测（推荐入口）

```powershell
uv run python gui_detector.py
```

### 2.2 训练

```powershell
uv run python train.py
```

> `train.py` 里模型加载路径是写死的，请按你的环境修改（例如改成 `yolo26n.pt` 或你自己的 `best.pt`）。

### 2.3 单图推理

```powershell
uv run python run_image.py
```

### 2.4 摄像头实时推理

```powershell
uv run python run.py
```

> `run.py` 里 `model_path` 也是写死路径，运行前请先改成你的模型路径。

---

## 3. 数据集结构

当前数据集位于 `resource/`，结构如下：

```text
resource/
├─ data.yaml
├─ classes.txt
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ train/
   ├─ val/
   ├─ train.cache
   └─ val.cache
```

`resource/data.yaml` 当前配置：

```yaml
path: D:\project\Python\YOLO_code\resource
train: images/train
val: images/val
nc: 6
names:
  - energy_arrow
  - label
  - box
  - stain
  - fold
  - brakeage
```

标签文件格式为 YOLO 常规格式（每行）：

```text
class_id x_center y_center width height
```

坐标为相对比例（0~1）。

---

## 4. 导出 requirements.txt

### 4.1 从 `uv.lock` 导出（推荐，可复现）

只导出基础依赖：

```powershell
uv export --format requirements-txt --output-file requirements.txt
```

导出含 GPU + OCR 的完整依赖：

```powershell
uv export --format requirements-txt --all-extras --output-file requirements-full.txt
```

### 4.2 从当前虚拟环境冻结（备选）

```powershell
.venv\Scripts\python.exe -m pip freeze > requirements-freeze.txt
```

---

## 5. 目录说明（主要文件）

- `gui_detector.py`：GUI 检测主程序（当前版本含颜色识别逻辑）
- `train.py`：训练脚本
- `run_image.py`：单图推理脚本
- `run.py`：摄像头实时推理脚本
- `resource/`：数据集目录
- `util/energy_grade_analyzer.py`：能效等级颜色分析模块
- `pyproject.toml`：依赖与项目配置

