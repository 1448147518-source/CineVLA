# CineVLA: 闭环视觉反馈相机轨迹生成

CineVLA 是一个闭环视觉-语言-动作系统。模型在每一步运动后真正"看一眼"附近的 RGB 环境，用这个真实的视觉信号来修正后续轨迹。


![Teaser Image](./assets/CineVLA.png)

## 架构

```
Phase 1: 初始规划
  RGB 帧序列 + 文本 [+ 音乐] → Video Perception → Planner → 初始轨迹

Phase 2: 闭环执行
  camera 走到 p_t → 捕捉当前帧
  Perception(当前帧) → z_t（真实环境感知）
  z_t vs 预测的 ẑ_t → 感知误差
  Refiner → 修正剩余轨迹 + 预测下一帧状态
  camera 走到修正后的 p_{t+1}
```

音乐可选，仅在 Planner 顶层交叉注意力中注入节奏特征，使轨迹跟随节拍律动。

## 组件

| 组件 | 功能 |
|------|------|
| Video Perception | CLIP ViT-B/32 + 时序 Transformer，从 RGB 帧序列中提取 3D 感知特征 |
| Planner | 因果 Transformer，从帧序列特征 + 文本 + 音乐节奏生成初始轨迹 |
| Music Encoder | librosa 提取 BPM/节拍/onset，编码为 30 帧节奏特征 |
| Refiner | 轻量 Transformer，用真实帧特征 vs 预测特征的误差修正轨迹 |

## 安装

```bash
conda create --name cinevla python=3.10
conda activate cinevla
pip install -r requirements.txt --break-system-packages
```

## 训练

```bash
python train.py default --workspace workspace --exp_name run1
```

## 推理

```bash
# 单张图片（不含音乐）
python eval.py default --image_path "scene.jpg" --text "镜头推进..." --resume "ckpt.safetensors"

# 带音乐律动（舞蹈拍摄等场景）
python eval.py default --image_path "scene.jpg" --text "..." \
    --music_path "bgm.mp3" --resume "ckpt.safetensors"

# 帧序列 / MP4 视频
python eval.py default --image_path "frames/" --text "..." --resume "ckpt.safetensors"
python eval.py default --image_path "video.mp4" --text "..." --resume "ckpt.safetensors"
```

## 可视化

CineVLA 提供两个独立的可视化工具，位于 `visualise/` 目录下。

### 轨迹可视化（推理阶段自动开启）

推理结束后自动生成，结果保存在 `./results/`：

- `trajectory.png` — 3D 相机轨迹图，包含路径连线、相机坐标框架（RGB 三轴）、起止点标注、SLERP 稠密插值路径
- `refinement_error.png` — 闭环修正过程中每步的感知误差（MSE），虚线标注 0.01 修正阈值

```bash
python eval.py default --image_path "scene.jpg" --text "镜头推进..." --resume "ckpt.safetensors"
# 输出 → ./results/trajectory.png  +  refinement_error.png
```

### 潜态环境变化可视化（训练/推理可选）

通过 PCA 将 512 维环境感知特征降至 2 维，对比模型预测的潜态 `z_pred` 与实际观测的 `z_real`。启用后结果保存在 `./pred_latent/`：

- `latent_space.png` — PCA 二维散点图，z_real（蓝色圆）与 z_pred（红色叉）用灰线连接，左右分栏展示训练/推理阶段
- `latent_error.png` — 预测误差（MSE）随时步变化，对数纵坐标

**推理时启用：**

```bash
python eval.py default --image_path "scene.jpg" --text "..." --resume "ckpt.safetensors" --vis_latent
```

**训练时启用：**

```bash
python train.py default --workspace workspace --exp_name run1 \
    --vis_latent --vis_latent_every 50
```

`--vis_latent_every` 控制每多少步记录一次潜态数据（默认 100），训练结束后自动生成 PCA 总结图。只有 refiner 预训练阶段和联合训练阶段会记录潜态（planner 预训练阶段不含 z_pred/z_real 对比）。

### 输出目录

```
CineVLA/
├── results/               # 轨迹可视化（推理时自动生成）
│   ├── trajectory.png
│   └── refinement_error.png
├── pred_latent/           # 潜态可视化（--vis_latent 启用后生成）
│   ├── latent_space.png
│   └── latent_error.png
└── visualise/             # 可视化源码
    ├── __init__.py
    ├── trajectory.py
    └── latent.py
```

## 数据集结构

```
dataset/
├── train_valid.txt              # 一行一个样本路径: {VideoID}/{ShotID}
├── 1_0000/                      # VideoID 目录
│   ├── shot_0070_rgb.png        # 第一帧 RGB 图像（任意分辨率，训练时 resize 到 224×224）
│   ├── shot_0070_caption.json   # 文本描述
│   ├── shot_0070_transforms_cleaning.json  # 相机轨迹
│   ├── shot_0070_music.mp3       # （可选）音乐文件
│   └── shot_0070_frames/         # （可选）多帧序列目录
│       ├── 000.png
│       ├── 001.png
│       └── ...
├── 1_0001/
│   └── ...
└── ...
```

### 文件格式

**`_caption.json`**

```json
{
    "Movement": "相机右移后上仰...",
    "Detailed Interaction": "...",
    "Concise Interaction": "简洁的导演意图描述"
}
```

训练时随机选一个 key，优先 `Concise Interaction`。

**`_transforms_cleaning.json`**

```json
{
    "w": 1920,
    "h": 1080,
    "fl_x": 1200.0, "fl_y": 1200.0,
    "cx": 960.0,  "cy": 540.0,
    "frames": [
        {"transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
        ...
    ]
}
```

`frames` 数组至少 120 帧（等距采样 30 帧做轨迹）。`transform_matrix` 为 4×4 c2w 矩阵。

**`_frames/` 目录（可选）**

放多个 PNG 帧文件（文件名排序后按序读取，取前 8 帧）。如果不存在该目录，训练时自动通过随机裁剪 + 亮度扰动从单帧生成伪多帧序列。

