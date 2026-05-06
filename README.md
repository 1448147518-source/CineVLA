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
# 帧序列目录
python eval.py default --image_path "shot_0070_frames/" --text "镜头推进..." --resume "ckpt.safetensors"

# MP4 视频
python eval.py default --image_path "shot_0070_video.mp4" --text "镜头推进..." --resume "ckpt.safetensors"

# 带音乐律动（舞蹈拍摄等场景）
python eval.py default --image_path "frames/" --text "..." \
    --music_path "bgm.mp3" --resume "ckpt.safetensors"
```

注意：推理不再支持单张图片输入。请使用 `_frames/` 目录或 `.mp4` 视频文件。

## Benchmark 评估

对测试集批量评估，计算 CLaTr 标准指标（FCD、PRDC、CLaTr Score），结果保存为 CSV。

```bash
python -m evaluate.eval_benchmark --resume ckpt.safetensors --data_path DataDoP/train
```

评估流程：
1. 加载 CineVLA 模型权重
2. 遍历测试集所有样本，逐个生成轨迹
3. 用 `TrajectoryEncoder` 分别编码生成轨迹和 GT 轨迹，得到特征向量
4. 用 CLIP 文本编码器提取文本特征
5. 聚合全部样本特征，计算：

| 指标 | 含义 |
|------|------|
| `clatr/fcd` | Frechet 距离 — 生成轨迹与真实轨迹的分布差异（越低越好） |
| `clatr/precision` | 生成轨迹落在真实流形内的比例 |
| `clatr/recall` | 真实轨迹被生成轨迹覆盖的比例 |
| `clatr/density` | 真实流形邻域内生成样本密度 |
| `clatr/coverage` | 真实样本被生成样本覆盖的比例 |
| `clatr/clatr_score` | 轨迹-文本余弦对齐度（0–100，越高越好） |

结果输出：`./metrics/benchmark.csv`

### 自行训练 TrajectoryEncoder

首次使用时 `TrajectoryEncoder` 为随机初始化，可添加训练损失使其学到更好的轨迹特征表示。将编码器权重保存为 `*_traj_enc.safetensors` 与 checkpoint 同名，评估脚本会自动加载。

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
│   ├── shot_0070_video.mp4       # 视频文件（与 _frames/ 二选一，同时存在优先 video）
│   └── shot_0070_frames/         # 多帧序列目录（与 video 二选一）
│       ├── 000.png
│       ├── 001.png
│       └── ...
├── 1_0001/
│   └── ...
└── ...
```

每个样本**必须**包含 `_video.mp4` 或 `_frames/` 目录之一（两者都有时优先使用 video）。不再支持从单张图片生成伪序列的降级模式；缺少两者时训练/推理直接报错退出。

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

**`_frames/` 目录**

放多个 PNG 帧文件（文件名排序后按序读取，取前 8 帧）。必须至少包含 8 帧，不足则报错退出。不进行任何补帧或伪序列生成。

**`_video.mp4` 文件（与 `_frames/` 二选一）**

MP4 视频文件，自动等距抽取 8 帧。视频总帧数必须 ≥ 8，否则报错退出。若同时存在 `_video.mp4` 和 `_frames/`，优先使用视频。

