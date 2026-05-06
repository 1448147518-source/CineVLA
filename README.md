# CineVLA: 闭环视觉反馈相机轨迹生成

CineVLA 是一个闭环视觉-语言-动作系统。模型在每一步运动后真正"看一眼"附近的 RGB 环境，用这个真实的视觉信号来修正后续轨迹。

## 架构

```
Phase 1: 初始规划
  image_0 + text + music → Planner → 初始轨迹 [p₁...p₃₀]

Phase 2: 闭环执行（每步循环）
  camera 走到 p_t
    ↓
  捕捉 image_t（真实环境 RGB）
    ↓
  Perception Encoder → z_t（真实感知）
    ↓
  比较 z_t vs 预测的 ẑ_t → 感知误差
    ↓
  Refiner → 修正剩余轨迹 + 预测下一帧环境
    ↓
  camera 走到 refined p_{t+1}
```

## 三个组件

| 组件 | 功能 |
|------|------|
| Perception Encoder | CLIP ViT-B/32，每步编码当前摄像头 RGB → 环境潜在向量 z |
| Planner | 因果 Transformer，从初始帧 + 文本 + 音乐生成初始轨迹 |
| Refiner | 轻量 Transformer，用真实感知 vs 预测感知的误差修正后续轨迹 |

## 安装

```bash
conda create --name cinevla python=3.10
conda activate cinevla
pip install -r requirements.txt --break-system-packages
```

## 训练

```bash
# 三阶段：Planner → Refiner → Joint
accelerate launch --config_file acc_configs/gpu1.yaml train.py default \
    --workspace workspace --exp_name run1
```

## 推理

```bash
python eval.py default \
    --resume "workspace/run1/model.safetensors" \
    --image_path "scene.jpg" \
    --text "镜头缓缓推进... "
```

## 数据格式

每个样本需要：`_rgb.png` + `_depth.npy` + `_caption.json` + `_transforms_cleaning.json`
