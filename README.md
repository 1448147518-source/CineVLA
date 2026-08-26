# CineVLA: 闭环视觉反馈相机轨迹生成

CineVLA 是一个闭环视觉-语言-动作系统。模型在每一步运动后真正"看一眼"附近的 RGB 环境，用这个真实的视觉信号来修正后续轨迹。


![Teaser Image](./assets/CineVLA.png)

## 架构

```
Phase 1: 初始规划（因果）
  frame_0（仅第一帧） + 文本 → Video Perception(causal) → Planner → 初始轨迹

Phase 2: 闭环执行（因果）
  camera 走到 p_t → 捕捉当前帧 img_t
  Perception(img_0...img_t, causal) → z_t（仅感知 ≤t 帧，无未来信息泄露）
  z_t vs 上一步预测的 ẑ_t → 感知误差
  Refiner → 修正剩余轨迹 + 预测下一帧状态 ẑ_{t+1}
  camera 走到修正后的 p_{t+1}
```

时序 Transformer 采用**因果掩码**：frame_i 的自注意力仅允许看到 frame_0...frame_i，无法注意到后续帧。这保证了闭环修正中每一步的感知严格依赖于已观测帧。离线评测将视频帧按时间顺序提供给控制器；部署时应由相机/仿真器在每步执行后追加新观测，而不是预先提供未来帧。

推理遵循统一的 `CameraEnvironment` 接口：`reset() → observe → plan → step(pose) → observe`。当前默认的 `OfflineReplayEnv` 逐帧回放真实视频，适合验证因果 rollout 与日志链路；它不会按提交的相机位姿重新渲染画面，因此不能作为 action-conditioned 闭环性能的实验结论。后续可替换为 Blender、Unity 或真实相机适配器。

渲染器/真实相机适配规范见 [环境接口文档](docs/environment.md)。

## 组件

| 组件 | 功能 |
|------|------|
| Video Perception | CLIP ViT-B/32 + 时序 Transformer，从 RGB 帧序列中提取 3D 感知特征 |
| Planner | 因果 Transformer，从帧序列特征 + 文本生成初始轨迹 |
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

训练会以固定随机种子切分 train/validation，并在每个 epoch 写入验证损失。每个阶段各自保存 `best_{stage}.safetensors`；联合训练阶段的最佳模型同时保存为 `best.safetensors`。`last.safetensors` 每个 epoch 覆盖保存，旁边的 `last.training.pt` 保存优化器、学习率调度器、阶段/epoch、随机数状态；以 `--resume path/to/last.safetensors` 恢复时会自动续训。`config.json` 记录完整实验配置。

## 推理

```bash
# 帧序列目录
python eval.py default --image_path "shot_0070_frames/" --text "镜头推进..." --resume "ckpt.safetensors"

# MP4 视频
python eval.py default --image_path "shot_0070_video.mp4" --text "镜头推进..." --resume "ckpt.safetensors"

```

注意：推理不再支持单张图片输入。请使用 `_frames/` 目录或 `.mp4` 视频文件。

## Benchmark 评估

对测试集批量评估，默认计算可解释的相机位姿误差；若提供经过训练的轨迹-文本编码器，额外计算 CLaTr 标准分布指标，结果保存为 CSV。

```bash
python -m evaluate.eval_benchmark --resume ckpt.safetensors --data_path DataDoP/train
```

评估流程：
1. 加载 CineVLA 模型权重
2. 遍历测试集所有样本，逐个生成轨迹
3. 计算旋转和位移的直接误差（主指标）
4. 可选：用已训练的 `TrajectoryEncoder` 聚合分布指标

| 指标 | 含义 |
| `pose/rotation_deg` | 四元数测地旋转误差（度，越低越好） |
| `pose/translation_l1` / `pose/translation_l2` | 数据集归一化坐标系中的位移误差（越低越好） |
|------|------|
| `clatr/fcd` | Frechet 距离 — 生成轨迹与真实轨迹的分布差异（越低越好） |
| `clatr/precision` | 生成轨迹落在真实流形内的比例 |
| `clatr/recall` | 真实轨迹被生成轨迹覆盖的比例 |
| `clatr/density` | 真实流形邻域内生成样本密度 |
| `clatr/coverage` | 真实样本被生成样本覆盖的比例 |
| `clatr/clatr_score` | 轨迹-文本余弦对齐度（0–100，越高越好） |
| `captions/precision` | 运动模式分割精确率 — 生成轨迹中正确复现 GT 运动模式的比例 |
| `captions/recall` | 运动模式分割召回率 — GT 运动模式被生成轨迹覆盖的比例 |
| `captions/fscore` | 运动模式分割 F1 分数 |

结果输出：`./metrics/benchmark.csv`

### 可选的 TrajectoryEncoder 指标

随机初始化的 `TrajectoryEncoder` 不具有评估含义，因此不会再自动启用。只有在独立训练并固定该编码器后，才通过 `--trajectory_encoder path/to/encoder.safetensors` 启用 FCD、PRDC、CLaTr Score；论文中应说明其训练数据与冻结策略。

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

`frames` 数组至少 120 帧（等距采样 30 帧做轨迹）。`transform_matrix` 为 4×4 c2w 矩阵；焦距字段使用 `fl_x`、`fl_y`（兼容历史数据中的 `fy`）。

**`_frames/` 目录**

放多个 PNG 帧文件（文件名排序后按序读取，取前 8 帧）。必须至少包含 8 帧，不足则报错退出。不进行任何补帧或伪序列生成。

**`_video.mp4` 文件（与 `_frames/` 二选一）**

MP4 视频文件，自动等距抽取 8 帧。视频总帧数必须 ≥ 8，否则报错退出。若同时存在 `_video.mp4` 和 `_frames/`，优先使用视频。

---

> **README 维护约定**：所有新增功能均需同步更新本文档。命令行参数、模块结构、输出路径等如有变动，须在对应章节反映。
