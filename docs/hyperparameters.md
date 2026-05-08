# 超参数速查表

## 轨迹

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pose_length` | 30 | 轨迹帧数 |
| `pose_dim` | 7 | 位姿维度（四元数 4 + 平移 3） |
| `dense_frames` | 120 | SLERP 稠密插值帧数 |

## 感知

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `perception_dim` | 512 | 感知隐空间维度 |
| `image_size` | 224 | 输入图像尺寸 |
| `num_frames` | 8 | 输入帧序列长度 |

## 模型结构

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `planner_hidden_dim` | 256 | Planner Transformer 隐层维度 |
| `planner_num_layers` | 6 | Planner 解码器层数 |
| `planner_num_heads` | 4 | Planner 注意力头数 |
| `planner_text_ca_layers` | 3 | 文本交叉注意力的层数 |
| `refiner_hidden_dim` | 256 | Refiner Transformer 隐层维度 |
| `refiner_num_layers` | 4 | Refiner 编码器层数 |
| `refiner_num_heads` | 4 | Refiner 注意力头数 |
| `refiner_lookahead` | 5 | 每次修正的最大帧数 |
| `music_dim` | 128 | 音乐特征维度 |
| `music_seq_len` | 30 | 音乐序列长度 |
| `music_ca_layers` | 2 | 音乐交叉注意力的 Planner 顶层数 |

## 训练

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 4 | Micro-batch 大小 |
| `lr` | 1e-4 | 学习率 |
| `planner_pretrain_epochs` | 30 | 阶段 1 epoch 数 |
| `refiner_pretrain_epochs` | 20 | 阶段 2 epoch 数 |
| `joint_epochs` | 50 | 阶段 3 epoch 数 |
| `grad_accum` | 4 | 梯度累积步数 |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |
| `warmup_ratio` | 0.05 | LR warmup 占比 |
| `mixed_precision` | bf16 | 混合精度类型 |

## Loss — Planner

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_rot` | 1.0 | 测地线旋转损失权重 |
| `lambda_trans` | 0.5 | L1 平移损失权重 |
| `lambda_rel` | 0.05 | 相对位姿一致性权重 |
| `lambda_smooth` | 0.1 | 轨迹平滑度权重 |
| `rel_window_size` | 5 | 因果滑动窗口大小 |
| `lambda_rot_smooth` | 0.5 | 平滑项中旋转权重 |
| `lambda_rel_t` | 0.1 | 相对项中平移权重 |

## Loss — Refiner

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_pose_delta` | 1.0 | 修正量 L1 正则权重 |
| `lambda_z_pred` | 0.1 | 感知特征预测权重 |
| `lambda_rel_ref` | 0.02 | 修正轨迹相对一致性 |
| `lambda_smooth_ref` | 0.05 | 修正轨迹平滑度 |

## 课程

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pose_start_len` | 10 | 起始轨迹长度 |
| `curriculum_ramp_epochs` | None | 增长周期（None = 阶段 epoch 的 67%） |

## 推理

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `closed_loop_steps` | 30 | 最大闭环步数 |
| `cfg_scale` | 2.0 | CFG 引导强度 |

## 可视化

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vis_latent` | False | 是否启用隐空间 PCA 可视化 |
| `vis_latent_every` | 100 | 每 N 步记录一次隐状态 |
