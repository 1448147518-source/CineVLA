# Planner 组合损失

> 代码: `planner_loss(...)` — `core/losses.py:208`

## 公式

$$\mathcal{L}_{\text{planner}}(N_{\text{eff}}) = \lambda_{\text{rot}} \mathcal{L}_{\text{rot}} + \lambda_{\text{trans}} \mathcal{L}_{\text{trans}} + \lambda_{\text{rel}} \mathcal{L}_{\text{rel}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}$$

四个损失项均仅在前 $N_{\text{eff}}$ 帧上计算。

## 权重

| 超参数 | 值 | 作用 |
|--------|-----|------|
| $\lambda_{\text{rot}}$ | 1.0 | 旋转主监督（弧度） |
| $\lambda_{\text{trans}}$ | 0.5 | 平移权重较低，因 1 rad ≈ 归一化后的典型平移量 |
| $\lambda_{\text{rel}}$ | 0.05 | 辅助项——不能压倒绝对位姿主损失 |
| $\lambda_{\text{smooth}}$ | 0.1 | 辅助项——平滑是先验约束，不是目标 |

## 渐进式课程

仅在 Planner 预训练阶段生效。

$$N_{\text{eff}}(\text{epoch}) = N_{\text{start}} + \bigl\lfloor (N_{\text{full}} - N_{\text{start}}) \cdot \min\!\bigl(1,\; \tfrac{\text{epoch}}{\text{ramp\_epochs}}\bigr) \bigr\rfloor$$

| 参数 | 默认值 | 含义 |
|------|--------|------|
| $N_{\text{start}}$ | 10 | 起始轨迹长度 |
| $N_{\text{full}}$ | 30 | 最终轨迹长度（= `pose_length`） |
| `ramp_epochs` | 阶段 epoch 数的 67% | 线性增长持续时长 |

```
epoch 0:  N_eff = 10   只监督前 10 帧 → 学习短程运动模式
epoch 10: N_eff = 20
epoch 20: N_eff = 30   开始监督全部 30 帧 → 学习长程轨迹结构
epoch 29: N_eff = 30   保持全长度
```

## 实现

```python
# train.py — forward_planner
eff_N = getattr(self, 'effective_pose_length', None)  # 由 run_stage 每 epoch 设置
loss, comps = planner_loss(
    out['poses'], gt_poses, effective_N=eff_N, ...
)

# losses.py — planner_loss
if effective_N is not None and effective_N < poses_pred.shape[1]:
    poses_pred = poses_pred[:, :effective_N, :]
    poses_gt   = poses_gt[:, :effective_N, :]
```

## 在训练流程中的位置

此损失在阶段 1（Planner 预训练）和阶段 3（联合训练）中使用。阶段 1 启用渐进课程，阶段 3 使用完整 $N=30$。
