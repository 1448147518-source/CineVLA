# Refiner 组合损失

> 代码: `refiner_loss(...)` — `core/losses.py:264`

## 公式

$$\mathcal{L}_{\text{refiner}} = \lambda_{\text{pose}} \mathcal{L}_{\text{pose-delta}} + \lambda_{z} \mathcal{L}_{\text{z-pred}} + \lambda_{\text{rel}}^{(r)} \mathcal{L}_{\text{rel}}^{(r)} + \lambda_{\text{smooth}}^{(r)} \mathcal{L}_{\text{smooth}}^{(r)}$$

## 修正后轨迹 L1 监督

$$\mathcal{L}_{\text{pose}} = \frac{1}{K \cdot 7} \sum_{k=1}^{K} \bigl\|\hat{P}^{(r)}_{t+k} - P^*_{t+k}\bigr\|_1$$

训练输入 $\hat{P}_{t+k}$ 来自 Planner，监督目标是真值轨迹块 $P^*_{t+k}$。这样 Refiner 学习纠正 Planner 的真实误差，而不是在真值输入上退化为恒等映射。

用 L1 而非 MSE 的原因：L1 对偶尔较大的修正量更宽容，不会平方放大惩罚。

## 感知特征预测

$$\mathcal{L}_{\text{z-pred}} = \bigl\|\hat{z}_{t+1} - z_{t+1}\bigr\|_2^2$$

Refiner 从上下文中预测下一帧的感知特征。这个辅助任务迫使 Refiner 学习环境的几何结构。

## 修正后轨迹的几何约束

$\mathcal{L}_{\text{rel}}^{(r)}$ 和 $\mathcal{L}_{\text{smooth}}^{(r)}$ 与基础版本相同，作用在 $K$ 帧修正后块上，确保 Refiner 的修正不会破坏局部几何一致性或引入抖动。

当 $K < 2$ 时相对一致性返回零，$K < 3$ 时平滑度返回零。

## 权重

| 超参数 | 值 | 为什么比 Planner 低 |
|--------|-----|---------------------|
| $\lambda_{\text{pose}}$ | 1.0 | 主要正则项 |
| $\lambda_z$ | 0.1 | 辅助任务（与 v3 保持不变） |
| $\lambda_{\text{rel}}^{(r)}$ | 0.02 | 块内帧对数稀疏（K ≤ 5），信号较弱 |
| $\lambda_{\text{smooth}}^{(r)}$ | 0.05 | 帧数少，二阶约束信号更弱 |

## 变长块训练

```python
for each batch:
    t = randint(0, N-2)                              # 已观测时刻
    K = randint(1, min(refiner_lookahead, N-1-t))    # 随机块长度 1~5

    z_real     = perc['features'][:, frame_idx, :]   # 帧 t 的感知特征（实际）
    z_pred     = perc['features'][:, max(frame_idx-1, 0), :]  # 仅历史观测
    planned    = planner(perc, text)[:, t+1:t+1+K, :] # Planner 的未来预测
    target     = gt_poses[:, t+1:t+1+K, :]            # GT 未来轨迹
    z_target   = perc['features'][:, next_frame_idx, :]       # z_{t+1}

    refined, z_next = refiner(z_real, z_pred, planned, text_feats)
```

Refiner 学会处理 1~5 帧的不同长度剩余轨迹，推理时无论剩余多少帧都能处理。

## 在训练流程中的位置

此损失在阶段 2（Refiner 预训练）和阶段 3（联合训练）中使用。阶段 2 中 Planner 预测轨迹会 detach；阶段 3 中 Refiner loss 会反传到 Planner，使二者共同适配。
