# 训练流程

## 三阶段概览

```
阶段 1: Planner 预训练  (30 epochs)
    ├── 训练: Perception + Planner
    ├── Refiner 不参与前向
    ├── 渐进课程: N_eff 从 10 线性增长到 30
    └── 目标: 从感知+文本+音乐生成准确的初始轨迹

阶段 2: Refiner 预训练 (20 epochs)
    ├── 训练: Perception + Refiner
    ├── Planner 产生的候选轨迹 detach（不接收梯度）
    ├── 随机采样 chunk: 起点 t 随机, 长度 K ∈ [1, 5]
    └── 目标: 用因果观测修正 Planner 的候选轨迹，并预测下一观测特征

阶段 3: 联合训练   (50 epochs)
    ├── 训练: Perception + Planner + Refiner 全部可学习参数
    ├── 两条前向路径: loss_p + loss_r
    └── 目标: 两个模块通过共享的 Perception 协同优化
```

## 日志追踪

每个阶段独立追踪的 loss 分量：

| 阶段 | keys | 日志内容 |
|------|------|---------|
| 1 — Planner | `loss`, `L_rot`, `L_trans`, `L_rel`, `L_smooth` | 总 loss + 4 个分量 |
| 2 — Refiner | `loss`, `loss_pose`, `loss_z`, `loss_rel`, `loss_smooth` | 总 loss + 4 个分量 |
| 3 — Joint | `loss`, `loss_p`, `loss_r` | 总 loss + Planner/Refiner 各自总 loss |

Joint 阶段的详细分量（`p_L_rot`, `r_loss_pose` 等）可通过 per-step wandb logging 查看。

## 保存策略

```python
if avg_loss < best_loss:                    # 仅当 epoch 平均 loss 改进时
    acc.wait_for_everyone()                 # 多 GPU 同步
    save_file(model.state_dict(),
              'workspace/exp_name/best.safetensors')

# 训练结束后保存最终模型
save_file(model.state_dict(),
          'workspace/exp_name/model.safetensors')
```

保存格式为 safetensors（安全、快速、无 pickle 风险）。
