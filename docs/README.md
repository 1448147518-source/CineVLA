# CineVLA 文档索引

## 损失函数设计

| 文档 | 内容 |
|------|------|
| [loss/overview.md](loss/overview.md) | 设计原则、符号表、v3→v4 变更对照 |
| [loss/rotation.md](loss/rotation.md) | 测地线旋转损失 — SO(3) 流形距离 |
| [loss/translation.md](loss/translation.md) | L1 平移损失 — 解耦监督 |
| [loss/relative-pose.md](loss/relative-pose.md) | 相对位姿一致性损失 — 因果滑动窗口帧对 |
| [loss/smoothness.md](loss/smoothness.md) | 轨迹平滑度损失 — 二阶加速度惩罚 |
| [loss/planner.md](loss/planner.md) | Planner 组合损失 + 渐进式课程 |
| [loss/refiner.md](loss/refiner.md) | Refiner 组合损失 + 变长块训练 |

## 训练策略

| 文档 | 内容 |
|------|------|
| [training/overview.md](training/overview.md) | 三阶段训练流程、保存策略 |
| [training/infrastructure.md](training/infrastructure.md) | 优化器、学习率调度、CFG、混合精度、Fallback |
| [training/parameters.md](training/parameters.md) | 冻结 vs 可学习参数、参数量汇总 |
| [training/gradient-flow.md](training/gradient-flow.md) | 三阶段梯度流向图 |

## 参考

| 文档 | 内容 |
|------|------|
| [hyperparameters.md](hyperparameters.md) | 全部超参数速查表 |
| [references.md](references.md) | 参考文献 |
