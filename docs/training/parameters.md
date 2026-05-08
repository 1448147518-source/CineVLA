# 冻结 vs 可学习参数

## Perception — VideoPerceptionEncoder

| 子模块 | 参数量级 | 状态 | 原因 |
|--------|---------|------|------|
| CLIP ViT-B/32 | ~88M | **冻结** | 预训练视觉特征已足够好 |
| frame_proj | ~0.5M | 可学习 | 将 CLIP 512 维投影到任务空间 |
| temporal_pos | ~0.1M | 可学习 | 时序位置编码 |
| temporal (3 层 Transformer) | ~3M | 可学习 | 跨帧时序聚合 |
| temporal_norm | ~2K | 可学习 | LayerNorm |

## Planner

| 子模块 | 参数量级 | 状态 | 原因 |
|--------|---------|------|------|
| CLIP TextModel (ViT-L/14) | ~123M | **冻结** | 预训练文本语义已足够 |
| MusicEncoder | ~2M | 可学习 | 需学习节奏→轨迹映射 |
| z_proj | ~0.1M | 可学习 | 感知特征投影 (512→256) |
| text_proj | ~0.2M | 可学习 | 文本特征投影 (768→256) |
| music_proj | ~0.03M | 可学习 | 音乐特征投影 (128→256) |
| traj_queries | ~0.01M | 可学习 | 30 个轨迹查询 token |
| pos_embed | ~0.01M | 可学习 | 位置编码 |
| 6 × _PlannerLayer | ~8M | 可学习 | 因果 Transformer 解码器 |
| pose_head | ~0.2M | 可学习 | 输出投影 (256→256→7) |

## Refiner

所有参数均可学习：

| 子模块 | 参数量级 |
|--------|---------|
| z_proj | ~0.1M |
| pose_proj | ~0.01M |
| text_proj | ~0.2M |
| error_encoder (2 层 MLP) | ~0.3M |
| transformer (4 层) | ~6M |
| text_cross + text_norm | ~0.2M |
| pose_delta_head | ~0.1M |
| z_pred_head | ~0.1M |
| pos_embed | ~0.02M |

## 参数量汇总

| 类别 | 参数量 | 占比 |
|------|--------|------|
| 冻结 | ~211M (CLIP ViT + CLIP Text) | ~95% |
| 可学习 | ~22M (投影层 + Transformer + 输出头) | ~5% |

冻结大模型（CLIP），训练小模型。这是典型的 parameter-efficient fine-tuning 策略。
