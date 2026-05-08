# 三阶段梯度流向

## 阶段 1 — Planner 预训练

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  frames ──→ Perception ──→ perc ──→ Planner ──→ loss        │
│              │                       │                       │
│           ┌──┴──┐                 ┌──┴──┐                    │
│           │CLIP │ 冻结             │Text │ 冻结              │
│           │ViT  │                  │Enc  │                   │
│           └─────┘                 └─────┘                    │
│           frame_proj    ✓         z_proj         ✓           │
│           temporal      ✓         MusicEncoder   ✓           │
│           temporal_norm ✓         text_proj      ✓           │
│                                   music_proj     ✓           │
│                                   traj_queries   ✓           │
│                                   pos_embed      ✓           │
│                                   6 × layers     ✓           │
│                                   pose_head      ✓           │
│                                                              │
│  梯度: loss → Planner (trainable) → Perception (trainable)   │
└──────────────────────────────────────────────────────────────┘
```

## 阶段 2 — Refiner 预训练

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  frames ──→ Perception ──→ perc ──→ Refiner ──→ loss        │
│              │                       │                       │
│           ┌──┴──┐                 ┌──┴──┐                    │
│           │CLIP │ 冻结             │全部 │ 全部可学习        │
│           │ViT  │                  │参数 │                   │
│           └─────┘                 └─────┘                    │
│           frame_proj    ✓         z_proj, pose_proj ✓        │
│           temporal      ✓         text_proj       ✓          │
│           temporal_norm ✓         error_encoder   ✓          │
│                                   transformer     ✓          │
│  texts ──→ Planner.encode_text() ──→ text_feats             │
│              (冻结 text_encoder, 无梯度)                     │
│                                                              │
│  Planner 的 trainable params 不参与此阶段                    │
│  梯度: loss → Refiner (全部) → Perception (trainable)        │
└──────────────────────────────────────────────────────────────┘
```

## 阶段 3 — 联合训练

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  路径 A: frames → Perception → Planner → p_loss             │
│  路径 B: frames → Perception → Refiner → r_loss             │
│                                                              │
│            total_loss = p_loss + r_loss                      │
│                                                              │
│  Perception 同时接收来自两条路径的梯度（叠加）               │
│  Planner    仅接收 p_loss 的梯度                             │
│  Refiner    仅接收 r_loss 的梯度                             │
│  CLIP ViT + TextEncoder 始终无梯度                           │
│                                                              │
│  梯度:                                                       │
│    loss → Planner (trainable) + Refiner (trainable)          │
│         → Perception (trainable, 两条路径叠加)               │
└──────────────────────────────────────────────────────────────┘
```

## 关键点

- CLIP ViT 和 CLIP TextModel 在所有阶段始终冻结，不参与梯度计算
- 阶段 2 中 Planner 的 trainable 参数（如 z_proj、pose_head 等）不参与——`encode_text` 仅调用冻结的 text_encoder
- 阶段 3 中 Perception 同时接收两个 loss 的梯度，形成端到端的联合优化
