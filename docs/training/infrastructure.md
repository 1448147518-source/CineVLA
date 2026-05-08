# 训练基础设施

## 优化器与学习率

| 组件 | 配置 |
|------|------|
| 优化器 | AdamW, lr=1e-4, weight_decay=0.01 |
| 学习率调度 | Cosine 衰减 + 5% 线性 warmup |
| 总步数 | `(30 + 20 + 50) × len(dataloader)` |

```python
def lr_lambda(step):
    p = step / total_steps                    # 0 → 1
    if p < 0.05:                              # 前 5% 线性 warmup: 0 → base_lr
        return p / 0.05
    p = (p - 0.05) / 0.95                     # 后 95% cosine 衰减: base_lr → 0
    return 0.5 * (1 + cos(π * p))
```

## 批量与梯度

| 组件 | 配置 |
|------|------|
| Micro-batch | 4 |
| 梯度累积步数 | 4 |
| 有效 batch | 16 |
| 梯度裁剪 | max_norm = 1.0 |
| 混合精度 | bf16（HuggingFace Accelerate） |

## CFG (Classifier-Free Guidance)

**训练时** — 10% 概率将文本替换为空字符串：

```python
if random.random() < 0.1:
    texts = [''] * len(texts)
```

**推理时** — 批量前向 + CFG 外推：

```python
out = planner.forward(perc, [text, ''])  # 批量: [有文本, 空文本]
cond_plan   = out['poses'][0]
uncond_plan = out['poses'][1]
plan = uncond_plan + 2.0 * (cond_plan - uncond_plan)  # cfg_scale=2.0
```

## 数据

| 组件 | 配置 |
|------|------|
| 数据集 | CineVLADataset（RGB-only, 8 帧/样本, 30 帧位姿） |
| 划分文件 | `DataDoP/train_valid.txt` |
| 随机种子 | `random.seed(42)`（数据划分可复现） |
| Workers | 0 (CPU) / 2 (CUDA) |

## Fallback 模式

当 `accelerate` 库不可用时，使用 `_SimpleAccelerator`：

```python
class _SimpleAccelerator:
    def backward(self, loss):
        (loss / grad_accum_steps).backward()   # 缩放 loss

    @property
    def sync_gradients(self):
        return step % grad_accum_steps == 0     # 每 N 步同步

    def accumulate(self, model):
        yield                                    # no-op 上下文管理器
```

梯度累积对所有后端（CPU / 单 GPU / 多 GPU）行为一致。
