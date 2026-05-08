# 测地线旋转损失

> 代码: `geodesic_rotation_loss(q_pred, q_gt)` — `core/losses.py:28`

## 公式

$$\mathcal{L}_{\text{rot}} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} 2 \arccos\!\bigl(|\langle\hat{q}, q^*\rangle|\bigr)$$

## 计算步骤

| 步骤 | 操作 | 原因 |
|------|------|------|
| $\langle\hat{q}, q^*\rangle$ | 四维向量内积 $w_1 w_2 + x_1 x_2 + y_1 y_2 + z_1 z_2$ | 等于 $\cos(\theta/2)$，$\theta$ 为旋转角 |
| $|\cdot|$ | 取绝对值 | $q$ 和 $-q$ 表示同一旋转（双覆盖） |
| $\arccos$ | 反余弦 | 内积转弧度 |
| $2 \times$ | 乘以 2 | $\langle q_1, q_2 \rangle = \cos(\theta/2)$，故 $\theta = 2\arccos$ |

## 值域

$[0, \pi]$ 弧度。$0$ = 完全相同。$\pi$ = 方向相反（180°）。

## 为什么不用四元数 L2？

四元数存在于 $S^3$（四维超球面）上，嵌入空间 $\mathbb{R}^4$ 中的欧氏距离不等于 $S^3$ 上的测地线距离。

反例：$q=(1,0,0,0)$ 和 $-q=(-1,0,0,0)$ 的 L2 距离为 2.0，但它们是同一个旋转。

## 数值稳定性

内积在传入 $\arccos$ 前钳制到 $[-1+10^{-7},\; 1-10^{-7}]$：

```python
dot = torch.abs(torch.sum(q_pred * q_gt, dim=-1))
dot = dot.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
return 2 * torch.acos(dot).mean()
```

## 与 Chordal 距离的关系

Chordal（Frobenius）距离可作为计算更便宜的替代：

$$d_{\text{chordal}} = \|R(\hat{q}) - R(q^*)\|_F = 2\sqrt{2} \cdot \sin(d_{\text{rot}} / 2)$$

小角度时 $d_{\text{chordal}} \approx \sqrt{2} \cdot d_{\text{rot}}$，两者近似线性。大旋转时仅 geodesic 给出真实流形距离。
