# L1 平移损失

> 代码: `l1_translation_loss(t_pred, t_gt)` — `core/losses.py:49`

## 公式

$$\mathcal{L}_{\text{trans}} = \frac{1}{3 \cdot |\mathcal{T}|} \sum_{t \in \mathcal{T}} \bigl\|\hat{t} - t^*\bigr\|_1$$

PyTorch 的 `mean()` 对所有元素求平均（batch、帧、3 个坐标维），等价于自带 $1/3$ 因子：

```python
return t_pred.sub(t_gt).abs().mean()
```

## 值域

$\ge 0$，单位为场景归一化长度。数据集将所有轨迹的平移量除以 $\max_i \|t_i\|$，因此 $\mathcal{L}_{\text{trans}} \approx 0.01$ 意味着偏差约为场景空间尺度的 1%。

## 为什么用 L1 而非 MSE

| 特性 | L1 (MAE) | L2 (MSE) |
|------|----------|----------|
| 大误差梯度 | 恒定 $\pm 1$ | 正比于误差（被 outlier 挟持） |
| 小误差梯度 | 恒定（不够精细） | 正比于误差（精细调整） |
| $x=0$ 可导 | 否 | 是 |

相机轨迹数据通常经过 COLMAP 优化，比较干净。L1 在此场景下足够且更鲁棒。如果数据中有 COLMAP 失败的 outlier 帧，L1 的优势更明显。

## 与 Huber Loss 的关系

Huber 是 L1 和 L2 的平滑拼接：

$$H_\delta(x) = \begin{cases} \frac{1}{2}x^2 & |x| \leq \delta \\ \delta \cdot (|x| - \frac{1}{2}\delta) & |x| > \delta \end{cases}$$

当前版本使用纯 L1。如果后续数据显示大量中等误差样本，可切换为 Huber（$\delta \approx 0.01$）以在 $x=0$ 附近获得更平滑的梯度。
