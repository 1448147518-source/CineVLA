# 相对位姿一致性损失

> 代码: `relative_pose_loss(poses_pred, poses_gt, window_size, lambda_trans)` — `core/losses.py:91`

## 帧对集合

$$\mathcal{P} = \{\, (i, j) \mid 1 \le i < N,\; \max(0, i-W) \le j < i \,\}$$

该集合是**因果的**——帧 $i$ 仅与已经观测到的帧 $j < i$ 进行比较，不存在未来信息泄露。

## 相对位姿的构造

| 量 | 公式 | 含义 |
|----|------|------|
| $\hat{q}_{j \to i}$ | $\overline{\hat{q}_j} \circ \hat{q}_i$ | 预测的从帧 $j$ 到帧 $i$ 的相对旋转 |
| $q^*_{j \to i}$ | $\overline{q^*_j} \circ q^*_i$ | 真值相对旋转 |
| $\hat{t}_{j \to i}$ | $\hat{t}_i - \hat{t}_j$ | 预测的相对平移（全局坐标系） |
| $t^*_{j \to i}$ | $t^*_i - t^*_j$ | 真值相对平移 |

其中 $\overline{q} = (w, -x, -y, -z)$ 是四元数共轭（对单位四元数即逆），$\circ$ 是 Hamilton 积。

## 损失公式

$$\mathcal{L}_{\text{rel}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \Bigl[ d_{\text{rot}}\!\bigl(\hat{q}_{j\to i},\; q^*_{j\to i}\bigr) \;+\; \lambda_t \cdot \bigl\|\hat{t}_{j\to i} - t^*_{j\to i}\bigr\|_1 \Bigr]$$

## 与绝对位姿损失的区别

绝对位姿损失问的是"每一帧在正确的位置吗？"——它只看到孤立帧。

相对损失问的是"帧 $j$ 到帧 $i$ 的运动正确吗？"——它看到帧间关系。

| 仅用绝对损失 | 绝对 + 相对 |
|-------------|-----------|
| 帧 3 偏了 0.01m，帧 5 也偏了 0.01m | 同上 |
| 帧 3 和帧 5 之间的关系错了，但不被惩罚 | 显式惩罚帧对关系误差 |
| 误差无声累积 → 在帧 30 处产生漂移 | 帧对约束阻止累积 |

## 窗口大小的影响

$W=5$ 意味着每帧与最近 5 帧比较。增大 $W$ 增加约束密度但增加计算量（$O(NW)$）。训练早期 $N_{\text{eff}}$ 较小时，窗口自然变小。

## 实现注意

```python
for i in range(1, N):
    for j in range(max(0, i - W), i):
        # 双循环在 Python 中较慢，但对于 N≤30, W≤5 可接受
        # 共约 (N-1)*W/2 ≈ 72 次迭代/batch
```

当 $N<2$ 或 $W<1$ 时函数返回 0。
