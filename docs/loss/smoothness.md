# 轨迹平滑度损失（二阶）

> 代码: `trajectory_smoothness_loss(poses, lambda_rot_smooth)` — `core/losses.py:159`

## 平移加速度

速度和加速度的离散定义：

$$v_i = t_i - t_{i-1} \qquad\qquad a^{\text{trans}}_i = v_i - v_{i-1} = t_i - 2t_{i-1} + t_{i-2}$$

$$\mathcal{L}_{\text{acc}}^{\text{trans}} = \frac{1}{N-2} \sum_{i=2}^{N-1} \bigl\|a^{\text{trans}}_i\bigr\|_1$$

匀速直线运动时 $a_i = 0$，loss = 0。任何速度大小或方向的改变都产生正 loss。

## 旋转角加速度

$$\omega_i = d_{\text{rot}}(\hat{q}_i, \hat{q}_{i-1}) \qquad\qquad a^{\text{rot}}_i = \omega_i - \omega_{i-1}$$

$$\mathcal{L}_{\text{acc}}^{\text{rot}} = \frac{1}{N-2} \sum_{i=2}^{N-1} \bigl|a^{\text{rot}}_i\bigr|$$

$\omega_i$ 是从帧 $i-1$ 到 $i$ 的标量角速度（rad/frame），$a^{\text{rot}}_i$ 是角加速度。

## 合并

$$\mathcal{L}_{\text{smooth}} = \mathcal{L}_{\text{acc}}^{\text{trans}} + \lambda_{\text{rot}}^{\text{smooth}} \cdot \mathcal{L}_{\text{acc}}^{\text{rot}}$$

## 为什么用二阶而非一阶

| 阶数 | 惩罚量 | 问题 |
|------|--------|------|
| 一阶 | 速度 $\|t_i - t_{i-1}\|$ | 与 GT 冲突：快速运动理应具有大速度 |
| 二阶 | 加速度 $\|a_i\|$ | 不关心速度大小——快但平稳的运动 loss = 0 |

一阶约束相当于假设"相机应该静止"——不合理。二阶约束的假设是"相机应该惯性运动"——物理上合理。

## 为什么用 L1 而非 L2

| 范数 | 对急转弯的惩罚 | 产生的运动风格 |
|------|---------------|--------------|
| L1 | 线性增长 | 稀疏加速度——大部分平滑，偶尔合法急转弯 |
| L2 | 平方增长 | 全局柔和——所有变化被均匀压制 |

相机轨迹中偶尔的急转弯是合法的（如场景切换）。L1 允许这种稀疏性。

## 何时返回零

当 $N < 3$ 时函数返回 0：帧数不足以计算加速度（需要至少 3 帧才能有一个加速度值）。

在 Refiner 的 chunk 中，$K$ 通常为 1~5 帧，当 $K < 3$ 时平滑损失自动退化为零。
