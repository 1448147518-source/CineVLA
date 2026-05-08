# CineVLA v4 损失函数设计文档

## 1. 设计原则

借鉴 SOTA 三维重建工作的三条核心洞察（π³ ICLR 2026、LingBot-Map 2026、VGGT CVPR 2025）：

1. 旋转和平移本质上是不同的几何流形——**必须解耦监督**
2. 帧间相对结构与绝对坐标同样重要——**成对一致性约束防止轨迹漂移**
3. 相机运动应符合物理规律——**平滑正则项消除高频抖动**

## 2. 符号表

| 符号 | 含义 |
|------|------|
| $B$ | 批量大小 |
| $N$ | 轨迹长度（默认 30 帧） |
| $N_{\text{eff}}$ | 渐进课程下的有效轨迹长度（10 → 30） |
| $K$ | Refiner 块大小，从 $[1, \text{lookahead}]$ 随机采样 |
| $\hat{q}_i \in S^3$ | 模型预测的第 $i$ 帧单位四元数 (w,x,y,z) |
| $q^*_i \in S^3$ | 真值单位四元数 |
| $\hat{t}_i \in \mathbb{R}^3$ | 模型预测的第 $i$ 帧平移向量（场景归一化后） |
| $t^*_i \in \mathbb{R}^3$ | 真值平移向量 |
| $\hat{P}_i = [\hat{q}_i, \hat{t}_i]$ | 模型预测的完整 7 维位姿 |
| $\hat{P}^{(r)}_i$ | 经 Refiner 修正后的位姿 |
| $W$ | 因果滑动窗口大小（默认 5） |

## 3. 基础损失函数

### 3.1 测地线旋转损失

> 代码: `geodesic_rotation_loss(q_pred, q_gt)` — `core/losses.py:28`

$$\mathcal{L}_{\text{rot}} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} 2 \arccos\!\bigl(|\langle\hat{q}, q^*\rangle|\bigr)$$

**计算步骤：**

| 步骤 | 操作 | 原因 |
|------|------|------|
| $\langle\hat{q}, q^*\rangle$ | 四维向量内积 $w_1 w_2 + x_1 x_2 + y_1 y_2 + z_1 z_2$ | 等于 $\cos(\theta/2)$，其中 $\theta$ 是旋转角 |
| $|\cdot|$ | 取绝对值 | 四元数双覆盖：$q$ 和 $-q$ 表示同一旋转 |
| $\arccos$ | 反余弦 | 将内积转为弧度 |
| $2 \times$ | 乘以 2 | $\langle q_1, q_2 \rangle = \cos(\theta/2)$，故 $\theta = 2\arccos$ |

**值域**：$[0, \pi]$ 弧度。$0$ = 完全相同，$\pi$ = 方向相反（180°）。

**为什么不用四元数 L2？** 四元数存在于 $S^3$（四维超球面）上，嵌入空间 $\mathbb{R}^4$ 中的欧氏距离不等于 $S^3$ 上的测地线距离。例如 $q=(1,0,0,0)$ 和 $-q=(-1,0,0,0)$ 的 L2 距离为 2.0，但它们是同一个旋转。

**数值稳定性**：内积在传入 $\arccos$ 前被钳制到 $[-1+10^{-7},\; 1-10^{-7}]$，防止边界处产生 NaN 梯度。

---

### 3.2 L1 平移损失

> 代码: `l1_translation_loss(t_pred, t_gt)` — `core/losses.py:49`

$$\mathcal{L}_{\text{trans}} = \frac{1}{3 \cdot |\mathcal{T}|} \sum_{t \in \mathcal{T}} \bigl\|\hat{t} - t^*\bigr\|_1$$

PyTorch 的 `mean()` 对所有元素求平均（batch、帧、3 个坐标维），等价于自带 $1/3$ 因子。

**值域**：$\ge 0$，单位为场景归一化长度。数据集将所有轨迹的平移量除以 $\max_i \|t_i\|$，因此 $\mathcal{L}_{\text{trans}} \approx 0.01$ 意味着偏差约为场景空间尺度的 1%。

**为什么用 L1 而非 MSE？** L1 对大误差的梯度恒定为 $\pm 1$，不会被离群帧挟持训练方向。

---

### 3.3 相对位姿一致性损失

> 代码: `relative_pose_loss(poses_pred, poses_gt, window_size, lambda_trans)` — `core/losses.py:91`

#### 帧对集合

$$\mathcal{P} = \{\, (i, j) \mid 1 \le i < N,\; \max(0, i-W) \le j < i \,\}$$

该集合是**因果的**——帧 $i$ 仅与已经观测到的帧 $j < i$ 进行比较，不存在未来信息泄露。

#### 相对位姿的构造

| 量 | 公式 | 含义 |
|----|------|------|
| $\overline{q}$ | $(w, -x, -y, -z)$ | 四元数共轭（对单位四元数即逆） |
| $\circ$ | Hamilton 积 | 四元数乘法 |
| $\hat{q}_{j \to i}$ | $\overline{\hat{q}_j} \circ \hat{q}_i$ | 预测的从帧 $j$ 到帧 $i$ 的相对旋转 |
| $q^*_{j \to i}$ | $\overline{q^*_j} \circ q^*_i$ | 真值相对旋转 |
| $\hat{t}_{j \to i}$ | $\hat{t}_i - \hat{t}_j$ | 预测的相对平移（全局坐标系） |
| $t^*_{j \to i}$ | $t^*_i - t^*_j$ | 真值相对平移 |

#### 损失公式

$$\mathcal{L}_{\text{rel}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \Bigl[ d_{\text{rot}}\!\bigl(\hat{q}_{j\to i},\; q^*_{j\to i}\bigr) \;+\; \lambda_t \cdot \bigl\|\hat{t}_{j\to i} - t^*_{j\to i}\bigr\|_1 \Bigr]$$

#### 为什么需要它

绝对位姿损失问的是"每一帧在正确的位置吗？"——它只看到孤立帧。相对损失问的是"帧 $j$ 到帧 $i$ 的运动正确吗？"——它看到帧间关系。

| 仅用绝对损失 | 绝对 + 相对 |
|-------------|-----------|
| 帧 3 偏了 0.01m，帧 5 也偏了 0.01m | 同上 |
| 帧 3 和帧 5 之间的关系错了，但不被惩罚 | 显式惩罚帧对关系误差 |
| 误差无声累积 → 在帧 30 处产生漂移 | 帧对约束阻止累积 |

---

### 3.4 轨迹平滑度损失（二阶）

> 代码: `trajectory_smoothness_loss(poses, lambda_rot_smooth)` — `core/losses.py:159`

#### 平移加速度

$$v_i = t_i - t_{i-1} \qquad\qquad a^{\text{trans}}_i = v_i - v_{i-1} = t_i - 2t_{i-1} + t_{i-2}$$

$$\mathcal{L}_{\text{acc}}^{\text{trans}} = \frac{1}{N-2} \sum_{i=2}^{N-1} \bigl\|a^{\text{trans}}_i\bigr\|_1$$

**物理含义**：加速度向量的 L1 范数。匀速直线运动 $a_i = 0$，loss = 0。任何速度大小或方向的改变都产生正 loss。

#### 旋转角加速度

$$\omega_i = d_{\text{rot}}(\hat{q}_i, \hat{q}_{i-1}) \qquad\qquad a^{\text{rot}}_i = \omega_i - \omega_{i-1}$$

$$\mathcal{L}_{\text{acc}}^{\text{rot}} = \frac{1}{N-2} \sum_{i=2}^{N-1} \bigl|a^{\text{rot}}_i\bigr|$$

**物理含义**：$\omega_i$ 是从帧 $i-1$ 到 $i$ 的标量角速度（rad/frame），$a^{\text{rot}}_i$ 是角加速度——角速度的变化量。

#### 合并

$$\mathcal{L}_{\text{smooth}} = \mathcal{L}_{\text{acc}}^{\text{trans}} + \lambda_{\text{rot}}^{\text{smooth}} \cdot \mathcal{L}_{\text{acc}}^{\text{rot}}$$

**为什么用 L1 而非 L2？** L2 会过度惩罚偶尔的合法急转弯。L1 产生**稀疏加速度**——大部分时间平稳，只在必要时急转。

**为什么用二阶而非一阶？** 一阶（速度惩罚）与 GT 冲突：快速运动理应具有大速度。二阶（加速度惩罚）不关心速度大小——快速但平稳的运动 loss = 0。

---

## 4. 阶段一 — Planner 预训练

$$\mathcal{L}_{\text{planner}}(N_{\text{eff}}) = \lambda_{\text{rot}} \mathcal{L}_{\text{rot}} + \lambda_{\text{trans}} \mathcal{L}_{\text{trans}} + \lambda_{\text{rel}} \mathcal{L}_{\text{rel}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}$$

四个损失项均仅在前 $N_{\text{eff}}$ 帧上计算。

| 超参数 | 值 | 作用 |
|--------|-----|------|
| $\lambda_{\text{rot}}$ | 1.0 | 旋转主监督（弧度） |
| $\lambda_{\text{trans}}$ | 0.5 | 平移权重较低，因 1 rad ≈ 归一化后的典型平移量 |
| $\lambda_{\text{rel}}$ | 0.05 | 辅助项——不能压倒绝对位姿主损失 |
| $\lambda_{\text{smooth}}$ | 0.1 | 辅助项——平滑是先验约束，不是目标 |
| $\lambda_{\text{rot}}^{\text{smooth}}$ | 0.5 | 平滑项中旋转相对于平移的权重 |
| $\lambda_t$ | 0.1 | 相对损失中平移相对于旋转的权重 |
| $W$ | 5 | 因果滑动窗口大小 |

---

## 5. 阶段二 — Refiner 预训练

$$\mathcal{L}_{\text{refiner}} = \lambda_{\text{pose}} \mathcal{L}_{\text{pose-delta}} + \lambda_{z} \mathcal{L}_{\text{z-pred}} + \lambda_{\text{rel}}^{(r)} \mathcal{L}_{\text{rel}}^{(r)} + \lambda_{\text{smooth}}^{(r)} \mathcal{L}_{\text{smooth}}^{(r)}$$

### 5.1 修正量 L1 正则

$$\mathcal{L}_{\text{pose-delta}} = \frac{1}{K \cdot 7} \sum_{k=1}^{K} \bigl\|\hat{P}^{(r)}_{t+k} - \hat{P}_{t+k}\bigr\|_1$$

训练时 $\hat{P}_{t+k}$ 是真值轨迹块。Refiner 应学习一个接近恒等映射——仅在 $z_{\text{real}} \neq z_{\text{pred}}$ 指示感知不匹配时才输出有意义的修正。

**为什么用 L1？** 比 MSE 更适合做正则——偶尔较大的修正量不会被平方惩罚放大。

### 5.2 感知特征预测

$$\mathcal{L}_{\text{z-pred}} = \bigl\|\hat{z}_{t+1} - z_t\bigr\|_2^2$$

Refiner 必须从上下文中预测下一帧的感知特征。这个辅助任务迫使 Refiner 学习环境的几何结构——知道正确的轨迹意味着能预测看到的场景变化。

### 5.3 修正后轨迹的几何约束

$\mathcal{L}_{\text{rel}}^{(r)}$ 和 $\mathcal{L}_{\text{smooth}}^{(r)}$ 与 §3.3 和 §3.4 相同，作用在 $K$ 帧修正后块上。它们确保 Refiner 的修正不会破坏局部几何一致性或引入抖动。

当 $K < 2$ 时，两项均返回零（帧数不足无法计算帧对或加速度）。

| 超参数 | 值 | 为什么比 Planner 低 |
|--------|-----|---------------------|
| $\lambda_{\text{pose}}$ | 1.0 | 主要正则项——保持 delta 较小 |
| $\lambda_z$ | 0.1 | 辅助任务（与 v3 保持不变） |
| $\lambda_{\text{rel}}^{(r)}$ | 0.02 | 块内帧对数稀疏（K ≤ 5），信号较弱 |
| $\lambda_{\text{smooth}}^{(r)}$ | 0.05 | 同理——帧数少，二阶约束信号更弱 |

---

## 6. 阶段三 — 联合训练

$$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{planner}} + \mathcal{L}_{\text{refiner}}$$

两组损失均以完整权重计算。Planner 和 Refiner 端到端训练，Refiner 同时接收自身损失和通过 Planner（共享感知特征）间接传来的梯度。

联合训练阶段 $N_{\text{eff}} = N$（课程已结束，Planner 此时应能处理全长度轨迹）。

---

## 7. 渐进式课程

$$N_{\text{eff}}(\text{epoch}) = N_{\text{start}} + \bigl\lfloor (N_{\text{full}} - N_{\text{start}}) \cdot \min\!\bigl(1,\; \tfrac{\text{epoch}}{\text{ramp\_epochs}}\bigr) \bigr\rfloor$$

| 参数 | 默认值 | 含义 |
|------|--------|------|
| $N_{\text{start}}$ | 10 | 起始轨迹长度 |
| $N_{\text{full}}$ | 30 | 最终轨迹长度（= `pose_length`） |
| `ramp_epochs` | 阶段 epoch 数的 67% | 线性增长持续时长 |

在 epoch 0，Planner 仅学习短程（10 帧）运动模式；约 epoch 20 时能处理完整 30 帧结构。这与 LingBot-Map 的渐进视点训练（24 → 320 帧）思路一致。

---

## 8. 超参数速查表

### Planner

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_rot` | 1.0 | 测地线旋转损失权重 |
| `lambda_trans` | 0.5 | L1 平移损失权重 |
| `lambda_rel` | 0.05 | 相对位姿一致性权重 |
| `lambda_smooth` | 0.1 | 轨迹平滑度权重 |
| `rel_window_size` | 5 | 相对位姿因果窗口大小 |
| `lambda_rot_smooth` | 0.5 | 平滑项中旋转权重 |
| `lambda_rel_t` | 0.1 | 相对项中平移权重 |

### Refiner

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_pose_delta` | 1.0 | 修正量 L1 正则权重 |
| `lambda_z_pred` | 0.1 | 感知特征预测权重 |
| `lambda_rel_ref` | 0.02 | 修正轨迹相对一致性 |
| `lambda_smooth_ref` | 0.05 | 修正轨迹平滑度 |

### 课程

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pose_start_len` | 10 | 起始轨迹长度 |
| `curriculum_ramp_epochs` | `None` | 增长周期；`None` = 阶段 epoch 的 67% |

---

## 9. v3 → v4 变更对照

| 维度 | v3 | v4 |
|------|-----|-----|
| 旋转度量 | 7 维向量的 `MSE(q, q*)` | $2\arccos(|\langle\hat{q},q^*\rangle|)$ 测地线距离 |
| 平移度量 | 隐含在 7 维 MSE 中 | 显式 $\|\hat{t}-t^*\|_1$ |
| 帧间约束 | 无 | 因果滑动窗口相对位姿一致性 |
| 运动先验 | 无 | 二阶加速度惩罚 |
| Refiner 修正量范数 | `MSE(refined, planned)` | L1 |
| Refiner 几何约束 | 无 | 修正块上的相对一致性 + 平滑度 |
| 训练长度 | 固定 N=30 | 10 → 30 渐进增长 |
| Checkpoint 兼容 | — | `strict=False`，旧 ckpt 可加载，缺失键自动忽略 |
| 推理 | 不变 | 不变 —— 损失函数仅用于训练 |

## 10. 参考文献

- π³: *Scalable Permutation-Equivariant Visual Geometry Learning*, ICLR 2026. [arXiv:2507.13347](https://arxiv.org/abs/2507.13347)
- LingBot-Map: *Geometric Context Transformer for Streaming 3D Reconstruction*, 2026. [arXiv:2604.14141](https://arxiv.org/abs/2604.14141)
- VGGT: *Visual Geometry Grounded Transformer*, CVPR 2025 (Best Paper). [arXiv:2503.11651](https://arxiv.org/abs/2503.11651)
- Kendall & Cipolla: *Geometric Loss Functions for Camera Pose Regression with Deep Learning*, CVPR 2017.
- Hempel et al.: *Toward Robust and Unconstrained Full Range of Rotation Head Pose Estimation*, IEEE TIP 2024.
