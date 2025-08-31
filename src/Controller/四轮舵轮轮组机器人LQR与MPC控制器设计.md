# 四轮舵轮轮组机器人LQR与MPC控制器设计

## 1. 机器人运动学模型

四轮舵轮轮组机器人底盘在全局坐标系下的运动学模型为：

$$
\begin{aligned}
\dot{x} &= v_x \\
\dot{y} &= v_y \\
\dot{\theta} &= \omega
\end{aligned}
$$

其中：

- 状态向量 

$$
\mathbf{x} = [x, y, \theta]^T
$$

  表示机器人的位置和方向角

- 控制输入向量 

$$
\mathbf{u} = [v_x, v_y, \omega]^T
$$

表示底盘在全局坐标系下的线速度和角速度

## 2. 跟踪误差模型

### 2.1 误差定义

设目标轨迹为 
$$
\mathbf{x}_d(t) = [x_d(t), y_d(t), \theta_d(t)]^T
$$




目标控制输入为 
$$
\mathbf{u}_d(t) = [v_{xd}(t), v_{yd}(t), \omega_d(t)]^T
$$


定义误差状态在机器人坐标系下：

$$
\begin{bmatrix}
e_x \\
e_y \\
e_\theta
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta & \sin \theta & 0 \\
-\sin \theta & \cos \theta & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x_d - x \\
y_d - y \\
\theta_d - \theta
\end{bmatrix}
$$

### 2.2 误差动力学模型

通过对误差求导得到误差动力学模型：

$$
\begin{aligned}
\dot{e}_x &= \cos \theta (v_{xd} - v_x) + \sin \theta (v_{yd} - v_y) + \omega e_y \\
\dot{e}_y &= -\sin \theta (v_{xd} - v_x) + \cos \theta (v_{yd} - v_y) - \omega e_x \\
\dot{e}_\theta &= \omega_d - \omega
\end{aligned}
$$

### 2.3 线性化处理

围绕参考轨迹（误差为零）进行线性化，假设 $\theta \approx \theta_d$，$\omega \approx \omega_d$，定义控制偏差：

$$
\begin{aligned}
\delta v_x &= v_x - v_{xd} \\
\delta v_y &= v_y - v_{yd} \\
\delta \omega &= \omega - \omega_d
\end{aligned}
$$

线性化后误差动力学为：

$$
\dot{\mathbf{e}} = A(t) \mathbf{e} + B(t) \mathbf{\delta u}
$$

其中：
- $\mathbf{e} = [e_x, e_y, e_\theta]^T$
- $\mathbf{\delta u} = [\delta v_x, \delta v_y, \delta \omega]^T$
- 系统矩阵 $A(t)$ 和控制矩阵 $B(t)$ 为：

$$
A(t) = \begin{bmatrix}
0 & \omega_d(t) & 0 \\
-\omega_d(t) & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}, \quad
B(t) = \begin{bmatrix}
-\cos \theta_d(t) & -\sin \theta_d(t) & 0 \\
\sin \theta_d(t) & -\cos \theta_d(t) & 0 \\
0 & 0 & -1
\end{bmatrix}
$$

这是一个线性时变（LTV）系统。

## 3. LQR控制器设计

### 3.1 代价函数

LQR控制器的目标是最小化跟踪误差和控制偏差的代价函数：

$$
J = \int_0^\infty \left( \mathbf{e}^T Q \mathbf{e} + \mathbf{\delta u}^T R \mathbf{\delta u} \right) dt
$$

其中：
- $Q$ 为半正定状态权重矩阵
- $R$ 为正定控制权重矩阵
- 控制偏差 $\mathbf{\delta u}$ 的惩罚确保了控制输出的平滑性

### 3.2 Riccati方程求解

由于系统是时变的，需使用时变Riccati方程求解最优反馈增益：

$$
-\dot{P}(t) = A(t)^T P(t) + P(t) A(t) - P(t) B(t) R^{-1} B(t)^T P(t) + Q
$$

边界条件为 $P(t_f) = 0$（对于无限时间问题，通常取稳态解）。

### 3.3 最优控制律

最优控制律为：

$$
\mathbf{\delta u} = -K(t) \mathbf{e}, \quad K(t) = R^{-1} B(t)^T P(t)
$$

实际控制输入为：

$$
\mathbf{u} = \mathbf{u}_d + \mathbf{\delta u}
$$

## 4. MPC控制器设计

### 4.1 离散化模型

使用离散时间误差模型，采样时间为 $T$，离散化后：

$$
\mathbf{e}_{k+1} = A_k \mathbf{e}_k + B_k \mathbf{\delta u}_k
$$

其中 $A_k$ 和 $B_k$ 由 $A(t)$ 和 $B(t)$ 在时间 $kT$ 处离散化得到。

### 4.2 代价函数

定义有限时间域代价函数：

$$
J = \sum_{k=0}^{N-1} \left( \mathbf{e}_k^T Q \mathbf{e}_k + \mathbf{\delta u}_k^T R \mathbf{\delta u}_k \right) + \mathbf{e}_N^T P \mathbf{e}_N
$$

其中：
- $N$ 为预测步长
- $P$ 为终端代价矩阵（通常从Riccati方程得到）

### 4.3 优化问题

在每个时间步求解优化问题：

$$
\min_{\mathbf{\delta u}_0, \ldots, \mathbf{\delta u}_{N-1}} J
$$

约束条件：
$$
\mathbf{e}_{k+1} = A_k \mathbf{e}_k + B_k \mathbf{\delta u}_k, \quad k=0,\ldots,N-1
$$

以及可能的控制约束：
$$
\mathbf{\delta u}_{\min} \leq \mathbf{\delta u}_k \leq \mathbf{\delta u}_{\max}
$$

### 4.4 控制实施

应用第一个控制输入：

$$
\mathbf{u} = \mathbf{u}_d + \mathbf{\delta u}_0
$$

## 5. 设计要点说明

### 5.1 模型线性化
- 误差动力学线性化围绕参考轨迹
- 假设误差较小，适用于跟踪控制

### 5.2 控制平滑性
- 代价函数中直接惩罚控制偏差 $\mathbf{\delta u}$
- 减少控制量的剧烈变化，实现平滑性

### 5.3 实际应用考虑
- **LQR**：需离线计算或预知轨迹，计算效率高
- **MPC**：在线计算更灵活，可处理约束，但计算量大
- 权重矩阵 $Q$ 和 $R$ 需根据性能需求调整

### 5.4 实现流程
1. 获取当前状态和目标轨迹
2. 计算跟踪误差 $\mathbf{e}$
3. 求解控制偏差 $\mathbf{\delta u}$
4. 生成实际控制输入 $\mathbf{u} = \mathbf{u}_d + \mathbf{\delta u}$
5. 通过底层轮子控制算法将底盘速度命令转换为各轮子的转向和驱动信号

此框架为四轮舵轮机器人提供了完整的轨迹跟踪控制解决方案，兼顾跟踪精度和控制平滑性。