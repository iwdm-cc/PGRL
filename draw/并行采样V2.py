import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成模拟偏好向量和对应的目标值（你应替换为真实模型推理结果）
np.random.seed(42)
num_samples = 20
preferences = np.random.dirichlet([1, 1, 1], size=num_samples)
objectives = np.random.rand(num_samples, 3) * [100, 80, 50]  # 模拟的 Cmax, W, S

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.viridis(np.linspace(0, 1, num_samples))
for i in range(num_samples):
    ax.scatter(objectives[i, 0], objectives[i, 1], objectives[i, 2],
               color=colors[i], label=f'偏好{i+1}' if i < 10 else None, s=50)

ax.set_xlabel('最大完工时间 $C_{max}$')
ax.set_ylabel('总负载 $W$')
ax.set_zlabel('切换成本 $S$')
ax.set_title('基于不同偏好向量的并行推理解分布')
ax.view_init(elev=20, azim=45)
plt.tight_layout()
plt.savefig('并行偏好采样_调度解分布图.png', dpi=300)
plt.show()
