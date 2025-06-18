import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from mpl_toolkits.mplot3d import Axes3D

# 设置 matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 定义四个不同的 Dirichlet 参数 alpha
alpha_params = [
    [1.0, 1.0, 1.0],  # 均衡分布
    [5.0, 1.0, 1.0],  # 偏好第一个目标 Cmax
    [1.0, 5.0, 1.0],  # 偏好第二个目标 W
    [1.0, 1.0, 5.0]  # 偏好第三个目标 S
]
num_samples = 300

# 生成四张图片
for i, alpha in enumerate(alpha_params):
    # Step 1: 生成 Dirichlet 分布的偏好向量
    preferences = dirichlet.rvs(alpha, size=num_samples)

    # Step 2: 模拟每个偏好对应的目标结果
    results = []
    for w in preferences:
        # 添加随机扰动模拟实际结果
        Cmax = 100 - 40 * w[0] + np.random.normal(0, 3)
        W = 200 - 80 * w[1] + np.random.normal(0, 3)
        S = 50 - 20 * w[2] + np.random.normal(0, 1.5)
        results.append([Cmax, W, S])
    results = np.array(results)

    # 创建图形和子图
    fig = plt.figure(figsize=(14, 6))

    # 子图1：偏好向量分布
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(preferences[:, 0], preferences[:, 1], preferences[:, 2],
                c='dodgerblue', alpha=0.6, s=15)
    ax1.set_xlabel('$w_1$ (完成时间)', fontsize=12)
    ax1.set_ylabel('$w_2$ (总负载)', fontsize=12)
    ax1.set_zlabel('$w_3$ (切换成本)', fontsize=12)
    ax1.set_title(f'偏好向量分布 α=({alpha[0]}, {alpha[1]}, {alpha[2]})', fontsize=13)

    # 子图2：调度结果分布
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(results[:, 0], results[:, 1], results[:, 2],
                c='mediumorchid', alpha=0.7, s=15)
    ax2.set_xlabel('完成时间 ($C_{max}$)', fontsize=12)
    ax2.set_ylabel('总负载 ($W$)', fontsize=12)
    ax2.set_zlabel('切换成本 ($S$)', fontsize=12)
    ax2.set_title('调度结果分布', fontsize=13)

    plt.tight_layout()
    plt.savefig(f"experiment_alpha_{i + 1}.png", dpi=300)
    plt.close()

print("✅ 四张图像已保存：")
print("1. experiment_alpha_1.png - 均衡分布")
print("2. experiment_alpha_2.png - 偏好完成时间")
print("3. experiment_alpha_3.png - 偏好总负载")
print("4. experiment_alpha_4.png - 偏好切换成本")