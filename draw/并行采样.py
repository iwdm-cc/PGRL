import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# ==== 图1：并行偏好采样流程图 ====
def draw_parallel_inference_flow():
    fig, ax = plt.subplots(figsize=(10, 5))

    num_prefs = 5
    for i in range(num_prefs):
        ax.annotate(f'w{i + 1}', xy=(i * 2, 4), fontsize=12, ha='center', bbox=dict(boxstyle="round", fc="#aec7e8"))
        ax.annotate("策略网络", xy=(i * 2, 2.5), fontsize=10, ha='center',
                    bbox=dict(boxstyle="round", fc="#1f77b4", ec="black", lw=1), color='white')
        ax.annotate(f'sol{i + 1}', xy=(i * 2, 1), fontsize=12, ha='center', bbox=dict(boxstyle="round", fc="#2ca02c"))

        # arrows
        ax.annotate("", xy=(i * 2, 3.5), xytext=(i * 2, 3.0), arrowprops=dict(arrowstyle='->'))
        ax.annotate("", xy=(i * 2, 2.0), xytext=(i * 2, 1.5), arrowprops=dict(arrowstyle='->'))

    ax.text(num_prefs, 0.2, "多样化解集逼近 Pareto 前沿", fontsize=12, style='italic')
    ax.set_xlim(-1, num_prefs * 2)
    ax.set_ylim(0, 5)
    ax.axis('off')
    plt.title("并行偏好推理流程图", fontsize=14)
    plt.tight_layout()
    plt.show()


# ==== 图2：目标空间解集图 ====
def draw_objective_space_solutions():
    # 生成虚拟解点（三目标投影为二维）
    np.random.seed(42)
    solutions = np.random.rand(20, 2) * [30, 100]  # (C_max, W)

    plt.figure(figsize=(6, 6))
    plt.scatter(solutions[:, 0], solutions[:, 1], c='#ff7f0e', s=60, label='推理结果解')
    plt.xlabel('最大完工时间 C_max')
    plt.ylabel('机器总负载 W')
    plt.title('目标空间中的多解分布（近似Pareto前沿）')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def draw_parallel_inference_and_pareto():
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 2]})

    ## ------ 上图：并行偏好推理流程图 -------
    ax = axs[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title("推理阶段的并行偏好采样流程", fontsize=14)

    prefs = ['w₁', 'w₂', 'w₃', 'w₄', 'w₅']
    for i, pref in enumerate(prefs):
        x = i * 2
        # 偏好向量
        ax.text(x + 0.5, 5.5, pref, ha='center', bbox=dict(boxstyle="round", fc="#aec7e8"))

        # 策略网络
        ax.add_patch(mpatches.FancyBboxPatch((x, 4), 1, 0.8, boxstyle="round,pad=0.1", fc="#1f77b4"))
        ax.text(x + 0.5, 4.4, "策略网络", ha='center', color='white', fontsize=10)

        # 调度解
        ax.text(x + 0.5, 2.5, f"解{i+1}", ha='center', bbox=dict(boxstyle="round", fc="#2ca02c"))

        # 箭头
        ax.annotate('', xy=(x + 0.5, 5.3), xytext=(x + 0.5, 4.8),
                    arrowprops=dict(arrowstyle='->'))
        ax.annotate('', xy=(x + 0.5, 3.9), xytext=(x + 0.5, 3.0),
                    arrowprops=dict(arrowstyle='->'))

    ax.text(5.5, 1.2, "→ 多样解集逼近 Pareto 前沿", fontsize=12, style='italic')

    ## ------ 下图：目标空间解分布图 -------
    ax2 = axs[1]
    ax2.set_title("目标空间中解集分布（Cₘₐₓ vs W）", fontsize=14)
    ax2.set_xlabel("最大完工时间 Cₘₐₓ")
    ax2.set_ylabel("机器总负载 W")
    ax2.grid(True)

    np.random.seed(0)
    solutions = np.random.rand(20, 2) * [40, 100]
    colors = np.linspace(0.2, 1.0, len(solutions))

    scatter = ax2.scatter(solutions[:, 0], solutions[:, 1], c=colors, cmap='viridis', s=80)
    plt.colorbar(scatter, ax=ax2, label='偏好强度/方向')

    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_3d_pareto_solutions(save_path="图4_目标空间解集分布.png"):
    # 模拟多目标调度生成的20个解，分别表示：(Cmax, W, S)
    np.random.seed(42)
    Cmax = np.random.uniform(20, 50, size=20)  # 最大完工时间
    W = np.random.uniform(60, 120, size=20)    # 机器总负载
    S = np.random.uniform(5, 20, size=20)      # 切换成本/时间

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 偏好程度模拟颜色映射（可按偏好向量生成解时的顺序来赋值）
    colors = np.linspace(0.2, 1.0, len(Cmax))

    # 3D散点图
    sc = ax.scatter(Cmax, W, S, c=colors, cmap='viridis', s=80)

    # 坐标轴标签
    ax.set_xlabel('最大完工时间 Cₘₐₓ', labelpad=10)
    ax.set_ylabel('机器总负载 W', labelpad=10)
    ax.set_zlabel('切换成本 S', labelpad=10)

    # 图标题
    ax.set_title("多偏好调度解在三目标空间中的分布", fontsize=14)

    # 添加颜色条
    cb = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cb.set_label('偏好向量方向（示意）')

    # 调整角度视角
    ax.view_init(elev=30, azim=45)  # elev:上下，azim:左右旋转角度

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"已保存图像至：{save_path}")



if __name__ == "__main__":
    draw_parallel_inference_flow()
    draw_objective_space_solutions()
    draw_parallel_inference_and_pareto()
    # 调用绘图函数
    draw_3d_pareto_solutions()