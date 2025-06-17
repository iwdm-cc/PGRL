# -*- coding: utf-8 -*-
"""
多目标柔性车间调度可视化示例
包含三种方案的甘特图绘制与 Pareto 前沿分析（三目标冲突）

作者：ChatGPT
用途：科研可视化与调度对比分析
"""

import matplotlib.pyplot as plt

# 设置 matplotlib 中文显示（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# -----------------------------
# Step 1: 准备三个调度方案的数据
# 每项数据格式：(作业-工序, 使用机器, 开始时间, 结束时间, 颜色)
# -----------------------------
import matplotlib.pyplot as plt
import pandas as pd

# 设置 matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# -----------------------------
# Step 1: 准备三个调度方案的数据
# 每项数据格式：(作业-工序, 使用机器, 开始时间, 结束时间, 颜色)
# -----------------------------

# 定义更美观的配色方案
color_scheme = {
    "J1": "#3B82F6",  # 蓝色（主色）
    "J2": "#10B981",  # 翡翠绿
    "J3": "#8B5CF6",  # 紫色
}


# 为每个作业的所有工序使用相同色系但不同深浅的颜色
def get_color(job_id, alpha=1.0):
    base_color = color_scheme.get(job_id, "#9CA3AF")  # 默认灰色
    # 转换为RGBA并设置透明度
    from matplotlib.colors import to_rgba
    return to_rgba(base_color, alpha)


# 生成带透明度的颜色，用于区分不同工序
schedule_data = {
    "MinCmax": [
        ("J1-O1", "M1", 0, 3, get_color("J1", 0.9)),
        ("J1-O2", "M2", 3, 7, get_color("J1", 0.7)),
        ("J1-O3", "M3", 7, 10, get_color("J1", 0.5)),
        ("J2-O1", "M2", 0, 2, get_color("J2", 0.9)),
        ("J2-O2", "M3", 2, 4, get_color("J2", 0.7)),
        ("J2-O3", "M1", 4, 6, get_color("J2", 0.5)),
        ("J3-O1", "M3", 0, 2, get_color("J3", 0.9)),
        ("J3-O2", "M1", 2, 5, get_color("J3", 0.7)),
        ("J3-O3", "M2", 5, 8, get_color("J3", 0.5)),
    ],
    "MinLoad": [
        ("J1-O1", "M2", 0, 3, get_color("J1", 0.9)),
        ("J1-O2", "M3", 3, 6, get_color("J1", 0.7)),
        ("J1-O3", "M1", 6, 8, get_color("J1", 0.5)),
        ("J2-O1", "M2", 3, 5, get_color("J2", 0.9)),
        ("J2-O2", "M1", 8, 10, get_color("J2", 0.7)),
        ("J2-O3", "M3", 10, 12, get_color("J2", 0.5)),
        ("J3-O1", "M3", 0, 2, get_color("J3", 0.9)),
        ("J3-O2", "M1", 2, 4, get_color("J3", 0.7)),
        ("J3-O3", "M2", 5, 7, get_color("J3", 0.5)),
    ],
    "MinSwitch": [
        ("J1-O1", "M1", 0, 3, get_color("J1", 0.9)),
        ("J1-O2", "M1", 3, 5, get_color("J1", 0.7)),
        ("J1-O3", "M1", 5, 8, get_color("J1", 0.5)),
        ("J2-O1", "M2", 0, 2, get_color("J2", 0.9)),
        ("J2-O2", "M2", 2, 4, get_color("J2", 0.7)),
        ("J2-O3", "M2", 4, 6, get_color("J2", 0.5)),
        ("J3-O1", "M3", 0, 2, get_color("J3", 0.9)),
        ("J3-O2", "M3", 2, 5, get_color("J3", 0.7)),
        ("J3-O3", "M3", 5, 8, get_color("J3", 0.5)),
    ],
}


# -----------------------------
# Step 2: 甘特图绘图函数
# -----------------------------
def plot_gantt(data, title):
    """绘制甘特图"""
    fig, ax = plt.subplots(figsize=(12, 5))
    machines = sorted(set([item[1] for item in data]))
    machine_map = {m: i for i, m in enumerate(machines)}
    y_ticks = range(len(machines))

    for job, machine, start, end, color in data:
        # 绘制任务条，添加细微阴影效果
        ax.barh(machine_map[machine], end - start, left=start, color=color, edgecolor='black', alpha=0.9, linewidth=0.8)
        # 添加作业标签
        ax.text((start + end) / 2, machine_map[machine], job, va='center', ha='center',
                fontsize=9, color='white', fontweight='bold')

    # 设置坐标轴和标题
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(machines, fontsize=10)
    ax.set_xlabel("时间", fontsize=11)
    ax.set_title(f"调度方案：{title}", fontsize=13, pad=10)

    # 添加网格线增强可读性
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=get_color("J1"), label='作业 J1'),
                       Patch(facecolor=get_color("J2"), label='作业 J2'),
                       Patch(facecolor=get_color("J3"), label='作业 J3')]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True)

    # 微调布局
    plt.tight_layout()
    plt.show()


# -----------------------------
# Step 3: 绘制所有方案的甘特图
# -----------------------------
for scheme, data in schedule_data.items():
    plot_gantt(data, scheme)

# -----------------------------
# Step 4: 构建三目标性能数据并绘制 Pareto 图
# -----------------------------
pareto_df = pd.DataFrame({
    "方案": ["MinCmax", "MinLoad", "MinSwitch"],
    "完工时间": [10, 11, 16],  # Cmax
    "总负载": [8, 8, 7],  # 所有机器加工时间总和
    "切换成本": [4, 3, 4],  # 假设切换一次成本为1
})

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pareto_df["完工时间"], pareto_df["总负载"], pareto_df["切换成本"], c=['r', 'g', 'b'], s=100)

# 添加标签
for i, row in pareto_df.iterrows():
    ax.text(row["完工时间"], row["总负载"], row["切换成本"], row["方案"], fontsize=10)

# 设置坐标轴与标题
ax.set_xlabel("完工时间 (Cmax)")
ax.set_ylabel("总负载")
ax.set_zlabel("切换成本")
ax.set_title("多目标调度的 Pareto 前沿图")
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 模拟三方案的三目标数据（你可扩展更多行）
df = pd.DataFrame({
    "方案": ["MinCmax", "MinLoad", "MinSwitch"],
    "完工时间": [10, 11, 16],  # Cmax
    "总负载": [8, 8, 7],  # 所有机器加工时间总和
    "切换成本": [4, 3, 4],  # 假设切换一次成本为1
})

# 提取数值指标用于相关性计算
metrics = df[["完工时间", "总负载", "切换成本"]]

# 计算皮尔逊相关系数矩阵
corr_matrix = metrics.corr(method="pearson")

# 画出热力图
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar=True)
plt.title("三目标冲突关系热力图（相关系数）")
plt.tight_layout()
plt.show()
