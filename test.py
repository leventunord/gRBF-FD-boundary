import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def generate_semicircle_points(n_interior=300, n_boundary=50):
    """
    生成上半圆盘的内部点和边界点
    """
    # np.random.seed(42) # 固定随机种子以保证结果可复现
    
    # 1. 生成内部点 (r < 1, y > 0)
    # 使用 sqrt 保证在圆盘面上均匀分布
    r = np.sqrt(np.random.rand(n_interior))
    theta = np.random.rand(n_interior) * np.pi/2
    x_in = r * np.cos(theta)
    y_in = r * np.sin(theta)
    interior = np.column_stack((x_in, y_in))

    # 2. 生成圆弧边界点
    theta_b = np.pi/2 * np.random.rand(n_boundary)
    # theta_b = np.linspace(0, np.pi, n_boundary)
    x_b = np.cos(theta_b)
    y_b = np.sin(theta_b)
    boundary = np.column_stack((x_b, y_b))

    
    return interior, boundary

def get_stencil_knn(center_point, all_points, k):
    """
    策略1: 直接选取全局最近的 k 个邻居
    """
    # 计算中心点到所有点的距离
    dists = distance.cdist([center_point], all_points)[0]
    # 获取距离最小的 k 个点的索引
    idx = np.argsort(dists)[:k]
    return all_points[idx]

def get_stencil_interior_biased(center_point, interior_points, k):
    """
    策略2: 选取中心点自身 + (k-1) 个最近的内部点
    """
    # 计算中心点到所有内部点的距离
    dists = distance.cdist([center_point], interior_points)[0]
    # 获取距离最小的 k-1 个内部点的索引
    idx = np.argsort(dists)[:k-1]
    
    # 组合中心点和选出的内部点
    neighbors = interior_points[idx]
    stencil = np.vstack((center_point, neighbors))
    return stencil

# --- 数据准备 ---
k = 5
n_in = 10
n_bd = 5
interior_pts, arc_bnd_pts = generate_semicircle_points(n_in, n_bd)
all_pts = np.vstack((interior_pts, arc_bnd_pts))
target_angle = np.pi / 6
center_idx = np.abs(np.arctan2(arc_bnd_pts[:, 1], arc_bnd_pts[:, 0]) - target_angle).argmin()
center_point = arc_bnd_pts[center_idx]

stencil_1 = get_stencil_knn(center_point, all_pts, k)
stencil_2 = get_stencil_interior_biased(center_point, interior_pts, k)

# --- 封装核心绘图逻辑 ---
def draw_base_plot(ax, stencil_data):
    # 1. 绘制背景
    t_plot = np.linspace(0, np.pi/2, 200)
    ax.plot(np.cos(t_plot), np.sin(t_plot), color='gray', linestyle='--', alpha=0.3)
    
    # 2. 绘制点 (保存返回的对象以便提取图例)
    l1 = ax.scatter(interior_pts[:, 0], interior_pts[:, 1], c='black', s=45, alpha=0.6, label='Interior Node')
    l2 = ax.scatter(arc_bnd_pts[:, 0], arc_bnd_pts[:, 1], c='red', s=45, alpha=0.8, label='Boundary Node')
    l3 = ax.scatter(stencil_data[:, 0], stencil_data[:, 1], s=450, facecolors='none', edgecolors='blue', 
                    linewidth=1.5, marker='o', label='Stencil Neighbor', zorder=10)
    l4 = ax.scatter(center_point[0], center_point[1], s=600, facecolors='none', edgecolors='green', 
                    linewidth=2.0, marker='s', label='Center Node', zorder=11)

    # ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal') # 建议开启，保证半圆不变形
    return [l1, l2, l3, l4]

# --- 1. 导出左图 (不含图例) ---
fig_a, ax_a = plt.subplots(figsize=(6, 5))
draw_base_plot(ax_a, stencil_1)
plt.tight_layout()
fig_a.savefig("fig_left.pdf", bbox_inches='tight')

# --- 2. 导出右图 (不含图例) ---
fig_b, ax_b = plt.subplots(figsize=(6, 5))
handles = draw_base_plot(ax_b, stencil_2)
plt.tight_layout()
fig_b.savefig("fig_right.pdf", bbox_inches='tight')

# --- 3. 导出纯图例 ---
# 创建一个非常小的画布
fig_leg = plt.figure(figsize=(10, 0.5)) 
# 提取刚才绘图得到的标签
labels = [h.get_label() for h in handles]
# 在这个新画布上画图例，ncol=4 表示横向排成一排
fig_leg.legend(handles, labels, loc='center', ncol=4, frameon=False, fontsize=12)
fig_leg.savefig("legend.pdf", bbox_inches='tight')

print("已导出: fig_left.pdf, fig_right.pdf, legend.pdf")