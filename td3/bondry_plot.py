import matplotlib.pyplot as plt
import matplotlib.patches as patches

def check_pos(x, y):
    a = 0.5
    goal_ok = True
    # 检查位置是否在禁区内
    if (1.5 - a) < x < (4.5 + a) and (1.5 - a) < y < (4.5 + a):
        goal_ok = False
    if (-5 - a) < x < (-0.5 + a) and (1.5 - a) < y < (4.5 + a):
        goal_ok = False
    if (-5.5 - a) < x < (-2.5 + a) and (0.5 - a) < y < (5 + a):
        goal_ok = False
    if (0.5 - a) < x < (5 + a) and (-5.5 - a) < y < (-2.5 + a):
        goal_ok = False
    if (2.5 - a) < x < (5.5 + a) and (-5 - a) < y < (-0.5 + a):
        goal_ok = False
    if (-4.5 - a) < x < (-1.5 + a) and (-4.5 - a) < y < (-1.5 + a):
        goal_ok = False
    if (-7.5 - a) < x < (-5.5 + a) and (5.5 - a) < y < (7.5 + a):
        goal_ok = False
    if (-5.5 - a) < x < (-4.5 + a) and (3.0 - a) < y < (4.0 + a):
        goal_ok = False
    if (-5.5 - a) < x < (-4.5 + a) and (-7.0 - a) < y < (-6.0 + a):
        goal_ok = False
    if (4.5 - a) < x < (5.5 + a) and (5.0 - a) < y < (6.0 + a):
        goal_ok = False
    if (5.5 - a) < x < (6.5 + a) and (-6.5 - a) < y < (-5.5 + a):
        goal_ok = False

    if x > 6.5 or x < -6.5 or y > 6.5 or y < -6.5:
        goal_ok = False

    return goal_ok

def visualize_forbidden_zones():
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 定义扩展范围
    a = 0.5
    
    # 定义禁区列表：(x_min, x_max, y_min, y_max)
    forbidden_zones = [
        ((1.5 - a), (4.5 + a), (1.5 - a), (4.5 + a)),
        ((-5 - a), (-0.5 + a), (1.5 - a), (4.5 + a)),
        ((-5.5 - a), (-2.5 + a), (0.5 - a), (5 + a)),
        ((0.5 - a), (5 + a), (-5.5 - a), (-2.5 + a)),
        ((2.5 - a), (5.5 + a), (-5 - a), (-0.5 + a)),
        ((-4.5 - a), (-1.5 + a), (-4.5 - a), (-1.5 + a)),
        ((-7.5 - a), (-5.5 + a), (5.5 - a), (7.5 + a)),
        ((-5.5 - a), (-4.5 + a), (3.0 - a), (4.0 + a)),
        ((-5.5 - a), (-4.5 + a), (-7.0 - a), (-6.0 + a)),
        ((4.5 - a), (5.5 + a), (5.0 - a), (6.0 + a)),
        ((5.5 - a), (6.5 + a), (-6.5 - a), (-5.5 + a)),
    ]
    
    # 绘制每个禁区
    for x_min, x_max, y_min, y_max in forbidden_zones:
        rect = patches.Rectangle(
            (x_min, y_min),  # 左下角坐标
            x_max - x_min,   # 宽度
            y_max - y_min,   # 高度
            linewidth=1,
            edgecolor='r',
            facecolor='red',
            alpha=0.3  # 半透明
        )
        ax.add_patch(rect)
    
    # 绘制外部边界（未扩展的原始边界）
    outer_boundary = 6.5
    ax.plot([-outer_boundary, outer_boundary], [-outer_boundary, -outer_boundary], 'b--', label='Outer Boundary')
    ax.plot([-outer_boundary, outer_boundary], [outer_boundary, outer_boundary], 'b--')
    ax.plot([-outer_boundary, -outer_boundary], [-outer_boundary, outer_boundary], 'b--')
    ax.plot([outer_boundary, outer_boundary], [-outer_boundary, outer_boundary], 'b--')
    
    # 设置坐标轴范围
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    
    # 添加网格和标签
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Forbidden Zones Visualization (a=0.5)')
    ax.legend()
    
    # 设置坐标轴比例相等
    ax.set_aspect('equal', adjustable='box')
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    visualize_forbidden_zones()