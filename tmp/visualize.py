import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt

# 已知参数
a = 16 * 170
b = -170 / (2 * pi)
r_enter = 400
r_exit = 400

# 计算关键点
theta_enter = (r_enter - a) / b
theta_exit = (r_exit - a) / b - pi

x_enter = r_enter * cos(theta_enter)
y_enter = r_enter * sin(theta_enter)
x_exit = r_exit * cos(theta_exit)
y_exit = r_exit * sin(theta_exit)

# 求解结果
solution = {
    'r_r': 266.819133,
    'r_2r': 67.273939,
    'theta_2r': -2.352060,
    'theta_r': 0.890190
}

# 计算圆心
center_2r_x = 2 * solution['r_2r'] * cos(solution['theta_2r'])
center_2r_y = 2 * solution['r_2r'] * sin(solution['theta_2r'])
center_r_x = solution['r_r'] * cos(solution['theta_r'])
center_r_y = solution['r_r'] * sin(solution['theta_r'])

# 可视化
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# 螺旋轨迹
theta_range = np.linspace(-4*pi, 4*pi, 1000)
r_entry = a + b * theta_range
r_exit_traj = a + b * (theta_range + pi)

# 过滤正半径
valid_entry = r_entry > 0
valid_exit = r_exit_traj > 0

x_entry = r_entry[valid_entry] * np.cos(theta_range[valid_entry])
y_entry = r_entry[valid_entry] * np.sin(theta_range[valid_entry])
x_exit_traj = r_exit_traj[valid_exit] * np.cos(theta_range[valid_exit])
y_exit_traj = r_exit_traj[valid_exit] * np.sin(theta_range[valid_exit])

# 绘制螺旋线
ax.plot(x_entry, y_entry, 'b-', alpha=0.6, label='Entry trajectory', linewidth=2)
ax.plot(x_exit_traj, y_exit_traj, 'r-', alpha=0.6, label='Exit trajectory', linewidth=2)

# 标记关键点
ax.plot(x_enter, y_enter, 'bo', markersize=10, label=f'Enter point ({x_enter:.1f}, {y_enter:.1f})')
ax.plot(x_exit, y_exit, 'ro', markersize=10, label=f'Exit point ({x_exit:.1f}, {y_exit:.1f})')

# 绘制圆心
ax.plot(center_2r_x, center_2r_y, 'go', markersize=12, 
        label=f'Large circle center ({center_2r_x:.1f}, {center_2r_y:.1f})')
ax.plot(center_r_x, center_r_y, 'mo', markersize=12,
        label=f'Small circle center ({center_r_x:.1f}, {center_r_y:.1f})')

# 绘制圆
large_radius = 2 * solution['r_2r']
small_radius = solution['r_r']

circle_2r = plt.Circle((center_2r_x, center_2r_y), large_radius, 
                       fill=False, color='green', linewidth=3, alpha=0.8)
circle_r = plt.Circle((center_r_x, center_r_y), small_radius, 
                      fill=False, color='magenta', linewidth=3, alpha=0.8)
ax.add_patch(circle_2r)
ax.add_patch(circle_r)

# 绘制连接线
ax.plot([x_enter, center_2r_x], [y_enter, center_2r_y], 'g--', 
        linewidth=2, alpha=0.7, label=f'Enter-Center2r (dist={large_radius:.1f})')
ax.plot([x_exit, center_r_x], [y_exit, center_r_y], 'm--', 
        linewidth=2, alpha=0.7, label=f'Exit-Centerr (dist={small_radius:.1f})')
ax.plot([center_2r_x, center_r_x], [center_2r_y, center_r_y], 'k--', 
        linewidth=2, alpha=0.7, label=f'Center-Center (dist={sqrt((center_2r_x-center_r_x)**2+(center_2r_y-center_r_y)**2):.1f})')

# 计算并绘制切线
# 进入点切线
tangent_enter = (sin(theta_enter) * b + cos(theta_enter) * r_enter) / (cos(theta_enter) * b - sin(theta_enter) * r_enter)
tangent_scale = 80
ax.arrow(x_enter, y_enter, tangent_scale, tangent_scale * tangent_enter,
         head_width=15, head_length=20, fc='blue', ec='blue', alpha=0.8, label='Enter tangent')

# 退出点切线  
tangent_exit = (sin(theta_exit) * b + cos(theta_exit) * r_exit) / (cos(theta_exit) * b - sin(theta_exit) * r_exit)
ax.arrow(x_exit, y_exit, tangent_scale, tangent_scale * tangent_exit,
         head_width=15, head_length=20, fc='red', ec='red', alpha=0.8, label='Exit tangent')

# 设置图形属性
ax.set_xlim(-600, 600)
ax.set_ylim(-600, 600)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_title('Dragon Dance U-turn System Visualization\n' + 
            f'Large circle radius: {large_radius:.1f}, Small circle radius: {small_radius:.1f}')

plt.tight_layout()
plt.savefig('/home/oft/cumcm_24a/tmp/uturn_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("可视化完成，图像已保存为 uturn_visualization.png")