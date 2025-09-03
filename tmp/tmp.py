import numpy as np
from math import sin, cos, atan2, sqrt, pi
import sympy as sp
from sympy import sqrt as sp_sqrt
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class Uturn_system:
    def __init__(self, r_enter_point, r_exit_point, a, b, r_circle=450):
        self.r_circle = r_circle
        self.a = a
        self.b = b
        self.enter_point = Uturn_point(
            r_enter_point, on_entry_trajectory=True, a=self.a, b=self.b)
        self.exit_point = Uturn_point(
            r_exit_point, on_exit_trajectory=True, a=self.a, b=self.b)
        self.circle_point_enter = Uturn_point(
            r_circle, on_entry_trajectory=True, a=self.a, b=self.b)
        self.circle_exit_enter = Uturn_point(
            r_circle, on_exit_trajectory=True, a=self.a, b=self.b)

    def get_distance_twopoint(self, point1, point2):
        return sp_sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def solve_numerical(self):
        """数值求解方法"""
        def equations(vars):
            r_r_val, r_2r_val, theta_2r_val, theta_r_val = vars

            center_circle_2r = Uturn_point(r_2r_val, theta_2r_val)
            center_circle_r = Uturn_point(r_r_val, theta_r_val)
            x_2r = center_circle_2r.x
            y_2r = center_circle_2r.y
            x_r = center_circle_r.x
            y_r = center_circle_r.y

            # 距离约束
            dist_2r_enter = sqrt((x_2r - self.enter_point.x)
                                 ** 2 + (y_2r - self.enter_point.y)**2)
            dist_r_exit = sqrt((x_r - self.exit_point.x) **
                               2 + (y_r - self.exit_point.y)**2)
            dist_r_2r = sqrt((x_2r - x_r)**2 + (y_2r - y_r)**2)

            eq1 = dist_2r_enter - 2 * dist_r_exit
            eq2 = dist_r_2r - 3 * dist_r_exit

            # 切线垂直约束
            vec_2r_enter = np.array(
                [x_2r - self.enter_point.x, y_2r - self.enter_point.y])
            tangent_enter = np.array([1, self.enter_point.tangent])
            eq3 = np.dot(vec_2r_enter, tangent_enter)

            vec_r_exit = np.array(
                [x_r - self.exit_point.x, y_r - self.exit_point.y])
            tangent_exit = np.array([1, self.exit_point.tangent])
            eq4 = np.dot(vec_r_exit, tangent_exit)

            return [eq1, eq2, eq3, eq4]

        guess = [200, 100, 0, pi]
        solution = fsolve(equations, guess, xtol=1e-12, maxfev=5000)
        residual = equations(solution)
        error = sum(abs(r) for r in residual)

        print(f"  解: r_r={solution[0]:.3f}, r_2r={solution[1]:.3f}, "
              f"θ_2r={solution[2]:.3f}, θ_r={solution[3]:.3f}")
        print(f"\n解误差: {error:.2e}")
        return {
            'r_r': solution[0],
            'r_2r': solution[1],
            'theta_2r': solution[2],
            'theta_r': solution[3],
            'residual_error': error
        }

    def visualize(self, solution):
        """可视化几何元素"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))

        # 螺旋轨迹 - 从theta=0开始到恰好经过enter和exit点
        # 对于进入轨迹: r = a + b*theta, 计算enter点对应的theta值
        # 对于退出轨迹: r = a + b*(theta+pi), 计算exit点对应的theta值
        theta_max_entry = self.enter_point.theta
        theta_max_exit = self.exit_point.theta
        theta_range_entry = np.linspace(0, theta_max_entry, 1000)
        theta_range_exit = np.linspace(-pi, theta_max_exit, 1000)
        r_entry = self.a + self.b * theta_range_entry
        r_exit = self.a + self.b * (theta_range_exit + pi)

        x_entry = r_entry * np.cos(theta_range_entry)
        y_entry = r_entry * np.sin(theta_range_entry)
        x_exit = r_exit * np.cos(theta_range_exit)
        y_exit = r_exit * np.sin(theta_range_exit)

        # 绘制螺旋线
        ax.plot(x_entry, y_entry, 'b-', linewidth=2,
                alpha=0.7, label='Entry trajectory')
        ax.plot(x_exit, y_exit, 'r-', linewidth=2,
                alpha=0.7, label='Exit trajectory')

        # 绘制r_circle圆圈
        boundary_circle = plt.Circle((0, 0), self.r_circle,
                                     fill=False, color='orange', linewidth=2,
                                     linestyle='--', alpha=0.8, label=f'Boundary circle (r={self.r_circle})')
        ax.add_patch(boundary_circle)

        # 标记关键点
        ax.plot(self.enter_point.x, self.enter_point.y,
                'bo', markersize=8, label='Enter point')
        ax.plot(self.exit_point.x, self.exit_point.y,
                'ro', markersize=8, label='Exit point')

        r_r = solution['r_r']
        r_2r = solution['r_2r']
        theta_2r = solution['theta_2r']
        theta_r = solution['theta_r']
        x_2r = r_2r * cos(theta_2r)
        y_2r = r_2r * sin(theta_2r)
        x_r = r_r * cos(theta_r)
        y_r = r_r * sin(theta_r)
        dist_2r_enter = sqrt((x_2r - self.enter_point.x)
                             ** 2 + (y_2r - self.enter_point.y)**2)
        dist_r_exit = sqrt((x_r - self.exit_point.x) **
                           2 + (y_r - self.exit_point.y)**2)

        center_2r_x = r_2r * cos(theta_2r)
        center_2r_y = r_2r * sin(theta_2r)
        center_r_x = r_r * cos(theta_r)
        center_r_y = r_r * sin(theta_r)

        circle_2r = plt.Circle((center_2r_x, center_2r_y),
                               dist_2r_enter, fill=False, color='green', linewidth=2)
        circle_r = plt.Circle((center_r_x, center_r_y),
                              dist_r_exit, fill=False, color='magenta', linewidth=2)
        ax.add_patch(circle_2r)
        ax.add_patch(circle_r)

        ax.plot([self.enter_point.x, center_2r_x], [self.enter_point.y,
                center_2r_y], 'g--', alpha=0.7, label='Enter-Large center')
        ax.plot([self.exit_point.x, center_r_x], [self.exit_point.y,
                center_r_y], 'm--', alpha=0.7, label='Exit-Small center')
        ax.plot([center_2r_x, center_r_x], [center_2r_y, center_r_y],
                'k--', alpha=0.7, label='Center-Center')

        if hasattr(self.enter_point, 'tangent'):
            tangent_scale = 100
            ax.arrow(self.enter_point.x, self.enter_point.y,
                     tangent_scale, tangent_scale * self.enter_point.tangent,
                     head_width=20, head_length=30, fc='blue', ec='blue', alpha=0.7)
        if hasattr(self.exit_point, 'tangent'):
            tangent_scale = 100
            ax.arrow(self.exit_point.x, self.exit_point.y,
                     tangent_scale, tangent_scale * self.exit_point.tangent,
                     head_width=20, head_length=30, fc='red', ec='red', alpha=0.7)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title(
            'Dragon Dance Spiral Trajectory with Circle Centers', fontsize=14)
        plt.tight_layout()
        plt.show()


class Uturn_point:
    def __init__(self, r=None, theta=None,
                 on_entry_trajectory=False, on_exit_trajectory=False,
                 a=None, b=None):
        self.a = a
        self.b = b
        self.r = r
        self.theta = theta
        self.on_entry_trajectory = on_entry_trajectory
        self.on_exit_trajectory = on_exit_trajectory
        if on_entry_trajectory:
            self.feedback_entry_trajectory()
        if on_exit_trajectory:
            self.feedback_exit_trajectory()
        if self.r is not None and self.theta is not None:
            self.x = self.r * cos(self.theta)
            self.y = self.r * sin(self.theta)
        self.get_vector()

    def feedback_entry_trajectory(self):
        """Formula: r = a + b * theta"""
        if self.theta is None:
            self.theta = (self.r - self.a) / self.b
        elif self.r is None:
            self.r = self.a + self.b * self.theta

    def feedback_exit_trajectory(self):
        """Formula: r = a + b * (theta+pi)"""
        if self.theta is None:
            self.theta = (self.r - self.a) / self.b - pi
        elif self.r is None:
            self.r = self.a + self.b * (self.theta + pi)

    def get_vector(self):
        if self.on_entry_trajectory or self.on_exit_trajectory:
            if self.theta is not None and self.r is not None and self.b is not None:
                self.tangent = (sin(self.theta) * self.b + cos(self.theta) *
                                self.r) / (cos(self.theta) * self.b - sin(self.theta) * self.r)


def main():
    system = Uturn_system(
        r_enter_point=400,
        r_exit_point=400,
        a=16*170,
        b=-170/(2*pi)
    )

    print("开始求解...")
    solution = system.solve_numerical()
    print("数值求解结果：")
    for key, value in solution.items():
        print(f"{key}: {value:.6f}")

    print("\n=== 开始绘制可视化图形 ===")
    system.visualize(solution)


if __name__ == "__main__":
    main()
