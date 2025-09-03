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

        # self.Arc_circumference =
        self.r_r = solution[0]
        self.r_2r = solution[1]
        self.theta_2r = solution[2]
        self.theta_r = solution[3]

        self.center_2r = (self.r_2r * cos(self.theta_2r), -
                          self.r_2r * sin(self.theta_2r))
        self.center_r = (self.r_r * cos(self.theta_r), -
                         self.r_r * sin(self.theta_r))
        self.radius_2r = sqrt((self.center_2r[0] - self.enter_point.x)
                              ** 2 + (self.center_2r[1] - (-self.enter_point.y))**2)
        self.radius_r = sqrt((self.center_r[0] - self.exit_point.x)
                             ** 2 + (self.center_r[1] - (-self.exit_point.y))**2)
        self.get_arc()

    def calculate_external_tangent_points(self, c1, r1, c2, r2):
        """计算两个外切圆的公切线切点"""
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        d = sqrt(dx**2 + dy**2)
        ratio = r1 / (r1 + r2)
        tangent_x = c1[0] + ratio * dx
        tangent_y = c1[1] + ratio * dy

        return (tangent_x, tangent_y)

    def get_arc(self):
        self.tangent_point = self.calculate_external_tangent_points(
            self.center_2r, self.radius_2r, self.center_r, self.radius_r)
        # 计算各关键点的角度（镜像对称后）
        enter_angle = atan2(-self.enter_point.y - self.center_2r[1],
                            self.enter_point.x - self.center_2r[0])
        tangent_angle_large = atan2(self.tangent_point[1] - self.center_2r[1],
                                    self.tangent_point[0] - self.center_2r[0])
        tangent_angle_small = atan2(self.tangent_point[1] - self.center_r[1],
                                    self.tangent_point[0] - self.center_r[0])
        exit_angle = atan2(-self.exit_point.y - self.center_r[1],
                           self.exit_point.x - self.center_r[0])

        # 大圆弧方向：沿螺旋线方向，选择短弧
        angle_diff1 = tangent_angle_large - enter_angle
        if angle_diff1 > pi:
            angle_diff1 -= 2*pi
        elif angle_diff1 < -pi:
            angle_diff1 += 2*pi
        angles_arc1 = np.linspace(enter_angle, enter_angle + angle_diff1, 100)

        self.x_arc1 = self.center_2r[0] + self.radius_2r * np.cos(angles_arc1)
        self.y_arc1 = self.center_2r[1] + self.radius_2r * np.sin(angles_arc1)

        # 小圆弧方向：选择短弧，与大圆弧方向相反
        angle_diff2 = exit_angle - tangent_angle_small
        if angle_diff2 > pi:
            angle_diff2 -= 2*pi
        elif angle_diff2 < -pi:
            angle_diff2 += 2*pi

        # 判断大圆弧方向
        large_arc_clockwise = angle_diff1 < 0

        # 小圆弧与大圆弧方向相反
        if large_arc_clockwise:
            # 大圆弧顺时针，小圆弧逆时针
            angle_diff2 = abs(angle_diff2)
        else:
            # 大圆弧逆时针，小圆弧顺时针
            angle_diff2 = -abs(angle_diff2)

        angles_arc2 = np.linspace(
            tangent_angle_small, tangent_angle_small + angle_diff2, 100)

        self.x_arc2 = self.center_r[0] + self.radius_r * np.cos(angles_arc2)
        self.y_arc2 = self.center_r[1] + self.radius_r * np.sin(angles_arc2)

    def visualize_test(self, solution):
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        # 螺旋轨迹 - 从theta=0开始到恰好经过enter和exit点
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

        # 绘制螺旋线（镜像对称）
        ax.plot(x_entry, -y_entry, 'b-', linewidth=2,
                alpha=0.7, label='Entry trajectory')
        ax.plot(x_exit, -y_exit, 'r-', linewidth=2,
                alpha=0.7, label='Exit trajectory')

        # 绘制r_circle圆圈
        boundary_circle = plt.Circle((0, 0), self.r_circle,
                                     fill=False, color='orange', linewidth=2,
                                     linestyle='--', alpha=0.8, label=f'Boundary circle (r={self.r_circle})')
        ax.add_patch(boundary_circle)

        circle_2r_full = plt.Circle(self.center_2r, self.radius_2r, fill=False, color='green',
                                    linewidth=2, linestyle='--', alpha=0.6, label='Large circle (full)')
        circle_r_full = plt.Circle(self.center_r, self.radius_r, fill=False, color='magenta',
                                   linewidth=2, linestyle='--', alpha=0.6, label='Small circle (full)')
        ax.add_patch(circle_2r_full)
        ax.add_patch(circle_r_full)

        ax.plot(self.x_arc1, self.y_arc1, 'g-', linewidth=6,
                alpha=0.9, label='Path on large circle')
        ax.plot(self.x_arc2, self.y_arc2, 'm-', linewidth=6,
                alpha=0.9, label='Path on small circle')

        # 标记外切点
        ax.plot(self.tangent_point[0], self.tangent_point[1], 'ko', markersize=10,
                label='Tangent point (external)', zorder=6)
        # 标记关键点（镜像对称）
        ax.plot(self.enter_point.x, -self.enter_point.y,
                'bo', markersize=10, label='Enter point', zorder=5)
        ax.plot(self.exit_point.x, -self.exit_point.y,
                'ro', markersize=10, label='Exit point', zorder=5)
        # 绘制圆心
        ax.plot(self.center_2r[0], self.center_2r[1], 'g+', markersize=12,
                markeredgewidth=3, label='Circle centers')
        ax.plot(self.center_r[0], self.center_r[1], 'm+',
                markersize=12, markeredgewidth=3)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax.set_title(
            'Dragon Dance U-turn Path with Direction Arrows', fontsize=16, pad=20)
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


def test():
    system = Uturn_system(
        r_enter_point=250,
        r_exit_point=300,
        a=16*170,
        b=-170/(2*pi)
    )

    print("开始求解...")
    solution = system.solve_numerical()
    print("\n=== 开始绘制可视化图形 ===")
    system.visualize_test(solution)


if __name__ == "__main__":
    test()
