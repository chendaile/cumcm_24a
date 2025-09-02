import numpy as np
from math import sin, cos, atan2, sqrt, pi
import sympy as sp
from sympy import sqrt as sp_sqrt
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class Uturn_system:
    def __init__(self, r_enter_point, r_exit_point, a, b, r_circle=450):
        self.a = a
        self.b = b
        self.enter_point = Uturn_point(
            r_enter_point, on_entry_trajectory=True, a=self.a, b=self.b)
        self.exit_point = Uturn_point(
            r_exit_point, on_exit_trajectory=True, a=self.a, b=self.b)
        self.circle_point_enter = Uturn_point(
            r_circle, on_exit_trajectory=True, a=self.a, b=self.b)
        self.circle_exit_enter = Uturn_point(
            r_circle, on_exit_trajectory=True, a=self.a, b=self.b)

        r_2r = sp.Symbol('r_2r', real=True, positive=True)
        r_r = sp.Symbol('r_r', real=True, positive=True)
        theta_2r = sp.Symbol('theta_2r', real=True)
        theta_r = sp.Symbol('theta_r', real=True)
        self.center_circle_2r = Uturn_point(2*r_2r, theta_2r)
        self.center_circle_r = Uturn_point(r_r, theta_r)
        self.unknowns = {'r': r_r, 'r_2r': r_2r,
                         'theta_2r': theta_2r, 'theta_r': theta_r}

    def get_distance_twopoint(self, point1, point2):
        return sp_sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def solve(self):
        # 构建方程组
        distance_2r_enter = self.get_distance_twopoint(
            self.enter_point, self.center_circle_2r)
        distance_r_exit = self.get_distance_twopoint(
            self.exit_point, self.center_circle_r)
        distance_r_2r = self.get_distance_twopoint(
            self.center_circle_2r, self.center_circle_r)

        # 约束方程
        eq1 = sp.Eq(distance_2r_enter, distance_r_exit * 2)
        eq2 = sp.Eq(distance_r_2r, distance_r_exit * 3)

        # 切线垂直约束
        vector_2r_enter = sp.Matrix([
            self.center_circle_2r.x - self.enter_point.x,
            self.center_circle_2r.y - self.enter_point.y
        ])
        tangent_2r = sp.Matrix([1, self.enter_point.tangent])
        eq3 = sp.Eq(tangent_2r.dot(vector_2r_enter), 0)

        vector_r_exit = sp.Matrix([
            self.center_circle_r.x - self.exit_point.x,
            self.center_circle_r.y - self.exit_point.y
        ])
        tangent_r = sp.Matrix([1, self.exit_point.tangent])
        eq4 = sp.Eq(tangent_r.dot(vector_r_exit), 0)

        # 求解方程组
        variables = [self.unknowns['r'], self.unknowns['r_2r'],
                     self.unknowns['theta_2r'], self.unknowns['theta_r']]

        solution = sp.solve([eq1, eq2, eq3, eq4], variables)
        return solution

    def solve_numerical(self):
        """数值求解方法"""
        def equations(vars):
            r_r_val, r_2r_val, theta_2r_val, theta_r_val = vars

            # 计算圆心坐标
            x_2r = 2*r_2r_val * cos(theta_2r_val)
            y_2r = 2*r_2r_val * sin(theta_2r_val)
            x_r = r_r_val * cos(theta_r_val)
            y_r = r_r_val * sin(theta_r_val)

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

        # 初始猜测值
        initial_guess = [200, 100, 0, pi]  # r_r, r_2r, theta_2r, theta_r

        solution = fsolve(equations, initial_guess)
        return {
            'r_r': solution[0],
            'r_2r': solution[1],
            'theta_2r': solution[2],
            'theta_r': solution[3]
        }

    def visualize(self, solution):
        """可视化几何元素"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # 螺旋轨迹
        theta_range = np.linspace(-4*pi, 4*pi, 1000)
        r_entry = self.a + self.b * theta_range
        r_exit = self.a + self.b * (theta_range + pi)

        x_entry = r_entry * np.cos(theta_range)
        y_entry = r_entry * np.sin(theta_range)
        x_exit = r_exit * np.cos(theta_range)
        y_exit = r_exit * np.sin(theta_range)

        # 绘制螺旋线
        ax.plot(x_entry, y_entry, 'b-', alpha=0.6, label='Entry trajectory')
        ax.plot(x_exit, y_exit, 'r-', alpha=0.6, label='Exit trajectory')

        # 标记关键点
        ax.plot(self.enter_point.x, self.enter_point.y,
                'bo', markersize=8, label='Enter point')
        ax.plot(self.exit_point.x, self.exit_point.y,
                'ro', markersize=8, label='Exit point')

        # 计算并绘制圆心
        r_r = solution['r_r']
        r_2r = solution['r_2r']
        theta_2r = solution['theta_2r']
        theta_r = solution['theta_r']

        # 大圆圆心 (半径 = 2*r_2r)
        center_2r_x = 2*r_2r * cos(theta_2r)
        center_2r_y = 2*r_2r * sin(theta_2r)

        # 小圆圆心
        center_r_x = r_r * cos(theta_r)
        center_r_y = r_r * sin(theta_r)

        ax.plot(center_2r_x, center_2r_y, 'go', markersize=10,
                label=f'Large circle center (r={2*r_2r:.1f})')
        ax.plot(center_r_x, center_r_y, 'mo', markersize=10,
                label=f'Small circle center (r={r_r:.1f})')

        # 绘制圆
        circle_2r = plt.Circle((center_2r_x, center_2r_y),
                               2*r_2r, fill=False, color='green', linewidth=2)
        circle_r = plt.Circle((center_r_x, center_r_y),
                              r_r, fill=False, color='magenta', linewidth=2)
        ax.add_patch(circle_2r)
        ax.add_patch(circle_r)

        # 绘制连接线
        ax.plot([self.enter_point.x, center_2r_x], [self.enter_point.y,
                center_2r_y], 'g--', alpha=0.7, label='Enter-Center2r')
        ax.plot([self.exit_point.x, center_r_x], [self.exit_point.y,
                center_r_y], 'm--', alpha=0.7, label='Exit-Centerr')
        ax.plot([center_2r_x, center_r_x], [center_2r_y, center_r_y],
                'k--', alpha=0.7, label='Center-Center')

        # 绘制切线向量
        if hasattr(self.enter_point, 'tangent'):
            tangent_scale = 100
            tangent_end_x = self.enter_point.x + tangent_scale
            tangent_end_y = self.enter_point.y + tangent_scale * self.enter_point.tangent
            ax.arrow(self.enter_point.x, self.enter_point.y,
                     tangent_scale, tangent_scale * self.enter_point.tangent,
                     head_width=20, head_length=30, fc='blue', ec='blue', alpha=0.7)

        if hasattr(self.exit_point, 'tangent'):
            tangent_scale = 100
            tangent_end_x = self.exit_point.x + tangent_scale
            tangent_end_y = self.exit_point.y + tangent_scale * self.exit_point.tangent
            ax.arrow(self.exit_point.x, self.exit_point.y,
                     tangent_scale, tangent_scale * self.exit_point.tangent,
                     head_width=20, head_length=30, fc='red', ec='red', alpha=0.7)

        ax.set_xlim(-800, 800)
        ax.set_ylim(-800, 800)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title('Dragon Dance Spiral Trajectory with Circle Centers')

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
            if isinstance(self.r, (int, float)) and isinstance(self.theta, (int, float)):
                # 数值计算
                self.x = self.r * cos(self.theta)
                self.y = self.r * sin(self.theta)
            else:
                # 符号计算
                self.x = self.r * sp.cos(self.theta)
                self.y = self.r * sp.sin(self.theta)
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
                if isinstance(self.theta, (int, float)) and isinstance(self.r, (int, float)):
                    # 数值计算
                    self.tangent = (sin(self.theta) * self.b + cos(self.theta) *
                                    self.r) / (cos(self.theta) * self.b - sin(self.theta) * self.r)
                else:
                    # 符号计算
                    self.tangent = (sp.sin(self.theta) * self.b + sp.cos(self.theta) *
                                    self.r) / (sp.cos(self.theta) * self.b - sp.sin(self.theta) * self.r)
                    self.tangent_vector = sp.Matrix([1, self.tangent])
                    self.normal_vector = sp.Matrix([self.tangent, -1])


# 创建实例并求解
if __name__ == "__main__":
    system = Uturn_system(
        r_enter_point=400,
        r_exit_point=400,
        a=16*170,
        b=-170/(2*pi)
    )

    print("开始求解...")
    print(f"进入点坐标: ({system.enter_point.x}, {system.enter_point.y})")
    print(f"退出点坐标: ({system.exit_point.x}, {system.exit_point.y})")

    # 使用数值求解
    try:
        solution = system.solve_numerical()
        print("数值求解结果：")
        for key, value in solution.items():
            print(f"{key}: {value:.6f}")

        # 可视化结果
        system.visualize(solution)

    except Exception as e:
        print(f"数值求解出错: {e}")
