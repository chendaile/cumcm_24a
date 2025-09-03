import numpy as np
from math import sin, cos, atan, atan2, sqrt, pi
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class Uturn_system:
    def __init__(self, r_enter_point, r_exit_point, Spiral_spacing=170, r_circle=450):
        self.r_circle = r_circle
        self.a = 16*Spiral_spacing
        self.b = -Spiral_spacing / (pi*2)
        self.enter_point = Uturn_point(
            r_enter_point, on_entry_trajectory=True, a=self.a, b=self.b)
        self.exit_point = Uturn_point(
            r_exit_point, on_exit_trajectory=True, a=self.a, b=self.b)
        self.circle_point_enter = Uturn_point(
            r_circle, on_entry_trajectory=True, a=self.a, b=self.b)
        self.circle_exit_enter = Uturn_point(
            r_circle, on_exit_trajectory=True, a=self.a, b=self.b)
        self.enter_length = self.calculus_length(0, self.enter_point.theta)
        self.exit_length = self.calculus_length(-pi, self.exit_point.theta)

    def calculus_length(self, theta1, theta2):
        return self.a * (theta2 - theta1) + 1/2 * self.b * (theta2**2 - theta1**2)

    def get_distance_twopoint(self, point1, point2):
        return sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

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
        self.Arc_circumference = self.radius_2r * \
            abs(self.angle_diff_2r) + self.radius_r * abs(self.angle_diff_r)
        self.Arc_circumference_2r = self.radius_2r * abs(self.angle_diff_2r)
        self.Arc_circumference_r = self.radius_r * abs(self.angle_diff_r)
        self.enter_point.y = -self.enter_point.y
        self.exit_point.y = -self.exit_point.y

    def calculate_external_tangent_points(self, c1, r1, c2, r2):
        """计算两个外切圆的公切线切点"""
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
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
        self.angle_diff_2r = tangent_angle_large - enter_angle
        if self.angle_diff_2r > pi:
            self.angle_diff_2r -= 2*pi
        elif self.angle_diff_2r < -pi:
            self.angle_diff_2r += 2*pi
        angles_arc1 = np.linspace(
            enter_angle, enter_angle + self.angle_diff_2r, 100)

        self.x_arc1 = self.center_2r[0] + self.radius_2r * np.cos(angles_arc1)
        self.y_arc1 = self.center_2r[1] + self.radius_2r * np.sin(angles_arc1)

        # 小圆弧方向：选择短弧，与大圆弧方向相反
        self.angle_diff_r = exit_angle - tangent_angle_small
        if self.angle_diff_r > pi:
            self.angle_diff_r -= 2*pi
        elif self.angle_diff_r < -pi:
            self.angle_diff_r += 2*pi

        # 判断大圆弧方向
        large_arc_clockwise = self.angle_diff_2r < 0

        # 小圆弧与大圆弧方向相反
        if large_arc_clockwise:
            # 大圆弧顺时针，小圆弧逆时针
            self.angle_diff_r = abs(self.angle_diff_r)
        else:
            # 大圆弧逆时针，小圆弧顺时针
            self.angle_diff_r = -abs(self.angle_diff_r)

        angles_arc2 = np.linspace(
            tangent_angle_small, tangent_angle_small + self.angle_diff_r, 100)

        self.x_arc2 = self.center_r[0] + self.radius_r * np.cos(angles_arc2)
        self.y_arc2 = self.center_r[1] + self.radius_r * np.sin(angles_arc2)

    def get_position_at_distance(self, linear_pos):
        """根据累积距离获取最优路径上的坐标 - 核心路径映射方法"""

        # 段1: 螺旋进入段 [0, enter_length]
        if linear_pos <= self.enter_length:
            return self._get_spiral_entry_position(linear_pos)

        # 段2: 大圆弧段 [enter_length, enter_length + Arc_circumference_2r]
        elif linear_pos <= self.enter_length + self.Arc_circumference_2r:
            arc_progress = linear_pos - self.enter_length
            return self._interpolate_arc1_position(arc_progress)

        # 段3: 小圆弧段 [enter_length + Arc_circumference_2r, enter_length + Arc_circumference]
        elif linear_pos <= self.enter_length + self.Arc_circumference:
            arc_progress = linear_pos - self.enter_length - self.Arc_circumference_2r
            return self._interpolate_arc2_position(arc_progress)

        # 段4: 螺旋退出段 [enter_length + Arc_circumference, ∞]
        else:
            exit_distance = linear_pos - self.enter_length - self.Arc_circumference
            return self._get_spiral_exit_position(exit_distance)

    def _get_spiral_entry_position(self, linear_pos):
        """进入螺旋段坐标计算"""
        # 计算极坐标
        theta = (self.a - sqrt(self.a**2 - 2*linear_pos*(-self.b))) / (-self.b)
        r = self.a + self.b * theta
        # 转换为直角坐标
        x = r * cos(theta)
        y = r * sin(theta)
        return (x, -y)  # 镜像y坐标

    def _interpolate_arc1_position(self, arc_distance):
        """大圆弧段内插值获取坐标"""
        if self.Arc_circumference_2r == 0:
            return (self.x_arc1[0], self.y_arc1[0])

        # 将弧长距离转换为数组索引参数
        progress = arc_distance / self.Arc_circumference_2r
        progress = max(0, min(1, progress))  # 限制在[0,1]范围

        index = progress * (len(self.x_arc1) - 1)
        idx = int(index)
        frac = index - idx

        if idx >= len(self.x_arc1) - 1:
            return (self.x_arc1[-1], self.y_arc1[-1])

        x = self.x_arc1[idx] + frac * (self.x_arc1[idx+1] - self.x_arc1[idx])
        y = self.y_arc1[idx] + frac * (self.y_arc1[idx+1] - self.y_arc1[idx])

        return (x, y)

    def _interpolate_arc2_position(self, arc_distance):
        """小圆弧段内插值获取坐标"""
        if self.Arc_circumference_r == 0:
            return (self.x_arc2[0], self.y_arc2[0])

        progress = arc_distance / self.Arc_circumference_r
        progress = max(0, min(1, progress))

        index = progress * (len(self.x_arc2) - 1)
        idx = int(index)
        frac = index - idx

        if idx >= len(self.x_arc2) - 1:
            return (self.x_arc2[-1], self.y_arc2[-1])

        x = self.x_arc2[idx] + frac * (self.x_arc2[idx+1] - self.x_arc2[idx])
        y = self.y_arc2[idx] + frac * (self.y_arc2[idx+1] - self.y_arc2[idx])

        return (x, y)

    def _get_spiral_exit_position(self, exit_distance):
        """退出螺旋段坐标计算 - 盘出曲线，从内向外螺旋，与盘入对称"""
        # 盘出曲线的正确理解：
        # - 起点：内圈（大θ，小r）
        # - 终点：外圈（小θ，大r）
        # - 螺旋公式：r = a - bθ，当θ减小时，r增大
        
        # 调试信息
        param_value = self.exit_length - exit_distance
        print(f"DEBUG: exit_distance={exit_distance}, exit_length={self.exit_length}")
        print(f"DEBUG: 传入_get_spiral_entry_position的参数={param_value}")
        
        if param_value < 0:
            print(f"WARNING: 参数为负数，可能导致数学域错误")
            return (0, 0)  # 临时返回原点
        
        x, y = self._get_spiral_entry_position(param_value)
        return (-x, -y)


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
        r_enter_point=300,
        r_exit_point=300
    )

    print("开始求解...")
    system.solve_numerical()
    print(f"radius_2r: {system.radius_2r}, radius_r: {system.radius_r}")
    print(
        f"angle_diff_2r: {system.angle_diff_2r}, angle_diff_r: {system.angle_diff_r}")
    print(f"所得弧长为{system.Arc_circumference}cm")
    print("=== 开始绘制可视化图形 ===")
    system.visualize_test()


if __name__ == "__main__":
    test()
