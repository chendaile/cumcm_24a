import numpy as np
from math import sin, cos, atan2, sqrt, pi
import sympy as sp
from sympy import sqrt as sp_sqrt


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
            self.x = self.r * sp.cos(self.theta)
            self.y = self.r * sp.sin(self.theta)
        self.get_vector()

    # def add_reference_coor(self, reference_point_x, reference_point_y, rotation_angle=0):
    #     """Add reference coordinate system with optional rotation"""
    #     self.x_refer = self.x - reference_point_x
    #     self.y_refer = self.y - reference_point_y

    #     self.x_refer_rotated = self.x_refer * \
    #         cos(rotation_angle) + self.y_refer * sin(rotation_angle)
    #     self.y_refer_rotated = -self.x_refer * \
    #         sin(rotation_angle) + self.y_refer * cos(rotation_angle)
    #     self.r_refer, self.theta_refer = self.xy_to_polar(
    #         self.x_refer_rotated, self.y_refer_rotated)

    # def xy_to_polar(self, x, y):
    #     r = sqrt(x**2 + y**2)
    #     theta = atan2(y, x)
    #     return r, theta

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
            if self.theta is not None and self.r is not None:
                self.tangent = (sp.sin(self.theta) * self.b + sp.cos(self.theta) *
                                self.r) / (sp.cos(self.theta) * self.b - sp.sin(self.theta) * self.r)
                # For symbolic computation, we don't need to normalize here
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
    
    # 尝试数值求解
    try:
        solution = system.solve()
        print("求解结果：")
        print(solution)
    except Exception as e:
        print(f"求解出错: {e}")
        print("尝试数值求解...")
        # 可以尝试使用 nsolve 进行数值求解
