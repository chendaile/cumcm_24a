import numpy as np
from math import sin, cos, atan2, sqrt, pi
import sympy as sp


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
        return sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def solve(self):
        # 方程组
        distance_2r_enter = self.get_distance_twopoint(
            self.enter_point, self.center_circle_2r)
        distance_r_exit = self.get_distance_twopoint(
            self.exit_point, self.center_circle_r)
        distance_r_2r = self.get_distance_twopoint(
            self.center_circle_2r, self.center_circle_r)
        vector_2r_enter = np.array(
            [self.center_circle_2r.x - self.enter_point.x, self.center_circle_2r.y - self.enter_point.y])
        self.enter_point.tangent_vector * vector_2r_enter = 0
        vector_r_exit = np.array(
            [self.center_circle_r.x - self.exit_point.x, self.center_circle_r.y - self.exit_point.y])
        self.exit_point.tangent_vector * vector_r_exit = 0


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
        if self.r and self.theta:
            self.x = self.r * cos(self.theta)
            self.y = self.r * sin(self.theta)
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
        if self.on_entry_trajectory or self.on_entry_trajectory:
            self.tangent = (sin(self.theta) * self.b + cos(self.theta) *
                            self.r) / (cos(self.theta) * self.b - sin(self.theta) * self.r)
            self.tangent_vector = np.array(
                [1, self.tangent]) / sqrt(1 + self.tangent**2)
            self.normal_vector = np.array(
                [self.tangent, -1]) / sqrt(1 + self.tangent**2)
