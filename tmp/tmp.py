import numpy as np
from math import sin, cos, atan2, sqrt, pi


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

    def add_reference_coor(self, reference_point_x, reference_point_y, rotation_angle=0):
        """Add reference coordinate system with optional rotation"""
        self.x_refer = self.x - reference_point_x
        self.y_refer = self.y - reference_point_y

        self.x_refer_rotated = self.x_refer * \
            cos(rotation_angle) + self.y_refer * sin(rotation_angle)
        self.y_refer_rotated = -self.x_refer * \
            sin(rotation_angle) + self.y_refer * cos(rotation_angle)
        self.r_refer, self.theta_refer = self.xy_to_polar(
            self.x_refer_rotated, self.y_refer_rotated)

    def xy_to_polar(self, x, y):
        r = sqrt(x**2 + y**2)
        theta = atan2(y, x)
        return r, theta

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
        self.tangent = (sin(self.theta) * self.b + cos(self.theta) *
                        self.r) / (cos(self.theta) * self.b - sin(self.theta) * self.r)
        self.tangent_vector = np.array(
            [1, self.tangent]) / sqrt(1 + self.tangent**2)
        self.normal_vector =
