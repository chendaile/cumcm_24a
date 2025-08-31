import numpy 
import pandas
import matplotlib.pyplot as plt 
from math import pi, sqrt, atan, sin, cos

class BenchDragonNode:
    def __init__(self, length, index, front_velocity,
                 head_linear_pos: float = None, front_linear_pos: float = None, 
                 width=30, hole_distance=27.5):
        """
        All units are centimeters
        """
        self.length = length
        self.index = index
        self.width = width
        self.hole_distance = hole_distance

        if front_linear_pos is None:
            self.front_linear_pos = head_linear_pos - self.hole_distance
        else:
            self.front_linear_pos = front_linear_pos
        self.back_linear_pos = self.front_linear_pos - (self.length - 2*self.hole_distance)
        
        self.front_enter = False if self.front_linear_pos < 0 else True
        self.back_enter = False if self.back_linear_pos < 0 else True

        self.front_polar_pos = self.get_polar_pos(self.front_linear_pos) if self.front_enter else None
        self.back_polar_pos = self.get_polar_pos(self.back_linear_pos) if self.back_enter else None
        self.board_ang = self.get_board_ang()
        self.front_tangent_ang = self.get_tangent_ang(self.front_polar_pos)
        self.back_tangent_ang = self.get_tangent_ang(self.back_polar_pos)

    def get_polar_pos(self, linear_pos, a=16*55, b=55/(pi*2)):
        """
        The spiral formula is r = a - bθ
        """
        self.a = a; self.b = b
        theta = (a - sqrt(a**2 - 2*linear_pos*b)) / b
        r = a - b * theta
        return (r, theta)
    
    def get_board_ang(self):
        """
        result = arctan((r₂sin θ₂ - r₁sin θ₁) / (r₂cos θ₂ - r₁cos θ₁))
        """
        if self.back_enter and self.front_enter:
            r1, theta1 = self.front_polar_pos
            r2, theta2 = self.back_polar_pos
            return atan((r2*sin(theta2)-r1*sin(theta1)) / (r2*cos(theta2)-r1*cos(theta1))) 
        else:
            return None

    def get_tangent_ang(self, polar_pos):
        """
        result = -θ + arctan((a - bθ) / b)
        """
        if polar_pos:
            theta = polar_pos[1]
            return -theta + atan((self.a - self.b * theta) / self.b)
        else:
            return None

class BenchDragon:
    def __init__(self, moment):
        """
        Time unit is seconds
        """
        self.NodeSum = 223
        self.Nodes = {}
        
        HeadSpeed = 100
        self.Head_linear_pos = HeadSpeed * moment
        self.Nodes[1] = BenchDragonNode(341, 1, self.Head_linear_pos)
        for i in range(2, self.NodeSum+1):
            front_linear_pos_i = self.Nodes[i-1].back_linear_pos
            self.Nodes[i] = BenchDragonNode(220, i, front_linear_pos=front_linear_pos_i)
        
    