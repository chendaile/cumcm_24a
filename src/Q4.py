import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, sqrt, atan, sin, cos
from Q4_opti import Uturn_system


class BenchDragonNode:
    def __init__(self, length, index, front_velocity, a, b,
                 head_linear_pos: float = None, front_linear_pos: float = None,
                 width=30, hole_distance=27.5):
        """
        All units are centimeters
        """
        self.a = a
        self.b = b
        self.length = length
        self.index = index
        self.width = width
        self.hole_distance = hole_distance

        if front_linear_pos is None:
            self.front_linear_pos = head_linear_pos - self.hole_distance
        else:
            self.front_linear_pos = front_linear_pos
        self.back_linear_pos = self.front_linear_pos - \
            (self.length - 2*self.hole_distance)

        self.front_enter = False if self.front_linear_pos < 0 else True
        self.back_enter = False if self.back_linear_pos < 0 else True

        self.front_polar_pos = self.get_polar_pos(
            self.front_linear_pos) if self.front_enter else None
        self.back_polar_pos = self.get_polar_pos(
            self.back_linear_pos) if self.back_enter else None
        self.board_ang = self.get_board_ang()
        self.front_tangent_ang = self.get_tangent_ang(self.front_polar_pos)
        self.back_tangent_ang = self.get_tangent_ang(self.back_polar_pos)

        self.front_velocity = front_velocity
        self.back_velocity = self.get_back_velocity()

        self.front_xy = self.get_xy(self.front_polar_pos)
        self.back_xy = self.get_xy(self.back_polar_pos)

        self.corners = self.get_bench_corners()

    def get_polar_pos(self, linear_pos):
        """
        The spiral formula is r = a - bθ
        """
        theta = (self.a - sqrt(self.a**2 - 2*linear_pos*self.b)) / self.b
        r = self.a - self.b * theta
        return (r, theta)

    def get_xy(self, polar_pos):
        if polar_pos is None:
            return None
        r, theta = polar_pos
        x = r * cos(-theta)
        y = r * sin(-theta)
        return (x, y)

    def get_board_ang(self):
        """
        result = arctan((r₂sin θ₂ - r₁sin θ₁) / (r₂cos θ₂ - r₁cos θ₁))
        """
        if self.back_enter and self.front_enter:
            r1, theta1 = self.front_polar_pos
            r2, theta2 = self.back_polar_pos
            return atan((r2*sin(-theta2)-r1*sin(-theta1)) / (r2*cos(-theta2)-r1*cos(-theta1)))
        else:
            return None

    def get_tangent_ang(self, polar_pos):
        """
        对于螺旋线 r = a - bθ，切线角度 = θ + arctan(r / (dr/dθ))
        其中 dr/dθ = -b
        所以切线角度 = θ + arctan(r / (-b)) = θ - arctan(r / b)
        """
        if polar_pos:
            r, theta = polar_pos
            tangent_angle = theta - atan(r / self.b)
            return -tangent_angle
        else:
            return None

    def get_back_velocity(self):
        if self.board_ang:
            self.front_ang_diff_cos = abs(
                cos(self.front_tangent_ang - self.board_ang))
            self.back_ang_diff_cos = abs(
                cos(self.back_tangent_ang - self.board_ang))
            return self.front_velocity * self.front_ang_diff_cos / self.back_ang_diff_cos
        else:
            self.front_ang_diff_cos = None
            self.back_ang_diff_cos = None
            return None

    def get_bench_corners(self):
        """
        计算板凳的四个角点坐标
        板凳的实际长度是 self.length，从龙头把手到龙尾把手
        返回：[(x1,y1), (x2,y2), (x3,y3), (x4,y4)] 或 None
        """
        if not (self.front_xy and self.back_xy):
            return None

        front_x, front_y = self.front_xy
        back_x, back_y = self.back_xy

        dx = back_x - front_x
        dy = back_y - front_y

        center_length = sqrt(dx**2 + dy**2)
        if center_length == 0:
            return None

        dx_norm = dx / center_length
        dy_norm = dy / center_length

        head_x = front_x - dx_norm * self.hole_distance
        head_y = front_y - dy_norm * self.hole_distance
        tail_x = back_x + dx_norm * self.hole_distance
        tail_y = back_y + dy_norm * self.hole_distance

        perp_x = -dy_norm
        perp_y = dx_norm

        half_width = self.width / 2
        corners = [
            (head_x + perp_x * half_width, head_y + perp_y * half_width),  # 头左
            (head_x - perp_x * half_width, head_y - perp_y * half_width),  # 头右
            (tail_x - perp_x * half_width, tail_y - perp_y * half_width),  # 尾右
            (tail_x + perp_x * half_width, tail_y + perp_y * half_width)   # 尾左
        ]

        return corners

    def check_collision_with(self, other_node):
        """
        检查当前板凳是否与另一个板凳碰撞
        使用分离轴定理(SAT)进行矩形碰撞检测
        """
        if not (self.corners and other_node.corners):
            return False

        # 获取两个矩形的角点
        rect1 = self.corners
        rect2 = other_node.corners

        # 检查两个矩形是否分离
        return self._rectangles_intersect(rect1, rect2)

    def _rectangles_intersect(self, rect1, rect2):
        """
        使用分离轴定理检查两个矩形是否相交
        rect1, rect2: 四个角点的列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        def get_axes(rect):
            """获取矩形的两个轴（边的方向向量）"""
            axes = []
            for i in range(4):
                p1 = rect[i]
                p2 = rect[(i + 1) % 4]
                edge = (p2[0] - p1[0], p2[1] - p1[1])
                # 获取垂直向量作为投影轴
                axis = (-edge[1], edge[0])
                # 归一化
                length = sqrt(axis[0]**2 + axis[1]**2)
                if length > 0:
                    axes.append((axis[0]/length, axis[1]/length))
            return axes

        def project_rect(rect, axis):
            """将矩形投影到轴上"""
            dots = [rect[i][0] * axis[0] + rect[i][1] * axis[1]
                    for i in range(4)]
            return min(dots), max(dots)

        # 获取所有投影轴
        axes = get_axes(rect1) + get_axes(rect2)

        # 检查每个轴上的投影是否分离
        for axis in axes:
            min1, max1 = project_rect(rect1, axis)
            min2, max2 = project_rect(rect2, axis)

            # 如果在任何轴上都不重叠，则矩形分离
            if max1 < min2 or max2 < min1:
                return False

        # 所有轴上都有重叠，矩形相交
        return True


class BenchDragon:
    def __init__(self, moment, Spiral_spacing=55):
        """
        Time unit is seconds
        """
        self.a = 16*Spiral_spacing
        self.b = Spiral_spacing / (pi*2)
        self.Spiral_spacing = Spiral_spacing
        self.moment = moment
        self.NodeSum = 223
        self.Nodes = {}

        HeadSpeed = 100
        self.Head_linear_pos = HeadSpeed * moment
        self.Nodes[1] = BenchDragonNode(
            341, 1, 100, self.a, self.b, self.Head_linear_pos)
        for i in range(2, self.NodeSum+1):
            front_linear_pos_i = self.Nodes[i-1].back_linear_pos
            front_velocity_i = self.Nodes[i-1].back_velocity
            self.Nodes[i] = BenchDragonNode(
                220, i, front_velocity_i, self.a, self.b, front_linear_pos=front_linear_pos_i)

    def check_all_collisions(self):
        collisions = []
        has_collision = False

        active_nodes = []
        for i in range(1, self.NodeSum+1):
            node = self.Nodes[i]
            if node.front_xy and node.back_xy and node.corners:
                active_nodes.append((i, node))

        for i in range(len(active_nodes)):
            for j in range(i+1, len(active_nodes)):
                idx1, node1 = active_nodes[i]
                idx2, node2 = active_nodes[j]
                if abs(idx1 - idx2) <= 1:
                    continue
                if node1.check_collision_with(node2):
                    collisions.append((idx1, idx2))
                    has_collision = True

        return has_collision, collisions


class BenchDragon_opti():
    def __init__(self, r_enter_point, r_exit_point, Spiral_spacing=170):
        self.Spiral_spacing = Spiral_spacing
        self.a = 16*Spiral_spacing
        self.b = Spiral_spacing / (pi*2)
        self.Uturn_system = Uturn_system(
            r_enter_point, r_exit_point, self.a, -self.b)
        self.Uturn_system.solve_numerical()

    def check_moments(self, moment_range=range(1331-100, 1331+101)):
        for moment in moment_range:
            self.BenchDragon = BenchDragon(moment, self.Spiral_spacing)
            has_collision, _ = self.BenchDragon.check_all_collisions()
            if moment % 100 == 0:
                self.visualize_with_collisions()
            if has_collision:
                return

    def visualize_with_collisions(self):
        plt.figure(figsize=(12, 12))
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

        # 画螺旋轨道
        theta_track = np.linspace(0, 16*2*pi, 1000)
        r_track = self.a - self.b * theta_track
        x_track = r_track * np.cos(-theta_track)
        y_track = r_track * np.sin(-theta_track)
        plt.plot(x_track, y_track, 'k--', alpha=0.3,
                 linewidth=1, label='Spiral Track')

        # 画掉头空间边界圆 (直径900cm，半径450cm)
        circle_theta = np.linspace(0, 2*pi, 100)
        circle_radius = 450  # 半径450cm
        circle_x = circle_radius * np.cos(circle_theta)
        circle_y = circle_radius * np.sin(circle_theta)
        plt.plot(circle_x, circle_y, 'r-', alpha=0.6,
                 linewidth=2, label='Turning Space Boundary (D=900cm)')

        # 检查碰撞
        has_collision, collisions = self.check_all_collisions()
        collision_nodes = set()
        for pair in collisions:
            collision_nodes.update(pair)

        # 画板凳矩形
        for i in range(1, self.NodeSum+1):
            node = self.Nodes[i]
            if node.corners:
                corners = node.corners + [node.corners[0]]  # 闭合矩形
                x_coords = [p[0] for p in corners]
                y_coords = [p[1] for p in corners]

                # 碰撞的板凳用红色高亮
                color = 'red' if i in collision_nodes else 'blue'
                alpha = 0.8 if i in collision_nodes else 0.3
                linewidth = 2 if i in collision_nodes else 1

                plt.plot(x_coords, y_coords, color=color,
                         alpha=alpha, linewidth=linewidth)

                # 填充碰撞的板凳
                if i in collision_nodes:
                    plt.fill(x_coords, y_coords, color='red', alpha=0.2)

        # 画节点中心点
        front_x, front_y = [], []
        for i in range(1, self.NodeSum+1):
            node = self.Nodes[i]
            if node.front_xy:
                front_x.append(node.front_xy[0])
                front_y.append(node.front_xy[1])

                # 碰撞节点的中心点特殊标记
                if i in collision_nodes:
                    plt.scatter(node.front_xy[0], node.front_xy[1],
                                c='red', s=50, marker='x', linewidth=3)

        if front_x:
            plt.scatter(front_x, front_y, c='black', s=20, alpha=0.7)

        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title(f'Dragon Collision Detection (Spiral distance={self.Spiral_spacing}, t={self.moment}s)\n'
                  f'Collisions: {len(collisions)} pairs')
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')

        collision_legend = plt.Line2D([0], [0], color='red', linewidth=2,
                                      label=f'Colliding Benches ({len(collision_nodes)})')
        normal_legend = plt.Line2D([0], [0], color='blue', linewidth=1,
                                   label='Normal Benches')
        boundary_legend = plt.Line2D([0], [0], color='red', linewidth=2,
                                     label='Turning Space Boundary (D=900cm)')
        plt.legend(handles=[collision_legend, normal_legend, boundary_legend])
        plt.savefig(f'./output/Q4/optimized-{self.moment}s-{self.Spiral_spacing}cm.png',
                    dpi=800, bbox_inches='tight')


# def main():


# if __name__ == "__main__":
#     main()
