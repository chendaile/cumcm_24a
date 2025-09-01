import numpy as np
import pandas as pd
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

    def get_polar_pos(self, linear_pos, a=16*55, b=55/(pi*2)):
        """
        The spiral formula is r = a - bθ
        """
        self.a = a
        self.b = b
        theta = (a - sqrt(a**2 - 2*linear_pos*b)) / b
        r = a - b * theta
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


class BenchDragon:
    def __init__(self, moment, a=16*55, b=55/(pi*2)):
        """
        Time unit is seconds
        """
        self.a = a
        self.b = b
        self.moment = moment
        self.NodeSum = 223
        self.Nodes = {}

        HeadSpeed = 100
        self.Head_linear_pos = HeadSpeed * moment
        self.Nodes[1] = BenchDragonNode(341, 1, 100, self.Head_linear_pos)
        for i in range(2, self.NodeSum+1):
            front_linear_pos_i = self.Nodes[i-1].back_linear_pos
            front_velocity_i = self.Nodes[i-1].back_velocity
            self.Nodes[i] = BenchDragonNode(
                220, i, front_velocity_i, front_linear_pos=front_linear_pos_i)

    def get_moment_pos(self):
        moment_content = {}
        moment_content['龙头x (m)'], moment_content['龙头y (m)'] = self.Nodes[1].front_xy
        for i in range(2, self.NodeSum):
            moment_content[f'第{i-1}节龙身x (m)'], moment_content[f'第{i-1}节龙身y (m)'] = self.Nodes[i].front_xy \
                if self.Nodes[i].front_xy else None, None
        moment_content['龙尾x (m)'], moment_content['龙尾y (m)'] = self.Nodes[self.NodeSum].front_xy \
            if self.Nodes[self.NodeSum].front_xy else None, None
        moment_content['龙尾（后）x (m)'], moment_content['龙尾（后）y (m)'] = self.Nodes[self.NodeSum].back_xy \
            if self.Nodes[self.NodeSum].back_xy else None, None
        return moment_content

    def get_moment_velo(self):
        moment_content = {}
        moment_content['龙头 (m/s)'] = self.Nodes[1].front_velocity
        for i in range(2, self.NodeSum):
            moment_content[f'第{i-1}节龙身  (m/s)'] = self.Nodes[i].front_velocity
        moment_content['龙尾  (m/s)'] = self.Nodes[self.NodeSum].front_velocity
        moment_content['龙尾（后） (m/s)'] = self.Nodes[self.NodeSum].back_velocity
        return moment_content

    def show_allNodes(self):
        show_contents = {}
        for i in range(1, self.NodeSum+1):
            node_i = self.Nodes[i]
            content = {
                'front xy': node_i.front_xy,
                'back xy': node_i.back_xy,
                'front polar': node_i.front_polar_pos,
                'back polar': node_i.back_polar_pos,
                'front velocity': node_i.front_velocity,
                'back velocity': node_i.back_velocity,
                'front_ang_diff_cos': node_i.front_ang_diff_cos,
                'back_ang_diff_cos': node_i.back_ang_diff_cos
            }
            show_contents[f'节点{i}'] = content
        show_contents = pd.DataFrame(show_contents)
        show_contents.to_csv(
            f"./output/Q1/preview_contents-{self.moment}s.csv")

    def visualize_Nodes(self):
        plt.figure(figsize=(12, 12))
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

        theta_track = np.linspace(0, 16*2*pi, 1000)
        r_track = self.a - self.b * theta_track
        x_track = r_track * np.cos(-theta_track)
        y_track = r_track * np.sin(-theta_track)
        plt.plot(x_track, y_track, 'k--', alpha=0.3,
                 linewidth=1, label='Spiral Track')

        front_x, front_y = [], []
        back_x, back_y = [], []
        node_indices = []

        for i in range(1, self.NodeSum+1):
            node = self.Nodes[i]

            if node.front_xy is not None:
                front_x.append(node.front_xy[0])
                front_y.append(node.front_xy[1])
                node_indices.append(i)

            if node.back_xy is not None:
                back_x.append(node.back_xy[0])
                back_y.append(node.back_xy[1])

        if front_x:
            plt.scatter(front_x, front_y, c='red', s=30,
                        alpha=0.7, label='Node Front', marker='o')
        if back_x:
            plt.scatter(back_x, back_y, c='blue', s=30,
                        alpha=0.7, label='Node Back', marker='s')

        for i in range(1, self.NodeSum+1):
            node = self.Nodes[i]
            if node.front_xy is not None and node.back_xy is not None:
                plt.plot([node.front_xy[0], node.back_xy[0]],
                         [node.front_xy[1], node.back_xy[1]],
                         'gray', alpha=0.5, linewidth=1)
        for i in range(1, min(11, len(node_indices)+1)):
            node = self.Nodes[i]
            if node.front_xy is not None:
                plt.annotate(f'{i}', (node.front_xy[0], node.front_xy[1]),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(f'Dragon Nodes Distribution (t={self.moment}s)')
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')
        plt.savefig(f'./output/Q1/dragon_nodes_visualization-{self.moment}s.png',
                    dpi=800, bbox_inches='tight')
        plt.show()
        print(
            f"在轨道上的节点数量: {len([i for i in range(1, self.NodeSum+1) if self.Nodes[i].front_enter])}")
        print(
            f"可视化图片已保存至: ./output/Q1/dragon_nodes_visualization-{self.moment}s.png")


def main():
    pos_chart = {}
    velo_chart = {}
    for moment in range(0, 300):
        BenchDragon_moment = BenchDragon(moment)
        pos_chart[f"{moment} s"] = BenchDragon_moment.get_moment_pos()
        velo_chart[f"{moment} s"] = BenchDragon_moment.get_moment_velo()
    pos_chart = pd.DataFrame(pos_chart)
    velo_chart = pd.DataFrame(velo_chart)
    with pd.ExcelWriter("./output/Q1/result1.xlsx", engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        pos_chart.to_excel(writer, sheet_name='位置')
        velo_chart.to_excel(writer, sheet_name='速度')


def test():
    BenchDragon_t = BenchDragon(300)
    BenchDragon_t.show_allNodes()
    BenchDragon_t.visualize_Nodes()


if __name__ == "__main__":
    main()
