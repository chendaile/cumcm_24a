import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi, sqrt, atan, sin, cos, atan2
from scipy.optimize import minimize_scalar
import os


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

    def get_moment_pos(self):
        moment_content_x, moment_content_y = {}, {}
        if self.Nodes[1].front_xy:
            moment_content_x['龙头'] = round(
                self.Nodes[1].front_xy[0] / 100, 6)
            moment_content_y['龙头'] = round(
                self.Nodes[1].front_xy[1] / 100, 6)
        else:
            moment_content_x['龙头'], moment_content_y['龙头'] = None, None

        for i in range(2, self.NodeSum):
            if self.Nodes[i].front_xy:
                moment_content_x[f'第{i-1}节龙身'] = round(
                    self.Nodes[i].front_xy[0] / 100, 6)
                moment_content_y[f'第{i-1}节龙身'] = round(
                    self.Nodes[i].front_xy[1] / 100, 6)
            else:
                moment_content_x[f'第{i-1}节龙身'], moment_content_y[f'第{i-1}节龙身'] = None, None

        if self.Nodes[self.NodeSum].front_xy:
            moment_content_x['龙尾'] = round(
                self.Nodes[self.NodeSum].front_xy[0] / 100, 6)
            moment_content_y['龙尾'] = round(
                self.Nodes[self.NodeSum].front_xy[1] / 100, 6)
        else:
            moment_content_x['龙尾'], moment_content_y['龙尾'] = None, None

        if self.Nodes[self.NodeSum].back_xy:
            moment_content_x['龙尾（后）'] = round(
                self.Nodes[self.NodeSum].back_xy[0] / 100, 6)
            moment_content_y['龙尾（后）'] = round(
                self.Nodes[self.NodeSum].back_xy[1] / 100, 6)
        else:
            moment_content_x['龙尾（后）'], moment_content_y['龙尾（后）'] = None, None
        return moment_content_x, moment_content_y

    def get_moment_velo(self):
        moment_content = {}
        moment_content['龙头'] = round(self.Nodes[1].front_velocity /
                                     100, 6) if self.Nodes[1].front_velocity else None
        for i in range(2, self.NodeSum):
            moment_content[f'第{i-1}节龙身'] = round(
                self.Nodes[i].front_velocity / 100, 6) if self.Nodes[i].front_velocity else None
        moment_content['龙尾'] = round(self.Nodes[self.NodeSum].front_velocity /
                                     100, 6) if self.Nodes[self.NodeSum].front_velocity else None
        moment_content['龙尾（后）'] = round(self.Nodes[self.NodeSum].back_velocity /
                                        100, 6) if self.Nodes[self.NodeSum].back_velocity else None
        return moment_content

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

    def visualize_with_collisions(self):
        """
        可视化龙身并高亮显示碰撞的板凳
        """
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

        plt.savefig(f'./output/Q4/collision_visualization-{self.moment}s-{self.Spiral_spacing}cm.png',
                    dpi=800, bbox_inches='tight')

        return has_collision, collisions


class TurnAroundOptimizer:
    def __init__(self, a=16*55, b=55/(pi*2)):
        self.a = a  
        self.b = b  
        self.turn_space_radius = 450  
        
    def spiral_inbound(self, linear_pos):
        theta = (self.a - sqrt(self.a**2 - 2*linear_pos*self.b)) / self.b
        r = self.a - self.b * theta
        return r, theta
        
    def spiral_outbound(self, r, theta):
        theta_out = theta + pi
        r_out = r
        return r_out, theta_out
        
    def polar_to_cartesian(self, r, theta):
        x = r * cos(-theta)
        y = r * sin(-theta)
        return x, y
        
    def find_turn_start_position(self, turn_delay=0):
        head_speed = 100  
        base_turn_time = self.find_intersection_time()
        actual_turn_time = base_turn_time + turn_delay
        turn_linear_pos = head_speed * actual_turn_time
        
        r_start, theta_start = self.spiral_inbound(turn_linear_pos)
        x_start, y_start = self.polar_to_cartesian(r_start, theta_start)
        
        return {
            'linear_pos': turn_linear_pos,
            'polar': (r_start, theta_start),
            'cartesian': (x_start, y_start),
            'turn_radius': r_start
        }
        
    def find_intersection_time(self):
        def distance_to_boundary(t):
            linear_pos = 100 * t  
            r, theta = self.spiral_inbound(linear_pos)
            return abs(r - self.turn_space_radius)
        
        result = minimize_scalar(distance_to_boundary, bounds=(0, 50), method='bounded')
        return result.x
        
    def design_s_curve(self, start_pos, end_pos):
        x1, y1 = start_pos['cartesian']
        x2, y2 = end_pos['cartesian']
        
        dx = x2 - x1
        dy = y2 - y1
        distance = sqrt(dx**2 + dy**2)
        
        R2 = distance / 6
        R1 = 2 * R2
        
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        normal_x = -dy / distance
        normal_y = dx / distance
        
        offset1 = R1 * 0.5
        offset2 = R2 * 0.5
        
        center1_x = x1 + normal_x * offset1
        center1_y = y1 + normal_y * offset1
        
        center2_x = x2 - normal_x * offset2
        center2_y = y2 - normal_y * offset2
        
        tangent_x = (center1_x + center2_x) / 2
        tangent_y = (center1_y + center2_y) / 2
        
        return {
            'arc1': {
                'center': (center1_x, center1_y),
                'radius': R1,
                'start_angle': atan2(y1 - center1_y, x1 - center1_x),
                'end_angle': atan2(tangent_y - center1_y, tangent_x - center1_x)
            },
            'arc2': {
                'center': (center2_x, center2_y),
                'radius': R2,
                'start_angle': atan2(tangent_y - center2_y, tangent_x - center2_x),
                'end_angle': atan2(y2 - center2_y, x2 - center2_x)
            },
            'tangent_point': (tangent_x, tangent_y)
        }
        
    def calculate_curve_length(self, curve):
        arc1 = curve['arc1']
        arc2 = curve['arc2']
        
        angle1 = abs(arc1['end_angle'] - arc1['start_angle'])
        if angle1 > pi:
            angle1 = 2*pi - angle1
            
        angle2 = abs(arc2['end_angle'] - arc2['start_angle'])
        if angle2 > pi:
            angle2 = 2*pi - angle2
            
        length1 = arc1['radius'] * angle1
        length2 = arc2['radius'] * angle2
        
        return length1 + length2
        
    def optimize_turn_delay(self, max_delay=5.0):
        def objective(delay):
            try:
                start_pos = self.find_turn_start_position(delay)
                
                r_end = start_pos['polar'][0]
                theta_end = start_pos['polar'][1] + pi
                end_pos = {
                    'polar': (r_end, theta_end),
                    'cartesian': self.polar_to_cartesian(r_end, theta_end)
                }
                
                curve = self.design_s_curve(start_pos, end_pos)
                return self.calculate_curve_length(curve)
            except:
                return float('inf')
        
        result = minimize_scalar(objective, bounds=(0, max_delay), method='bounded')
        return result.x, result.fun
        
    def generate_path_points(self, curve, num_points=100):
        arc1 = curve['arc1']
        arc2 = curve['arc2']
        
        points1 = []
        start_angle1 = arc1['start_angle']
        end_angle1 = arc1['end_angle']
        
        angle_diff1 = end_angle1 - start_angle1
        if abs(angle_diff1) > pi:
            if angle_diff1 > 0:
                angle_diff1 -= 2*pi
            else:
                angle_diff1 += 2*pi
                
        for i in range(num_points//2):
            angle = start_angle1 + angle_diff1 * i / (num_points//2 - 1)
            x = arc1['center'][0] + arc1['radius'] * cos(angle)
            y = arc1['center'][1] + arc1['radius'] * sin(angle)
            points1.append((x, y))
            
        points2 = []
        start_angle2 = arc2['start_angle']
        end_angle2 = arc2['end_angle']
        
        angle_diff2 = end_angle2 - start_angle2
        if abs(angle_diff2) > pi:
            if angle_diff2 > 0:
                angle_diff2 -= 2*pi
            else:
                angle_diff2 += 2*pi
                
        for i in range(num_points//2):
            angle = start_angle2 + angle_diff2 * i / (num_points//2 - 1)
            x = arc2['center'][0] + arc2['radius'] * cos(angle)
            y = arc2['center'][1] + arc2['radius'] * sin(angle)
            points2.append((x, y))
            
        return points1 + points2
        
    def analyze_turn_optimization(self):
        optimal_delay, min_length = self.optimize_turn_delay()
        
        start_pos = self.find_turn_start_position(optimal_delay)
        
        r_end = start_pos['polar'][0]
        theta_end = start_pos['polar'][1] + pi
        end_pos = {
            'polar': (r_end, theta_end),
            'cartesian': self.polar_to_cartesian(r_end, theta_end)
        }
        
        optimal_curve = self.design_s_curve(start_pos, end_pos)
        path_points = self.generate_path_points(optimal_curve)
        
        results = {
            'optimal_delay': optimal_delay,
            'min_path_length': min_length,
            'turn_start_position': start_pos,
            'turn_end_position': end_pos,
            'curve_parameters': optimal_curve,
            'path_points': path_points,
            'turn_radius': start_pos['turn_radius']
        }
        
        return results
        
    def visualize_optimization(self, results):
        plt.figure(figsize=(15, 10))
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        
        theta_in = np.linspace(0, 16*2*pi, 2000)
        r_in = self.a - self.b * theta_in
        x_in = r_in * np.cos(-theta_in)
        y_in = r_in * np.sin(-theta_in)
        
        theta_out = theta_in + pi
        r_out = r_in
        x_out = r_out * np.cos(-theta_out)
        y_out = r_out * np.sin(-theta_out)
        
        plt.plot(x_in, y_in, 'b--', alpha=0.6, linewidth=2, label='Inbound Spiral')
        plt.plot(x_out, y_out, 'g--', alpha=0.6, linewidth=2, label='Outbound Spiral')
        
        circle = plt.Circle((0, 0), self.turn_space_radius, fill=False, 
                          color='red', linestyle=':', linewidth=2, label='Turn Space Boundary')
        plt.gca().add_patch(circle)
        
        path_points = results['path_points']
        if path_points:
            path_x, path_y = zip(*path_points)
            plt.plot(path_x, path_y, 'r-', linewidth=3, label='Optimal Turn Path')
            
        start_pos = results['turn_start_position']['cartesian']
        end_pos = results['turn_end_position']['cartesian']
        plt.plot(start_pos[0], start_pos[1], 'ro', markersize=8, label='Turn Start')
        plt.plot(end_pos[0], end_pos[1], 'go', markersize=8, label='Turn End')
        
        curve = results['curve_parameters']
        arc1_center = curve['arc1']['center']
        arc2_center = curve['arc2']['center']
        plt.plot(arc1_center[0], arc1_center[1], 'mx', markersize=6, label='Arc Centers')
        plt.plot(arc2_center[0], arc2_center[1], 'mx', markersize=6)
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.title(f'Dragon Turn-Around Path Optimization\n'
                 f'Optimal Delay: {results["optimal_delay"]:.3f}s, '
                 f'Min Path Length: {results["min_path_length"]:.2f}cm', fontsize=12)
        plt.xlabel('X (cm)', fontsize=11)
        plt.ylabel('Y (cm)', fontsize=11)
        
        if not os.path.exists('./output/Q4'):
            os.makedirs('./output/Q4')
            
        plt.savefig('./output/Q4/turn_optimization_visualization.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def export_results(self, results):
        if not os.path.exists('./output/Q4'):
            os.makedirs('./output/Q4')
            
        summary_data = {
            'Parameter': [
                'Optimal Turn Delay (s)',
                'Minimum Path Length (cm)', 
                'Turn Radius (cm)',
                'Turn Start X (cm)',
                'Turn Start Y (cm)',
                'Turn End X (cm)',
                'Turn End Y (cm)',
                'Arc1 Radius (cm)',
                'Arc2 Radius (cm)'
            ],
            'Value': [
                results['optimal_delay'],
                results['min_path_length'],
                results['turn_radius'],
                results['turn_start_position']['cartesian'][0],
                results['turn_start_position']['cartesian'][1],
                results['turn_end_position']['cartesian'][0],
                results['turn_end_position']['cartesian'][1],
                results['curve_parameters']['arc1']['radius'],
                results['curve_parameters']['arc2']['radius']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('./output/Q4/turn_optimization_summary.csv', index=False)
        
        path_data = {
            'Point_Index': range(len(results['path_points'])),
            'X_cm': [p[0] for p in results['path_points']],
            'Y_cm': [p[1] for p in results['path_points']]
        }
        
        path_df = pd.DataFrame(path_data)
        path_df.to_csv('./output/Q4/optimal_turn_path.csv', index=False)
        
        print("Results exported to:")
        print("- ./output/Q4/turn_optimization_summary.csv")
        print("- ./output/Q4/optimal_turn_path.csv")
        print("- ./output/Q4/turn_optimization_visualization.png")


class DragonTurnPathOptimizer:
    def __init__(self, spiral_spacing=170):
        self.spiral_spacing = spiral_spacing
        self.a = 16 * spiral_spacing
        self.b = spiral_spacing / (pi * 2)
        self.turn_space_radius = 450
        self.head_speed = 100
        
        # Calculate when dragon enters turn space
        self.base_enter_time = self._calculate_enter_time()
        
    def _calculate_enter_time(self):
        """Calculate time when dragon head enters turn space (r=450cm)"""
        target_theta = (self.a - self.turn_space_radius) / self.b
        linear_pos = (self.a**2 - (self.a - self.b*target_theta)**2) / (2*self.b)
        return linear_pos / self.head_speed
        
    def spiral_inbound(self, linear_pos):
        """Inbound spiral: r = a - bθ"""
        if linear_pos < 0:
            return None, None
        discriminant = self.a**2 - 2*linear_pos*self.b
        if discriminant < 0:
            return None, None
        theta = (self.a - sqrt(discriminant)) / self.b
        r = self.a - self.b * theta
        return r, theta
        
    def spiral_outbound(self, r_in, theta_in):
        """Outbound spiral: center symmetric about origin"""
        theta_out = theta_in + pi
        r_out = r_in
        return r_out, theta_out
        
    def polar_to_cartesian(self, r, theta):
        """Convert polar to cartesian coordinates"""
        if r is None or theta is None:
            return None, None
        x = r * cos(-theta)
        y = r * sin(-theta)
        return x, y
        
    def find_turn_positions(self, delay_time=0):
        """Find turn start and end positions"""
        turn_start_time = self.base_enter_time + delay_time
        turn_start_linear_pos = self.head_speed * turn_start_time
        
        r_start, theta_start = self.spiral_inbound(turn_start_linear_pos)
        if r_start is None:
            return None, None
            
        start_pos = {
            'polar': (r_start, theta_start),
            'cartesian': self.polar_to_cartesian(r_start, theta_start),
            'radius': r_start
        }
        
        r_end, theta_end = self.spiral_outbound(r_start, theta_start)
        end_pos = {
            'polar': (r_end, theta_end),
            'cartesian': self.polar_to_cartesian(r_end, theta_end),
            'radius': r_end
        }
        
        return start_pos, end_pos
        
    def design_s_curve(self, start_pos, end_pos):
        """Design S-curve path with two tangent arcs, R1 = 2*R2"""
        if start_pos is None or end_pos is None:
            return None
            
        x1, y1 = start_pos['cartesian']
        x2, y2 = end_pos['cartesian']
        
        # Distance between start and end points
        distance = sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Design two tangent arcs: R1 = 2*R2
        R2 = distance / 6  # Back arc radius
        R1 = 2 * R2       # Front arc radius (twice the back)
        
        # Calculate arc centers for S-curve
        # Direction vector from start to end
        dx = x2 - x1
        dy = y2 - y1
        
        # Perpendicular vector (normalized)
        perp_x = -dy / distance
        perp_y = dx / distance
        
        # Arc centers positioned to create smooth S-curve
        center_offset1 = R1 * 0.8
        center_offset2 = R2 * 0.8
        
        center1_x = x1 + perp_x * center_offset1
        center1_y = y1 + perp_y * center_offset1
        
        center2_x = x2 - perp_x * center_offset2
        center2_y = y2 - perp_y * center_offset2
        
        # Calculate tangent point (where two arcs connect)
        tangent_x = (center1_x + center2_x) / 2
        tangent_y = (center1_y + center2_y) / 2
        
        # Calculate arc angles
        start_angle1 = atan2(y1 - center1_y, x1 - center1_x)
        end_angle1 = atan2(tangent_y - center1_y, tangent_x - center1_x)
        
        start_angle2 = atan2(tangent_y - center2_y, tangent_x - center2_x)
        end_angle2 = atan2(y2 - center2_y, x2 - center2_x)
        
        # Calculate arc lengths
        angle1 = abs(end_angle1 - start_angle1)
        if angle1 > pi:
            angle1 = 2*pi - angle1
        angle2 = abs(end_angle2 - start_angle2)
        if angle2 > pi:
            angle2 = 2*pi - angle2
            
        arc1_length = R1 * angle1
        arc2_length = R2 * angle2
        total_length = arc1_length + arc2_length
        
        return {
            'arc1': {
                'center': (center1_x, center1_y),
                'radius': R1,
                'start_angle': start_angle1,
                'end_angle': end_angle1,
                'length': arc1_length
            },
            'arc2': {
                'center': (center2_x, center2_y),
                'radius': R2,
                'start_angle': start_angle2,
                'end_angle': end_angle2,
                'length': arc2_length
            },
            'tangent_point': (tangent_x, tangent_y),
            'total_length': total_length
        }
        
    def generate_s_curve_points(self, s_curve, num_points=200):
        """Generate discrete points along S-curve path"""
        if s_curve is None:
            return []
            
        points = []
        
        # First arc points
        arc1 = s_curve['arc1']
        cx1, cy1 = arc1['center']
        start_angle1 = arc1['start_angle']
        end_angle1 = arc1['end_angle']
        
        # Handle angle wrapping
        angle_diff1 = end_angle1 - start_angle1
        if abs(angle_diff1) > pi:
            if angle_diff1 > 0:
                angle_diff1 -= 2*pi
            else:
                angle_diff1 += 2*pi
                
        for i in range(num_points//2):
            t = i / (num_points//2 - 1)
            angle = start_angle1 + angle_diff1 * t
            x = cx1 + arc1['radius'] * cos(angle)
            y = cy1 + arc1['radius'] * sin(angle)
            points.append((x, y))
            
        # Second arc points
        arc2 = s_curve['arc2']
        cx2, cy2 = arc2['center']
        start_angle2 = arc2['start_angle']
        end_angle2 = arc2['end_angle']
        
        angle_diff2 = end_angle2 - start_angle2
        if abs(angle_diff2) > pi:
            if angle_diff2 > 0:
                angle_diff2 -= 2*pi
            else:
                angle_diff2 += 2*pi
                
        for i in range(num_points//2):
            t = i / (num_points//2 - 1)
            angle = start_angle2 + angle_diff2 * t
            x = cx2 + arc2['radius'] * cos(angle)
            y = cy2 + arc2['radius'] * sin(angle)
            points.append((x, y))
            
        return points
        
    def optimize_turn_delay(self, max_delay=10.0):
        """Optimize turn delay to minimize S-curve length"""
        def objective(delay):
            try:
                start_pos, end_pos = self.find_turn_positions(delay)
                if start_pos is None or end_pos is None:
                    return float('inf')
                    
                # Must be within turn space
                if start_pos['radius'] > self.turn_space_radius:
                    return float('inf')
                    
                s_curve = self.design_s_curve(start_pos, end_pos)
                if s_curve is None:
                    return float('inf')
                    
                return s_curve['total_length']
            except:
                return float('inf')
        
        result = minimize_scalar(objective, bounds=(0, max_delay), method='bounded')
        return result.x, result.fun
        
    def get_optimization_results(self):
        """Get complete optimization results"""
        optimal_delay, min_length = self.optimize_turn_delay()
        start_pos, end_pos = self.find_turn_positions(optimal_delay)
        optimal_s_curve = self.design_s_curve(start_pos, end_pos)
        path_points = self.generate_s_curve_points(optimal_s_curve)
        
        return {
            'base_enter_time': self.base_enter_time,
            'optimal_delay': optimal_delay,
            'optimal_turn_time': self.base_enter_time + optimal_delay,
            'min_path_length': min_length,
            'start_position': start_pos,
            'end_position': end_pos,
            's_curve': optimal_s_curve,
            'path_points': path_points
        }
        
    def visualize_optimization(self, results):
        """Visualize the optimized turn path"""
        plt.figure(figsize=(16, 12))
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        
        # Draw spiral tracks
        theta_range = np.linspace(0, 16*pi, 2000)
        r_in = self.a - self.b * theta_range
        
        valid_indices = r_in > 0
        theta_range = theta_range[valid_indices]
        r_in = r_in[valid_indices]
        
        x_in = r_in * np.cos(-theta_range)
        y_in = r_in * np.sin(-theta_range)
        
        # Outbound spiral (center symmetric)
        x_out = -x_in
        y_out = -y_in
        
        plt.plot(x_in, y_in, 'b--', alpha=0.6, linewidth=1.5, label='Inbound Spiral')
        plt.plot(x_out, y_out, 'g--', alpha=0.6, linewidth=1.5, label='Outbound Spiral')
        
        # Turn space boundary
        circle_theta = np.linspace(0, 2*pi, 100)
        circle_x = self.turn_space_radius * np.cos(circle_theta)
        circle_y = self.turn_space_radius * np.sin(circle_theta)
        plt.plot(circle_x, circle_y, 'r:', linewidth=2, label='Turn Space Boundary (R=450cm)')
        
        # Draw optimized S-curve path
        if results['path_points']:
            path_x, path_y = zip(*results['path_points'])
            plt.plot(path_x, path_y, 'r-', linewidth=4, label='Optimized S-Curve Path', zorder=5)
            
        # Mark start and end positions
        start_pos = results['start_position']
        end_pos = results['end_position']
        
        if start_pos and end_pos:
            sx, sy = start_pos['cartesian']
            ex, ey = end_pos['cartesian']
            plt.plot(sx, sy, 'ro', markersize=12, label='Turn Start', zorder=6)
            plt.plot(ex, ey, 'go', markersize=12, label='Turn End', zorder=6)
            
            # Mark arc centers and tangent point
            s_curve = results['s_curve']
            if s_curve:
                arc1_center = s_curve['arc1']['center']
                arc2_center = s_curve['arc2']['center']
                tangent_point = s_curve['tangent_point']
                
                plt.plot(arc1_center[0], arc1_center[1], 'mx', markersize=10, label='Arc Centers', zorder=6)
                plt.plot(arc2_center[0], arc2_center[1], 'mx', markersize=10, zorder=6)
                plt.plot(tangent_point[0], tangent_point[1], 'co', markersize=8, label='Tangent Point', zorder=6)
                
                # Draw arc circles (partial)
                circle1_theta = np.linspace(0, 2*pi, 100)
                circle1_x = arc1_center[0] + s_curve['arc1']['radius'] * np.cos(circle1_theta)
                circle1_y = arc1_center[1] + s_curve['arc1']['radius'] * np.sin(circle1_theta)
                plt.plot(circle1_x, circle1_y, 'm:', alpha=0.3, linewidth=1)
                
                circle2_x = arc2_center[0] + s_curve['arc2']['radius'] * np.cos(circle1_theta)
                circle2_y = arc2_center[1] + s_curve['arc2']['radius'] * np.sin(circle1_theta)
                plt.plot(circle2_x, circle2_y, 'm:', alpha=0.3, linewidth=1)
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        title_text = f'Dragon Turn Path Optimization (Pitch={self.spiral_spacing}cm)\nDelay: {results["optimal_delay"]:.2f}s, Path Length: {results["min_path_length"]:.2f}cm'
        plt.title(title_text, fontsize=12)
        plt.xlabel('X (cm)', fontsize=11)
        plt.ylabel('Y (cm)', fontsize=11)
        
        # Save figure
        if not os.path.exists('./output/Q4'):
            os.makedirs('./output/Q4')
            
        plt.savefig('./output/Q4/optimized_turn_path.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return True


def find_Uturn_moment(Spiral_spacing):
    print(f"=== 寻找碰撞-Space-{Spiral_spacing}cm ===")
    start_time = 600
    max_time = 15000

    for t in range(start_time, max_time):
        dragon = BenchDragon(t, Spiral_spacing)
        if dragon.Nodes[1].front_polar_pos[0] < 450:
            has_collision, collisions = dragon.check_all_collisions()
            print("\n生成碰撞可视化...")
            dragon.visualize_with_collisions()
            if has_collision:
                print(f"碰撞对数量: {len(collisions)}")
                print("碰撞的板凳对:")
                for pair in collisions:
                    print(f"  板凳 {pair[0]} 与 板凳 {pair[1]}")
                return True
            else:
                print("此时无碰撞")
                return False


def main():
    print("Dragon Turn Path Optimization - Question 4")
    print("=" * 50)
    
    # Use the new improved optimizer
    optimizer = DragonTurnPathOptimizer(spiral_spacing=170)
    
    print("Calculating optimal S-curve turn path...")
    results = optimizer.get_optimization_results()
    
    print(f"\n=== Optimization Results ===")
    print(f"Dragon enters turn space at: {results['base_enter_time']:.2f} s")
    print(f"Optimal turn delay: {results['optimal_delay']:.2f} s")  
    print(f"Optimal turn start time: {results['optimal_turn_time']:.2f} s")
    print(f"Minimum S-curve path length: {results['min_path_length']:.2f} cm")
    print(f"Turn start radius: {results['start_position']['radius']:.2f} cm")
    print(f"Turn start position: ({results['start_position']['cartesian'][0]:.2f}, {results['start_position']['cartesian'][1]:.2f}) cm")
    print(f"Turn end position: ({results['end_position']['cartesian'][0]:.2f}, {results['end_position']['cartesian'][1]:.2f}) cm")
    
    if results['s_curve']:
        print(f"\n=== S-Curve Parameters ===")
        print(f"Front arc radius (R1): {results['s_curve']['arc1']['radius']:.2f} cm") 
        print(f"Back arc radius (R2): {results['s_curve']['arc2']['radius']:.2f} cm")
        print(f"R1/R2 ratio: {results['s_curve']['arc1']['radius']/results['s_curve']['arc2']['radius']:.2f}")
        print(f"Front arc length: {results['s_curve']['arc1']['length']:.2f} cm")
        print(f"Back arc length: {results['s_curve']['arc2']['length']:.2f} cm")
    
    # Visualize the optimized path
    optimizer.visualize_optimization(results)
    
    print("\nCollision analysis:")
    find_Uturn_moment(170)


if __name__ == "__main__":
    main()
