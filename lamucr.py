import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import random


class ObstacleEnvironment:
    def __init__(self, width=100, height=100, obstacles=None):
        self.width = width
        self.height = height
        self.obstacles = obstacles if obstacles is not None else []
        self.obs_union = unary_union(self.obstacles) if self.obstacles else None
        self.transmit_power = 1000  # 发射功率
        self.noise_level = 0.1  # 噪声水平
        self.gain = 1.0  # 天线增益
        self.snr_constant = 1000  # SNR计算常数

    def is_position_safe(self, point):
        """检查位置是否安全（不在障碍物内）"""
        p = Point(point)
        return not any(p.intersects(obs) for obs in self.obstacles)

    def calculate_obstacle_loss(self, p1, p2):
        """计算两点间障碍物信号衰减"""
        if not self.obstacles:
            return 0

        line = LineString([p1, p2])
        total_length = line.length
        blocked_length = 0

        for obs in self.obstacles:
            intersection = line.intersection(obs)
            if not intersection.is_empty:
                blocked_length += intersection.length

        blockage_ratio = min(blocked_length / total_length, 1.0)
        return 200 * blockage_ratio

    def calculate_link_rate(self, p1, p2):
        """计算链路的通信速率"""
        # 距离（米）
        distance = np.linalg.norm(np.array(p1) - np.array(p2))

        # 自由空间路径损耗 (d^2)
        free_space_loss = distance ** 2

        # 障碍物信号衰减
        obstacle_loss = self.calculate_obstacle_loss(p1, p2)

        # 总损耗（假设发射功率、天线增益等归一化）
        total_loss = max(free_space_loss + obstacle_loss, 1e-3)

        # SNR与损耗成反比，速率与log(1+SNR)成正比
        snr = self.snr_constant / total_loss
        return np.log2(1 + snr)  # 简化速率计算


class PSOOptimizer:
    def __init__(self, environment, source, destination, num_relays,
                 swarm_size=50, max_iter=200, w=0.5, c1=1.5, c2=1.5):
        self.env = environment
        self.source = np.array(source)
        self.destination = np.array(destination)
        self.num_relays = num_relays
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.dim = 2 * num_relays
        self.x_min = 0
        self.x_max = environment.width
        self.y_min = 0
        self.y_max = environment.height

        self.positions = self._initialize_positions()
        self.velocities = self._initialize_velocities()

        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(swarm_size, float('-inf'))
        self.global_best_position = None
        self.global_best_score = float('-inf')

        self._evaluate_swarm()

    def _initialize_positions(self):
        """初始化粒子位置 - 确保安全位置"""
        positions = np.zeros((self.swarm_size, self.dim))

        for i in range(self.swarm_size):
            for j in range(self.num_relays):
                # 在可行区域内随机生成安全位置
                safe_position_found = False
                attempts = 0
                while not safe_position_found and attempts < 20:
                    x = random.uniform(self.x_min, self.x_max)
                    y = random.uniform(self.y_min, self.y_max)
                    if self.env.is_position_safe((x, y)):
                        positions[i, 2 * j] = x
                        positions[i, 2 * j + 1] = y
                        safe_position_found = True
                    attempts += 1

                if not safe_position_found:
                    # 如果找不到安全位置，放置在不靠近边界的地方
                    positions[i, 2 * j] = max(min(random.uniform(self.x_min, self.x_max),
                                                  self.x_max - 10), self.x_min + 10)
                    positions[i, 2 * j + 1] = max(min(random.uniform(self.y_min, self.y_max),
                                                      self.y_max - 10), self.y_min + 10)
        return positions

    def _initialize_velocities(self):
        """初始化粒子速度"""
        return np.random.uniform(-1, 1, (self.swarm_size, self.dim))

    def _get_relay_positions(self, particle):
        """从粒子位置提取中继坐标"""
        positions = []
        for i in range(self.num_relays):
            x = particle[2 * i]
            y = particle[2 * i + 1]
            positions.append([x, y])
        return positions

    def calculate_min_rate(self, particle):
        """计算粒子的最小一跳速率"""
        relay_positions = self._get_relay_positions(particle)
        all_positions = [self.source] + relay_positions + [self.destination]

        # 检查所有位置是否安全
        for pos in relay_positions:
            if not self.env.is_position_safe(pos):
                return float('-inf')  # 位置不安全，极大惩罚

        # 计算所有链路的速率
        rates = []
        for i in range(len(all_positions) - 1):
            rate = self.env.calculate_link_rate(all_positions[i], all_positions[i + 1])
            rates.append(rate)

        return min(rates)  # 最小一跳速率

    def _evaluate_swarm(self):
        """评估整个粒子群"""
        for i in range(self.swarm_size):
            score = self.calculate_min_rate(self.positions[i])

            # 修复无效粒子
            if score == float('-inf'):
                self._repair_particle(i)
                score = self.calculate_min_rate(self.positions[i])

            # 更新个体最优
            if score > self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = np.copy(self.positions[i])

            # 更新全局最优
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = np.copy(self.positions[i])

        # 确保至少有一个有效粒子
        if self.global_best_position is None or self.global_best_score == float('-inf'):
            # 随机选择一个粒子作为全局最优
            idx = np.random.randint(0, self.swarm_size)
            self.global_best_position = np.copy(self.positions[idx])
            self.global_best_score = self.calculate_min_rate(self.positions[idx])

    def _repair_particle(self, index):
        """修复不安全或无效的粒子位置"""
        for j in range(self.num_relays):
            x = self.positions[index, 2 * j]
            y = self.positions[index, 2 * j + 1]

            if not self.env.is_position_safe((x, y)):
                # 简单策略：将点移向安全区域的中心
                safe_x = max(min(x, self.x_max - 10), self.x_min + 10)
                safe_y = max(min(y, self.y_max - 10), self.y_min + 10)

                # 尝试找到安全位置
                if self.env.is_position_safe((safe_x, safe_y)):
                    self.positions[index, 2 * j] = safe_x
                    self.positions[index, 2 * j + 1] = safe_y
                else:
                    # 如果仍然不安全，随机生成新位置
                    while True:
                        new_x = random.uniform(self.x_min + 10, self.x_max - 10)
                        new_y = random.uniform(self.y_min + 10, self.y_max - 10)
                        if self.env.is_position_safe((new_x, new_y)):
                            self.positions[index, 2 * j] = new_x
                            self.positions[index, 2 * j + 1] = new_y
                            break

    def optimize(self):
        """执行PSO优化"""
        progress = tqdm(range(self.max_iter), desc="Optimizing")

        for iter in progress:
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social

                self.positions[i] += self.velocities[i]

                # 边界约束
                for j in range(self.num_relays):
                    if self.positions[i, 2 * j] < self.x_min:
                        self.positions[i, 2 * j] = self.x_min
                    elif self.positions[i, 2 * j] > self.x_max:
                        self.positions[i, 2 * j] = self.x_max

                    if self.positions[i, 2 * j + 1] < self.y_min:
                        self.positions[i, 2 * j + 1] = self.y_min
                    elif self.positions[i, 2 * j + 1] > self.y_max:
                        self.positions[i, 2 * j + 1] = self.y_max

            # 评估更新后的粒子群
            self._evaluate_swarm()
            progress.set_postfix({"MaxMinRate": self.global_best_score})

        return self.global_best_position, self.global_best_score


class Visualizer:
    def __init__(self, environment, source, destination, relay_positions=None):
        self.env = environment
        self.source = source
        self.destination = destination
        self.relay_positions = relay_positions

    def plot_environment(self):
        """可视化环境、障碍物和路径"""
        plt.figure(figsize=(10, 10))

        # 绘制边界
        plt.plot([0, self.env.width, self.env.width, 0, 0],
                 [0, 0, self.env.height, self.env.height, 0], 'k-')

        # 绘制障碍物
        for obs in self.env.obstacles:
            x, y = obs.exterior.xy
            plt.fill(x, y, alpha=0.5, fc='gray', ec='black')

        # 绘制起点和终点
        plt.plot(self.source[0], self.source[1], 'go', markersize=12, label='Source')
        plt.plot(self.destination[0], self.destination[1], 'ro', markersize=12, label='Destination')

        # 绘制中继和路径
        if self.relay_positions:
            positions = [self.source] + self.relay_positions + [self.destination]
            x = [p[0] for p in positions]
            y = [p[1] for p in positions]

            # 绘制路径
            plt.plot(x, y, 'b-', linewidth=2, label='Relay Path')
            plt.scatter(x[1:-1], y[1:-1], c='purple', s=100, label='Relay UAVs')

            # 计算并标注每跳速率
            for i in range(len(positions) - 1):
                rate = self.env.calculate_link_rate(positions[i], positions[i + 1])
                mid_x = (positions[i][0] + positions[i + 1][0]) / 2
                mid_y = (positions[i][1] + positions[i + 1][1]) / 2
                plt.text(mid_x, mid_y, f"{rate:.2f}", fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.7))

        plt.title('Multi-hop Relay Path Optimization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def plot_heatmap(self, min_rate):
        """可视化最小一跳速率的热力图"""
        x = np.linspace(0, self.env.width, 50)
        y = np.linspace(0, self.env.height, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # 计算每个点的安全性和障碍物衰减（简化）
        for i in range(len(x)):
            for j in range(len(y)):
                # 简化计算：到路径的距离
                dist_to_source = np.linalg.norm([x[i] - self.source[0], y[j] - self.source[1]])
                dist_to_dest = np.linalg.norm([x[i] - self.destination[0], y[j] - self.destination[1]])
                Z[j, i] = min(dist_to_source, dist_to_dest)

        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(label='Distance to Path')

        # 绘制路径
        positions = [self.source] + self.relay_positions + [self.destination]
        x_path = [p[0] for p in positions]
        y_path = [p[1] for p in positions]
        plt.plot(x_path, y_path, 'r-', linewidth=2, label='Optimal Path')

        plt.title(f'Optimal Relay Path (Min Rate: {min_rate:.2f})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()


# 示例用法
if __name__ == "__main__":
    # 创建障碍物环境 (100x100)
    env = ObstacleEnvironment(width=100, height=100)

    # 添加障碍物 (矩形和三角形)
    obstacles = [
        Polygon([(20, 20), (40, 20), (40, 40), (20, 40)]),
        Polygon([(70, 20), (90, 20), (90, 40), (70, 40)])
    ]
    env.obstacles = obstacles

    # 定义起点和终点
    source = (10, 10)
    destination = (90, 90)

    # 初始化PSO优化器 (3个中继)
    pso = PSOOptimizer(env, source, destination, num_relays=3,
                       swarm_size=30, max_iter=100, w=0.7, c1=1.2, c2=1.2)

    # 执行优化
    best_position, best_score = pso.optimize()
    print(f"Optimized min rate: {best_score:.4f}")

    # 提取中继位置
    relay_positions = []
    for i in range(pso.num_relays):
        x = best_position[2 * i]
        y = best_position[2 * i + 1]
        relay_positions.append([x, y])

    # 可视化结果
    vis = Visualizer(env, source, destination, relay_positions)
    vis.plot_environment()