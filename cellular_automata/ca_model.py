import numpy as np
import matplotlib.pyplot as plt

# ---------- 参数设置 ----------
grid_size = 20  # 网格大小（20x20 城市）
price_grid = np.random.randint(200, 400, size=(grid_size, grid_size))  # 每个格子的初始房价
developed_grid = np.zeros((grid_size, grid_size))  # 记录是否开发（0=未开发，1=已开发）

q_table = np.zeros((grid_size, grid_size))  # Q表，每个格子的价值估计

# 强化学习参数
alpha = 0.1      # 学习率
gamma = 0.9      # 折扣因子
epsilon = 0.1    # 探索率（初始时20%概率随机探索）
episodes = 500    # 模拟总轮数

# ---------- 模拟开发过程 ----------
development_history = []

for ep in range(episodes):
    # 决策开发哪个地块
    if np.random.rand() < epsilon:
        x, y = np.random.randint(0, grid_size, size=2)  # 随机探索
    else:
        x, y = np.unravel_index(np.argmax(q_table), q_table.shape)  # 利用Q值最大的格子

    # 模拟开发行为
    if developed_grid[x, y] == 0:
        developed_grid[x, y] = 1  # 标记为已开发
        reward = price_grid[x, y] - 100  # 奖励 = 房价 - 成本（假设成本250）
    else:
        reward = -10  # 已开发地块再次开发无效，给予惩罚

    # Q-learning更新
    old_value = q_table[x, y]
    next_max = np.max(q_table)  # 最大Q值（无实际转移，仅估值）
    q_table[x, y] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

    development_history.append((x, y))  # 记录开发历史

# ---------- 可视化开发结果 ----------
plt.figure(figsize=(6, 6))
plt.imshow(developed_grid, cmap="Greens")
plt.title("Final Developed Land (Green = Developed)", fontsize=14)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(False)
plt.colorbar(label="Developed")
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(q_table, cmap="coolwarm")
plt.title("Final Developed Land (Green = Developed)", fontsize=14)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(False)
plt.colorbar(label="Q Value")
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(price_grid, cmap="coolwarm")
plt.title("Final Developed Land (Green = Developed)", fontsize=14)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(False)
plt.colorbar(label="Q Value")
plt.show()