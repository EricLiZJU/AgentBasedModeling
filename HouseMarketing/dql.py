import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

device = torch.device("cpu")

# ----------------- 深度Q网络定义 -----------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # 三层全连接网络
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        # 使用ReLU激活函数
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ----------------- 强化学习智能体类 -----------------
class RLAgent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.9, epsilon=0.1, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size         # 状态空间维度
        self.action_size = action_size       # 动作空间维度
        self.memory = deque(maxlen=2000)     # 经验回放池
        self.gamma = gamma                   # 折扣因子
        self.epsilon = epsilon               # 探索率
        self.epsilon_min = epsilon_min       # 最小探索率
        self.epsilon_decay = epsilon_decay   # 探索衰减
        self.learning_rate = lr              # 学习率

        # 初始化DQN模型
        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        # 记录经验
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # ε-贪婪策略选择动作
        if np.random.rand() <= self.epsilon:
            return random.choice([-1, 0, 1])  # 随机探索动作
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_values = self.model(state_tensor)
        return [-1, 0, 1][torch.argmax(action_values).item()]  # 选择Q值最大的动作

    def replay(self, batch_size=32):
        # 从经验中学习
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            target_f = self.model(state_tensor)
            target_f[0][[0, 1, 2].index(action)] = target  # 仅更新被选中动作的Q值

            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state_tensor))
            loss.backward()
            self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ----------------- ABM代理类 -----------------

# 购房者：使用RL决定是否参与摇号
class Buyer(Agent):
    def __init__(self, unique_id, model, income):
        super().__init__(unique_id, model)
        self.rl_agent = RLAgent(state_size=2, action_size=2)  # 2个状态，2个动作（报名 or 不报名）
        self.income = income
        self.won_lottery = False
        self.houseprice_income_ratio = 21.4 #房价收入比

    def step(self):
        self.won_lottery = False
        state = np.array([self.model.price_cap, self.income])  # 状态包含当前限价和收入
        action = self.rl_agent.act(state)  # 选择是否报名摇号

        if action == 1:
            self.model.lottery_pool.append(self)
            self.model.participated_lottery_count += 1
            if np.random.rand() <= self.model.won_possibility:
                self.won_lottery = True

        # 简单的购房回报逻辑：买到且房价可承受，则正奖励
        reward = 0
        if action == 1 and not self.won_lottery:
            reward = -1
        if self.won_lottery and self.model.price_cap > self.income * self.houseprice_income_ratio:
            reward = -2
        elif self.won_lottery and self.model.price_cap < self.income * self.houseprice_income_ratio:
            reward = 5000000000000000

        next_state = np.array([self.model.price_cap, self.income])
        self.rl_agent.remember(state, action, reward, next_state, False)

# 开发商：使用RL控制新房供应数量
class Developer(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.rl_agent = RLAgent(state_size=2, action_size=3)  # 状态：[限价，供需比]
        self.units = np.random.normal(loc=20)  # 初始房源

    def step(self):
        state = np.array([self.model.price_cap, len(self.model.lottery_pool) / max(1, self.model.total_units)])
        action = self.rl_agent.act(state)
        self.model.total_units += action*1  # 控制新房供应增减


        # 简单利润模型：限价 × 房源数量 - 成本
        profit = self.model.price_cap * self.units - 300 * self.units
        reward = profit  # 归一化利润作为奖励

        next_state = np.array([self.model.price_cap, len(self.model.lottery_pool) / max(1, self.model.total_units)])
        self.rl_agent.remember(state, action, reward, next_state, False)

# 政府：使用RL调整限价
class Government(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.rl_agent = RLAgent(state_size=2, action_size=3)  # 状态：[供需比, 当前限价]

    def step(self):
        state = np.array([len(self.model.lottery_pool) / max(1, self.model.total_units), self.model.price_cap])
        action = self.rl_agent.act(state)
        self.model.price_cap += action*10  # 动作为限价调整

        reward = -abs(state[0] - 1) * 500  # 越接近供需平衡，奖励越高
        next_state = np.array([len(self.model.lottery_pool) / max(1, self.model.total_units), self.model.price_cap])
        self.rl_agent.remember(state, action, reward, next_state, False)

# ----------------- 主模型类 -----------------
class HousingMarket(Model):
    def __init__(self, num_buyers=100, num_developers=10, initial_price_cap=395):
        super().__init__()
        self.num_buyers = num_buyers
        self.num_developers = num_developers
        self.price_cap = initial_price_cap
        self.total_units = 0
        self.lottery_pool = []
        self.participated_lottery_count = 1
        self.won_possibility = 0

        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(model_reporters={"Price Cap": "price_cap", "Total Units": "total_units",
                                                            "Participated Lottery Count": "participated_lottery_count",
                                                            "Won Possibility": "won_possibility"})

        # 添加政府
        self.government = Government(0, self)
        self.schedule.add(self.government)

        # 添加开发商
        for i in range(1, num_developers + 1):
            dev = Developer(i, self)
            self.schedule.add(dev)
            self.total_units += dev.units

        # 添加购房者
        for i in range(num_developers + 1, num_developers + num_buyers + 1):
            income = np.random.normal(loc=250)  # 随机收入
            buyer = Buyer(i, self, income)
            self.schedule.add(buyer)

    def step(self):
        self.won_possibility = self.total_units / self.participated_lottery_count
        print("Participate Count:" + str(self.participated_lottery_count))
        print("Won Possibility:" + str(self.won_possibility))
        self.lottery_pool = []  # 清空摇号池
        self.participated_lottery_count = 1
        self.schedule.step()    # 所有代理执行动作
        self.datacollector.collect(self)  # 收集数据
        print("Total Units: " + str(self.total_units))

# ----------------- 运行模型 -----------------
Model = HousingMarket()
for i in range(100):  # 模拟50期
    Model.step()

# ----------------- 可视化结果 -----------------
results = Model.datacollector.get_model_vars_dataframe()
print(results)
plt.figure(figsize=(10,5))
plt.plot(results["Price Cap"], label="Price Cap")
plt.legend()
plt.title("Evolution of Price Cap and Housing Supply", fontsize=14)
plt.xlabel("Simulation Step", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(results["Total Units"], label="Total Units")
plt.legend()
plt.title("Evolution of Price Cap and Housing Supply", fontsize=14)
plt.xlabel("Simulation Step", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(results["Won Possibility"][1:], label="Won Possibility")
plt.legend()
plt.title("Evolution of Price Cap and Housing Supply", fontsize=14)
plt.xlabel("Simulation Step", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()