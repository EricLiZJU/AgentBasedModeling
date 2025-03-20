import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

device = torch.device("mps")

# ----------------- DQL 神经网络 -----------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ----------------- 强化学习代理 -----------------
class RLAgent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = lr

        self.model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([-10, 0, 10])
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_values = self.model(state_tensor)
        return [-10, 0, 10][torch.argmax(action_values).item()]

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            target_f = self.model(state_tensor)
            target_f[0][[0, 1, 2].index(action)] = target

            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state_tensor))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ----------------- ABM 代理 -----------------
class Buyer(Agent):
    def __init__(self, unique_id, model, income):
        super().__init__(unique_id, model)
        self.rl_agent = RLAgent(state_size=2, action_size=2)
        self.income = income
        self.won_lottery = False

    def step(self):
        state = np.array([self.model.price_cap, self.income])
        action = self.rl_agent.act(state)

        if action == 1:
            self.model.lottery_pool.append(self)

        reward = 0
        if self.won_lottery and self.model.price_cap > self.income * 5:
            reward = -1
        elif self.won_lottery and self.model.price_cap < self.income * 5:
            reward = 1

        next_state = np.array([self.model.price_cap, self.income])
        self.rl_agent.remember(state, action, reward, next_state, False)

class Developer(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.rl_agent = RLAgent(state_size=2, action_size=3)
        self.units = np.random.randint(10, 20)

    def step(self):
        state = np.array([self.model.price_cap, len(self.model.lottery_pool) / max(1, self.model.total_units)])
        action = self.rl_agent.act(state)
        self.units += action

        profit = self.model.price_cap * self.units - 100 * self.units
        reward = profit / 1000
        next_state = np.array([self.model.price_cap, len(self.model.lottery_pool) / max(1, self.model.total_units)])
        self.rl_agent.remember(state, action, reward, next_state, False)

class Government(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.rl_agent = RLAgent(state_size=2, action_size=3)

    def step(self):
        state = np.array([len(self.model.lottery_pool) / max(1, self.model.total_units), self.model.price_cap])
        action = self.rl_agent.act(state)
        self.model.price_cap += action
        reward = -abs(state[0] - 1)
        next_state = np.array([len(self.model.lottery_pool) / max(1, self.model.total_units), self.model.price_cap])
        self.rl_agent.remember(state, action, reward, next_state, False)

class HousingMarket(Model):
    def __init__(self, num_buyers=100, num_developers=5, initial_price_cap=300):
        super().__init__()
        self.num_buyers = num_buyers
        self.num_developers = num_developers
        self.price_cap = initial_price_cap
        self.total_units = 20
        self.lottery_pool = []

        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(model_reporters={"Price Cap": "price_cap", "Total Units": "total_units"})

        self.government = Government(0, self)
        self.schedule.add(self.government)

        for i in range(1, num_developers + 1):
            dev = Developer(i, self)
            self.schedule.add(dev)
            self.total_units += dev.units

        for i in range(num_developers + 1, num_developers + num_buyers + 1):
            income = np.random.randint(100, 500)
            buyer = Buyer(i, self, income)
            self.schedule.add(buyer)

    def step(self):
        self.lottery_pool = []
        self.schedule.step()
        self.datacollector.collect(self)

# 运行模型
model = HousingMarket()
for i in range(50):
    model.step()

# 结果可视化
results = model.datacollector.get_model_vars_dataframe()
print(results)
plt.figure(figsize=(10,5))
plt.plot(results["Price Cap"], label="Price Cap")
plt.legend()
plt.title("限价与市场供应变化 (政府+开发商+购房者 强化学习)")
plt.show()