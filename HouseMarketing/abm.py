import random

import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# 定义购房者代理
class HomeBuyer(Agent):
    """一个简单的购房者代理"""
    def __init__(self, unique_id, model, income, is_lottery_winner=False):
        super().__init__(unique_id, model)
        self.income = income
        self.is_lottery_winner = is_lottery_winner
        self.participated = False

        self.q_table = np.zeros((3, 2))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2

    def step(self):
        state = self.get_state()
        action = self.choose_action(state)
        reward = self.execute_action(action)

        next_state = self.get_state()
        self.update_q_table(state, action, reward, next_state)


    def get_state(self):
        if self.income > 1000:
            return 0
        elif self.income > 500:
            return 1
        else:
            return 2

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice([0, 1])
        else:
            return np.argmax(self.q_table[state])

    def execute_action(self, action):
        self.participated = False
        if action == 0:   # 参与摇号
            self.participated = True
            self.model.participated_count += 1
            if self.is_lottery_winner and self.income > self.model.house_price:
                self.buy_house()
                return 5
            elif self.is_lottery_winner and self.income <= self.model.house_price:
                return -2
            else:
                return 0

        elif action == 1:   # 等待观望
            return 0

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error


    def buy_house(self):
        # 这里可以根据购房者的收入、房价等因素模拟购房行为
        if self.income > self.model.house_price:
            self.model.buyers += 1

# 定义开发商代理
class Developer(Agent):
    """一个简单的开发商代理"""
    def __init__(self, unique_id, model, house_supply):
        super().__init__(unique_id, model)
        self.house_supply = house_supply

    def step(self):
        # 开发商提供房屋
        self.model.house_supply = self.house_supply

# 定义政府代理
class Government(Agent):
    """政府代理，控制房价和摇号政策"""
    def __init__(self, unique_id, model, price_cap):
        super().__init__(unique_id, model)
        self.price_cap = price_cap

    def step(self):
        # 设置房价限价
        self.model.house_price = min(self.model.house_price, self.price_cap)

        # 执行摇号
        self.run_lottery()

    def run_lottery(self):
        # 摇号过程：随机选择一定比例的购房者为中签者
        for buyer in self.model.buyers_list:
            buyer.is_lottery_winner = False
            if buyer.participated:
                if random.random() < 0.2:  # 20%的购房者中签
                    buyer.is_lottery_winner = True

# 定义模型
class HousingMarketModel(Model):
    """模拟购房摇号与限价过程的模型"""
    def __init__(self, num_buyers, num_developers, price_cap, house_supply):
        self.num_agents = num_buyers + num_developers + 1  # 购房者+开发商+政府
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(10, 10, True)
        self.house_price = 1000  # 初始房价
        self.buyers_list = []
        self.buyers = 0
        self.participated_count = 0

        # 收集数据
        self.datacollector = DataCollector(
            model_reporters={"Buyer_Count": "buyers", "Participated_Count": "participated_count"}
        )

        # 创建购房者
        for i in range(num_buyers):
            buyer = HomeBuyer(i, self, income=random.randint(100, 3000))
            self.schedule.add(buyer)
            self.buyers_list.append(buyer)

        # 创建开发商
        for i in range(num_developers):
            developer = Developer(i + num_buyers, self, house_supply=100)
            self.schedule.add(developer)

        # 创建政府
        government = Government(self.num_agents, self, price_cap)
        self.schedule.add(government)


    def step(self):
        self.buyers = 0
        self.participated_count = 0
        self.schedule.step()
        self.datacollector.collect(self)

# 运行模型
model = HousingMarketModel(num_buyers=100, num_developers=2, price_cap=1500, house_supply=100)
for i in range(10):
    model.step()

# 查看数据
buyer_count_data = model.datacollector.get_model_vars_dataframe()
print(buyer_count_data)