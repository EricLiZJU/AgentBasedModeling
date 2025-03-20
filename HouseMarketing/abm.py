from mesa import Agent, Model
from mesa.time import RandomActivation  # 旧版仍然使用这个
from mesa.datacollection import DataCollector
import numpy as np
import matplotlib.pyplot as plt


class Buyer(Agent):
    """购房者代理"""

    def __init__(self, unique_id, model, income, demand):
        super().__init__(unique_id, model)
        self.income = income  # 收入水平
        self.demand = demand  # 购房需求 (0-1)
        self.won_lottery = False  # 是否中签

    def step(self):
        # 决定是否参与摇号
        if self.model.price_cap < self.income * 5:  # 价格可负担
            self.model.lottery_pool.append(self)


class Developer(Agent):
    """开发商代理"""

    def __init__(self, unique_id, model, initial_units):
        super().__init__(unique_id, model)
        self.units = initial_units  # 初始房源数量

    def step(self):
        # 房屋供应策略
        if self.model.price_cap > 400:  # 限价较高，愿意供应
            self.units += np.random.randint(10, 20)
        else:  # 限价低，减少供应
            self.units += np.random.randint(0, 10)


class Government(Agent):
    """政府代理"""

    def __init__(self, unique_id, model, initial_price_cap):
        super().__init__(unique_id, model)
        self.price_cap = initial_price_cap  # 价格上限

    def step(self):
        # 动态调整限价
        if len(self.model.lottery_pool) > self.model.total_units:
            self.model.price_cap += 10  # 供不应求，提高限价
        elif len(self.model.lottery_pool) < self.model.total_units / 2:
            self.model.price_cap -= 10  # 供大于求，降低限价


class HousingMarket(Model):
    """杭州新房限价与摇号市场模型"""

    def __init__(self, num_buyers=100000, num_developers=10, initial_price_cap=200):
        super().__init__()
        self.num_buyers = num_buyers
        self.num_developers = num_developers
        self.price_cap = initial_price_cap  # 限价
        self.total_units = 50000  # 市场上供应的房源总数
        self.lottery_pool = []  # 摇号购房者池

        self.schedule = RandomActivation(self)  # 旧版仍然使用这个
        self.datacollector = DataCollector(model_reporters={"Price Cap": "price_cap", "Total Units": "total_units"})

        # 创建政府
        self.government = Government(0, self, initial_price_cap)
        self.schedule.add(self.government)

        # 创建开发商
        for i in range(1, num_developers + 1):
            dev = Developer(i, self, np.random.randint(10, 20))
            self.schedule.add(dev)
            self.total_units += dev.units

        # 创建购房者
        for i in range(num_developers + 1, num_developers + num_buyers + 1):
            income = np.random.randint(100, 500)  # 随机收入
            demand = np.random.random()  # 购房需求
            buyer = Buyer(i, self, income, demand)
            self.schedule.add(buyer)

    def step(self):
        """执行一个周期的仿真"""
        self.lottery_pool = []  # 清空摇号池
        self.schedule.step()

        # 摇号逻辑
        np.random.shuffle(self.lottery_pool)
        winners = self.lottery_pool[:min(len(self.lottery_pool), self.total_units)]
        for buyer in winners:
            buyer.won_lottery = True

        # 统计数据
        self.total_units = sum([agent.units for agent in self.schedule.agents if isinstance(agent, Developer)])
        self.datacollector.collect(self)


# 运行模型
model = HousingMarket()
for i in range(50):  # 运行50轮
    model.step()

# 结果可视化
results = model.datacollector.get_model_vars_dataframe()
plt.figure(figsize=(10, 5))
plt.plot(results["Price Cap"], label="Price Cap")
plt.plot(results["Total Units"], label="Total Units")
plt.legend()
plt.title("Price Cap and Total Units Change")
plt.show()