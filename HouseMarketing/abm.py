import random
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

    def step(self):
        # 如果是摇号中签者，表示购房者有资格购买房产
        if self.is_lottery_winner:
            self.buy_house()

    def buy_house(self):
        # 这里可以根据购房者的收入、房价等因素模拟购房行为
        if self.income > self.model.house_price:
            print(f"购房者 {self.unique_id} 购买了房子。")
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

        # 收集数据
        self.datacollector = DataCollector(
            model_reporters={"Buyer_Count": "buyers"}
        )

        # 创建购房者
        for i in range(num_buyers):
            buyer = HomeBuyer(i, self, income=random.randint(500, 1500))
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
        self.schedule.step()
        self.datacollector.collect(self)

# 运行模型
model = HousingMarketModel(num_buyers=100, num_developers=2, price_cap=1500, house_supply=100)
for i in range(10):
    model.step()

# 查看数据
buyer_count_data = model.datacollector.get_model_vars_dataframe()
print(buyer_count_data)