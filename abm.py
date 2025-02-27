from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import pandas as pd

from calculation import *

"""
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_colwidth', None)  # 显示完整内容，不截断列
"""

class VaccinationAgent(Agent):

    SUSCEPTIBLE = 0
    INFECTED = 1
    IMMUNE = 2

    move_distance_list = range(-200, 200)

    def __init__(self, unique_id, is_medical_staff, model):
        super().__init__(model)
        self.unique_id = unique_id
        self.is_medical_staff = is_medical_staff                      #是否为健康工作者
        self.state = self.SUSCEPTIBLE
        self.vaccinated = False
        self.x = random.randrange(self.model.grid.width)
        self.y = random.randrange(self.model.grid.height)

    def move(self):
        dx = random.choice(self.move_distance_list)
        dy = random.choice(self.move_distance_list)
        new_position = (self.x + dx) % self.model.grid.width, (self.y + dy) % self.model.grid.height
        self.model.grid.move_agent(self, new_position)

    def infect(self):
        if self.state == self.SUSCEPTIBLE:
            if random.random() < self.model.infection_probability:
                self.state = self.INFECTED
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
            for neighbor in neighbors:
                if isinstance(neighbor, VaccinationAgent) and neighbor.state == VaccinationAgent.INFECTED:
                    if random.random() < 0.5:      # 密切接触者感染概率
                        self.state = self.INFECTED
                        break

    def vaccinate(self):
        if self.state == self.SUSCEPTIBLE and random.random() < self.model.vaccination_probability:
            self.state = self.IMMUNE
            self.vaccinated = True

    def step(self):
        self.move()
        self.infect()
        self.vaccinate()

class VaccinationModel(Model):

    def __init__(self,
                 width,
                 height,
                 num_agents,
                 infection_probability,
                 vaccination_probability,
                 initial_infected_probability,
                 OR_strategy_1,
                 OR_strategy_2,
                 medical_staff_ratio,
                 medical_staff_recommendation_probability):
        super().__init__()
        self.count = 0                                                                               # 天数
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.infection_probability = infection_probability[self.count]                               # 感染概率
        self.vaccination_probability = vaccination_probability                        # 疫苗接种概率
        self.initial_infected_probability = initial_infected_probability                             # 初始感染率
        self.OR_strategy_1 = OR_strategy_1
        self.OR_strategy_2 = OR_strategy_2
        self.medical_staff_ratio = medical_staff_ratio                                               # 健康工作者比例
        self.medical_staff_recommendation_probability = medical_staff_recommendation_probability     # 健康工作者推荐概率

        for i in range(self.num_agents):

            if random.random() < self.medical_staff_ratio:
                is_medical_staff = True
            else:
                is_medical_staff = False

            a = VaccinationAgent(i, is_medical_staff, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        initial_infected = random.sample(range(self.num_agents), int(self.num_agents * self.initial_infected_probability))
        for idx in initial_infected:
            self.schedule.agents[idx].state = VaccinationAgent.INFECTED

        self.datacollector = DataCollector(
            agent_reporters={"State": "state", "Vaccinated": "vaccinated", "is_medical_staff": "is_medical_staff"},
        )

    def change_vaccination_probability(self, new_vaccination_probability):
        self.vaccination_probability = new_vaccination_probability

    # 接种者的三种推荐策略
    # 策略一：健康工作者推荐策略
    def step_strategy_1(self):
        if self.count % 7 == 0:   # 每隔7天推荐一次
            P = calculate_OR_to_recommended_vaccinate_probability(self.OR_strategy_1,
                                                                  self.num_agents,
                                                                  self.vaccination_probability,
                                                                  self.medical_staff_recommendation_probability)
            # print(P)
            self.change_vaccination_probability(P)
            # print(self.vaccination_probability)
            # self.datacollector.collect(self)
            # self.schedule.step()

    # 策略二：免费疫苗策略
    def step_strategy_2(self, initial_vaccination_probability):
        initial_vaccination_probability = initial_vaccination_probability
        P = calculate_OR_to_recommended_vaccinate_probability(self.OR_strategy_2,
                                                              self.num_agents,
                                                              initial_vaccination_probability,
                                                              self.medical_staff_recommendation_probability)
        self.change_vaccination_probability(P)
        # self.datacollector.collect(self)
        # self.schedule.step()

    def step(self):
        self.count += 1
        self.step_strategy_1()
        self.schedule.step()
        self.datacollector.collect(self)
