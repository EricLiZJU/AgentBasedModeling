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

    def __init__(self, unique_id, is_medical_staff, model):
        super().__init__(model)
        self.unique_id = unique_id
        self.is_medical_staff = is_medical_staff                      #是否为健康工作者
        self.state = self.SUSCEPTIBLE
        self.vaccinated = False
        self.x = random.randrange(self.model.grid.width)
        self.y = random.randrange(self.model.grid.height)

    def move(self):
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        new_position = (self.x + dx) % self.model.grid.width, (self.y + dy) % self.model.grid.height
        self.model.grid.move_agent(self, new_position)

    def infect(self):
        if self.state == self.SUSCEPTIBLE:
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
            for neighbor in neighbors:
                if isinstance(neighbor, VaccinationAgent) and neighbor.state == VaccinationAgent.INFECTED:
                    if random.random() < self.model.infection_probability:
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
                 OR,
                 medical_staff_ratio,
                 medical_staff_recommendation_probability):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.infection_probability = infection_probability                                           # 感染概率
        self.vaccination_probability = vaccination_probability                                       # 疫苗接种概率
        self.initial_infected_probability = initial_infected_probability # 初始感染率
        self.OR = OR
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


    def step(self):
        P = calculate_OR_to_recommended_vaccinate_probability(self.OR,
                                                              self.num_agents,
                                                              self.vaccination_probability,
                                                              self.medical_staff_recommendation_probability)
        print(P)
        self.change_vaccination_probability(P)
        print(self.vaccination_probability)
        self.datacollector.collect(self)
        self.schedule.step()