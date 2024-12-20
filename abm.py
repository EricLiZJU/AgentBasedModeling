from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random

class VaccinationAgent(Agent):

    SUSCEPTIBLE = 0
    INFECTED = 1
    IMMUNE = 2

    def __init__(self, unique_id, model):
        super().__init__(model)
        self.unique_id = unique_id
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

    def __init__(self, width, height, num_agents, infection_probability, vaccination_probability, initial_infected_probability):
        super().__init__()
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.infection_probability = infection_probability                    # 感染概率
        self.vaccination_probability = vaccination_probability                # 疫苗接种概率
        self.initial_infected_probability = initial_infected_probability      # 初始感染率

        for i in range(self.num_agents):
            a = VaccinationAgent(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        initial_infected = random.sample(range(self.num_agents), int(self.num_agents * self.initial_infected_probability))
        for idx in initial_infected:
            self.schedule.agents[idx].state = VaccinationAgent.INFECTED

        self.datacollector = DataCollector(
            agent_reporters={"State": "state", "Vaccinated": "vaccinated"},
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()