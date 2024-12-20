from abm import VaccinationAgent, VaccinationModel
import numpy as np
import matplotlib.pyplot as plt

width = 1000
height = 1000
num_agents = 735
infection_probability = 0.2
vaccination_probability = 0.005
initial_infection_probability = 0.4

model = VaccinationModel(
    width=width,
    height=height,
    num_agents=num_agents,
    infection_probability=infection_probability,
    vaccination_probability=vaccination_probability,
    initial_infected_probability=initial_infection_probability
)


for i in range(60):
    model.step()

agent_data = model.datacollector.get_agent_vars_dataframe()
print(agent_data)

state_counts = agent_data.groupby(["Step", "State"]).size().unstack(fill_value=0)
print(state_counts)

# 绘制健康、感染和免疫代理的分布情况
fig, ax = plt.subplots(figsize=(10, 6))
state_counts.plot(kind='bar', stacked=True, ax=ax)

# 设置图形标题和标签
ax.set_title("Agent State Distribution Over Time")
ax.set_xlabel("Time Steps")
ax.set_ylabel("Number of Agents")
ax.set_xticklabels(state_counts.index, rotation=45)

# 显示图形
plt.tight_layout()
plt.show()
