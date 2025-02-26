from abm import VaccinationAgent, VaccinationModel
import numpy as np
import matplotlib.pyplot as plt
from SVIRS import svirs
import pandas as pd

# SVIRS模型
S, V, I, R = svirs()

width = 1000
height = 10000
num_agents = 7350                      # 全体人数
infection_probability = I              # 自然感染概率
vaccination_probability = V            # 疫苗自然接种率
initial_infection_probability = 0.001      # 初始感染率
infection_days = 90                       # 流感季持续时间
medical_staff_ratio = 0.0117              # 健康工作者比例
# medical_staff_recommendation_probability = 0.33  # 健康工作者推荐概率
medical_staff_recommendation_probability_list = np.arange(0.1, 0.6, 0.02)
OR_strategy_1 = 2
# OR_strategy_1_list = np.arange(1, 5, 0.01)
OR_strategy_2 = 2.24

immune_count_list = []
infected_count_list = []
susceptible_count_list = []


for i in medical_staff_recommendation_probability_list:

    medical_staff_recommendation_probability = i
    print('medical_staff_recommendation_probability: ', i)

    model = VaccinationModel(
        width=width,
        height=height,
        num_agents=num_agents,
        infection_probability=infection_probability,
        vaccination_probability=vaccination_probability,
        initial_infected_probability=initial_infection_probability,
        OR_strategy_1=OR_strategy_1,
        OR_strategy_2=OR_strategy_2,
        medical_staff_ratio = medical_staff_ratio,
        medical_staff_recommendation_probability = medical_staff_recommendation_probability
    )


    for i in range(infection_days):
        model.step()

    agent_data = model.datacollector.get_agent_vars_dataframe()
    #print(agent_data)

    state_counts = agent_data.groupby(["Step", "State"]).size().unstack(fill_value=0)
    print(state_counts)

    immune_count = state_counts[2].iloc[infection_days-1] / num_agents
    infected_count = state_counts[1].iloc[infection_days-1] / num_agents
    susceptable_count = state_counts[0].iloc[infection_days-1] / num_agents

    immune_count_list.append(immune_count)
    infected_count_list.append(infected_count)
    susceptible_count_list.append(susceptable_count)


"""
model = VaccinationModel(
        width=width,
        height=height,
        num_agents=num_agents,
        infection_probability=infection_probability,
        vaccination_probability=vaccination_probability,
        initial_infected_probability=initial_infection_probability,
        OR_strategy_1=OR_strategy_1,
        OR_strategy_2=OR_strategy_2,
        medical_staff_ratio = medical_staff_ratio,
        medical_staff_recommendation_probability = medical_staff_recommendation_probability
    )


for i in range(infection_days):
    model.step()

agent_data = model.datacollector.get_agent_vars_dataframe()
#print(agent_data)

state_counts = agent_data.groupby(["Step", "State"]).size().unstack(fill_value=0)
print(state_counts)

immune_count = state_counts[2].iloc[infection_days-1] / num_agents
infected_count = state_counts[1].iloc[infection_days-1] / num_agents
susceptable_count = state_counts[0].iloc[infection_days-1] / num_agents

immune_count_list.append(immune_count)
infected_count_list.append(infected_count)
susceptible_count_list.append(susceptable_count)





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
"""

fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(medical_staff_recommendation_probability_list, immune_count_list, label="immune")
plt.plot(medical_staff_recommendation_probability_list, infected_count_list, label="infected")
plt.plot(medical_staff_recommendation_probability_list, susceptible_count_list, label="susceptible")

plt.legend()
plt.show()
