from abm import VaccinationAgent, VaccinationModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

width = 1000
height = 1000
num_agents = 735                          # 全体人数
infection_probability = 0.2               # 自然感染概率
vaccination_probability = 0.008         # 疫苗自然接种率
initial_infection_probability = 0.4       # 初始感染率
infection_days = 90                       # 流感季持续时间
medical_staff_ratio = 0.0117              # 健康工作者比例
medical_staff_recommendation_probability = 0.08  # 健康工作者推荐概率
OR = 6.69


model = VaccinationModel(
    width=width,
    height=height,
    num_agents=num_agents,
    infection_probability=infection_probability,
    vaccination_probability=vaccination_probability,
    initial_infected_probability=initial_infection_probability,
    OR=OR,
    medical_staff_ratio = medical_staff_ratio,
    medical_staff_recommendation_probability = medical_staff_recommendation_probability
)


for i in range(infection_days):
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
