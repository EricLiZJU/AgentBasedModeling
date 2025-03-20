import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 定义SVIRS模型的微分方程
def svirs_model(y, t, beta, gamma, rho, eta, delta):
    S, V, I, R = y
    N = S + V + I + R  # 总人口数

    # 微分方程
    dSdt = -beta * S * I / N + rho * R  # 易感群体的变化
    dVdt = eta * S - delta * V  # 接种群体的变化
    dIdt = beta * S * I / N - gamma * I  # 感染群体的变化
    dRdt = gamma * I - rho * R  # 恢复群体的变化

    return [dSdt, dVdt, dIdt, dRdt]

def svirs():
    # 参数设置
    beta = 0.3  # 传播率
    gamma = 0.1  # 恢复率
    rho = 0.05  # 恢复后变为易感个体的速率
    eta = 0.02  # 接种率
    delta = 0.01  # 疫苗失效率

    # 初始条件
    S0 = 0.99  # 易感个体的比例
    V0 = 0.01  # 接种疫苗个体的比例
    I0 = 0.001  # 感染个体的比例
    R0 = 0  # 恢复个体的比例

    # 总人口比例（S+V+I+R）
    initial_conditions = [S0, V0, I0, R0]

    # 时间步长
    t = np.linspace(0, 200, 200)  # 模拟200天

    # 求解ODE
    solution = odeint(svirs_model, initial_conditions, t, args=(beta, gamma, rho, eta, delta))

    # 提取解
    S, V, I, R = solution.T

    print(S, V, I, R)

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible (S)')
    plt.plot(t, V, label='Vaccinated (V)')
    plt.plot(t, I, label='Infected (I)')
    plt.plot(t, R, label='Recovered (R)')
    plt.xlabel('Time (days)')
    plt.ylabel('Proportion of Population')
    plt.legend()
    plt.title('SVIRS Model Dynamics')
    plt.grid(True)
    plt.show()


    return S, V, I, R

if __name__ == "__main__":
    S, V, I, R = svirs()
    print(I)