import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def log_model(x, a, b, c):
    return a * np.log(b * x) + c


def calculate_OR_to_recommended_vaccinate_probability(OR,
                                                      num_agents,
                                                      vaccination_probability,
                                                      medical_staff_recommendation_probability):
    p = medical_staff_recommendation_probability
    num_recommended_agents = num_agents * p                         # A+C
    num_unrecommended_agents = num_agents - num_recommended_agents  # B+D
    B = int(num_unrecommended_agents * vaccination_probability)
    D = num_unrecommended_agents - B

    m = OR * (B / D)
    A = int(m * medical_staff_recommendation_probability * num_agents / (1 + m))

    recommended_vaccination_probability = (A + B) / num_agents
    # print(A, B, D)
    return recommended_vaccination_probability

if __name__=='__main__':
    res_list = []
    P = np.arange(0, 0.2, 0.001)
    for j in P:
        for i in range(int(90/7)):
            j = calculate_OR_to_recommended_vaccinate_probability(6.69, 735, j, 0.08)
        res_list.append(j)

    # 使用 curve_fit 拟合模型
    params, covariance = curve_fit(log_model, P, res_list, p0=[0.05, 10, 0.5])  # p0 为初始猜测值

    # 提取拟合得到的参数
    a_fit, b_fit, c_fit = params

    print(f"拟合参数：a = {a_fit}, b = {b_fit}, c = {c_fit}")

    # 绘制数据点和拟合曲线
    plt.scatter(P, res_list, label='Data', color='red', s=10)  # 绘制实际数据
    plt.plot(P, log_model(P, *params), label='Fitted Curve', color='blue', lw=2)  # 绘制拟合曲线

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()



"""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(P, res_list)
    plt.show()
"""

