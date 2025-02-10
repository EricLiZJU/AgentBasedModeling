import numpy as np

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
    P = 0.02
    for i in range(int(90/7)):
        P = calculate_OR_to_recommended_vaccinate_probability(6.69, 735, P, 0.08)
        print(P)

