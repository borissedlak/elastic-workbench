import numpy as np
from scipy.optimize import minimize

from agent.ES_Registry import ServiceType
from agent.LGBN import LGBN
from agent.SLO_Registry import SLO_Registry, to_avg_SLO_F

slo_registry = SLO_Registry()
slos = slo_registry.get_SLOs_for_client("C_1", ServiceType.QR)

ass = {"C_1": 70}

lgbn = LGBN()

# Define the latency equation as a function of pixel
# def latency(pixel):
#     return 0.165 * pixel - 55.384

# Define the abstract reward function phi(state), which is a function of pixel and latency
# def phi(state):
#     pixel, latency_value = state
#     # Replace this with your actual phi function
#     # Example (just as a placeholder):
#     # phi = -((pixel - 10)**2 + (latency_value - 5)**2)
#     return -((pixel - 10)**2 + (latency_value - 5)**2)  # Example: A simple quadratic function

# Define the objective function to minimize (negative of the reward function to maximize)
def objective(pixel):
    full_state = lgbn.get_expected_state({"pixel": pixel[0], "cores": 1}, ass)
    return - (to_avg_SLO_F(slo_registry.calculate_slo_fulfillment(full_state, slos))) # -phi((pixel, latency_value))  # We negate because minimize() minimizes the function

# Initial guess for pixel (you can change this value as needed)
initial_pixel = np.array([500])

# Optimize the pixel value
result = minimize(objective, initial_pixel, bounds=[(360, 1080)])  # Adjust bounds based on your domain

# Extract the optimal pixel and corresponding latency
optimal_pixel = result.x[0]
# optimal_latency = latency(optimal_pixel)

print(f"Optimal pixel value: {optimal_pixel}")
# print(f"Corresponding latency value: {optimal_latency}")
print(f"Maximum reward: {-result.fun}")
