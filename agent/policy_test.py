import numpy as np
from scipy.optimize import minimize


# Function 1: Exponential decay from linear start
def soft_clip(x, max):
    return x * np.exp(-1.5 * x) + max * (1 - np.exp(-1.5 * x))

def objective(x):
    pixel, cores = x
    p_latency = 0.049 * pixel - 6.624
    throughput = (1000 / p_latency) * cores
    slo_pixel = soft_clip(pixel / 500, 1.0)
    slo_latency = soft_clip(1 - (p_latency - 70) / 70, 1.0)
    slo_completion = soft_clip(throughput / 50, 1.0)
    slo_f = (slo_pixel + slo_latency + slo_completion) / 3
    return -slo_f  # because we want to maximize




if __name__ == '__main__':

    bounds = [(360, 1080), (1, 8)]

    # Initial guess
    x0 = [390, 4]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    # Extract result
    pixel = result.x[0]
    cores = round(result.x[1])  # round to nearest int
    p_latency = 0.049 * pixel - 6.624
    throughput = (1000 / p_latency) * cores
    slo_pixel = soft_clip(pixel / 800, 1.0)
    slo_latency = soft_clip(1 - (p_latency - 70) / 70, 1.0)
    slo_completion = soft_clip(throughput / 50, 1.0)
    slo_f = (slo_pixel + slo_latency + slo_completion) / 3

    # Output
    print("Status:", result.message)
    print("Optimal SLO F:", slo_f)
    print("Pixel-based SLO:", slo_pixel)
    print("Latency-based SLO:", slo_latency)
    print("Completion-based SLO:", slo_completion)
    print("Optimal pixel:", pixel)
    print("Optimal cores (rounded):", cores)
    print("Optimal latency:", p_latency)
    print("Estimated throughput:", throughput)
