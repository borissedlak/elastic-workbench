from pulp import LpProblem, LpVariable, LpMaximize, LpStatus, value

# Create the problem
problem = LpProblem("MaximizeSLOF", LpMaximize)

# Variables
pixel = LpVariable("pixel", lowBound=360, upBound=1080, cat="Integer")
cores = LpVariable("cores", lowBound=1, upBound=8, cat="Integer")
latency = LpVariable("latency", cat="Continuous")
throughput = LpVariable("throughput", cat="Continuous")
completion_rate = LpVariable("completion_rate", cat="Continuous")

slo_pixel = LpVariable("slo_pixel", lowBound=0.0, upBound=1.0, cat="Continuous")
slo_latency = LpVariable("slo_latency", lowBound=0.0, upBound=1.0, cat="Continuous")
slo_f = LpVariable("slo_f", lowBound=0.0, upBound=1.0, cat="Continuous")

# Introduce new variable for the product of latency * cores
# latency_cores_product = LpVariable("latency_cores_product", lowBound=0, cat="Continuous")

# Constraints
problem += latency == 0.049 * pixel - 6.624
problem += throughput == (1 / 1000) * latency * cores
problem += completion_rate == 0.049 * pixel - 6.624

problem += slo_pixel == (1 / 800) * pixel
problem += slo_latency == 1 - (latency - 70) / 70

# Use an auxiliary variable to approximate the product latency * cores
# problem += latency_cores_product == latency * cores

# Combine slo_pixel and slo_latency into slo_f
problem += slo_f == (slo_pixel + slo_latency) / 2

# Objective: maximize slo_f
problem += slo_f

# Solve
problem.solve()

# Output
print("Status:", LpStatus[problem.status])
print("Optimal SLO F:", value(slo_f))
print("Pixel-based SLO:", value(slo_pixel))
print("Latency-based SLO:", value(slo_latency))
print("Optimal pixel:", value(pixel))
print("Optimal latency:", value(latency))
print("Optimal cores:", value(cores))
