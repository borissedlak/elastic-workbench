import numpy as np
import matplotlib.pyplot as plt

# Tuned Sigmoid
def tuned_sigmoid(x):
    sharpness = 10  # make it sharper
    return 1 / (1 + np.exp(-sharpness * (x - 0.5)))  # center at 0.5

# Tuned Tanh
def tuned_tanh(x):
    sharpness = 3  # smaller than sigmoid to match shapes better
    return (np.tanh(sharpness * (x - 0.5)) + 1) / 2  # center at 0.5 and scale to [0,1]

# Generate x values
x = np.linspace(-0.5, 1.5, 500)  # wider view around [0,1]

# Compute tuned activations
y_sigmoid = tuned_sigmoid(x)
y_tanh = tuned_tanh(x)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y_sigmoid, label='Tuned Sigmoid', linestyle='-')
plt.plot(x, y_tanh, label='Tuned Tanh', linestyle='--')

plt.title('Tuned Sigmoid and Tanh (0 below 0, 1 above 1)')
plt.xlabel('x')
plt.ylabel('Activation Output')
plt.axvline(x=0, color='grey', linestyle=':', label='x=0')
plt.axvline(x=1, color='grey', linestyle='--', label='x=1')
plt.grid(True)
plt.legend()
plt.show()
