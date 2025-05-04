import numpy as np
import matplotlib.pyplot as plt

# Sample data (x: input, y: output)
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Mean values
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculating coefficients
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
slope = numerator / denominator
intercept = y_mean - slope * x_mean

print(f"Line Equation: y = {slope:.2f}x + {intercept:.2f}")

# Predictions
y_pred = slope * x + intercept

# Plotting
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression")
plt.grid(True)
plt.show()
