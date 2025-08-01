import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Example data (replace with your dataset)
data = np.loadtxt("C:/Users/shoun/Downloads/SDSU Jetson Project - 8008.csv", delimiter=",", skiprows=1)
birdseye_x, birdseye_y, x_8005, y_8005 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

# Define nonlinear function to fit
def model(inputs, a, b, c, d, e):
    x, y = inputs
    return a * x**2 + b * y**2 + c * x + d * y + e

# Prepare inputs and fit Birdseye_X
inputs = np.array([birdseye_x, birdseye_y])
params_x, _ = curve_fit(model, inputs, x_8005)

# Prepare inputs and fit Birdseye_Y
params_y, _ = curve_fit(model, inputs, y_8005)

# Predict
x_8005_pred = model(inputs, *params_x)
y_8005_pred = model(inputs, *params_y)

# Plot results
plt.scatter(x_8005, x_8005_pred, label="X Prediction")
plt.scatter(y_8005, y_8005_pred, label="Y Prediction")
plt.legend()
plt.show()

print(params_x, params_y)