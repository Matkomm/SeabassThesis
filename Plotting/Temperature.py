import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
""""
# Data points: temperature (T) and fish speed (S)
T = np.array([12, 22.86, 30])
S = np.array([0.5, 1, 0.8])

# Quadratic model function
def quadratic_model(T, a, b):
    return a * (T - 22.86)**2 + b

# Fit the model to the data
params, _ = curve_fit(quadratic_model, T, S)
a, b = params

# Generate temperature range for the model
T_fit = np.linspace(10, 32, 100)
S_fit = quadratic_model(T_fit, a, b)

# Plot the data and the fitted model
plt.figure(figsize=(10, 6))
plt.plot(T, S, 'bo', label='Data Points')
plt.plot(T_fit, S_fit, 'r-', label='Quadratic Fit')
plt.axvline(22.86, color='k', linestyle='--', label='Optimal Temperature (22.86°C)')
plt.xlabel('Water Temperature (°C)')
plt.ylabel('Fish Speed (Normalized)')
plt.legend()
plt.title('Fish Speed vs. Water Temperature')
plt.show()

print(f"Model Parameters: a = {a}, b = {b}")

"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the hypothetical data points based on the observed pattern
T = np.array([8,10,12, 16, 20, 22.86, 25, 28, 30, 32, 34])
S = np.array([0.55,0.6,0.65, 0.75, 0.9, 1.15, 0.95, 0.88, 0.8, 0.78,0.75])  # Adjusted to follow the observed trend

# Define a quartic model function (4th-degree polynomial)
def quartic_model(T, a, b, c, d, e):
    return a * (T - 22.86)**4 + b * (T - 22.86)**3 + c * (T - 22.86)**2 + d * (T - 22.86) + e

# Fit the model to the data
params, _ = curve_fit(quartic_model, T, S)
a, b, c, d, e = params

# Generate temperature range for the model
T_fit = np.linspace(8, 34, 100)
S_fit = quartic_model(T_fit, a, b, c, d, e)

# Plot the data and the fitted model
plt.figure(figsize=(10, 6))
plt.plot(T, S, 'bo', label='Data Points')
plt.plot(T_fit, S_fit, 'r-', label='Quartic Fit')
plt.axvline(22.86, color='k', linestyle='--', label='Optimal Temperature (22.86°C)')
plt.xlabel('Water Temperature (°C)')
plt.ylabel('Fish Speed (Normalized)')
plt.legend()
plt.title('Fish Speed vs. Water Temperature (Quartic Model)')
plt.show()

print(f"Model Parameters: a = {a}, b = {b}, c = {c}, d = {d}, e = {e}")

a = 1.5437135525254706e-05
b = 5.295833314637193e-05
c = -0.004566765360787012
d = -0.0003563208751696658
e = 1.003878811498009
T = 12
S = a * (T - 22.86)**4 + b * (T - 22.86)**3 + c * (T - 22.86)**2 + d * (T - 22.86) + e
print(S)