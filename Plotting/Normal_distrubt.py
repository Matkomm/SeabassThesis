import numpy as np
import matplotlib.pyplot as plt

# Depth preferences data
depth_preferences = {
    'morning': (1.5, 1),
    'noon': (4.5, 1),
    'afternoon': (4.5, 1.5),
    'evening': (2, 2),
    'night': (4, 2.5)
}

# Generating x values for the plot
x = np.linspace(-5, 10, 1000)

# Plotting the normal distributions
plt.figure(figsize=(10, 6))
for time_of_day, (mean, std) in depth_preferences.items():
    y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    plt.plot(x, y, label=f'{time_of_day} (mean={mean}, std={std})')

# Adding titles and labels
plt.title('Depth Preferences at Different Times of the Day')
plt.xlabel('Depth')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
