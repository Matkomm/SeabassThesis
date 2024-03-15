import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle

# Define the vertices for the fish shape
fish_path = Path([
    (-1, 0), (0, 1), (1, 0), (0, -1), (-0.5, 0),
    (0, -0.5), (0.5, 0), (0, 0.5), (-0.5, 0)
])

# Create a marker style object using the fish shape
fish_marker = MarkerStyle(fish_path)

# Generate some sample data
x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]

# Plot the data using the custom fish marker
plt.scatter(x, y, marker=fish_marker, s=300, color='blue')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Fish Marker Scatter Plot')
plt.grid(True)
plt.show()
