
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

fish_data = pd.read_csv(r'C:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Master\Simulation_data\afternoon_test2.csv')
#fish_data = pd.read_csv(r'C:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Master\Simulation_data\density_testing.csv')

# Parameters for the cylinder
radius = 6.37
height = 8
theta = np.linspace(0, 2*np.pi, 100)
z = np.linspace(0, height, 100)
theta_grid, z_grid = np.meshgrid(theta, z)
x_grid = radius * np.cos(theta_grid)
y_grid = radius * np.sin(theta_grid)

#Filter out the data for Fish x
fish6_data = fish_data[fish_data['fish_id'] == 6]

dx = np.diff(fish6_data['pos_x'])
dy = np.diff(fish6_data['pos_y'])
dz = np.diff(fish6_data['pos_z'])


distances = np.sqrt(dx**2 + dy**2 + dz**2)
path_length = np.sum(distances)

# Print the total path length
print(f'The total path length of Fish 6 is: {path_length}')


# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

#Create a colormap based on time step
norm = plt.Normalize(fish6_data['time_step'].min(), fish6_data['time_step'].max())
colors = cm.jet(norm(fish6_data['time_step']))
points = np.array([fish6_data['pos_x'], fish6_data['pos_y'], fish6_data['pos_z']]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = Line3DCollection(segments, cmap='jet', norm=norm)
lc.set_array(fish6_data['time_step'][1:].values)
ax.add_collection(lc)

# Add a colorbar
cbar = fig.colorbar(lc, shrink=0.5, aspect=5)
cbar.set_label('Time Step')
cbar.set_ticks(np.linspace(fish6_data['time_step'].min(), fish6_data['time_step'].max(), num=10))
cbar.set_ticklabels([str(int(t)) for t in cbar.get_ticks()])

ax.plot(fish6_data['pos_x'], fish6_data['pos_y'], fish6_data['pos_z'], label='Fish 6 Path')

ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='c')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.legend()

ax.set_xlim([-radius, radius])
ax.set_ylim([-radius, radius])
ax.set_zlim([height, 0])

plt.show()