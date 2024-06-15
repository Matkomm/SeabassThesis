from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Load fish data
# fish_data = pd.read_csv(r'C:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Master\Simulation_data\afternoon_test2.csv')
# fish_data = pd.read_csv(r'C:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Master\Simulation_data\test_10000.csv')
fish_data = pd.read_csv(r'C:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Master\Simulation_data\morning_final1.csv')
#fish_data = pd.read_csv(r'C:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Master\Simulation_data\noon_final1.csv')
#fish_data = pd.read_csv(r'C:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Master\Simulation_data\afternoon_final1.csv')
#fish_data = pd.read_csv(r'C:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Master\Simulation_data\evening_final1.csv')

# Parameters for the cylinder
radius = 6.37
height = 8
theta = np.linspace(0, 2*np.pi, 100)
z = np.linspace(0, height, 100)
theta_grid, z_grid = np.meshgrid(theta, z)
x_grid = radius * np.cos(theta_grid)
y_grid = radius * np.sin(theta_grid)

# Filter out the data for Fish x
fish6_data = fish_data[fish_data['fish_id'] == 7].iloc[10:].reset_index(drop=True)

dx = np.diff(fish6_data['pos_x'])
dy = np.diff(fish6_data['pos_y'])
dz = np.diff(fish6_data['pos_z'])

# Calculate distances and speeds
distances = np.sqrt(dx**2 + dy**2 + dz**2)
speeds = distances / fish6_data['fish_size'].iloc[:-1]  # speed in body lengths per second
path_length = np.sum(distances)

# Print the total path length
print(f'The total path length of Fish 6 is: {path_length}')

# 3D Plot with subplots for different elevation angles
fig = plt.figure(figsize=(18, 6))

elevations = [0, 30, 90]

for i, elev in enumerate(elevations):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    ax.view_init(elev, 45)

    # Create a colormap based on speed
    norm = plt.Normalize(speeds.min(), speeds.max())
    colors = cm.jet(norm(speeds))
    points = np.array([fish6_data['pos_x'], fish6_data['pos_y'], fish6_data['pos_z']]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(segments, cmap='jet', norm=norm)
    lc.set_array(speeds)
    ax.add_collection(lc)

    # Plot the cylinder
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='c')

    # Plot the fish path
    ax.plot(fish6_data['pos_x'], fish6_data['pos_y'], fish6_data['pos_z']) #label='Fish 6 Path')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.legend()

    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([height, 0])

# Add a colorbar
cbar = fig.colorbar(lc, ax=fig.get_axes(), shrink=0.5, aspect=10)
cbar.set_label('Speed (Body Lengths per Second)')
cbar.set_ticks(np.linspace(speeds.min(), speeds.max(), num=10))
cbar.set_ticklabels([f'{t:.2f}' for t in cbar.get_ticks()])

plt.savefig(r'C:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Figures\Autosaved\Path_Morning_speed.png', bbox_inches='tight')
plt.show()
