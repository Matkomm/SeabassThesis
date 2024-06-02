import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_rotation_axes():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Axis lengths
    axis_length = 3
    
    # Draw main axes
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, label='X axis (Roll)')
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, label='Y axis (Pitch)')
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, label='Z axis (Yaw)')

    # Parameters for circular arrows
    t = np.linspace(0, np.pi/2, 100)  # Parameter for circular section
    r = 0.5  # Radius of the circles

    # Yaw (rotation around Z-axis, blue)
    x_yaw = r * np.sin(t)
    y_yaw = r * np.cos(t)
    z_yaw = np.zeros_like(t) + axis_length * 0.9
    ax.plot(x_yaw, y_yaw, z_yaw, 'b')

    # Pitch (rotation around Y-axis, green)
    x_pitch = r * np.sin(t)
    y_pitch = np.zeros_like(t) + axis_length * 0.9
    z_pitch = r * np.cos(t)
    ax.plot(x_pitch, y_pitch, z_pitch, 'g')

    # Roll (rotation around X-axis, red)
    x_roll = np.zeros_like(t) + axis_length * 0.9
    y_roll = r * np.cos(t)
    z_roll = r * np.sin(t)
    ax.plot(x_roll, y_roll, z_roll, 'r')

    # Add a "Fish" arrow at a specific location with orientation
    fish_length = 3  # Length of the fish arrow
    fish_position = [0, 0, 0]  # Position in space
    fish_orientation = [1.5, 1.5, 0]  # Orientation vector
    ax.quiver(*fish_position, *fish_orientation, color='orange', arrow_length_ratio=0.2, label='Fish')

    # Set limits, labels, and legend
    ax.set_xlim([-1, 3])
    ax.set_ylim([-1, 3])
    ax.set_zlim([-1, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Show the plot
    plt.show()

# Example usage
plot_rotation_axes()
