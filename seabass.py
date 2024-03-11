import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Fish:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity

    def update_position(self, cage):
        new_position = self.position + self.velocity
        if not cage.is_inside(new_position):
            self.velocity *= -1
            new_position = self.position + self.velocity
        self.position = new_position

class SeaCage:
    def __init__(self, radius, depth):
        self.radius = radius
        self.depth = depth

    def is_inside(self, position):
        radial_distance = np.sqrt(position[0]**2 + position[1]**2)
        return radial_distance <= self.radius and 0 <= position[2] <= self.depth

class Simulation:
    def __init__(self, num_fish, cage_radius, cage_depth):
        self.fish = []
        for _ in range(num_fish):
            # Randomly generate position in cylindrical coordinates and convert to Cartesian
            radius = np.random.uniform(0, cage_radius)
            angle = np.random.uniform(0, 2 * np.pi)
            depth = np.random.uniform(0, cage_depth)
            position = np.array([radius * np.cos(angle), radius * np.sin(angle), depth])
            velocity = (np.random.rand(3) - 0.5) / 10
            self.fish.append(Fish(position, velocity))
        self.cage = SeaCage(cage_radius, cage_depth)

    def run(self, num_steps, visualize=False):
        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim3d([-self.cage.radius, self.cage.radius])
            ax.set_ylim3d([-self.cage.radius, self.cage.radius])
            ax.set_zlim3d([0, self.cage.depth])
            x = np.linspace(-self.cage.radius, self.cage.radius, 100)
            z = np.linspace(0, self.cage.depth, 100)
            Xc, Zc = np.meshgrid(x, z)
            Yc = np.sqrt(self.cage.radius**2 - Xc**2)
            ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='blue')
            ax.plot_surface(Xc, -Yc, Zc, alpha=0.3, color='blue')

        for _ in range(num_steps):
            if visualize:
                ax.clear()
                ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='blue')
                ax.plot_surface(Xc, -Yc, Zc, alpha=0.3, color='blue')

            for fish in self.fish:
                fish.update_position(self.cage)
                if visualize:
                    ax.scatter(fish.position[0], fish.position[1], fish.position[2], color='red')
            
            if visualize:
                plt.pause(0.05)

        if visualize:
            plt.show()

# Parameters for the simulation
num_fish = 50
cage_radius = 10  # Example radius
cage_depth = 5  # Example depth
num_steps = 100  # Example number of steps

# Run the simulation with visualization
simulation = Simulation(num_fish, cage_radius, cage_depth)
simulation.run(num_steps, visualize=True)
