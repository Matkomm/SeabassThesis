import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Fish:
    def __init__(self, position, velocity, size):
        self.position = position
        self.velocity = velocity
        self.size = size

    def update_position(self, cage):
        # Hypothetically adjust velocity based on a reference velocity concept
        ref_velocity_direction = np.random.rand(3) - 0.5  # This would be based on actual dynamics in the full model
        ref_velocity_magnitude = 0.05  # Also a placeholder value
        self.velocity = ref_velocity_direction / np.linalg.norm(ref_velocity_direction) * ref_velocity_magnitude

        new_position = self.position + self.velocity * self.size
        if not cage.is_inside(new_position):
            self.velocity *= -1
            new_position = self.position + self.velocity * self.size
        self.position = new_position

class SeaCage:
    def __init__(self, radius, depth):
        self.radius = radius
        self.depth = depth

    def is_inside(self, position):
        radial_distance = np.sqrt(position[0]**2 + position[1]**2)
        return radial_distance <= self.radius and 0 <= position[2] <= self.depth

class Simulation:
    def __init__(self, num_fish, cage_radius, cage_depth, fish_size_mean, fish_size_var):
        self.fish = []
        for _ in range(num_fish):
            radius = np.random.uniform(0, cage_radius)
            angle = np.random.uniform(0, 2 * np.pi)
            depth = np.random.uniform(0, cage_depth)
            position = np.array([radius * np.cos(angle), radius * np.sin(angle), depth])
            velocity = (np.random.rand(3) - 0.5) / 10
            size = np.random.normal(fish_size_mean, fish_size_var)
            self.fish.append(Fish(position, velocity, size))
        self.cage = SeaCage(cage_radius, cage_depth)

    def run(self, num_steps, visualize=False):
        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim3d([-self.cage.radius, self.cage.radius])
            ax.set_ylim3d([-self.cage.radius, self.cage.radius])
            ax.set_zlim3d([0, self.cage.depth])

            # For visualizing the cage boundary
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
                    ax.quiver(fish.position[0], fish.position[1], fish.position[2], 
                              fish.velocity[0] * fish.size, fish.velocity[1] * fish.size, 
                              fish.velocity[2] * fish.size, length=fish.size, color='red')
            
            if visualize:
                plt.pause(0.05)

        if visualize:
            plt.show()

# Simulation parameters
num_fish = 50
cage_radius = 10  # Example radius
cage_depth = 5    # Example depth
num_steps = 100   # Example number of steps
fish_size_mean = 5  # Mean fish size in meters
fish_size_var = 0.05  # Fish size variation in meters

# Run the simulation with visualization
simulation = Simulation(num_fish, cage_radius, cage_depth, fish_size_mean, fish_size_var)
simulation.run(num_steps, visualize=True)




#Stochatic component
    def get_rotation_matrix(self, velocity):
        # Create a rotation matrix from the velocity vector
        if norm(velocity) > 0:
            # Normalize the velocity vector to use as a basis for rotation
            direction = velocity / norm(velocity)
            # We need two perpendicular vectors to the direction for a full basis
            # For simplicity, we can use a simple trick if the direction is not vertical
            if abs(direction[2]) != 1:
                # Create a vector that is not collinear
                non_collinear = np.array([0, 0, 1])
                # Use cross product to find a vector perpendicular to the direction
                v1 = np.cross(direction, non_collinear)
                # Normalize v1
                v1 /= norm(v1)
                # The second perpendicular vector is perpendicular to both v1 and direction
                v2 = np.cross(direction, v1)
                # Now we have an orthonormal basis
                rotation_matrix = np.array([v1, v2, direction])
            else:
                # For the vertical direction, choose a different approach
                rotation_matrix = np.eye(3)  # This should be replaced with an appropriate rotation matrix
            return rotation_matrix
        else:
            # If the velocity is zero, we return an identity matrix
            return np.eye(3)

    def stochastic_component(self, sigma=0.25):
        # Generate a stochastic component vector with the given sigma
        random_vector = np.random.normal(0, sigma, 2)
        # We need to expand this to a 3D vector, the third component is 0
        stochastic_vector = np.array([random_vector[0], random_vector[1], 0])
        # Get the rotation matrix based on the current orientation
        rotation_matrix = self.get_rotation_matrix(self.velocity)
        # Apply the rotation matrix to the stochastic vector
        V_ST = rotation_matrix.dot(stochastic_vector)
        print('stochastic',V_ST)
        return V_ST

class Simulation:
    def __init__(self, num_fish, cage_radius, cage_depth):
        self.fish = []
        for fish_id in range(num_fish):
            size = np.random.uniform(0.3, 0.4)  # meters
            radius = np.random.uniform(0, cage_radius)
            angle = np.random.uniform(0, 2 * np.pi)
            depth = np.random.uniform(0, cage_depth)
            position = np.array([radius * np.cos(angle), radius * np.sin(angle), depth])  # Start position
            signs = np.random.choice([-1, 1], size=3)  # Choose random direction
            velocity = np.random.normal(0.5 * size, 0.2 * size, 3) * signs  # Start velocity (BL/S), normal distribution
            self.fish.append(Fish(position, velocity, tau=0.6, size=size, fish_id=fish_id))
        self.cage = SeaCage(cage_radius, cage_depth)

    def run(self, num_steps, visualize=False):
        frames = []
        dt = 1.0 / 10  # Time step

        # Initialize fish positions for frame 0
        fish_positions = {f.id: [f.position] for f in self.fish}

        for step in range(num_steps):
            for fish in self.fish:
                fish.update_neighbors(self.fish, react_dist)
                fish.update_position(dt)
                fish_positions[fish.id].append(fish.position)

            if step % 10 == 0:  # Adjust granularity for smoother animation
                frame_data = [py.Scatter3d(x=[pos[0] for pos in fish_positions[fid]],
                                           y=[pos[1] for pos in fish_positions[fid]],
                                           z=[pos[2] for pos in fish_positions[fid]],
                                           mode='markers',
                                           marker=dict(size=6, opacity=0.8))  # Adjust size based on fish size
                            for fid in fish_positions]
                frames.append(py.Frame(data=frame_data, name=str(step)))

        if visualize:
            fig = py.Figure(
                data=frames[0].data,
                frames=frames,
                layout=py.Layout(
                    scene=dict(
                        xaxis=dict(range=[-self.cage.radius, self.cage.radius]),
                        yaxis=dict(range=[-self.cage.radius, self.cage.radius]),
                        zaxis=dict(range=[0, self.cage.depth]),
                        aspectratio=dict(x=1, y=1, z=0.5)),
                    updatemenus=[dict(
                        type="buttons",
                        buttons=[dict(label="Play",
                                      method="animate",
                                      args=[None, {"frame": {"duration": 500, "redraw": True},
                                                   "fromcurrent": True}]),
                                 dict(label="Pause",
                                      method="animate",
                                      args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                     "mode": "immediate"}])])]))
            fig.show()

# Example usage:
simulation = Simulation(num_fish=10, cage_radius=10, cage_depth=5)
simulation.run(100, visualize=True)

class Simulation:
    def __init__(self, num_fish, cage_radius, cage_depth):
        self.fish = []
        for fish_id in range(num_fish):
            size = np.random.uniform(0.3, 0.4) #meters
            radius = np.random.uniform(0, cage_radius)
            angle = np.random.uniform(0, 2 * np.pi)
            depth = np.random.uniform(0, cage_depth)
            position = np.array([radius * np.cos(angle), radius * np.sin(angle), depth]) #Start position
            signs = np.random.choice([-1, 1], size=3) #Chose random direction
            velocity = np.random.normal(0.5 * size, 0.2 * size, 3)*signs #Start velocity (BL/S), normal distribution
            self.fish.append(Fish(position, velocity, tau=0.6, size=size, fish_id=fish_id))
        self.cage = SeaCage(cage_radius, cage_depth)

    def run(self, num_steps, visualize=False):
        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim3d([-self.cage.radius, self.cage.radius])
            ax.set_ylim3d([-self.cage.radius, self.cage.radius])
            ax.set_zlim3d([0, self.cage.depth])
            x = np.linspace(-self.cage.radius, self.cage.radius, 1000)
            z = np.linspace(0, self.cage.depth, 1000)
            Xc, Zc = np.meshgrid(x, z)
            Yc = np.sqrt(self.cage.radius**2 - Xc**2)
            #ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='blue')
            #ax.plot_surface(Xc, -Yc, Zc, alpha=0.3, color='blue')

        dt = 1.0/10  #Time step
        for _ in range(num_steps):
            for fish in self.fish:
                fish.update_neighbors(self.fish, react_dist)
                #print('neighbors',fish.neighbors)
            if visualize:
                ax.clear()
                ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='blue')
                ax.plot_surface(Xc, -Yc, Zc, alpha=0.3, color='blue')
                ax.set_xlim3d([-self.cage.radius, self.cage.radius])
                ax.set_ylim3d([-self.cage.radius, self.cage.radius])
                ax.set_zlim3d([0, self.cage.depth])
                ax.set_xlabel('X-direction')
                ax.set_ylabel('Y-direction')
                ax.set_zlabel('Z-direction')

            for fish in self.fish:
                fish.update_position(dt)
                if visualize:
                    ax.scatter(fish.position[0], fish.position[1], fish.position[2], color='red', marker='>', s=fish.size*100)

            if visualize:
                plt.pause(0.05)

        if visualize:
            plt.show()



# Run the simulation with visualization
simulation = Simulation(num_fish, cage_radius, cage_depth)
simulation.run(num_steps, visualize=True)
