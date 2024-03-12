import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# Constants and coefficients will go here
# For example:
k_values = {"k_C": 1, "k_F": 1, "k_T": 1, "k_L": 1, "k_SO": 1, "k_ST": 1}  # Placeholder values

class Fish:
    def __init__(self, position, velocity, tau):
        self.position = position
        self.velocity = velocity
        self.tau = tau
        # Assume k_values are global or passed to the fish

    def velocity_ode(self, t, y):
        # Unpack the current position and velocity from y
        position, velocity = y[:3], y[3:]
        print("pos: ",position)
        print("velo:", velocity)
        # Calculate the velocity components based on current state
        # Placeholder functions for velocity components (these need to be defined properly)
        V_c = 2  # Replace with actual calculation
        V_f = 0.4 # Replace with actual calculation
        V_T = 1  # Replace with actual calculation
        V_L = 0  # Replace with actual calculation
        V_SO = 0.3  # Replace with actual calculation
        V_ST = 0.1  # Replace with actual calculation

        # Compute new velocity using the reference velocity equation
        r_dot_ref = self.tau * velocity + (1 - self.tau) * (k_values["k_C"] * V_c + k_values["k_F"] * V_f + 
                                                            k_values["k_T"] * V_T + k_values["k_L"] * V_L + 
                                                            k_values["k_SO"] * V_SO + k_values["k_ST"] * V_ST)
        
        # The output is the derivative of position and velocity
        return np.concatenate((velocity, r_dot_ref))

    def update_position(self, dt):
        # Initial conditions for solve_ivp [position, velocity]
        y0 = np.concatenate((self.position, self.velocity))
        print("Y0:", y0)
        # Time span for the ODE solver for one step of the simulation
        t_span = (0, float(dt))  # Ensure dt is cast to float

        # Solve the ODE
        sol = solve_ivp(self.velocity_ode, t_span, y0, method='RK45')

        # Update fish position and velocity with the last solution step
        self.position, self.velocity = sol.y[:3, -1], sol.y[3:, -1]



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
            self.fish.append(Fish(position, velocity, tau = 0.6))
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

        dt = 1.0/10
        for _ in range(num_steps):
            if visualize:
                ax.clear()
                ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='blue')
                ax.plot_surface(Xc, -Yc, Zc, alpha=0.3, color='blue')

            for fish in self.fish:
                fish.update_position(dt)
                if visualize:
                    ax.scatter(fish.position[0], fish.position[1], fish.position[2], color='red')
            
            if visualize:
                plt.pause(0.05)

        if visualize:
            plt.show()

# Parameters for the simulation
num_fish = 1
cage_radius = 10  # Example radius
cage_depth = 5  # Example depth
num_steps = 100  # Example number of steps

# Run the simulation with visualization
simulation = Simulation(num_fish, cage_radius, cage_depth)
simulation.run(num_steps, visualize=True)
