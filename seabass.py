import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from numpy.linalg import norm


# Constants/parameters and coefficients
k_values = {"k_C": 6/6, "k_F": 4/6, "k_T": 2/6, "k_L": 3/6, "k_SO": 5/6, "k_ST": 1/6}  #Hierarchical coefficients
dpref_bottom = 0.5  #meters
dpref_surface = 0.5  #meters
dpref_wall = 1  #meters
dpref_fish = 0.2 #meters, trying to change this to body length of each induvidual fish
react_dist = 0.6  #meters
max_speed = 1.2  #m/s
average_speed = 0.6  #m/s

# Parameters for the simulation
num_fish = 2
cage_radius = 10  # Example radius
cage_depth = 5  # Example depth
num_steps = 100  # Example number of steps

class Fish:
    def __init__(self, position, velocity, tau, size):
        self.position = position
        self.velocity = velocity
        self.tau = tau
        self.size = size

        #Behaviour: response to cage and water surface
    def v_cs(self, position):
        d_surf = cage_depth-position[2]
        if d_surf <= dpref_surface :
            v_cs = np.array([0, 0, -1]) * (dpref_surface - d_surf)
        else:
            v_cs = np.array([0, 0, 0])
        return v_cs
    
    def v_cb(self, position):
        d_bottom = position[2]
        if d_bottom <= dpref_bottom :
            v_cb = np.array([0, 0, 1]) * (dpref_bottom - d_bottom)
        else:
            v_cb = np.array([0, 0, 0])
        return v_cb
    
    def v_cw(self, position):
        d_wall = cage_radius - np.sqrt(position[0]**2 + position[1]**2)
        if d_wall <= dpref_wall:
            # Create a unit vector pointing towards the center of the cage
            # This requires normalizing the horizontal component of the position vector
            direction_to_center = np.array([-position[0], -position[1], 0])
            # Normalize this vector to create a unit vector
            norm = np.linalg.norm(direction_to_center[:2])
            if norm != 0:  # Check to prevent division by zero
                unit_vector_to_center = direction_to_center / norm
            else:
                unit_vector_to_center = direction_to_center
            v_cw = unit_vector_to_center * (dpref_wall - d_wall)
        else:
            v_cw = np.array([0, 0, 0])
        return v_cw

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
        return V_ST

    def velocity_ode(self, t, y):
        # Unpack the current position and velocity from y
        position, velocity = y[:3], y[3:]

        V_c = (self.v_cs(position) + self.v_cb(position) + self.v_cw(position))
        V_f = 0  # Replace with actual calculation
        V_T = 0  # Replace with actual calculation
        V_L = 0  # Replace with actual calculation
        V_SO = 0  # Replace with actual calculation
        V_ST = self.stochastic_component()

        # Compute new velocity using the reference velocity equation
        r_dot_ref = self.tau * velocity + (1 - self.tau) * (k_values["k_C"] * V_c + k_values["k_F"] * V_f +
                                                            k_values["k_T"] * V_T + k_values["k_L"] * V_L +
                                                            k_values["k_SO"] * V_SO + k_values["k_ST"] * V_ST)

        # The output is the derivative of position and velocity)
        return np.concatenate((velocity, r_dot_ref))


    def update_position(self, dt):
        # Initial conditions for solve_ivp [position, velocity]
        y0 = np.concatenate((self.position, self.velocity))
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

"""    def is_inside(self, position):
        radial_distance = np.sqrt(position[0]**2 + position[1]**2)
        return radial_distance <= self.radius and 0 <= position[2] <= self.depth"""

class Simulation:
    def __init__(self, num_fish, cage_radius, cage_depth):
        self.fish = []
        for _ in range(num_fish):
            size = np.random.uniform(0.3, 0.4) #meters
            radius = np.random.uniform(0, cage_radius)
            angle = np.random.uniform(0, 2 * np.pi)
            depth = np.random.uniform(0, cage_depth)
            position = np.array([radius * np.cos(angle), radius * np.sin(angle), depth]) #Start position
            velocity = np.random.normal(0.5 * size, 0.2 * size, 3) #Start velocity (BL/S), normal distribution
            print(velocity)
            self.fish.append(Fish(position, velocity, tau=0.6, size=size))
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

        dt = 1.0/10  #Time step
        for _ in range(num_steps):
            if visualize:
                ax.clear()
                ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='blue')
                ax.plot_surface(Xc, -Yc, Zc, alpha=0.3, color='blue')

            for fish in self.fish:
                fish.update_position(dt)
                if visualize:
                    # Visualize fish as ellipsoid
                    u = np.linspace(0, 2 * np.pi, 100)
                    v = np.linspace(0, np.pi, 100)
                    x = fish.size * np.outer(np.cos(u), np.sin(v)) + fish.position[0]
                    y = fish.size * np.outer(np.sin(u), np.sin(v)) + fish.position[1]
                    z = fish.size * np.outer(np.ones(np.size(u)), np.cos(v)) + fish.position[2]
                    ax.plot_surface(x, y, z, color='red')

            if visualize:
                plt.pause(0.05)

        if visualize:
            plt.show()



# Run the simulation with visualization
simulation = Simulation(num_fish, cage_radius, cage_depth)
simulation.run(num_steps, visualize=True)
