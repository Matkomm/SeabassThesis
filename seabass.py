
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from numpy.linalg import norm
import os
#matplotlib.use('Qt5Agg')

directory = 'Simulation_data'
file_path = os.path.join(directory, 'fish_data1.csv') #Change to file name here to store new simulation data

# Constants/parameters and coefficients for fish simulation
k_values = {"k_C": 1/200, "k_F": 4/6, "k_T": 2/6, "k_L": 3/6, "k_SO": 5/600, "k_ST": 1/250}  #Velocity coefficients
dpref_bottom = 0.5  #meters
dpref_surface = 0.5  #meters
dpref_wall = 0.5  #meters
dpref_fish = 0.3 #meters, trying to change this to body length of each induvidual fish
react_dist = 0.5  #meters
#max_speed = 1.2  #BL/s
fish_size_upper = 0.26  #meters
fish_size_lower = 0.24  #meters
ave_velocity_xy = 0.2  #BL/s
ave_velocity_z = 0.08  #BL/s

# Parameters for the simulation
num_fish = 100
cage_radius = 6.37  
cage_depth = 8 
num_steps = 10000 #Number of steps
dt = 1.0/1 #Time step
elev = 0  #Elevation angle for the 3D plot
time_of_day = 'noon'  # 'morning', 'noon', 'afternoon', 'night'
#(morning : 6-10h, noon : 10-14h, afternoon : 14-18h and night : 20-6h)

# Speed factors for different times of the day
Speed_factor = {
    'morning': 1.2, 
    'noon': 1,      
    'afternoon': 0.8, 
    'night': 1      
}

class Fish:
    def __init__(self, position, velocity, tau, size, fish_id):
        self.id = fish_id
        self.position = position
        self.velocity = velocity
        self.tau = tau
        self.size = size
        self.r_dot_prev = np.zeros(3)
        self.delta_l = 0.66 * self.size  #Based on 0.66 Body Lengths as the minimum preferred distance
        self.delta_h = 3 * self.size  #Based on 3 Body Lengths as the maximum reaction distance

    #Behaviour: response to cage and water surface
    def v_cs(self, position):
        d_surf = cage_depth-position[2]
        if d_surf <= dpref_surface : #Dobbelcheck this
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
    
    def v_cw(self, position):#Create a unit vector pointing towards the center of the cage
        d_wall = cage_radius - np.sqrt(position[0]**2 + position[1]**2)
        if d_wall <= dpref_wall:
            direction_to_center = np.array([-position[0], -position[1], 0])
            norm = np.linalg.norm(direction_to_center[:2])
            if norm != 0: 
                unit_vec_center = direction_to_center/norm
            else:
                unit_vec_center = direction_to_center
            v_cw = unit_vec_center * (dpref_wall - d_wall)
        else:
            v_cw = np.array([0, 0, 0])
        return v_cw

    #Stochastic component
    def get_rotation_matrix(self, velocity):
        if norm(velocity) > 0:
            direction = velocity / norm(velocity)
            if abs(direction[2]) != 1:
                non_collinear = np.array([0, 0, 1])
                v1 = np.cross(direction, non_collinear)
                v1 /= norm(v1)
                v2 = np.cross(direction, v1)
                rotation_matrix = np.array([direction, v1, v2]).T  
            else:
                v1 = np.array([1, 0, 0])
                v2 = np.array([0, 1, 0])
                if direction[2] < 0:  #Pointing down
                    direction = -direction  
                rotation_matrix = np.array([direction, v1, v2]).T
            return rotation_matrix
        return np.eye(3)

    def stochastic_component(self, sigma=0.25):
        random_vector = np.array([np.random.normal(0, sigma), np.random.normal(0, sigma), np.random.normal(0, sigma)]) #Changed x from 1 to normal distribution
        rotation_matrix = self.get_rotation_matrix(self.velocity)
        V_ST = rotation_matrix.dot(random_vector)
        return V_ST

    def update_neighbors(self, all_fish, threshold_distance):
        self.neighbors = [f for f in all_fish if 0 < np.linalg.norm(f.position - self.position) <= threshold_distance]
        #print('neighbors', self.id, self.neighbors)


    def social_response(self, neighbors):
        self.neighbors = []  # This should be replaced with actual neighbor fish
        v_so = np.zeros(3)  
        for neighbor in neighbors:
            dij = neighbor.position - self.position  #Distance vector between fish i and j
            rij_dot = neighbor.velocity  
            #print('dij',dij)
            if np.linalg.norm(dij) <= self.delta_l:
                v_so_j = dij * (self.delta_l - np.linalg.norm(dij)) #If too close, swim away from the neighbor
            elif self.delta_l <= np.linalg.norm(dij) <= self.delta_h:
                v_so_j = 0.5 * rij_dot * (np.linalg.norm(dij) - self.delta_h) / (self.delta_h - self.delta_l)#If within preferred distance, try aligning with the neighbor
            else:
                #Otherwise, no response
                v_so_j = np.zeros(3)
            
            v_so += v_so_j  
        
        return v_so / len(neighbors) if neighbors else v_so  
    
    def velocity_ode(self, t, y):
        #Unpack the current position and velocity from y
        position, velocity = y[:3], y[3:]
        V_c = (self.v_cs(position) + self.v_cb(position) + self.v_cw(position))
        V_f = 0  #Food
        V_T = 0  #Temperature
        V_L = 0  #Light response, replaced by time of day?
        V_SO = self.social_response(self.neighbors)
        V_ST = self.stochastic_component() #Very computing heavy!!
        #print('V_SO', self.id, V_SO)
        #print('V_ST', self.id, V_ST)
        #print('V_c', self.id, V_c)

        V_c = V_c / np.linalg.norm(V_c) if np.linalg.norm(V_c) > 1.0 else V_c
        V_SO = V_SO / np.linalg.norm(V_SO) if np.linalg.norm(V_SO) > 1.0 else V_SO
        V_ST = V_ST / np.linalg.norm(V_ST) if np.linalg.norm(V_ST) > 1.0 else V_ST
        
        #Compute new velocity using the reference velocity equation
        r_dot_ref = self.tau * self.r_dot_prev + (1 - self.tau) * (k_values["k_C"] * V_c + k_values["k_F"] * V_f +
                                                            k_values["k_T"] * V_T + k_values["k_L"] * V_L +
                                                            k_values["k_SO"] * V_SO + k_values["k_ST"] * V_ST)
        self.r_dot_prev = r_dot_ref
        #print('ref',self.r_dot_prev)
        #print('velocity', velocity)
        return np.concatenate((velocity, r_dot_ref))


    def update_position(self, dt):
        #Initial conditions y0 = [position, velocity]
        y0 = np.concatenate((self.position, self.velocity))
        t_span = (0, float(dt)) #One time step for the ODE
        sol = solve_ivp(self.velocity_ode, t_span, y0, method='RK45')
        #Update fish position and velocity with the last solution step
        self.position, self.velocity = sol.y[:3, -1], sol.y[3:, -1]
        #print('velocity', self.velocity)

class SeaCage:
    def __init__(self, radius, depth):
        self.radius = radius
        self.depth = depth


class Simulation:
    def __init__(self, num_fish, cage_radius, cage_depth): #initialize
        self.fish = []
        for fish_id in range(num_fish):
            size = np.random.uniform(fish_size_lower, fish_size_upper) 
            radius = np.random.uniform(0, cage_radius-dpref_wall) #Prevent fish from starting too close to the wall
            angle = np.random.uniform(0,  2*np.pi) 
            depth = np.random.uniform(dpref_bottom, cage_depth-dpref_surface)
            signs = np.random.choice([-1, 1], 3)#Randomly choose direction of velocity
            sign = np.random.choice([-1, 1]) 

            if time_of_day == 'night' and num_fish != 1: 
                z = dpref_bottom + (cage_depth - dpref_bottom - dpref_surface)*(fish_id/(num_fish - 1))
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                position = np.array([x, y, z])
            else:
                position = np.array([radius * np.cos(angle), radius * np.sin(angle), depth]) #Start position
                

            if time_of_day == 'morning' or time_of_day == 'noon':
                velocity = np.random.normal(ave_velocity_xy * size, 0.03 * size, 3)*signs
                velocity[2] = np.random.normal(ave_velocity_z * size, 0.01 * size)
            else:
                velocity = np.random.normal(ave_velocity_xy * size, 0.03 * size, 3)*signs  # Start velocity (BL/S), normal distribution
                velocity[2] = np.random.normal(ave_velocity_z * size, 0.01 * size)*sign  #Start speed in z-direction
            print('fish', fish_id, 'position', position, 'velocity', velocity, 'size', size )

            if time_of_day == 'morning':
                velocity *= Speed_factor['morning']
            elif time_of_day == 'noon':
                velocity *= Speed_factor['noon']
            elif time_of_day == 'afternoon':
                velocity *= Speed_factor['afternoon']
            else:
                velocity *= Speed_factor['night']
            self.fish.append(Fish(position, velocity, tau=0.6, size=size, fish_id=fish_id))
            #print('fish', fish_id, 'position', position, 'velocity', velocity, 'size', size )
        self.cage = SeaCage(cage_radius, cage_depth)
        

    def run(self, num_steps, visualize=False, log_data=False):
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

        if log_data:
            if not os.path.exists(directory):
                os.makedirs(directory)
            data_file = open(file_path, 'w')
            headers = ['time_step', 'fish_id'] + [f'pos_{dim}' for dim in ['x', 'y', 'z']] + [f'vel_{dim}' for dim in ['x', 'y', 'z']]
            data_file.write(','.join(headers) + '\n')
        else:
            data_file = None

        simulated_time = 0  #Initialize simulated time
        for step in range(num_steps):
            update_progress((step + 1) / num_steps) 
            for fish in self.fish:
                fish.update_neighbors(self.fish, react_dist) #react_dist could be updated to delta_h 

                if log_data:
                        fish_data = [str(step), str(fish.id)] + list(map(str, fish.position)) + list(map(str, fish.velocity))
                        data_file.write(','.join(fish_data) + '\n')

            if visualize:
                ax.clear()
                ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='blue')
                ax.plot_surface(Xc, -Yc, Zc, alpha=0.3, color='blue')
                ax.set_xlim3d([-self.cage.radius, self.cage.radius])
                ax.set_ylim3d([-self.cage.radius, self.cage.radius])
                ax.set_zlim3d([0, self.cage.depth])

                angle = step % 360
                ax.view_init(elev)# azim=angle)

                ax.set_xlabel('X-direction')
                ax.set_ylabel('Y-direction')
                ax.set_zlabel('Z-direction')

                simulated_time = step*dt
                ax.set_title(f"Simulated Time: {simulated_time:.2f} seconds")

            for fish in self.fish:
                fish.update_position(dt)
                if visualize:
                    #ax.scatter(fish.position[0], fish.position[1], fish.position[2], color='red', marker='>', s=fish.size*100)
                    ax.quiver(fish.position[0], fish.position[1], fish.position[2], fish.velocity[0], fish.velocity[1], fish.velocity[2], color='red', length=fish.size, normalize=True)

            if visualize:
                plt.pause(0.05)

        if visualize:
            plt.show()


def update_progress(progress):
    bar_length = 50
    block = int(round(bar_length * progress))

    bar = "█" * block + "-" * (bar_length - block)
    print(f"\rProgress: [{bar}] {progress * 100:.2f}%", end='')

    #New line on complete
    if progress >= 1:
        print()
        
#Run the simulation with visualization
simulation = Simulation(num_fish, cage_radius, cage_depth)
simulation.run(num_steps, visualize=False, log_data=False)