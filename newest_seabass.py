import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
import os


directory = 'Simulation_data'
file_path = os.path.join(directory, 'evening_final10.csv') #Change to file name here to store new simulation data

dpref_bottom = 0.5  #meters
dpref_surface = 0.5  #meters
dpref_wall = 0.8  #meters
max_direction_change = np.radians(60)
fish_size_upper = 0.26  #meters
fish_size_lower = 0.24  #meters
ave_velocity_xy = 0  
ave_velocity_z = 0  
Divide_cell_size = 10  #Adjustable, smaller number, less cells


# Parameters for the simulation
num_fish =  300
cage_radius = 6.37  
cage_depth = 8 
num_steps = 300 #Number of steps
dt = 1.0/1 #Time step
elev = 0  #Elevation angle for the 3D plot
time_of_day = 'morning'  
Temperature = 22.86  #Temperature 12-30°C
#(morning : 6-10h, noon : 10-13h, afternoon : 13-17h, evening :17-20h and night : 20-6h)

visualize=True #Visualize the simulation
log_data=False #Log the data to a file

free_will_percent = {'morning': 0.5, 'noon': 0.8, 'afternoon': 0.6, 'evening': 0.7, 'night': 0.9} #Percentage of fish that swim freely (0-1)
free_will_percentage = free_will_percent[time_of_day] 
#Depth preferences for different times of the day (preferred depth, standard deviation)
depth_preferences = {'morning': (1.2, 1), 'noon': (4.7, 1.2), 'afternoon': (4.5, 1.5), 'evening': (1.5, 2), 'night': (4, 2)
}

#Temperature: response to temperature 
def temperature_coefficient(T):
    if T < 12 or T > 30:
        raise ValueError("Temperature out of range. The simulation only runs for temperatures between 12°C and 30°C.")
    a = 1.5437e-05
    b = 5.2958e-05
    c = -0.004567
    d = -0.0003563
    e = 1.0039
    return a * (T - 22.86)**4 + b * (T - 22.86)**3 + c * (T - 22.86)**2 + d * (T - 22.86) + e

class Fish:
    def __init__(self, position, velocity, tau, size, fish_id):
        self.id = fish_id
        self.position = position
        self.velocity = velocity
        self.tau = tau
        self.size = size
        self.r_dot_prev = np.zeros(3)
        self.neighbors = []
        self.dist_pref_fish = 0.66 * self.size  #Based on 0.66 Body Lengths as the minimum preferred distance
        self.dist_rect_fish = 3 * self.size  #Based on 3 Body Lengths as the maximum reaction distance
        self.max_speed = 1.5*self.size
        self.time_of_day = time_of_day
        self.characteristic_velocity = 0
        # Initialize depth target and duration
        self.depth_preferences = self.set_depth_preferences(time_of_day)
        self.depth_target, self.target_duration = self.initialize_depth_target(self.depth_preferences)
        
        while self.characteristic_velocity < 0.2*self.size: #Remove "dead fish"
            if time_of_day == 'morning':
                self.characteristic_velocity = 0.75*self.size + 0.1 * self.size * np.random.normal()
            elif time_of_day == 'noon':
                self.characteristic_velocity = 0.49* self.size + 0.1 * self.size * np.random.normal()
            elif time_of_day == 'afternoon':
                self.characteristic_velocity = 0.46*self.size + 0.1 * self.size * np.random.normal()
            elif time_of_day == 'evening':
                self.characteristic_velocity = 0.47*self.size + 0.1 * self.size * np.random.normal()
            else: #Night should not be used 
                self.characteristic_velocity = 0.51*self.size + 0.1 * self.size * np.random.normal()

        self.characteristic_velocity *= temperature_coefficient(Temperature)       
     #The np.random.normal() function generates a random float drawn from a standard normal distribution 
     #(mean of 0 and standard deviation of 1).

    #Behaviour: response to cage and water surface
    def v_cs(self, position):
        d_surf = position[2]
        if d_surf <= dpref_surface : 
            v_cs = np.array([0, 0, 1]) * (dpref_surface - d_surf)
        else:
            v_cs = np.array([0, 0, 0])
        return v_cs
    
    def v_cb(self, position):
        d_bottom = cage_depth-position[2]
        if d_bottom <= dpref_bottom :
            v_cb = np.array([0, 0, -1]) * (dpref_bottom - d_bottom)
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

    #Depth preference
    def set_depth_preferences(self, time_of_day):
            return depth_preferences[time_of_day]

    def initialize_depth_target(self, depth_preferences):
        free_will = np.random.uniform(0, 1)
        target_duration = np.random.randint(20, 60)  #Duration of target depth
        if free_will <= free_will_percentage:  #% chance of free swimming
            target_depth = None
        else:
            preferred_depth, std_dev = depth_preferences
            target_depth = np.random.normal(preferred_depth, std_dev)
            if target_depth < dpref_surface:
                target_depth = dpref_surface
            elif target_depth > cage_depth - dpref_bottom:
                target_depth = cage_depth - dpref_bottom  
        return target_depth, target_duration 

    def update_depth_target(self):
        if self.target_duration <= 0:
            self.depth_target, self.target_duration = self.initialize_depth_target(self.depth_preferences)
        else:
            self.target_duration -= 1

    def v_dp(self):
        if self.depth_target is None:
            return np.array([0, 0, 0])
        else:
            depth_difference = self.position[2] - self.depth_target
            return np.array([0, 0, -np.sign(depth_difference) * min(abs(depth_difference), 1)])


    #Stochastic component
    def get_rotation_matrix(self):
        psi = self.position[3]
        theta = self.position[4]
        cosPsi = np.cos(psi)
        sinPsi = np.sin(psi)
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)

        R = np.array([
            [cosPsi * cosTheta, -sinPsi, -cosPsi * sinTheta],
            [sinPsi * cosTheta, cosPsi, -sinPsi * sinTheta],
            [sinTheta, 0, cosTheta]
        ])
        return R

    def stochastic_component(self, sigma=0.25):
        random_vector = sigma * np.random.randn(3)
        random_vector[0] = 1.0 
        random_vector /= np.linalg.norm(random_vector) 
        rotation_matrix = self.get_rotation_matrix()
        V_ST = np.dot(rotation_matrix, random_vector)
        return V_ST
    
    def update_neighbors(self, all_fish, threshold_distance):
        self.neighbors = [f for f in all_fish if 0 < np.linalg.norm(f.position[:3] - self.position[:3]) <= threshold_distance]


    def social_response(self, neighbors):
        v_so = np.zeros(3)  
        num_neighbors = 0  
        for neighbor in neighbors:
            dij = neighbor.position[:3] - self.position[:3]  #Distance vector between fish i and j
            distance = np.linalg.norm(dij)
            rj_dot = neighbor.velocity  

            if distance <= self.dist_pref_fish:
                v_so_j = dij * (self.dist_pref_fish - distance) 
                v_so += v_so_j
                num_neighbors += 1
            elif self.dist_pref_fish < distance <= self.dist_rect_fish:
                v_so_j = 0.5 * rj_dot * (distance - self.dist_rect_fish) / (self.dist_rect_fish - self.dist_pref_fish)
                v_so += v_so_j
                num_neighbors += 1

        if num_neighbors > 0:
            v_so /= num_neighbors  #Average the contributions

        return v_so

    def update_orientation(self):
        psi = self.position[3]
        theta = self.position[4]

        #Calculate the reference angles based on the new velocity
        psi_ref = np.arctan2(self.velocity[1], self.velocity[0])
        theta_ref = np.arctan2(self.velocity[2], np.sqrt(self.velocity[0]**2 + self.velocity[1]**2))

        psi_dot = self._angle_difference(psi, psi_ref)
        theta_dot = self._angle_difference(theta, theta_ref)

        #constraint the change in angles
        psi_dot = np.clip(psi_dot, -max_direction_change, max_direction_change)
        theta_dot = np.clip(theta_dot, -max_direction_change, max_direction_change)

        new_psi = self._constrain_angle(psi + psi_dot)
        new_theta = self._constrain_angle(theta + theta_dot)

        self.position[3] = new_psi
        self.position[4] = new_theta

    def _constrain_angle(self, angle):
        #Constrain angle to the range [-pi, pi]
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _angle_difference(self, angle1, angle2):
        #Compute the difference between two angles and constrain it to [-pi, pi]
        difference = angle2 - angle1
        while difference < -np.pi:
            difference += 2 * np.pi
        while difference > np.pi:
            difference -= 2 * np.pi
        return difference

    def velocity_ode(self, position):

        quotas = {'horizontal': 1, 'vertical': 0.5}

        V_c = (self.v_cs(position[:3]) + self.v_cb(position[:3]) + self.v_cw(position[:3]))
        V_DP = self.v_dp()
        V_SO = self.social_response(self.neighbors) 
        V_ST = self.stochastic_component() 

        r_dot_change = np.zeros(3)

        r_dot_change[:2] += self.apply_quota(V_c[:2], quotas, 'horizontal')
        r_dot_change[2] += self.apply_quota(V_c[2], quotas, 'vertical')
     
        r_dot_change[:2] += self.apply_quota(V_SO[:2], quotas, 'horizontal')
        r_dot_change[2] += self.apply_quota(V_SO[2], quotas, 'vertical')
        
        r_dot_change[:2] += self.apply_quota(V_ST[:2], quotas, 'horizontal')
        r_dot_change[2] += self.apply_quota(V_ST[2], quotas, 'vertical')

        r_dot_change[:2] += self.apply_quota(V_DP[:2], quotas, 'horizontal')
        r_dot_change[2] += self.apply_quota(V_DP[2], quotas, 'vertical')
       
        if np.linalg.norm(r_dot_change) > self.characteristic_velocity:
           r_dot_change *= self.characteristic_velocity * np.random.uniform(0.8, 1.2)
        #Reference velocity
        r_dot_ref = self.tau * self.r_dot_prev + (1 - self.tau) * r_dot_change

        self.r_dot_prev = r_dot_ref

        return r_dot_ref

    def apply_quota(self, behavior_contribution, quotas, direction):
        magnitude = np.linalg.norm(behavior_contribution)
        if magnitude > 0:
            scale_factor = min(quotas[direction], magnitude)
            quotas[direction] -= scale_factor 
            return behavior_contribution / magnitude * scale_factor
        return np.zeros_like(behavior_contribution)

    def euler_step(self, dt):
            self.update_orientation() 

            self.position[:3] = self.position[:3] + self.velocity * dt

            new_velocity = self.velocity_ode(self.position[:3])

            self.velocity = new_velocity

            speed = norm(self.velocity)
            if speed > self.max_speed:
                self.velocity = self.max_speed
           
class SeaCage: 
    def __init__(self, radius, depth):
        self.radius = radius
        self.depth = depth


class Simulation:
    def __init__(self, num_fish, cage_radius, cage_depth): #initialize
        self.fish = []
        self.num_fish = num_fish
        self.cage_radius = cage_radius
        self.cage_depth = cage_depth
        self.cell_size = max(self.cage_radius, self.cage_depth) / Divide_cell_size
        self.grid_dimensions = np.ceil(np.array([2*self.cage_radius, 2*self.cage_radius, self.cage_depth]) / self.cell_size).astype(int)
        print('grid_dimensions', self.grid_dimensions)
        self.grid = [[[[] for _ in range(self.grid_dimensions[2])] for _ in range(self.grid_dimensions[1])] for _ in range(self.grid_dimensions[0])]
        for fish_id in range(num_fish):
            size = np.random.uniform(fish_size_lower, fish_size_upper) 
            radius = np.random.uniform(0, cage_radius-dpref_wall) #Prevent fish from starting too close to the wall
            angle = np.random.uniform(0,  2*np.pi) 
            depth = np.random.uniform(dpref_surface, cage_depth - dpref_bottom)
            signs = np.random.choice([-1, 1], 3)#Randomly choose direction of velocity
            sign = np.random.choice([-1, 1]) 

            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            psi = np.arctan2(y, x)
            theta = np.arctan2(np.sqrt(x**2 + y**2), depth)
            position = np.array([x, y, depth, psi, theta]) #Start position
            
            velocity = np.random.normal(ave_velocity_xy * size, 0.03 * size, 3)*signs
            velocity[2] = np.random.normal(ave_velocity_z * size, 0.01 * size)*sign
            
            cell_x = int((position[0] + self.cage_radius) / self.cell_size)
            cell_y = int((position[1] + self.cage_radius) / self.cell_size)
            cell_z = int(position[2] / self.cell_size)

            if cell_x < 0 or cell_x >= self.grid_dimensions[0] or cell_y < 0 or cell_y >= self.grid_dimensions[1] or cell_z < 0 or cell_z >= self.grid_dimensions[2]:
                print(f"Fish {fish_id} is out of grid bounds!")
                continue

            new_fish = Fish(position, velocity, tau=0.6, size=size, fish_id=fish_id)
            self.grid[cell_x][cell_y][cell_z].append(new_fish)
            self.fish.append(new_fish)

        self.cage = SeaCage(cage_radius, cage_depth)

    def update_grid(self):
    # Clear and update the grid
        for x in range(self.grid_dimensions[0]):
            for y in range(self.grid_dimensions[1]):
                for z in range(self.grid_dimensions[2]):
                    self.grid[x][y][z].clear()
        for fish in self.fish:
            cell_x = int((fish.position[0] + self.cage_radius) / self.cell_size)
            cell_y = int((fish.position[1] + self.cage_radius) / self.cell_size)
            cell_z = int(fish.position[2] / self.cell_size)
            self.grid[cell_x][cell_y][cell_z].append(fish)

    def find_neighbors(self, fish):
        neighbors = []
        cell_x = int((fish.position[0] + self.cage_radius) / self.cell_size)
        cell_y = int((fish.position[1] + self.cage_radius) / self.cell_size)
        cell_z = int(fish.position[2] / self.cell_size)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    nx, ny, nz = cell_x + dx, cell_y + dy, cell_z + dz
                    if 0 <= nx < self.grid_dimensions[0] and 0 <= ny < self.grid_dimensions[1] and 0 <= nz < self.grid_dimensions[2]:
                        neighbors.extend([f for f in self.grid[nx][ny][nz] if f != fish])
        return neighbors   

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
            headers = ['time_step', 'fish_id', 'fish_size'] + [f'pos_{dim}' for dim in ['x', 'y', 'z']] + [f'vel_{dim}' for dim in ['x', 'y', 'z']]
            data_file.write(','.join(headers) + '\n')
        else:
            data_file = None

        simulated_time = 0  #Initialize simulated time
        
        for step in range(num_steps):
            update_progress((step + 1) / num_steps) 
            for fish in self.fish:
                fish.neighbors = self.find_neighbors(fish)  #Find neighbors using the grid, all fish in nearest 26 cubes
                fish.update_depth_target()
                

                if log_data:
                    fish_data = [str(step), str(fish.id), str(fish.size)] + list(map(str, fish.position[:3])) + list(map(str, fish.velocity))
                    data_file.write(','.join(fish_data) + '\n')
                    
            if visualize:
                ax.clear()
                ax.plot_surface(Xc, Yc, Zc, alpha=0.3, color='blue')
                ax.plot_surface(Xc, -Yc, Zc, alpha=0.3, color='blue')
                ax.set_xlim3d([-self.cage.radius, self.cage.radius])
                ax.set_ylim3d([-self.cage.radius, self.cage.radius])
                ax.set_zlim3d([self.cage.depth, 0])

                angle = step % 360
                ax.view_init(elev)# azim=angle)

                ax.set_xlabel('X-direction')
                ax.set_ylabel('Y-direction')
                ax.set_zlabel('Depth')

                simulated_time = step*dt
                ax.set_title(f"Simulated Time: {simulated_time:.2f} seconds")
            """"
            # Drawing the grid
                for i in range(self.grid_dimensions[0]):
                    for j in range(self.grid_dimensions[1]):
                        for k in range(self.grid_dimensions[2]):
                            # Calculate the boundaries of the cell
                            x = [-self.cage_radius + i * self.cell_size, -self.cage_radius + (i+1) * self.cell_size]
                            y = [-self.cage_radius + j * self.cell_size, -self.cage_radius + (j+1) * self.cell_size]
                            z = [k * self.cell_size, (k+1) * self.cell_size]

                            # Draw lines to represent the grid cell
                            ax.plot([x[0], x[1]], [y[0], y[0]], [z[0], z[0]], 'gray', linewidth=0.5, linestyle=':')
                            ax.plot([x[0], x[1]], [y[1], y[1]], [z[0], z[0]], 'gray', linewidth=0.5, linestyle=':')
                            ax.plot([x[0], x[0]], [y[0], y[1]], [z[0], z[0]], 'gray', linewidth=0.5, linestyle=':')
                            ax.plot([x[1], x[1]], [y[0], y[1]], [z[0], z[0]], 'gray', linewidth=0.5, linestyle=':')
"""
            for fish in self.fish:
                fish.euler_step(dt)
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

    if progress >= 1:
        print()
        
#Run the simulation with visualization
simulation = Simulation(num_fish, cage_radius, cage_depth)
simulation.run(num_steps, visualize, log_data)