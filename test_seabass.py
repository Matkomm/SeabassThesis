import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from numpy.linalg import norm
import os


directory = 'Simulation_data'
file_path = os.path.join(directory, 'neighbors_test_prev.csv') #Change to file name here to store new simulation data

dpref_bottom = 0.5  #meters
dpref_surface = 0.5  #meters
dpref_wall = 0.8  #meters
dpref_fish = 0.3 #meters
react_dist = 0.5  #meters
max_direction_change = np.radians(60)
fish_size_upper = 0.26  #meters
fish_size_lower = 0.24  #meters
ave_velocity_xy = 0  #BL/s
ave_velocity_z = 0  #BL/s

# Parameters for the simulation
num_fish =  500
cage_radius = 6.37  
cage_depth = 8 
num_steps = 240 #Number of steps
dt = 1.0/1 #Time step
elev = 0  #Elevation angle for the 3D plot
time_of_day = 'afternoon'  # 'morning', 'noon', 'afternoon', 'night'
#(morning : 6-10h, noon : 10-14h, afternoon : 14-18h and night : 20-6h)

# Speed factors for different times of the day
Speed_factor = {
    'morning': 1, 
    'noon': 1,      
    'afternoon': 1, 
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
        self.dist_pref_fish = 0.66 * self.size  #Based on 0.66 Body Lengths as the minimum preferred distance
        self.dist_rect_fish = 3 * self.size  #Based on 3 Body Lengths as the maximum reaction distance
        self.max_speed = 1.5*self.size
        if time_of_day == 'morning':
            self.characteristic_velocity = 0.71*self.size + 0.1 * self.size * np.random.normal()
        elif time_of_day == 'noon':
            self.characteristic_velocity = 0.48* self.size + 0.1 * self.size * np.random.normal()
        elif time_of_day == 'afternoon':
            self.characteristic_velocity = 0.45*self.size + 0.1 * self.size * np.random.normal()
        else:
            self.characteristic_velocity = 0.48 * self.size + 0.1 * self.size * np.random.normal()

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
        #random_vector =  np.random.uniform(-sigma, sigma, 3)
        random_vector[0] = 1.0 
        random_vector /= np.linalg.norm(random_vector) 
        rotation_matrix = self.get_rotation_matrix()
        V_ST = np.dot(rotation_matrix, random_vector)
        return V_ST
    
    def update_neighbors(self, all_fish, threshold_distance):
        self.neighbors = [f for f in all_fish if 0 < np.linalg.norm(f.position[:3] - self.position[:3]) <= threshold_distance]
        #print('neighbors', self.id, self.neighbors)


    def social_response(self, neighbors):
        self.neighbors = []  
        v_so = np.zeros(3)  
        for neighbor in neighbors:
            dij = neighbor.position[:3] - self.position[:3]  #Distance vector between fish i and j
            rij_dot = neighbor.velocity  
            #print('dij',dij)
            if np.linalg.norm(dij) <= self.dist_pref_fish:
                v_so_j = dij * (self.dist_pref_fish - np.linalg.norm(dij)) #If too close, swim away from the neighbor
            elif self.dist_pref_fish <= np.linalg.norm(dij) <= self.dist_rect_fish:
                v_so_j = 0.5 * rij_dot * (np.linalg.norm(dij) - self.dist_rect_fish) / (self.dist_rect_fish - self.dist_pref_fish)#If within preferred distance, try aligning with the neighbor
            else:
                v_so_j = np.zeros(3)
            
            v_so += v_so_j  
        
        return v_so / len(neighbors) if neighbors else v_so  
    
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

        # Update the fish's orientation
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

        quotas = {'horizontal': 1, 'vertical': 1}

        V_c = (self.v_cs(position[:3]) + self.v_cb(position[:3]) + self.v_cw(position[:3]))
        V_f = 0  #Food
        V_T = 0  #Temperature
        V_L = 0  #Light response, replaced by time of day?
        V_SO = self.social_response(self.neighbors)
        V_ST = self.stochastic_component() 
        #print('V_ST', V_ST, 'V_SO', V_SO, 'V_c', V_c)

        r_dot_change = np.zeros(3)

        r_dot_change[:2] += self.apply_quota(V_c[:2], quotas, 'horizontal')
        r_dot_change[2] += self.apply_quota(V_c[2], quotas, 'vertical')
     
        r_dot_change[:2] += self.apply_quota(V_SO[:2], quotas, 'horizontal')
        r_dot_change[2] += self.apply_quota(V_SO[2], quotas, 'vertical')
        
        r_dot_change[:2] += self.apply_quota(V_ST[:2], quotas, 'horizontal')
        r_dot_change[2] += self.apply_quota(V_ST[2], quotas, 'vertical')
       
        #Normalize the change in velocity to prevent it from exceeding the characteristic velocity
        if np.linalg.norm(r_dot_change) > self.characteristic_velocity:
            r_dot_change *= self.characteristic_velocity 

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
        for fish_id in range(num_fish):
            size = np.random.uniform(fish_size_lower, fish_size_upper) 
            radius = np.random.uniform(0, cage_radius-dpref_wall) #Prevent fish from starting too close to the wall
            angle = np.random.uniform(0,  2*np.pi) 
            depth = np.random.uniform(dpref_surface, cage_depth - dpref_bottom)
            signs = np.random.choice([-1, 1], 3)#Randomly choose direction of velocity
            sign = np.random.choice([-1, 1]) 

            if time_of_day == 'night' and num_fish != 1: 
                z = dpref_bottom + (cage_depth - dpref_bottom - dpref_surface)*(fish_id/(num_fish - 1))
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                psi = np.arctan2(y, x)
                theta = np.arctan2(np.sqrt(x**2 + y**2), z)
                position = np.array([x, y, z, psi, theta])
            else:
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                psi = np.arctan2(y, x)
                theta = np.arctan2(np.sqrt(x**2 + y**2), depth)
                position = np.array([x, y, depth, psi, theta]) #Start position
                

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
            headers = ['time_step', 'fish_id', 'fish_size'] + [f'pos_{dim}' for dim in ['x', 'y', 'z']] + [f'vel_{dim}' for dim in ['x', 'y', 'z']]
            data_file.write(','.join(headers) + '\n')
        else:
            data_file = None

        simulated_time = 0  #Initialize simulated time
        for step in range(num_steps):
            update_progress((step + 1) / num_steps) 
            for fish in self.fish:
                fish.update_neighbors(self.fish, react_dist) #react_dist could be updated to dist_rect_fish 

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
                ax.set_zlabel('Z-direction')

                simulated_time = step*dt
                ax.set_title(f"Simulated Time: {simulated_time:.2f} seconds")

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
simulation.run(num_steps, visualize=False, log_data=True)