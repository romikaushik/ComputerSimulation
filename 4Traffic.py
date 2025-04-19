import numpy as np
import matplotlib.pyplot as plt
import random

class TrafficSimulation:
    """
    A class to simulate simple one-lane traffic flow using a cellular automaton model.
    """

    def __init__(self, road_length, car_density, iterations):
        """
        Initializes the simulation.

        Parameters:
        road_length (int): The length of the road (number of positions).
        car_density (float): The fraction of positions occupied by cars (between 0 and 1).
        iterations (int): The number of time steps to run the simulation.
        """
        self.road_length = road_length
        self.car_density = car_density
        self.iterations = iterations
        self.road = np.zeros(road_length, dtype=int)  # Initialize road as an empty array
        self.initialize_road()

    def initialize_road(self):
        """
        Randomly places cars on the road according to the specified density.
        """
        num_cars = int(self.car_density * self.road_length)  # Calculate number of cars
        car_positions = random.sample(range(self.road_length), num_cars)  # Select unique positions
        for pos in car_positions:
            self.road[pos] = 1  # Place a car in each selected position

    def update(self):
        """
        Updates the road for one time step.

        Moves cars forward if the next position is empty. The road is circular,
        meaning cars that move beyond the last position wrap around to the start.

        Returns:
        float: The average speed (fraction of moving cars).
        """
        new_road = np.zeros_like(self.road)  # Create a new empty road for the next state
        moved_cars = 0  # Counter for moved cars
        total_cars = np.sum(self.road)  # Count total number of cars
        
        for i in range(self.road_length):
            if self.road[i] == 1:  # If there is a car at position i
                next_pos = (i + 1) % self.road_length  # Calculate next position (circular road)
                if self.road[next_pos] == 0:  # If the next position is empty, move the car
                    new_road[next_pos] = 1
                    moved_cars += 1
                else:  # If the next position is occupied, keep the car in place
                    new_road[i] = 1
        
        self.road = new_road  # Update the road state
        avg_speed = moved_cars / total_cars if total_cars > 0 else 0  # Compute average speed
        return avg_speed

    def run_simulation(self):
        """
        Runs the traffic simulation for the specified number of iterations.

        Returns:
        list: A list containing the average speed at each time step.
        """
        avg_speeds = []
        for _ in range(self.iterations):
            avg_speed = self.update()  # Update road and compute speed
            avg_speeds.append(avg_speed)
        return avg_speeds  # Return recorded speeds over time

def plot_average_speed_vs_density(road_length, iterations, densities):
    """
    Plots the steady-state average speed of cars as a function of car density.

    Parameters:
    road_length (int): The length of the road.
    iterations (int): The number of time steps per simulation.
    densities (list or numpy array): A list of density values to test.
    """
    avg_speeds = []
    
    for density in densities:
        sim = TrafficSimulation(road_length, density, iterations)  # Create a new simulation
        speeds = sim.run_simulation()
        avg_speeds.append(np.mean(speeds[-10:]))  # Take last 10 time steps for steady-state speed
    
    plt.plot(densities, avg_speeds, marker='o')  # Plot density vs. speed
    plt.xlabel('Car Density')
    plt.ylabel('Steady-State Average Speed')
    plt.title('Traffic Flow Simulation')
    plt.grid()
    plt.show()

# User input and execution
if __name__ == "__main__":
    road_length = int(input("Enter road length: "))  # Get road length from user
    iterations = int(input("Enter number of iterations: "))  # Get number of iterations
    car_density = float(input("Enter car density (0 to 1): "))  # Get car density
    
    simulation = TrafficSimulation(road_length, car_density, iterations)  # Initialize simulation
    speeds = simulation.run_simulation()  # Run simulation
    
    # Print the final average speed, calculated from the last 10 iterations
    print(f"Final average speed: {np.mean(speeds[-10:]):.2f}")
    
    plot_choice = input("Plot speed vs density graph? (y/n): ").strip().lower()  # Ask user if they want a graph
    if plot_choice == 'y':
        densities = np.linspace(0, 1, 20)  # Generate 20 density values from 0 to 1
        plot_average_speed_vs_density(road_length, iterations, densities)  # Plot the graph
