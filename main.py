"""
Tidied up docstrings written
Computer Simulation 2025
Romi Kaushik s2578970
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import json
import csv

# Constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
AU = 1.496e11  # Astronomical unit in meters
DAY_TO_SEC = 24 * 60 * 60  # Conversion factor from days to seconds
MAX_DISTANCE_FROM_SUN = 100 * AU  # Max plot range for visualization

class CelestialBody:
    """
    Represents a celestial body (e.g., planet, sun) in the solar system.

    Attributes:
        name (str): Name of the celestial body.
        mass (float): Mass in kilograms.
        radius (float): Radius in meters.very
        orbital_radius (float): Distance from the central body in meters.
        orbital_period (float): Known orbital period in Earth years.
        patch_radius (float): Radius for rendering the body in visualizations.
        colour (str): Color used for plotting.
        r (np.array): 2D position vector.
        v (np.array): 2D velocity vector.
        force, acceleration, previous_acceleration: Vectors used for dynamics.
        orbit_period (float): Calculated orbital period from simulation.
        last_angle (float): Previous angle used for tracking period.
        path (list): Historical path of the body for plotting.
    """

    def __init__(self, body_name, body_data, central_mass):
        self.name = body_name
        self.mass = body_data["mass"]
        self.radius = body_data["radius"] * 1e3
        self.orbital_radius = body_data["orbital_radius"] * 1e3
        self.orbital_period = body_data["orbital_period"] / 365.25
        self.patch_radius = body_data["patch_radius"]
        self.colour = body_data["colour"]

        self.r = np.array([self.orbital_radius, 0], dtype=np.float64)
        self.v = np.array([0, math.sqrt(G * central_mass / self.orbital_radius) if self.orbital_radius > 0 else 0], dtype=np.float64)

        self.force = np.zeros(2, dtype=np.float64)
        self.acceleration = np.zeros(2, dtype=np.float64)
        self.previous_acceleration = np.zeros(2, dtype=np.float64)

        self.orbit_period = None
        self.last_angle = math.atan2(self.r[1], self.r[0])
        self.path = []

    def calculate_force(self, solar_system):
        """
        Computes the net gravitational force on this body from all others in the system.
        """
        self.force.fill(0)
        for body in solar_system:
            if body is not self:
                r_vec = body.r - self.r
                distance = np.linalg.norm(r_vec)
                if distance > 0:
                    force_mag = G * self.mass * body.mass / (distance ** 2)
                    self.force += force_mag * (r_vec / distance)

    def calculate_acceleration(self, solar_system):
        """
        Calculates the net acceleration based on current force.
        """
        self.calculate_force(solar_system)
        self.acceleration = self.force / self.mass

    def update_position_velocity(self, dt, method):
        """
        Updates the body's position and velocity using one of several numerical integration methods.

        Args:
            dt (float): Timestep in seconds.
            method (str): Integration method (BEEMAN, EULER-CROMER, DIRECT-EULER).
        """
        if method == "BEEMAN":
            self.r += self.v * dt + (1/6) * (4 * self.acceleration - self.previous_acceleration) * dt**2
            new_acceleration = self.acceleration
            self.v += (1/6) * (2 * new_acceleration + 5 * self.acceleration - self.previous_acceleration) * dt
            self.previous_acceleration = self.acceleration
        elif method == "EULER-CROMER":
            self.v += self.acceleration * dt
            self.r += self.v * dt
        elif method == "DIRECT-EULER":
            self.r += self.v * dt
            self.v += self.acceleration * dt

        self.path.append(self.r.copy())

    def calculate_orbital_period(self, current_time):
        """
        Roughly calculates orbital period based on angle crossing.
        """
        angle = math.atan2(self.r[1], self.r[0])
        if self.last_angle < 0 and angle > 0:
            if self.orbit_period is None:
                self.orbit_period = current_time / (365.25 * DAY_TO_SEC)
        self.last_angle = angle

    def calculate_period(self, dt):
        """
        Refines the orbital period by scanning the full path history.

        Args:
            dt (float): Timestep in days.
        Returns:
            float: Estimated orbital period in years.
        """
        self.orbit_period = None
        if self.orbital_radius != 0:
            for i in range(len(self.path)):
                angle = math.degrees(math.atan2(self.path[i][1], self.path[i][0]))
                if -5 < angle < 0:
                    self.orbit_period = i * (dt / 365.25)
                    break
                if i == len(self.path) - 1 and self.orbit_period is None:
                    if angle < 0:
                        angle += 360
                    self.orbit_period = i * (dt / 365.25) / (angle / 360)
                    break
        return self.orbit_period

class Satellite(CelestialBody):
    """
    Represents a user-defined satellite launched from Earth.

    Additional Attributes:
        closest_approach_to_mars (float): Minimum distance to Mars in simulation.
        time_to_mars (float): Time taken to reach closest approach in days.
    """

    def __init__(self, initial_position, initial_velocity, central_mass):
        self.name = "Satellite"
        self.mass = 500
        self.colour = "black"
        self.r = np.array(initial_position, dtype=np.float64)
        self.v = np.array(initial_velocity, dtype=np.float64)
        self.force = np.zeros(2, dtype=np.float64)
        self.acceleration = np.zeros(2, dtype=np.float64)
        self.previous_acceleration = np.zeros(2, dtype=np.float64)
        self.closest_approach_to_mars = float('inf')
        self.time_to_mars = None
        self.path = []

    def update_closest_approach(self, mars, current_time):
        """
        Tracks the minimum distance to Mars and the time it occurs.

        Args:
            mars (CelestialBody): The Mars object.
            current_time (float): Current time in seconds.
        """
        distance_to_mars = np.linalg.norm(self.r - mars.r)
        if distance_to_mars < self.closest_approach_to_mars:
            self.closest_approach_to_mars = distance_to_mars
            self.time_to_mars = current_time / DAY_TO_SEC

class SolarSystem:
    """
    Simulates a solar system with planets and an optional satellite.

    Attributes:
        bodies (list): List of CelestialBody objects.
        satellite (Satellite): Satellite object if present.
    """

    def __init__(self, body_data):
        central_body = max(body_data, key=lambda k: body_data[k]["mass"])
        central_mass = body_data[central_body]["mass"]
        self.bodies = [CelestialBody(name, body_data[name], central_mass) for name in body_data]
        self.satellite = None

    def add_satellite(self, initial_position, initial_velocity):
        """
        Adds a user-defined satellite to the solar system.

        Args:
            initial_position (np.array): Initial position vector.
            initial_velocity (np.array): Initial velocity vector.
        """
        central_mass = max(self.bodies, key=lambda b: b.mass).mass
        self.satellite = Satellite(initial_position, initial_velocity, central_mass)
        self.bodies.append(self.satellite)

    def simulate(self, T, dt, method, animate=False):
        """
        Runs the simulation for the specified duration and integration method.

        Args:
            T (float): Total time in days.
            dt (float): Timestep in days.
            method (str): Integration method name.
            animate (bool): Whether to show real-time animation.

        Returns:
            tuple: List of times and corresponding total system energies.
        """
        steps = int(T / dt)
        dt_sec = dt * DAY_TO_SEC

        mars = next((b for b in self.bodies if b.name.lower() == "mars"), None)
        if mars is None:
            raise ValueError("Mars not found in the simulation data. Check the input file.")

        times = []
        energies = []

        for step in range(steps):
            total_energy = 0
            for body in self.bodies:
                body.calculate_acceleration(self.bodies)
                body.update_position_velocity(dt_sec, method)
                if body.name.lower() != "satellite":
                    body.calculate_orbital_period(T)
                if self.satellite and mars:
                    self.satellite.update_closest_approach(mars, step * dt_sec)

                ke = 0.5 * body.mass * np.linalg.norm(body.v) ** 2
                pe = sum([-G * body.mass * other.mass / np.linalg.norm(other.r - body.r)
                          for other in self.bodies if other is not body])
                total_energy += ke + pe

            times.append(step * dt_sec)
            energies.append(total_energy)

        if self.satellite:
            print(f"Closest approach to Mars: {self.satellite.closest_approach_to_mars / 1e6:.2f} million km")
            print(f"Time to Mars: {self.satellite.time_to_mars:.2f} days")

        with open(f"{method}.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(zip(times, energies))

        return times, energies

    def show_orbital_periods(self, dt):
        """
        Displays actual vs simulated orbital periods and relative error.
        """
        for body in self.bodies:
            if body.name in ["Sun", "Satellite"]:
                continue
            body.calculate_period(dt)
            error = 100 * abs(body.orbit_period - body.orbital_period) / body.orbital_period
            print(f"The orbital period of {body.name}, actual = {body.orbital_period:.3} years, "
                  f"measured in the simulation = {body.orbit_period:.3} years, %error = {error:.3}%")

    def animate_3d(self):
        """
        Displays a 3D orbital animation using matplotlib.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        lines = []
        for body in self.bodies:
            path = np.array(body.path)
            z = np.zeros(len(path))
            line, = ax.plot(path[:, 0], path[:, 1], z, label=body.name, color=body.colour)
            lines.append(line)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        plt.title("3D Orbital Animation")
        plt.show()

    def plot_graphs(self, method_list, times, energies_list):
        """
        Plots total energy and orbital paths for visualization.

        Args:
            method_list (list): List of integration method names.
            times (list): Time steps.
            energies_list (list): Corresponding energy data for each method.
        """
        plt.figure(figsize=(12, 6))
        for method, energies in zip(method_list, energies_list):
            plt.plot(times, energies, label=f"Energy ({method})")
        plt.xlabel("Time (s)")
        plt.ylabel("Total Energy (J)")
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 8))
        for body in self.bodies:
            path = np.array(body.path)
            plt.plot(path[:, 0], path[:, 1], label=body.name)

        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Orbital Paths")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()

def run_satellite_experiment(body_data):
    """
    Run an experiment to simulate the closest approach of a satellite to Mars.
    
    The function varies the satellite's initial velocity and angle, simulates 
    the motion for each combination, and records the closest approach to Mars. 
    The best combination of velocity and angle is reported.

    Parameters:
    - body_data (dict): A dictionary containing data about the celestial bodies in the system.

    Returns:
    - None: Outputs the best result to the console.
    """
    print("\n--- Running Satellite to Mars Experiment ---\n")
    T = 2 * 365.25  # simulate 2 years
    dt = 1  # timestep in days
    initial_position = np.array([1 * AU + 6.37e6, 0])  # initial position slightly outside Earth's orbit

    # Best result placeholder
    best_result = {
        "velocity": None,
        "angle": None,
        "closest_approach": float('inf'),
        "time_to_mars": None
    }

    # Loop over velocity and angle combinations
    for speed in range(28000, 35000, 1000):  # speeds in m/s
        for angle_deg in range(-10, 11, 5):  # angles from -10° to 10°
            angle_rad = np.radians(angle_deg)  # convert angle to radians
            vx = 0 + speed * np.cos(angle_rad)  # calculate x velocity component
            vy = speed * np.sin(angle_rad)  # calculate y velocity component
            initial_velocity = np.array([vx, vy])

            # Create solar system and add satellite
            solar_system = SolarSystem(body_data)
            solar_system.add_satellite(initial_position, initial_velocity)
            try:
                # Simulate the system
                solar_system.simulate(T, dt, "BEEMAN")
                satellite = solar_system.satellite

                # Output closest approach for current velocity and angle
                print(f"Speed: {speed} m/s, Angle: {angle_deg}° --> Closest to Mars: {satellite.closest_approach_to_mars / 1e6:.2f} million km in {satellite.time_to_mars:.1f} days")

                # Update best result if current is better
                if satellite.closest_approach_to_mars < best_result["closest_approach"]:
                    best_result = {
                        "velocity": speed,
                        "angle": angle_deg,
                        "closest_approach": satellite.closest_approach_to_mars,
                        "time_to_mars": satellite.time_to_mars
                    }

            except Exception as e:
                print(f"Simulation failed for speed {speed}, angle {angle_deg}: {e}")

    # Output the best result from the experiment
    print("\n--- Best Result ---")
    print(f"Velocity: {best_result['velocity']} m/s")
    print(f"Angle: {best_result['angle']}°")
    print(f"Closest approach to Mars: {best_result['closest_approach'] / 1e6:.2f} million km")
    print(f"Time to reach Mars: {best_result['time_to_mars']:.2f} days (~{best_result['time_to_mars']/30:.2f} months)")
    print("Compare to NASA Perseverance: ~203 days")

def main():
    """
    Main function to simulate the solar system and experiment with satellite's path to Mars.

    The user is prompted to enter the total simulation time and timestep. Then, 
    the system is simulated with different numerical methods (Euler-Cromer, Direct-Euler, Beeman). 
    The results are visualized and orbital periods are shown. Finally, an experiment to find 
    the best satellite trajectory to Mars is conducted.

    Parameters:
    - None

    Returns:
    - None
    """
    # Load celestial body data from the input JSON file
    with open('input.json', 'r') as file:
        body_data = json.load(file)

    # Create the solar system with loaded data
    solar_system = SolarSystem(body_data)
    
    # Get user input for simulation time and timestep
    T = float(input("Enter total simulation time (years): ")) * 365.25  # convert years to days
    dt = float(input("Enter timestep (days): "))  # timestep in days

    # Satellite's initial position and velocity
    initial_position = np.array([1 * AU + 6.37e6, 0])  # Earth's orbit + satellite altitude
    initial_velocity = np.array([0, 30000])  # Initial velocity (30000 m/s along y-axis)

    # Add satellite to the system
    solar_system.add_satellite(initial_position, initial_velocity)
    
    # Run simulations with different methods
    ec_times, ec_energies = solar_system.simulate(T, dt, "EULER-CROMER")
    de_times, de_energies = solar_system.simulate(T, dt, "DIRECT-EULER")
    b_times, b_energies = solar_system.simulate(T, dt, "BEEMAN", animate=True)

    # Show orbital periods of bodies
    solar_system.show_orbital_periods(dt)

    # 3D animation of the orbital paths
    solar_system.animate_3d()

    # Plot energy and orbital paths
    solar_system.plot_graphs(["BEEMAN", "EULER-CROMER", "DIRECT-EULER"], b_times, [b_energies, ec_energies, de_energies])

    # Run experiment to determine best satellite path to Mars
    run_satellite_experiment(body_data)

# Run the main program when the script is executed
if __name__ == "__main__":
    main()
