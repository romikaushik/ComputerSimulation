import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Gravitational constant
G = 6.67430e-11  # m^3 kg^-1 s^-2

class CelestialBody:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

class OrbitalSimulation:
    def __init__(self, mars, phobos, time_step, num_steps):
        self.mars = mars
        self.phobos = phobos
        self.dt = time_step
        self.num_steps = num_steps

        # Preallocate position arrays
        self.positions_mars = np.zeros((num_steps, 2))
        self.positions_phobos = np.zeros((num_steps, 2))
        self.kinetic_energies = np.zeros(num_steps)

    def compute_gravitational_force(self):
        r = self.phobos.position - self.mars.position
        distance = np.linalg.norm(r)
        
        if distance == 0:
            return np.array([0, 0])  # Avoid division by zero
        
        force_magnitude = G * self.mars.mass * self.phobos.mass / distance**2
        force = -force_magnitude * (r / distance)  # Force direction correction
        return force

    def update_positions(self):
        for i in range(self.num_steps):
            force = self.compute_gravitational_force()
            
            acceleration_phobos = force / self.phobos.mass
            acceleration_mars = -force / self.mars.mass
            
            self.phobos.velocity += acceleration_phobos * self.dt
            self.mars.velocity += acceleration_mars * self.dt
            
            self.phobos.position += self.phobos.velocity * self.dt
            self.mars.position += self.mars.velocity * self.dt
            
            # Store positions
            self.positions_mars[i] = self.mars.position
            self.positions_phobos[i] = self.phobos.position
            
            # Compute kinetic energy
            kinetic_energy = 0.5 * self.mars.mass * np.linalg.norm(self.mars.velocity)**2 + \
                             0.5 * self.phobos.mass * np.linalg.norm(self.phobos.velocity)**2
            self.kinetic_energies[i] = kinetic_energy
        
        # Debugging: Print first few positions to verify motion
        print("First 10 Mars positions:\n", self.positions_mars[:10])
        print("First 10 Phobos positions:\n", self.positions_phobos[:10])

    def animate_orbit(self):
        fig, ax = plt.subplots()
        
        # Determine dynamic plot limits based on Phobos' motion
        lim = np.max(np.abs(self.positions_phobos)) * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')

        mars_dot, = ax.plot([], [], 'ro', markersize=10, label='Mars')
        phobos_dot, = ax.plot([], [], 'bo', markersize=5, label='Phobos')
        ax.legend()

        def init():
            mars_dot.set_data([], [])
            phobos_dot.set_data([], [])
            return mars_dot, phobos_dot

        def update(frame):
            mars_dot.set_data(self.positions_mars[frame, 0], self.positions_mars[frame, 1])
            phobos_dot.set_data(self.positions_phobos[frame, 0], self.positions_phobos[frame, 1])
            return mars_dot, phobos_dot

        ani = animation.FuncAnimation(fig, update, frames=self.num_steps, init_func=init, interval=20, blit=True)
        plt.show()

    def plot_orbit_static(self):
        """ Debugging: Plot the entire trajectory statically before animating """
        plt.figure(figsize=(6, 6))
        plt.plot(self.positions_phobos[:, 0], self.positions_phobos[:, 1], label="Phobos Orbit")
        plt.scatter(self.positions_mars[:, 0], self.positions_mars[:, 1], color='red', label="Mars", s=100)
        plt.legend()
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Phobos Orbit around Mars")
        plt.axis("equal")
        plt.show()

# Initialize Mars and Phobos
mars = CelestialBody(6.417e23, [0, 0], [0, 0])  # Mars at origin
phobos = CelestialBody(1.0659e16, [9.376e6, 0], [0, 2.14e3])  # Phobos at orbit distance

# Run simulation with smaller time step for accuracy
sim = OrbitalSimulation(mars, phobos, time_step=1, num_steps=10000)
sim.update_positions()

# Debugging: First test a static orbit plot
sim.plot_orbit_static()

plt.close()  # Close the static plot to prevent an empty graph

# Run animation (should work now)
sim.animate_orbit()
