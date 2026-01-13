"""
Simple Physics Simulation
A basic physics engine demonstrating gravity, velocity, friction, and collisions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Particle:
    """Represents a single particle in the simulation."""
    
    def __init__(self, x, y, vx=0, vy=0, radius=0.5, mass=1.0, color='blue', friction_coefficient=0.1):
        """
        Initialize a particle.
        
        Parameters:
        - x, y: Initial position
        - vx, vy: Initial velocity
        - radius: Size of the particle
        - mass: Mass of the particle
        - color: Color for visualization
        - friction_coefficient: Coefficient of friction (0 = no friction, higher = more friction)
        """
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.mass = mass
        self.color = color
        self.friction_coefficient = friction_coefficient
    
    def update(self, dt, gravity=-9.8, bounds=None, surface_friction=0.0):
        """
        Update particle position and velocity based on physics.
        
        Parameters:
        - dt: Time step
        - gravity: Gravitational acceleration (negative = downward)
        - bounds: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max' for boundaries
        - surface_friction: Additional friction from the surface (applied when on ground)
        """
        # Calculate total friction coefficient (particle friction + surface friction)
        total_friction = self.friction_coefficient + surface_friction
        
        # Apply friction (opposes motion, reduces velocity)
        # Friction force is proportional to velocity magnitude
        if total_friction > 0:
            speed = np.sqrt(self.vx**2 + self.vy**2)
            if speed > 0:
                # Friction acceleration opposes velocity direction
                friction_ax = -total_friction * (self.vx / speed) * speed
                friction_ay = -total_friction * (self.vy / speed) * speed
                
                # Apply friction to velocity
                self.vx += friction_ax * dt
                self.vy += friction_ay * dt
                
                # Stop very small velocities to prevent jitter
                if abs(self.vx) < 0.01:
                    self.vx = 0
                if abs(self.vy) < 0.01:
                    self.vy = 0
        
        # Apply gravity to vertical velocity
        self.vy += gravity * dt
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Handle collisions with boundaries
        if bounds:
            # Left and right walls
            if self.x - self.radius < bounds['x_min']:
                self.x = bounds['x_min'] + self.radius
                self.vx *= -0.8  # Bounce with some energy loss (damping)
            elif self.x + self.radius > bounds['x_max']:
                self.x = bounds['x_max'] - self.radius
                self.vx *= -0.8
            
            # Bottom and top walls
            if self.y - self.radius < bounds['y_min']:
                self.y = bounds['y_min'] + self.radius
                self.vy *= -0.8
            elif self.y + self.radius > bounds['y_max']:
                self.y = bounds['y_max'] - self.radius
                self.vy *= -0.8


class PhysicsSimulation:
    """Main physics simulation engine."""
    
    def __init__(self, width=10, height=10, gravity=-9.8, surface_friction=0.0):
        """
        Initialize the simulation.
        
        Parameters:
        - width: Width of the simulation area
        - height: Height of the simulation area
        - gravity: Gravitational acceleration
        - surface_friction: Coefficient of friction for the surface (0 = no friction)
        """
        self.width = width
        self.height = height
        self.gravity = gravity
        self.surface_friction = surface_friction
        self.particles = []
        self.bounds = {
            'x_min': 0,
            'x_max': width,
            'y_min': 0,
            'y_max': height
        }
        self.dt = 0.01  # Time step (smaller = more accurate but slower)
    
    def add_particle(self, particle):
        """Add a particle to the simulation."""
        self.particles.append(particle)
    
    def update(self):
        """Update all particles in the simulation."""
        for particle in self.particles:
            # Apply surface friction when particle is on or near the ground
            # (you can modify this logic for more complex friction behavior)
            surface_friction = self.surface_friction if particle.y - particle.radius <= self.bounds['y_min'] + 0.1 else 0.0
            particle.update(self.dt, self.gravity, self.bounds, surface_friction)
    
    def run_animation(self, duration=10, interval=10):
        """
        Run the simulation with visualization.
        
        Parameters:
        - duration: How long to run the simulation (seconds)
        - interval: Animation frame interval (milliseconds)
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('Physics Simulation')
        ax.grid(True, alpha=0.3)
        
        # Create circles for each particle
        circles = []
        for particle in self.particles:
            circle = plt.Circle((particle.x, particle.y), particle.radius, 
                              color=particle.color, fill=True)
            ax.add_patch(circle)
            circles.append(circle)
        
        def animate(frame):
            """Animation function called for each frame."""
            self.update()
            for i, particle in enumerate(self.particles):
                circles[i].center = (particle.x, particle.y)
            return circles
        
        # Create animation
        anim = FuncAnimation(fig, animate, interval=interval, blit=True, 
                           frames=int(duration * 1000 / interval))
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run a simple example simulation."""
    # Create simulation with surface friction
    sim = PhysicsSimulation(width=10, height=10, gravity=-9.8, surface_friction=0.5)
    
    # Add some particles with different initial conditions and friction coefficients
    # Particle 1: Dropped from top, low friction (slippery)
    sim.add_particle(Particle(x=2, y=9, vx=0, vy=0, radius=0.5, color='blue', friction_coefficient=0.1))
    
    # Particle 2: Thrown with initial velocity, medium friction
    sim.add_particle(Particle(x=5, y=8, vx=3, vy=2, radius=0.5, color='red', friction_coefficient=0.3))
    
    # Particle 3: Another particle, high friction (sticky)
    sim.add_particle(Particle(x=8, y=9, vx=-2, vy=0, radius=0.5, color='green', friction_coefficient=0.5))
    
    # Run the animation
    print("Starting physics simulation with friction...")
    print("Notice how particles slow down due to friction!")
    print("Close the window to stop.")
    sim.run_animation(duration=10, interval=10)


if __name__ == "__main__":
    main()
