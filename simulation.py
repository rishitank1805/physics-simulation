"""
Simple Physics Simulation
A basic physics engine demonstrating gravity, velocity, and collisions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Particle:
    """Represents a single particle in the simulation."""
    
    def __init__(self, x, y, vx=0, vy=0, radius=0.5, mass=1.0, color='blue'):
        """
        Initialize a particle.
        
        Parameters:
        - x, y: Initial position
        - vx, vy: Initial velocity
        - radius: Size of the particle
        - mass: Mass of the particle
        - color: Color for visualization
        """
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.mass = mass
        self.color = color
    
    def update(self, dt, gravity=-9.8, bounds=None):
        """
        Update particle position and velocity based on physics.
        
        Parameters:
        - dt: Time step
        - gravity: Gravitational acceleration (negative = downward)
        - bounds: Dictionary with 'x_min', 'x_max', 'y_min', 'y_max' for boundaries
        """
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
    
    def __init__(self, width=10, height=10, gravity=-9.8):
        """
        Initialize the simulation.
        
        Parameters:
        - width: Width of the simulation area
        - height: Height of the simulation area
        - gravity: Gravitational acceleration
        """
        self.width = width
        self.height = height
        self.gravity = gravity
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
            particle.update(self.dt, self.gravity, self.bounds)
    
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
    # Create simulation
    sim = PhysicsSimulation(width=10, height=10, gravity=-9.8)
    
    # Add some particles with different initial conditions
    # Particle 1: Dropped from top
    sim.add_particle(Particle(x=2, y=9, vx=0, vy=0, radius=0.5, color='blue'))
    
    # Particle 2: Thrown with initial velocity
    sim.add_particle(Particle(x=5, y=8, vx=3, vy=2, radius=0.5, color='red'))
    
    # Particle 3: Another particle
    sim.add_particle(Particle(x=8, y=9, vx=-2, vy=0, radius=0.5, color='green'))
    
    # Run the animation
    print("Starting physics simulation...")
    print("Close the window to stop.")
    sim.run_animation(duration=10, interval=10)


if __name__ == "__main__":
    main()
