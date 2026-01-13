"""
Fluid Dynamics Simulation
A simple grid-based fluid simulation using simplified Navier-Stokes equations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class FluidSimulation:
    """Grid-based fluid dynamics simulation."""
    
    def __init__(self, width=100, height=100, viscosity=0.01, diffusion=0.0, dt=0.1):
        """
        Initialize the fluid simulation.
        
        Parameters:
        - width: Grid width (number of cells)
        - height: Grid height (number of cells)
        - viscosity: Fluid viscosity (higher = more viscous/thicker)
        - diffusion: Density diffusion rate
        - dt: Time step
        """
        self.width = width
        self.height = height
        self.viscosity = viscosity
        self.diffusion = diffusion
        self.dt = dt
        
        # Velocity fields (u = x-velocity, v = y-velocity)
        self.u = np.zeros((height, width))
        self.v = np.zeros((height, width))
        self.u_prev = np.zeros((height, width))
        self.v_prev = np.zeros((height, width))
        
        # Density field (what we visualize)
        self.density = np.zeros((height, width))
        self.density_prev = np.zeros((height, width))
    
    def add_density(self, x, y, amount, radius=3):
        """
        Add density (like adding dye or smoke) at a specific location.
        
        Parameters:
        - x, y: Position in grid coordinates
        - amount: Amount of density to add
        - radius: Radius of the density source
        """
        # Clamp coordinates to grid bounds
        x = max(0, min(self.width - 1, int(x)))
        y = max(0, min(self.height - 1, int(y)))
        
        # Add density in a circular area
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist_sq = dx*dx + dy*dy
                if dist_sq <= radius * radius:
                    cy = max(0, min(self.height - 1, y + dy))
                    cx = max(0, min(self.width - 1, x + dx))
                    self.density[cy, cx] += amount * (1 - dist_sq / (radius * radius))
    
    def add_velocity(self, x, y, vx, vy, radius=3):
        """
        Add velocity (like a force or wind) at a specific location.
        
        Parameters:
        - x, y: Position in grid coordinates
        - vx, vy: Velocity to add
        - radius: Radius of the velocity source
        """
        # Clamp coordinates to grid bounds
        x = max(0, min(self.width - 1, int(x)))
        y = max(0, min(self.height - 1, int(y)))
        
        # Add velocity in a circular area
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist_sq = dx*dx + dy*dy
                if dist_sq <= radius * radius:
                    cy = max(0, min(self.height - 1, y + dy))
                    cx = max(0, min(self.width - 1, x + dx))
                    weight = 1 - dist_sq / (radius * radius)
                    self.u[cy, cx] += vx * weight
                    self.v[cy, cx] += vy * weight
    
    def set_bounds(self, b, field):
        """
        Set boundary conditions (no-slip at walls).
        
        Parameters:
        - b: Boundary type (1 for velocity, 0 for density)
        - field: The field to set boundaries for
        """
        # Top and bottom boundaries
        field[0, :] = -field[1, :] if b == 1 else field[1, :]
        field[-1, :] = -field[-2, :] if b == 1 else field[-2, :]
        
        # Left and right boundaries
        field[:, 0] = -field[:, 1] if b == 1 else field[:, 1]
        field[:, -1] = -field[:, -2] if b == 1 else field[:, -2]
        
        # Corner boundaries (average of neighbors)
        field[0, 0] = 0.5 * (field[1, 0] + field[0, 1])
        field[0, -1] = 0.5 * (field[1, -1] + field[0, -2])
        field[-1, 0] = 0.5 * (field[-2, 0] + field[-1, 1])
        field[-1, -1] = 0.5 * (field[-2, -1] + field[-1, -2])
    
    def linear_solve(self, b, x, x0, a, c, iterations=20):
        """
        Solve linear system using Gauss-Seidel iteration.
        Used for diffusion and pressure projection.
        """
        for _ in range(iterations):
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + 
                           a * (x[2:, 1:-1] + x[:-2, 1:-1] + 
                                x[1:-1, 2:] + x[1:-1, :-2])) / c
            self.set_bounds(b, x)
    
    def diffuse(self, b, x, x0, diff, dt):
        """
        Diffuse a field (spread it out over time).
        """
        a = dt * diff * (self.width - 2) * (self.height - 2)
        self.linear_solve(b, x, x0, a, 1 + 4 * a)
    
    def advect(self, b, d, d0, u, v, dt):
        """
        Advect a field (move it with the velocity).
        """
        dt0_x = dt * (self.width - 2)
        dt0_y = dt * (self.height - 2)
        
        for j in range(1, self.height - 1):
            for i in range(1, self.width - 1):
                # Trace back along velocity to find where density came from
                x = i - dt0_x * u[j, i]
                y = j - dt0_y * v[j, i]
                
                # Clamp to grid bounds
                x = max(0.5, min(self.width - 1.5, x))
                y = max(0.5, min(self.height - 1.5, y))
                
                # Bilinear interpolation
                i0 = int(x)
                j0 = int(y)
                i1 = i0 + 1
                j1 = j0 + 1
                
                s1 = x - i0
                t1 = y - j0
                s0 = 1 - s1
                t0 = 1 - t1
                
                d[j, i] = (s0 * (t0 * d0[j0, i0] + t1 * d0[j1, i0]) +
                           s1 * (t0 * d0[j0, i1] + t1 * d0[j1, i1]))
        
        self.set_bounds(b, d)
    
    def project(self, u, v, p, div):
        """
        Project velocity to be divergence-free (incompressible).
        """
        # Calculate divergence
        h = 0.5 / min(self.width, self.height)
        div[1:-1, 1:-1] = -0.5 * h * (
            u[1:-1, 2:] - u[1:-1, :-2] +
            v[2:, 1:-1] - v[:-2, 1:-1]
        )
        p.fill(0)
        self.set_bounds(0, div)
        self.set_bounds(0, p)
        
        # Solve for pressure
        self.linear_solve(0, p, div, 1, 4)
        
        # Subtract pressure gradient from velocity
        u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
        v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
        
        self.set_bounds(1, u)
        self.set_bounds(1, v)
    
    def update(self):
        """Update the fluid simulation for one time step."""
        # Swap current and previous fields
        self.u, self.u_prev = self.u_prev, self.u
        self.v, self.v_prev = self.v_prev, self.v
        self.density, self.density_prev = self.density_prev, self.density
        
        # Diffuse velocity (viscosity)
        self.diffuse(1, self.u, self.u_prev, self.viscosity, self.dt)
        self.diffuse(1, self.v, self.v_prev, self.viscosity, self.dt)
        
        # Project velocity (make it incompressible)
        p = np.zeros((self.height, self.width))
        div = np.zeros((self.height, self.width))
        self.project(self.u, self.v, p, div)
        
        # Advect velocity
        self.advect(1, self.u, self.u_prev, self.u_prev, self.v_prev, self.dt)
        self.advect(1, self.v, self.v_prev, self.u_prev, self.v_prev, self.dt)
        
        # Project again
        self.project(self.u, self.v, p, div)
        
        # Diffuse density (if diffusion > 0)
        if self.diffusion > 0:
            self.diffuse(0, self.density, self.density_prev, self.diffusion, self.dt)
        
        # Advect density
        self.advect(0, self.density, self.density_prev, self.u, self.v, self.dt)
        
        # Decay density over time (fade effect)
        self.density *= 0.995
    
    def run_animation(self, duration=30, interval=50, interactive=True):
        """
        Run the simulation with visualization.
        
        Parameters:
        - duration: How long to run the simulation (seconds)
        - interval: Animation frame interval (milliseconds)
        - interactive: If True, allows mouse interaction to add density/velocity
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Initialize with some density
        self.add_density(self.width // 2, self.height // 2, 100, radius=5)
        
        # Mouse interaction
        mouse_x, mouse_y = self.width // 2, self.height // 2
        mouse_pressed = False
        
        def on_mouse_move(event):
            nonlocal mouse_x, mouse_y
            if event.inaxes == ax:
                mouse_x = int(event.xdata * self.width / 10)
                mouse_y = int(event.ydata * self.height / 10)
        
        def on_mouse_press(event):
            nonlocal mouse_pressed
            if event.inaxes == ax:
                mouse_pressed = True
        
        def on_mouse_release(event):
            nonlocal mouse_pressed
            mouse_pressed = False
        
        if interactive:
            fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
            fig.canvas.mpl_connect('button_press_event', on_mouse_press)
            fig.canvas.mpl_connect('button_release_event', on_mouse_release)
        
        im = ax.imshow(self.density, cmap='hot', origin='lower', 
                      vmin=0, vmax=100, interpolation='bilinear')
        ax.set_title('Fluid Dynamics Simulation\n(Move mouse to add density, click to add velocity)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Density')
        
        frame_count = [0]
        
        def animate(frame):
            # Add density at mouse position
            if interactive and mouse_pressed:
                self.add_density(mouse_x, mouse_y, 50, radius=3)
                # Add upward velocity when clicking
                self.add_velocity(mouse_x, mouse_y, 0, -2, radius=5)
            
            # Add some continuous sources for demonstration
            if frame_count[0] % 10 == 0:
                # Add density at center
                self.add_density(self.width // 2, self.height // 2, 30, radius=2)
                # Add some velocity (convection)
                self.add_velocity(self.width // 2, self.height // 2, 
                                 np.sin(frame * 0.1) * 0.5, -1, radius=3)
            
            # Update simulation
            self.update()
            
            # Update visualization
            im.set_array(self.density)
            frame_count[0] += 1
            return [im]
        
        anim = FuncAnimation(fig, animate, interval=interval, 
                           frames=int(duration * 1000 / interval), 
                           blit=True, repeat=True)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run a fluid dynamics simulation."""
    print("Creating fluid dynamics simulation...")
    print("Move your mouse to add density, click to add velocity!")
    
    # Create simulation with low viscosity (water-like)
    sim = FluidSimulation(width=100, height=100, viscosity=0.01, 
                         diffusion=0.0, dt=0.1)
    
    # Run the animation
    sim.run_animation(duration=30, interval=50, interactive=True)


if __name__ == "__main__":
    main()
