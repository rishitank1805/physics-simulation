"""
Advanced Fluid Dynamics Simulation
A sophisticated 2D fluid solver using Navier-Stokes equations with pressure projection.
Features realistic fluid flow, viscosity, and advanced visualization.
"""

import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle
import time

# Set appearance mode
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AdvancedFluidSolver:
    """Advanced 2D fluid solver using Navier-Stokes equations."""
    
    def __init__(self, width=100, height=100, viscosity=0.01, dt=0.1):
        """
        Initialize the fluid solver.
        
        Parameters:
        - width, height: Grid dimensions
        - viscosity: Fluid viscosity (higher = more viscous)
        - dt: Time step
        """
        self.width = width
        self.height = height
        self.viscosity = viscosity
        self.dt = dt
        
        # Velocity fields (staggered grid)
        self.u = np.zeros((height, width + 1))  # x-velocity
        self.v = np.zeros((height + 1, width))  # y-velocity
        self.u_prev = np.zeros_like(self.u)
        self.v_prev = np.zeros_like(self.v)
        
        # Pressure field
        self.pressure = np.zeros((height, width))
        
        # Density field (for visualization)
        self.density = np.zeros((height, width))
        self.density_prev = np.zeros((height, width))
        
        # Source locations for adding fluid
        self.sources = []
    
    def add_velocity_source(self, x, y, vx, vy, radius=5):
        """Add velocity at a specific location."""
        y_idx = int(y)
        x_idx = int(x)
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist_sq = dx*dx + dy*dy
                if dist_sq <= radius * radius:
                    cy = max(0, min(self.height - 1, y_idx + dy))
                    cx = max(0, min(self.width - 1, x_idx + dx))
                    weight = 1 - dist_sq / (radius * radius)
                    
                    if 0 <= cx < self.width:
                        self.u[cy, cx] += vx * weight
                        self.u[cy, cx + 1] += vx * weight
                    if 0 <= cy < self.height:
                        self.v[cy, cx] += vy * weight
                        self.v[cy + 1, cx] += vy * weight
    
    def add_density_source(self, x, y, amount, radius=5):
        """Add density (fluid) at a specific location."""
        y_idx = int(y)
        x_idx = int(x)
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist_sq = dx*dx + dy*dy
                if dist_sq <= radius * radius:
                    cy = max(0, min(self.height - 1, y_idx + dy))
                    cx = max(0, min(self.width - 1, x_idx + dx))
                    weight = 1 - dist_sq / (radius * radius)
                    self.density[cy, cx] += amount * weight
    
    def set_boundary_conditions(self, b, field, is_velocity=False):
        """Apply boundary conditions (no-slip for velocity, zero for density)."""
        if is_velocity:
            # No-slip boundary conditions
            field[0, :] = -field[1, :]
            field[-1, :] = -field[-2, :]
            field[:, 0] = -field[:, 1]
            field[:, -1] = -field[:, -2]
        else:
            # Zero boundary for density
            field[0, :] = field[1, :]
            field[-1, :] = field[-2, :]
            field[:, 0] = field[:, 1]
            field[:, -1] = field[:, -2]
        
        # Corner conditions
        field[0, 0] = 0.5 * (field[1, 0] + field[0, 1])
        field[0, -1] = 0.5 * (field[1, -1] + field[0, -2])
        field[-1, 0] = 0.5 * (field[-2, 0] + field[-1, 1])
        field[-1, -1] = 0.5 * (field[-2, -1] + field[-1, -2])
    
    def linear_solve(self, b, x, x0, a, c, iterations=20, is_velocity=False):
        """Solve linear system using Gauss-Seidel iteration."""
        for _ in range(iterations):
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + 
                           a * (x[2:, 1:-1] + x[:-2, 1:-1] + 
                                x[1:-1, 2:] + x[1:-1, :-2])) / c
            self.set_boundary_conditions(b, x, is_velocity)
    
    def diffuse(self, b, x, x0, diff, dt, is_velocity=False):
        """Diffuse a field (viscosity effect)."""
        a = dt * diff * (self.width - 2) * (self.height - 2)
        self.linear_solve(b, x, x0, a, 1 + 4 * a, is_velocity=is_velocity)
    
    def advect(self, b, d, d0, u, v, dt, is_velocity=False):
        """Advect a field along the velocity."""
        dt0_x = dt * (self.width - 2)
        dt0_y = dt * (self.height - 2)
        
        for j in range(1, self.height - 1):
            for i in range(1, self.width - 1):
                # Trace back along velocity
                x = i - dt0_x * u[j, i]
                y = j - dt0_y * v[j, i]
                
                # Clamp to grid
                x = max(0.5, min(self.width - 1.5, x))
                y = max(0.5, min(self.height - 1.5, y))
                
                # Bilinear interpolation
                i0, j0 = int(x), int(y)
                i1, j1 = i0 + 1, j0 + 1
                
                s1, t1 = x - i0, y - j0
                s0, t0 = 1 - s1, 1 - t1
                
                d[j, i] = (s0 * (t0 * d0[j0, i0] + t1 * d0[j1, i0]) +
                          s1 * (t0 * d0[j0, i1] + t1 * d0[j1, i1]))
        
        self.set_boundary_conditions(b, d, is_velocity)
    
    def project(self, u, v, p, div):
        """Project velocity to be divergence-free (incompressible)."""
        # Calculate divergence
        h = 0.5 / min(self.width, self.height)
        div[1:-1, 1:-1] = -0.5 * h * (
            u[1:-1, 2:] - u[1:-1, :-2] +
            v[2:, 1:-1] - v[:-2, 1:-1]
        )
        
        p.fill(0)
        self.set_boundary_conditions(0, div, False)
        self.set_boundary_conditions(0, p, False)
        
        # Solve for pressure
        self.linear_solve(0, p, div, 1, 4, is_velocity=False)
        
        # Subtract pressure gradient
        u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
        v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
        
        self.set_boundary_conditions(1, u, True)
        self.set_boundary_conditions(1, v, True)
    
    def step(self):
        """Perform one simulation step."""
        # Swap fields
        self.u, self.u_prev = self.u_prev, self.u
        self.v, self.v_prev = self.v_prev, self.v
        self.density, self.density_prev = self.density_prev, self.density
        
        # Diffuse velocity (viscosity)
        self.diffuse(1, self.u, self.u_prev, self.viscosity, self.dt, is_velocity=True)
        self.diffuse(1, self.v, self.v_prev, self.viscosity, self.dt, is_velocity=True)
        
        # Project velocity
        p = np.zeros((self.height, self.width))
        div = np.zeros((self.height, self.width))
        self.project(self.u, self.v, p, div)
        
        # Advect velocity
        self.advect(1, self.u, self.u_prev, self.u_prev, self.v_prev, self.dt, is_velocity=True)
        self.advect(1, self.v, self.v_prev, self.u_prev, self.v_prev, self.dt, is_velocity=True)
        
        # Project again
        self.project(self.u, self.v, p, div)
        
        # Advect density
        self.advect(0, self.density, self.density_prev, self.u, self.v, self.dt, is_velocity=False)
        
        # Decay density
        self.density *= 0.995
    
    def get_velocity_magnitude(self):
        """Get velocity magnitude field for visualization."""
        u_center = 0.5 * (self.u[:, :-1] + self.u[:, 1:])
        v_center = 0.5 * (self.v[:-1, :] + self.v[1:, :])
        return np.sqrt(u_center**2 + v_center**2)


class AdvancedFluidApp(ctk.CTk):
    """Advanced fluid simulation application."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Advanced Fluid Dynamics Simulation")
        self.geometry("1600x950")
        self.minsize(1400, 800)
        
        # Create fluid solver
        self.solver = AdvancedFluidSolver(width=120, height=120, viscosity=0.01, dt=0.1)
        
        # Simulation state
        self.is_running = True
        self.mouse_x = self.solver.width // 2
        self.mouse_y = self.solver.height // 2
        self.mouse_pressed = False
        self.last_update = time.time()
        
        # Setup UI
        self.create_widgets()
        
        # Start simulation
        self.update_simulation()
    
    def create_widgets(self):
        """Create UI widgets."""
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left side - Visualization
        left_frame = ctk.CTkFrame(main_container)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        title = ctk.CTkLabel(
            left_frame,
            text="🌊 Advanced Fluid Dynamics",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(15, 10))
        
        # Matplotlib figure
        self.fig = Figure(figsize=(11, 10), facecolor='#1a1a1a')
        self.ax = self.fig.add_subplot(111, facecolor='#1a1a1a')
        self.ax.set_xlim(0, self.solver.width)
        self.ax.set_ylim(0, self.solver.height)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Create custom colormap for fluid
        colors = ['#000000', '#001122', '#003366', '#0066AA', '#00AAFF', '#66DDFF', '#FFFFFF']
        n_bins = 256
        self.fluid_cmap = LinearSegmentedColormap.from_list('fluid', colors, N=n_bins)
        
        # Initial visualization
        self.im = self.ax.imshow(
            self.solver.density,
            cmap=self.fluid_cmap,
            origin='lower',
            interpolation='bilinear',
            vmin=0,
            vmax=100,
            extent=[0, self.solver.width, 0, self.solver.height]
        )
        
        # Velocity vectors (quiver plot)
        x = np.arange(0, self.solver.width, 5)
        y = np.arange(0, self.solver.height, 5)
        X, Y = np.meshgrid(x, y)
        self.quiver = None
        
        self.fig.tight_layout()
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, left_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Mouse interaction
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
        # Right side - Controls
        right_frame = ctk.CTkFrame(main_container, width=450)
        right_frame.pack(side="right", fill="y", padx=(10, 0))
        right_frame.pack_propagate(False)
        
        controls_title = ctk.CTkLabel(
            right_frame,
            text="⚙️ Controls",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        controls_title.pack(pady=(20, 25))
        
        # Viscosity control
        visc_frame = ctk.CTkFrame(right_frame)
        visc_frame.pack(fill="x", padx=20, pady=10)
        
        visc_label = ctk.CTkLabel(
            visc_frame,
            text="🧪 Viscosity",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        visc_label.pack(pady=(15, 5))
        
        self.visc_value_label = ctk.CTkLabel(
            visc_frame,
            text=f"{self.solver.viscosity:.4f}",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#4A9EFF"
        )
        self.visc_value_label.pack()
        
        self.visc_slider = ctk.CTkSlider(
            visc_frame,
            from_=0.0,
            to=0.1,
            number_of_steps=1000,
            command=self.on_viscosity_changed
        )
        self.visc_slider.set(self.solver.viscosity)
        self.visc_slider.pack(fill="x", padx=20, pady=(10, 15))
        
        # Visualization mode
        mode_frame = ctk.CTkFrame(right_frame)
        mode_frame.pack(fill="x", padx=20, pady=10)
        
        mode_label = ctk.CTkLabel(
            mode_frame,
            text="🎨 Visualization Mode",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        mode_label.pack(pady=(15, 10))
        
        self.viz_mode = ctk.StringVar(value="density")
        density_radio = ctk.CTkRadioButton(
            mode_frame,
            text="Density (Fluid)",
            variable=self.viz_mode,
            value="density",
            command=self.on_viz_mode_changed
        )
        density_radio.pack(pady=5)
        
        velocity_radio = ctk.CTkRadioButton(
            mode_frame,
            text="Velocity Magnitude",
            variable=self.viz_mode,
            value="velocity",
            command=self.on_viz_mode_changed
        )
        velocity_radio.pack(pady=5)
        
        combined_radio = ctk.CTkRadioButton(
            mode_frame,
            text="Combined",
            variable=self.viz_mode,
            value="combined",
            command=self.on_viz_mode_changed
        )
        combined_radio.pack(pady=5)
        
        # Show vectors checkbox
        self.show_vectors = ctk.BooleanVar(value=False)
        vectors_check = ctk.CTkCheckBox(
            mode_frame,
            text="Show Velocity Vectors",
            variable=self.show_vectors,
            command=self.on_viz_mode_changed
        )
        vectors_check.pack(pady=(10, 15))
        
        # Control buttons
        buttons_frame = ctk.CTkFrame(right_frame)
        buttons_frame.pack(fill="x", padx=20, pady=20)
        
        button_container = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        button_container.pack(pady=10)
        
        self.clear_button = ctk.CTkButton(
            button_container,
            text="🗑️ Clear",
            command=self.clear_simulation,
            font=ctk.CTkFont(size=14, weight="bold"),
            width=140,
            height=40,
            fg_color="#E74C3C",
            hover_color="#C0392B"
        )
        self.clear_button.pack(side="left", padx=5)
        
        self.pause_button = ctk.CTkButton(
            button_container,
            text="⏸️ Pause",
            command=self.toggle_pause,
            font=ctk.CTkFont(size=14, weight="bold"),
            width=140,
            height=40,
            fg_color="#3498DB",
            hover_color="#2980B9"
        )
        self.pause_button.pack(side="left", padx=5)
        
        # Info label
        info_label = ctk.CTkLabel(
            right_frame,
            text="💡 Move mouse to add fluid\nClick and drag to add velocity",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            justify="center"
        )
        info_label.pack(pady=(10, 20))
    
    def on_viscosity_changed(self, value):
        """Handle viscosity slider change."""
        self.solver.viscosity = value
        self.visc_value_label.configure(text=f"{value:.4f}")
    
    def on_viz_mode_changed(self):
        """Handle visualization mode change."""
        pass  # Will be handled in update_visualization
    
    def on_mouse_move(self, event):
        """Handle mouse movement."""
        if event.inaxes == self.ax:
            self.mouse_x = int(event.xdata)
            self.mouse_y = int(event.ydata)
    
    def on_mouse_press(self, event):
        """Handle mouse press."""
        if event.inaxes == self.ax:
            self.mouse_pressed = True
    
    def on_mouse_release(self, event):
        """Handle mouse release."""
        self.mouse_pressed = False
    
    def clear_simulation(self):
        """Clear the simulation."""
        self.solver.u.fill(0)
        self.solver.v.fill(0)
        self.solver.density.fill(0)
        self.solver.pressure.fill(0)
    
    def toggle_pause(self):
        """Toggle simulation pause/play."""
        self.is_running = not self.is_running
        if self.is_running:
            self.pause_button.configure(text="⏸️ Pause")
        else:
            self.pause_button.configure(text="▶️ Resume")
    
    def update_simulation(self):
        """Update simulation and visualization."""
        if self.is_running:
            # Add density at mouse position
            if self.mouse_pressed:
                self.solver.add_density(self.mouse_x, self.mouse_y, 50, radius=8)
                # Add upward velocity when clicking
                self.solver.add_velocity_source(
                    self.mouse_x, self.mouse_y, 0, -3, radius=8
                )
            else:
                # Continuous source
                self.solver.add_density(self.mouse_x, self.mouse_y, 20, radius=5)
            
            # Step simulation
            self.solver.step()
        
        # Update visualization
        self.update_visualization()
        
        # Schedule next update
        self.after(16, self.update_simulation)  # ~60 FPS
    
    def update_visualization(self):
        """Update the visualization."""
        try:
            mode = self.viz_mode.get()
            
            if mode == "density":
                data = self.solver.density
                vmax = 100
            elif mode == "velocity":
                data = self.solver.get_velocity_magnitude()
                vmax = 5
            else:  # combined
                data = self.solver.density + 0.3 * self.solver.get_velocity_magnitude() * 20
                vmax = 100
            
            self.im.set_array(data)
            self.im.set_clim(0, vmax)
            
            # Update velocity vectors if enabled
            if self.show_vectors.get():
                # Sample velocity field
                step = 5
                x = np.arange(0, self.solver.width, step)
                y = np.arange(0, self.solver.height, step)
                X, Y = np.meshgrid(x, y)
                
                # Get velocities at sample points
                U = np.zeros_like(X)
                V = np.zeros_like(Y)
                for i, xi in enumerate(x):
                    for j, yj in enumerate(y):
                        if 0 <= int(xi) < self.solver.width and 0 <= int(yj) < self.solver.height:
                            U[j, i] = 0.5 * (self.solver.u[int(yj), int(xi)] + 
                                            self.solver.u[int(yj), int(xi)+1])
                            V[j, i] = 0.5 * (self.solver.v[int(yj), int(xi)] + 
                                            self.solver.v[int(yj)+1, int(xi)])
                
                # Remove old quiver
                if self.quiver is not None:
                    self.quiver.remove()
                
                # Create new quiver
                self.quiver = self.ax.quiver(
                    X, Y, U, V,
                    scale=20,
                    color='white',
                    alpha=0.6,
                    width=0.003
                )
            else:
                if self.quiver is not None:
                    self.quiver.remove()
                    self.quiver = None
            
            self.canvas.draw()
        except Exception as e:
            print(f"Visualization error: {e}")


def main():
    """Main function."""
    try:
        app = AdvancedFluidApp()
        app.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
