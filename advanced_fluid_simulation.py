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
import time

# Set appearance mode
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AdvancedFluidSolver:
    """Advanced 2D fluid solver using Navier-Stokes equations."""
    
    def __init__(self, width=100, height=100, viscosity=0.01, dt=0.1):
        """Initialize the fluid solver."""
        # Validate and set dimensions
        self.width = max(3, int(width))  # Minimum 3 for boundary conditions
        self.height = max(3, int(height))
        
        # Validate viscosity
        self.viscosity = max(0.0, float(viscosity))
        
        # Validate time step
        if dt <= 0 or not np.isfinite(dt):
            dt = 0.1
        self.dt = float(dt)
        
        # Velocity fields (centered grid for simplicity)
        self.u = np.zeros((height, width))  # x-velocity
        self.v = np.zeros((height, width))  # y-velocity
        self.u_prev = np.zeros_like(self.u)
        self.v_prev = np.zeros_like(self.v)
        
        # Pressure field
        self.pressure = np.zeros((height, width))
        
        # Density field (for visualization)
        self.density = np.zeros((height, width))
        self.density_prev = np.zeros((height, width))
    
    def add_velocity_source(self, x, y, vx, vy, radius=5):
        """Add velocity at a specific location."""
        # Validate inputs
        if not np.isfinite(x) or not np.isfinite(y):
            return
        if not np.isfinite(vx) or not np.isfinite(vy):
            return
        if radius <= 0:
            return
        
        y_idx = int(np.clip(y, 0, self.height - 1))
        x_idx = int(np.clip(x, 0, self.width - 1))
        
        radius_sq = radius * radius
        if radius_sq < 1e-10:
            return
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist_sq = dx*dx + dy*dy
                if dist_sq <= radius_sq:
                    cy = int(np.clip(y_idx + dy, 0, self.height - 1))
                    cx = int(np.clip(x_idx + dx, 0, self.width - 1))
                    weight = max(0, 1 - dist_sq / radius_sq)
                    self.u[cy, cx] += vx * weight
                    self.v[cy, cx] += vy * weight
    
    def add_density_source(self, x, y, amount, radius=5):
        """Add density (fluid) at a specific location."""
        # Validate inputs
        if not np.isfinite(x) or not np.isfinite(y):
            return
        if not np.isfinite(amount):
            return
        if radius <= 0:
            return
        
        y_idx = int(np.clip(y, 0, self.height - 1))
        x_idx = int(np.clip(x, 0, self.width - 1))
        
        radius_sq = radius * radius
        if radius_sq < 1e-10:
            return
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist_sq = dx*dx + dy*dy
                if dist_sq <= radius_sq:
                    cy = int(np.clip(y_idx + dy, 0, self.height - 1))
                    cx = int(np.clip(x_idx + dx, 0, self.width - 1))
                    weight = max(0, 1 - dist_sq / radius_sq)
                    self.density[cy, cx] += amount * weight
    
    def set_bounds(self, b, x, is_velocity=False):
        """Apply boundary conditions."""
        if is_velocity:
            # No-slip boundary
            x[0, :] = -x[1, :]
            x[-1, :] = -x[-2, :]
            x[:, 0] = -x[:, 1]
            x[:, -1] = -x[:, -2]
        else:
            # Zero boundary for density
            x[0, :] = x[1, :]
            x[-1, :] = x[-2, :]
            x[:, 0] = x[:, 1]
            x[:, -1] = x[:, -2]
        
        # Corner conditions
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
        x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
        x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])
    
    def linear_solve(self, b, x, x0, a, c, iterations=20, is_velocity=False):
        """Solve linear system using Gauss-Seidel iteration."""
        # Handle division by zero
        if abs(c) < 1e-10:
            x[:] = x0
            return
        
        # Ensure iterations is valid
        iterations = max(1, int(iterations))
        
        for _ in range(iterations):
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + 
                           a * (x[2:, 1:-1] + x[:-2, 1:-1] + 
                                x[1:-1, 2:] + x[1:-1, :-2])) / c
            self.set_bounds(b, x, is_velocity)
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(x)):
            x[:] = x0
    
    def diffuse(self, b, x, x0, diff, dt, is_velocity=False):
        """Diffuse a field (viscosity effect)."""
        if diff <= 0 or abs(diff) < 1e-10:
            x[:] = x0
            return
        
        # Validate dimensions
        if self.width < 3 or self.height < 3:
            x[:] = x0
            return
        
        # Validate dt
        if dt <= 0 or not np.isfinite(dt):
            dt = 0.1
        
        a = dt * diff * max(1, (self.width - 2)) * max(1, (self.height - 2))
        c = 1 + 4 * a
        if c < 1e-10:
            x[:] = x0
            return
        
        self.linear_solve(b, x, x0, a, c, is_velocity=is_velocity)
    
    def advect(self, b, d, d0, u, v, dt, is_velocity=False):
        """Advect a field along the velocity."""
        # Validate dimensions
        if self.width < 3 or self.height < 3:
            d[:] = d0
            self.set_bounds(b, d, is_velocity)
            return
        
        # Validate dt
        if dt <= 0 or not np.isfinite(dt):
            dt = 0.1
        
        dt0_x = dt * max(1, (self.width - 2))
        dt0_y = dt * max(1, (self.height - 2))
        
        for j in range(1, self.height - 1):
            for i in range(1, self.width - 1):
                # Check for NaN or Inf in velocity
                if not np.isfinite(u[j, i]):
                    u[j, i] = 0
                if not np.isfinite(v[j, i]):
                    v[j, i] = 0
                
                # Trace back along velocity
                x_pos = i - dt0_x * u[j, i]
                y_pos = j - dt0_y * v[j, i]
                
                # Clamp to grid
                x_pos = max(0.5, min(self.width - 1.5, x_pos))
                y_pos = max(0.5, min(self.height - 1.5, y_pos))
                
                # Bilinear interpolation
                i0, j0 = int(x_pos), int(y_pos)
                i1, j1 = min(i0 + 1, self.width - 1), min(j0 + 1, self.height - 1)
                
                # Ensure indices are valid
                i0 = max(0, min(i0, self.width - 1))
                j0 = max(0, min(j0, self.height - 1))
                i1 = max(0, min(i1, self.width - 1))
                j1 = max(0, min(j1, self.height - 1))
                
                s1, t1 = x_pos - int(x_pos), y_pos - int(y_pos)
                s0, t0 = 1 - s1, 1 - t1
                
                # Check for valid values
                if (np.isfinite(d0[j0, i0]) and np.isfinite(d0[j1, i0]) and
                    np.isfinite(d0[j0, i1]) and np.isfinite(d0[j1, i1])):
                    d[j, i] = (s0 * (t0 * d0[j0, i0] + t1 * d0[j1, i0]) +
                              s1 * (t0 * d0[j0, i1] + t1 * d0[j1, i1]))
                else:
                    d[j, i] = d0[j, i]  # Fallback to original value
        
        self.set_bounds(b, d, is_velocity)
    
    def project(self, u, v, p, div):
        """Project velocity to be divergence-free (incompressible)."""
        # Validate dimensions
        if self.width < 3 or self.height < 3:
            return
        
        # Calculate divergence with safe division
        min_dim = min(self.width, self.height)
        if min_dim < 1:
            return
        
        h = 0.5 / min_dim
        if h < 1e-10 or not np.isfinite(h):
            h = 0.1  # Safe default
        
        div[1:-1, 1:-1] = -0.5 * h * (
            u[1:-1, 2:] - u[1:-1, :-2] +
            v[2:, 1:-1] - v[:-2, 1:-1]
        )
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(div)):
            div.fill(0)
        
        p.fill(0)
        self.set_bounds(0, div, False)
        self.set_bounds(0, p, False)
        
        # Solve for pressure
        self.linear_solve(0, p, div, 1, 4, is_velocity=False)
        
        # Subtract pressure gradient with safe division
        if h > 1e-10:
            u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
            v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
        
        # Check for NaN or Inf
        if not np.all(np.isfinite(u)):
            u.fill(0)
        if not np.all(np.isfinite(v)):
            v.fill(0)
        
        self.set_bounds(1, u, True)
        self.set_bounds(1, v, True)
    
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
        self.density *= 0.998
    
    def get_velocity_magnitude(self):
        """Get velocity magnitude field for visualization."""
        vel_mag = np.sqrt(self.u**2 + self.v**2)
        # Replace NaN or Inf with 0
        vel_mag = np.nan_to_num(vel_mag, nan=0.0, posinf=0.0, neginf=0.0)
        return vel_mag


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
        self.last_mouse_x = self.mouse_x
        self.last_mouse_y = self.mouse_y
        
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
        pass
    
    def on_mouse_move(self, event):
        """Handle mouse movement."""
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.mouse_x = int(np.clip(event.xdata, 0, self.solver.width - 1))
            self.mouse_y = int(np.clip(event.ydata, 0, self.solver.height - 1))
    
    def on_mouse_press(self, event):
        """Handle mouse press."""
        if event.inaxes == self.ax:
            self.mouse_pressed = True
            self.last_mouse_x = self.mouse_x
            self.last_mouse_y = self.mouse_y
    
    def on_mouse_release(self, event):
        """Handle mouse release."""
        self.mouse_pressed = False
    
    def clear_simulation(self):
        """Clear the simulation."""
        self.solver.u.fill(0)
        self.solver.v.fill(0)
        self.solver.density.fill(0)
        self.solver.pressure.fill(0)
        self.solver.u_prev.fill(0)
        self.solver.v_prev.fill(0)
        self.solver.density_prev.fill(0)
    
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
            self.solver.add_density(self.mouse_x, self.mouse_y, 30, radius=6)
            
            # Add velocity when clicking
            if self.mouse_pressed:
                # Calculate velocity from mouse movement
                vx = (self.mouse_x - self.last_mouse_x) * 0.5
                vy = (self.mouse_y - self.last_mouse_y) * 0.5
                self.solver.add_velocity_source(self.mouse_x, self.mouse_y, vx, -vy, radius=8)
                self.last_mouse_x = self.mouse_x
                self.last_mouse_y = self.mouse_y
            else:
                # Small upward velocity for continuous flow
                self.solver.add_velocity_source(self.mouse_x, self.mouse_y, 0, -1, radius=5)
            
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
                vel_mag = self.solver.get_velocity_magnitude()
                data = self.solver.density + 0.3 * vel_mag * 20
                vmax = 100
            
            self.im.set_array(data)
            self.im.set_clim(0, vmax)
            
            # Update velocity vectors if enabled
            if self.show_vectors.get():
                # Sample velocity field
                step = 8
                x = np.arange(step, self.solver.width - step, step)
                y = np.arange(step, self.solver.height - step, step)
                X, Y = np.meshgrid(x, y)
                
                # Get velocities at sample points
                U = np.zeros_like(X)
                V = np.zeros_like(Y)
                for i, xi in enumerate(x):
                    for j, yj in enumerate(y):
                        xi_int = int(xi)
                        yj_int = int(yj)
                        if 0 <= xi_int < self.solver.width and 0 <= yj_int < self.solver.height:
                            U[j, i] = self.solver.u[yj_int, xi_int]
                            V[j, i] = self.solver.v[yj_int, xi_int]
                
                # Remove old quiver
                if self.quiver is not None:
                    self.quiver.remove()
                
                # Create new quiver
                self.quiver = self.ax.quiver(
                    X, Y, U, V,
                    scale=30,
                    color='white',
                    alpha=0.7,
                    width=0.003,
                    headwidth=3,
                    headlength=4
                )
            else:
                if self.quiver is not None:
                    self.quiver.remove()
                    self.quiver = None
            
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback
            traceback.print_exc()


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
