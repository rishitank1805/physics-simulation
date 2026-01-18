"""
Tap/Faucet Fluid Flow Simulation with Modern UI
Simulates fluid flow from a tap with adjustable height, cross-section, and fluid properties.
Uses Torricelli's law to calculate exit velocity.
Built with CustomTkinter for a modern, realistic interface.
"""

import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
import threading
import time


# Set appearance mode and color theme
ctk.set_appearance_mode("dark")  # "light" or "dark"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"


class FluidType:
    """Defines properties of different fluid types."""
    
    FLUIDS = {
        'water': {
            'density': 1000,
            'viscosity': 0.001,
            'color': '#4A90E2',
            'name': 'Water',
            'icon': '💧'
        },
        'oil': {
            'density': 900,
            'viscosity': 0.1,
            'color': '#8B4513',
            'name': 'Oil',
            'icon': '🛢️'
        },
        'honey': {
            'density': 1420,
            'viscosity': 10.0,
            'color': '#FFD700',
            'name': 'Honey',
            'icon': '🍯'
        },
        'alcohol': {
            'density': 789,
            'viscosity': 0.0012,
            'color': '#FF6B6B',
            'name': 'Alcohol',
            'icon': '🍷'
        },
        'mercury': {
            'density': 13590,
            'viscosity': 0.0015,
            'color': '#C0C0C0',
            'name': 'Mercury',
            'icon': '⚗️'
        }
    }
    
    @classmethod
    def get_fluid(cls, name):
        """Get fluid properties by name."""
        return cls.FLUIDS.get(name.lower(), cls.FLUIDS['water'])
    
    @classmethod
    def list_fluids(cls):
        """List all available fluid types."""
        return list(cls.FLUIDS.keys())


class TapFlowSimulation:
    """Simulates fluid flow from a tap."""
    
    def __init__(self, tap_height=5.0, tap_area=0.0001, fluid_type='water', gravity=9.81):
        """Initialize the tap flow simulation."""
        self.tap_height = tap_height
        self.tap_area = tap_area
        self.fluid_type_name = fluid_type
        self.fluid = FluidType.get_fluid(fluid_type)
        self.gravity = gravity
        
        self.exit_velocity = self.calculate_exit_velocity()
        self.flow_rate = self.tap_area * self.exit_velocity
        
        self.particles = []
        self.max_particles = 500
        
        self.sim_width = 10
        self.sim_height = 10
        self.tap_x = self.sim_width / 2
        self.tap_y = self.sim_height - 0.5
    
    def calculate_exit_velocity(self):
        """Calculate exit velocity using Torricelli's law: v = sqrt(2gh)"""
        # Handle edge cases
        if self.tap_height <= 0:
            return 0.0
        if self.gravity <= 0:
            return 0.0
        velocity = np.sqrt(2 * self.gravity * self.tap_height)
        # Check for NaN or Inf
        if not np.isfinite(velocity):
            return 0.0
        return velocity
    
    def set_tap_height(self, height):
        """Update tap height and recalculate velocity."""
        self.tap_height = max(0.1, height)
        self.exit_velocity = self.calculate_exit_velocity()
        self.flow_rate = self.tap_area * self.exit_velocity
    
    def set_tap_area(self, area):
        """Update tap cross-sectional area and recalculate flow rate."""
        self.tap_area = max(0.00001, area)
        self.flow_rate = self.tap_area * self.exit_velocity
    
    def set_fluid_type(self, fluid_type):
        """Change fluid type."""
        self.fluid_type_name = fluid_type
        self.fluid = FluidType.get_fluid(fluid_type)
    
    def get_info(self):
        """Get current simulation parameters and calculated values."""
        return {
            'tap_height': self.tap_height,
            'tap_area': self.tap_area,
            'tap_diameter': 2 * np.sqrt(self.tap_area / np.pi) * 1000,
            'fluid_type': self.fluid['name'],
            'density': self.fluid['density'],
            'viscosity': self.fluid['viscosity'],
            'exit_velocity': self.exit_velocity,
            'flow_rate': self.flow_rate,
            'flow_rate_lpm': self.flow_rate * 60000
        }
    
    def add_particle(self):
        """Add a new particle at the tap opening."""
        if len(self.particles) < self.max_particles:
            spread = 0.1
            vx = np.random.uniform(-spread, spread)
            # Handle division by zero or very small tap_height
            if self.tap_height > 1e-6:
                vy = -self.exit_velocity * (self.sim_height / self.tap_height)
            else:
                vy = -self.exit_velocity
            
            particle = {
                'x': self.tap_x + np.random.uniform(-0.05, 0.05),
                'y': self.tap_y,
                'vx': vx,
                'vy': vy,
                'age': 0
            }
            self.particles.append(particle)
    
    def update_particles(self, dt):
        """Update particle positions based on physics."""
        particles_per_second = self.flow_rate * 1000
        if np.random.random() < particles_per_second * dt:
            self.add_particle()
        
        # Validate dt
        if dt <= 0 or not np.isfinite(dt):
            dt = 0.01
        
        for particle in self.particles[:]:
            # Handle division by zero or very small tap_height
            if self.tap_height > 1e-6:
                particle['vy'] += self.gravity * dt * (self.sim_height / self.tap_height)
            else:
                particle['vy'] += self.gravity * dt
            
            drag_coefficient = 0.02 * self.fluid['viscosity']
            speed = np.sqrt(particle['vx']**2 + particle['vy']**2)
            if speed > 1e-10:  # Avoid issues with very small speeds
                drag_factor = max(0, 1 - drag_coefficient * dt)  # Prevent negative
                particle['vx'] *= drag_factor
                particle['vy'] *= drag_factor
            
            particle['x'] += particle['vx'] * dt
            particle['y'] += particle['vy'] * dt
            particle['age'] += dt
            
            if particle['y'] < -1 or particle['age'] > 5 or \
               particle['x'] < -1 or particle['x'] > self.sim_width + 1:
                self.particles.remove(particle)


class App(ctk.CTk):
    """Main application window with modern UI."""
    
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Fluid Dynamics - Tap Flow Simulation")
        self.geometry("1600x900")
        self.minsize(1200, 700)
        
        # Create simulation
        self.simulation = TapFlowSimulation(
            tap_height=5.0,
            tap_area=0.0001,
            fluid_type='water'
        )
        
        # Setup UI
        self.create_widgets()
        
        # Animation state
        self.is_running = True
        self.update_simulation()
    
    def create_widgets(self):
        """Create and layout all UI widgets."""
        # Main container
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left side - Simulation canvas
        left_frame = ctk.CTkFrame(main_container)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Title for simulation area
        title_label = ctk.CTkLabel(
            left_frame, 
            text="🔬 Fluid Flow Visualization",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(15, 10))
        
        # Matplotlib figure
        self.fig = Figure(figsize=(10, 9), facecolor='#1a1a1a')
        self.ax = self.fig.add_subplot(111, facecolor='#1a1a1a')
        self.ax.set_xlim(0, self.simulation.sim_width)
        self.ax.set_ylim(0, self.simulation.sim_height)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (m)', color='white', fontsize=11)
        self.ax.set_ylabel('Y (m)', color='white', fontsize=11)
        self.ax.set_title('Tap Flow Simulation', color='white', fontsize=14, fontweight='bold', pad=15)
        self.ax.tick_params(colors='white')
        self.ax.grid(True, alpha=0.2, color='white')
        
        # Draw tap
        tap_width = 0.3
        tap_height_viz = 0.5
        self.tap_rect = Rectangle(
            (self.simulation.tap_x - tap_width/2, self.simulation.sim_height - tap_height_viz),
            tap_width, tap_height_viz,
            facecolor='#4a4a4a', edgecolor='white', linewidth=2
        )
        self.ax.add_patch(self.tap_rect)
        
        # Tap opening
        tap_opening_radius = np.sqrt(self.simulation.tap_area / np.pi) * 50
        self.tap_opening = Circle(
            (self.simulation.tap_x, self.simulation.tap_y),
            tap_opening_radius,
            facecolor='#2a2a2a', edgecolor='white', linewidth=2
        )
        self.ax.add_patch(self.tap_opening)
        
        # Particle scatter
        self.scatter = self.ax.scatter([], [], s=40, alpha=0.8)
        
        self.fig.tight_layout()
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, left_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Right side - Controls panel
        right_frame = ctk.CTkFrame(main_container, width=450)
        right_frame.pack(side="right", fill="y", padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Title
        title = ctk.CTkLabel(
            right_frame,
            text="⚙️ Control Panel",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title.pack(pady=(20, 25))
        
        # Tap Height Control
        height_frame = ctk.CTkFrame(right_frame)
        height_frame.pack(fill="x", padx=20, pady=10)
        
        height_label = ctk.CTkLabel(
            height_frame,
            text="📏 Tap Height",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        height_label.pack(pady=(15, 5))
        
        self.height_value_label = ctk.CTkLabel(
            height_frame,
            text=f"{self.simulation.tap_height:.2f} m",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#4A9EFF"
        )
        self.height_value_label.pack()
        
        self.height_slider = ctk.CTkSlider(
            height_frame,
            from_=0.1,
            to=20.0,
            number_of_steps=190,
            command=self.on_height_changed
        )
        self.height_slider.set(self.simulation.tap_height)
        self.height_slider.pack(fill="x", padx=20, pady=(10, 15))
        
        # Tap Area Control
        area_frame = ctk.CTkFrame(right_frame)
        area_frame.pack(fill="x", padx=20, pady=10)
        
        area_label = ctk.CTkLabel(
            area_frame,
            text="🔘 Cross-Sectional Area",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        area_label.pack(pady=(15, 5))
        
        self.area_value_label = ctk.CTkLabel(
            area_frame,
            text=f"{self.simulation.tap_area*10000:.2f} cm²",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#4A9EFF"
        )
        self.area_value_label.pack()
        
        self.area_slider = ctk.CTkSlider(
            area_frame,
            from_=0.01,
            to=10.0,
            number_of_steps=999,
            command=self.on_area_changed
        )
        self.area_slider.set(self.simulation.tap_area * 10000)
        self.area_slider.pack(fill="x", padx=20, pady=(10, 15))
        
        # Fluid Type Selection
        fluid_frame = ctk.CTkFrame(right_frame)
        fluid_frame.pack(fill="x", padx=20, pady=10)
        
        fluid_label = ctk.CTkLabel(
            fluid_frame,
            text="🧪 Fluid Type",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        fluid_label.pack(pady=(15, 10))
        
        self.fluid_combo = ctk.CTkComboBox(
            fluid_frame,
            values=[f"{FluidType.get_fluid(f)['icon']} {FluidType.get_fluid(f)['name']}" 
                   for f in FluidType.list_fluids()],
            font=ctk.CTkFont(size=14),
            dropdown_font=ctk.CTkFont(size=14),
            width=300
        )
        self.fluid_combo.set(f"{self.simulation.fluid['icon']} {self.simulation.fluid['name']}")
        self.fluid_combo.configure(command=self.on_fluid_changed)
        self.fluid_combo.pack(pady=(0, 15))
        
        # Calculated Values Display
        values_frame = ctk.CTkFrame(right_frame)
        values_frame.pack(fill="x", padx=20, pady=10)
        
        values_title = ctk.CTkLabel(
            values_frame,
            text="📊 Calculated Values",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        values_title.pack(pady=(15, 15))
        
        # Exit Velocity
        self.velocity_frame = self.create_value_display(
            values_frame, "Exit Velocity", "0.00 m/s", "#00D4AA"
        )
        
        # Flow Rate
        self.flow_rate_frame = self.create_value_display(
            values_frame, "Flow Rate", "0.00 L/min", "#FF6B6B"
        )
        
        # Tap Diameter
        self.diameter_frame = self.create_value_display(
            values_frame, "Tap Diameter", "0.00 mm", "#FFD93D"
        )
        
        # Fluid Properties
        properties_frame = ctk.CTkFrame(right_frame)
        properties_frame.pack(fill="x", padx=20, pady=10)
        
        properties_title = ctk.CTkLabel(
            properties_frame,
            text="🔬 Fluid Properties",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        properties_title.pack(pady=(15, 15))
        
        self.density_frame = self.create_value_display(
            properties_frame, "Density", "0 kg/m³", "#9B59B6"
        )
        
        self.viscosity_frame = self.create_value_display(
            properties_frame, "Viscosity", "0 Pa·s", "#E67E22"
        )
        
        # Control Buttons
        buttons_frame = ctk.CTkFrame(right_frame)
        buttons_frame.pack(fill="x", padx=20, pady=20)
        
        button_container = ctk.CTkFrame(buttons_frame, fg_color="transparent")
        button_container.pack(pady=10)
        
        self.reset_button = ctk.CTkButton(
            button_container,
            text="🔄 Reset",
            command=self.reset_simulation,
            font=ctk.CTkFont(size=14, weight="bold"),
            width=140,
            height=40,
            fg_color="#E74C3C",
            hover_color="#C0392B"
        )
        self.reset_button.pack(side="left", padx=5)
        
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
        
        # Formula display
        formula_label = ctk.CTkLabel(
            right_frame,
            text="Formula: v = √(2gh)",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        formula_label.pack(pady=(0, 20))
        
        # Update initial display
        self.update_display()
    
    def create_value_display(self, parent, label_text, value_text, color):
        """Create a styled value display widget."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=15, pady=8)
        
        label = ctk.CTkLabel(
            frame,
            text=label_text + ":",
            font=ctk.CTkFont(size=13)
        )
        label.pack(side="left", padx=(0, 10))
        
        value = ctk.CTkLabel(
            frame,
            text=value_text,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=color
        )
        value.pack(side="right")
        
        return value
    
    def on_height_changed(self, value):
        """Handle tap height slider change."""
        self.simulation.set_tap_height(value)
        self.height_value_label.configure(text=f"{value:.2f} m")
        self.update_display()
    
    def on_area_changed(self, value):
        """Handle tap area slider change."""
        area = value / 10000.0
        self.simulation.set_tap_area(area)
        self.area_value_label.configure(text=f"{value:.2f} cm²")
        self.update_display()
    
    def on_fluid_changed(self, selection):
        """Handle fluid type selection change."""
        if selection:
            try:
                fluid_name = selection.split(' ', 1)[1].lower()
                self.simulation.set_fluid_type(fluid_name)
                self.update_display()
            except (IndexError, AttributeError):
                pass  # Ignore errors during initialization
    
    def update_display(self):
        """Update all displayed values."""
        info = self.simulation.get_info()
        
        self.velocity_frame.configure(text=f"{info['exit_velocity']:.2f} m/s")
        self.flow_rate_frame.configure(text=f"{info['flow_rate_lpm']:.2f} L/min")
        self.diameter_frame.configure(text=f"{info['tap_diameter']:.2f} mm")
        self.density_frame.configure(text=f"{info['density']:.0f} kg/m³")
        self.viscosity_frame.configure(text=f"{info['viscosity']:.4f} Pa·s")
    
    def update_simulation(self):
        """Update simulation physics and visualization."""
        if self.is_running:
            dt = 0.02
            self.simulation.update_particles(dt)
            self.update_visualization()
        
        # Schedule next update
        self.after(20, self.update_simulation)
    
    def update_visualization(self):
        """Update the matplotlib visualization."""
        try:
            # Update tap opening size
            tap_opening_radius = np.sqrt(self.simulation.tap_area / np.pi) * 50
            self.tap_opening.set_radius(tap_opening_radius)
            
            # Update particles
            if self.simulation.particles:
                x_coords = [p['x'] for p in self.simulation.particles]
                y_coords = [p['y'] for p in self.simulation.particles]
                self.scatter.set_offsets(np.c_[x_coords, y_coords])
                self.scatter.set_color(self.simulation.fluid['color'])
                self.scatter.set_sizes([40] * len(self.simulation.particles))
            else:
                self.scatter.set_offsets(np.empty((0, 2)))
            
            self.canvas.draw()
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def reset_simulation(self):
        """Reset simulation to default values."""
        self.simulation.set_tap_height(5.0)
        self.simulation.set_tap_area(0.0001)
        self.simulation.set_fluid_type('water')
        self.simulation.particles.clear()
        
        self.height_slider.set(5.0)
        self.area_slider.set(1.0)
        self.fluid_combo.set("💧 Water")
        
        self.height_value_label.configure(text="5.00 m")
        self.area_value_label.configure(text="1.00 cm²")
        
        self.update_display()
    
    def toggle_pause(self):
        """Toggle simulation pause/play."""
        self.is_running = not self.is_running
        if self.is_running:
            self.pause_button.configure(text="⏸️ Pause")
        else:
            self.pause_button.configure(text="▶️ Resume")


def main():
    """Main function to run the application."""
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
