# Simple Physics Simulation

A simple and understandable physics simulation in Python demonstrating basic mechanics: gravity, velocity, acceleration, friction, and collisions.

## Features

- **Particle System**: Create particles with position, velocity, mass, and size
- **Gravity**: Realistic gravitational acceleration
- **Friction**: Coefficient of friction for particles and surfaces
- **Collisions**: Bouncing off walls with energy damping
- **Visualization**: Real-time animation using matplotlib

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Particle Physics Simulation
Run the particle physics simulation:
```bash
python simulation.py
```

### Fluid Dynamics Simulation
Run the fluid dynamics simulation:
```bash
python fluid_simulation.py
```
Move your mouse to add density (like smoke/dye), and click to add velocity (like wind/force)!

### Tap Flow Simulation

**With Modern CustomTkinter UI (Recommended):**
```bash
python tap_flow_simulation_ui.py
```
Features a beautiful, modern dark-themed interface with smooth sliders, styled buttons, and real-time parameter display!

## How It Works

### Particle Class
Each particle has:
- **Position** (x, y): Where it is in space
- **Velocity** (vx, vy): How fast and in what direction it's moving
- **Mass**: Affects how forces act on it
- **Radius**: Size for collision detection
- **Friction Coefficient**: How much friction the particle experiences (0 = no friction, higher = more friction)

### Physics Update
Each frame:
1. Friction reduces velocity (opposes motion)
2. Gravity accelerates particles downward
3. Velocity updates position
4. Collisions with boundaries reverse velocity (with damping)

### Customization

You can easily modify the simulation:

```python
# Create your own simulation with surface friction
sim = PhysicsSimulation(width=20, height=20, gravity=-9.8, surface_friction=0.3)

# Add custom particles with different friction coefficients
sim.add_particle(Particle(x=5, y=10, vx=2, vy=0, radius=1.0, color='purple', friction_coefficient=0.2))

# Run it
sim.run_animation(duration=15, interval=10)
```

### Friction Parameters
- **Particle friction_coefficient**: Each particle can have its own friction (0.0 = no friction, 0.5 = high friction)
- **Surface friction**: Applied by the simulation environment, especially when particles are on the ground
- Higher friction values cause particles to slow down faster

## Fluid Dynamics Simulation

The fluid dynamics simulation uses a grid-based approach with simplified Navier-Stokes equations.

### Features
- **Grid-based simulation**: Uses a 2D grid to represent fluid velocity and density
- **Navier-Stokes equations**: Implements simplified version of fluid dynamics equations
- **Viscosity**: Controls how thick/viscous the fluid is
- **Advection**: Density and velocity move with the flow
- **Diffusion**: Optional spreading of density over time
- **Interactive**: Move mouse to add density, click to add velocity

### How It Works

The simulation solves three main equations each frame:

1. **Diffusion**: Spreads velocity and density (viscosity effect)
2. **Advection**: Moves density and velocity along the flow
3. **Projection**: Makes velocity divergence-free (incompressible fluid)

### Customization

```python
# Create fluid simulation
sim = FluidSimulation(
    width=150,           # Grid width
    height=150,          # Grid height
    viscosity=0.01,      # Higher = thicker fluid (0.01 = water-like, 0.1 = honey-like)
    diffusion=0.0,       # Density spreading (0 = no diffusion)
    dt=0.1               # Time step
)

# Add density (like smoke or dye)
sim.add_density(x=50, y=50, amount=100, radius=5)

# Add velocity (like wind or force)
sim.add_velocity(x=50, y=50, vx=2, vy=-1, radius=5)

# Run it
sim.run_animation(duration=30, interval=50, interactive=True)
```

### Parameters
- **viscosity**: Controls fluid thickness (0.01 = water, 0.1 = honey, 1.0 = very thick)
- **diffusion**: How much density spreads (0 = no spreading, higher = more spreading)
- **dt**: Time step (smaller = more accurate but slower)

## Tap Flow Simulation

A practical fluid mechanics simulation that calculates exit velocity from a tap/faucet based on height, cross-sectional area, and fluid properties.

### Features
- **Adjustable Tap Height**: Change the height of the tap (affects exit velocity)
- **Variable Cross-Section**: Adjust the tap opening area (affects flow rate)
- **Multiple Fluid Types**: Choose from water, oil, honey, alcohol, mercury
- **Physics-Based Calculation**: Uses Torricelli's law (v = √(2gh)) to calculate exit velocity
- **Real-time Visualization**: See fluid particles flowing from the tap
- **Modern CustomTkinter UI**: Beautiful dark-themed interface with smooth animations, styled widgets, and professional design
- **Interactive Controls**: Intuitive sliders, dropdown menus, and color-coded value displays
- **Realistic Appearance**: Modern, polished UI that looks professional and realistic

### How It Works

The simulation uses **Torricelli's law** to calculate exit velocity:
```
v = √(2gh)
```
Where:
- `v` = exit velocity (m/s)
- `g` = gravitational acceleration (9.81 m/s²)
- `h` = height of tap above ground (m)

The flow rate is then calculated as:
```
Q = A × v
```
Where:
- `Q` = flow rate (m³/s)
- `A` = cross-sectional area of tap (m²)
- `v` = exit velocity (m/s)

### Available Fluid Types

- **Water**: Density 1000 kg/m³, Viscosity 0.001 Pa·s
- **Oil**: Density 900 kg/m³, Viscosity 0.1 Pa·s
- **Honey**: Density 1420 kg/m³, Viscosity 10.0 Pa·s
- **Alcohol**: Density 789 kg/m³, Viscosity 0.0012 Pa·s
- **Mercury**: Density 13590 kg/m³, Viscosity 0.0015 Pa·s

### Interactive Controls

**CustomTkinter UI (tap_flow_simulation_ui.py):**
- **Height Slider**: Smooth, modern slider to adjust tap height (0.1 - 20.0 m)
- **Area Slider**: Adjust tap cross-sectional area (0.01 - 10 cm²)
- **Fluid Dropdown**: Beautiful dropdown with emoji icons for each fluid type
- **Reset Button**: Styled red button to reset all parameters
- **Pause/Resume Button**: Blue button to pause or resume the simulation
- **Real-time Display**: Color-coded value displays that update automatically
- **Dark Theme**: Modern dark interface with professional styling
- **Smooth Animations**: Fluid particle animations at 50 FPS

### Customization

```python
# Create tap flow simulation
sim = TapFlowSimulation(
    tap_height=5.0,      # Height in meters
    tap_area=0.0001,     # Area in m² (0.0001 m² = 1 cm²)
    fluid_type='water',  # 'water', 'oil', 'honey', 'alcohol', 'mercury'
    gravity=9.81         # Gravitational acceleration
)

# Adjust parameters
sim.set_tap_height(10.0)      # Change height
sim.set_tap_area(0.0002)      # Change area (2 cm²)
sim.set_fluid_type('honey')   # Change fluid

# Get calculated values
info = sim.get_info()
print(f"Exit Velocity: {info['exit_velocity']:.2f} m/s")
print(f"Flow Rate: {info['flow_rate_lpm']:.2f} L/min")

# Run simulation
sim.run_animation(duration=30, interval=20)
```

### Understanding the Results

- **Higher tap height** → Higher exit velocity (more potential energy)
- **Larger cross-section** → Higher flow rate (more area for fluid to pass through)
- **Different fluids** → Same velocity (Torricelli's law doesn't depend on density), but different flow patterns due to viscosity
- **Viscosity** affects how the fluid flows after exiting (honey flows slower, water flows faster)

## Future Enhancements

### Particle Physics
- Multiple particle interactions
- Collision detection between particles
- Different force types (springs, air resistance, etc.)
- Static vs kinetic friction
- 3D visualization
- More complex shapes
- Interactive controls

### Fluid Dynamics
- Temperature effects (buoyancy)
- Multiple fluid types
- Obstacles and boundaries
- Surface tension
- 3D fluid simulation
- Better visualization (velocity vectors, streamlines)
- Performance optimizations

### Tap Flow
- Pressure effects (Bernoulli's equation for pressurized systems)
- Multiple taps
- Flow visualization improvements (streamlines, velocity vectors)
- Real-time parameter sliders
- Export flow data
- Different tap shapes (rectangular, etc.)
- Flow rate history graphs
