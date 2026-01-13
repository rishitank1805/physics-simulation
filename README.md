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

Run the simulation:
```bash
python simulation.py
```

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

## Future Enhancements

This simple foundation can be extended with:
- Multiple particle interactions
- Collision detection between particles
- Different force types (springs, air resistance, etc.)
- Static vs kinetic friction
- 3D visualization
- More complex shapes
- Interactive controls
