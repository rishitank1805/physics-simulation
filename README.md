# Simple Physics Simulation

A simple and understandable physics simulation in Python demonstrating basic mechanics: gravity, velocity, acceleration, and collisions.

## Features

- **Particle System**: Create particles with position, velocity, mass, and size
- **Gravity**: Realistic gravitational acceleration
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

### Physics Update
Each frame:
1. Gravity accelerates particles downward
2. Velocity updates position
3. Collisions with boundaries reverse velocity (with damping)

### Customization

You can easily modify the simulation:

```python
# Create your own simulation
sim = PhysicsSimulation(width=20, height=20, gravity=-9.8)

# Add custom particles
sim.add_particle(Particle(x=5, y=10, vx=2, vy=0, radius=1.0, color='purple'))

# Run it
sim.run_animation(duration=15, interval=10)
```

## Future Enhancements

This simple foundation can be extended with:
- Multiple particle interactions
- Collision detection between particles
- Different force types (springs, friction, etc.)
- 3D visualization
- More complex shapes
- Interactive controls
