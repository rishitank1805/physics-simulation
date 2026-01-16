"""
Test cases for physics simulations
Tests particle physics, tap flow, and advanced fluid dynamics simulations.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import simulation modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from simulation import Particle, PhysicsSimulation
except ImportError as e:
    print(f"Warning: Could not import simulation module: {e}")
    Particle = None
    PhysicsSimulation = None

try:
    from tap_flow_simulation_ui import TapFlowSimulation, FluidType
except ImportError as e:
    print(f"Warning: Could not import tap_flow_simulation_ui module: {e}")
    TapFlowSimulation = None
    FluidType = None

try:
    from advanced_fluid_simulation import AdvancedFluidSolver
except ImportError as e:
    print(f"Warning: Could not import advanced_fluid_simulation module: {e}")
    AdvancedFluidSolver = None


@unittest.skipIf(Particle is None or PhysicsSimulation is None, "Particle physics classes not available")
class TestParticlePhysics(unittest.TestCase):
    """Test cases for particle physics simulation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.particle = Particle(x=5.0, y=10.0, vx=2.0, vy=3.0, radius=0.5, mass=1.0)
        self.sim = PhysicsSimulation(width=20, height=20, gravity=-9.8, surface_friction=0.1)
    
    def test_particle_initialization(self):
        """Test particle initialization."""
        self.assertEqual(self.particle.x, 5.0)
        self.assertEqual(self.particle.y, 10.0)
        self.assertEqual(self.particle.vx, 2.0)
        self.assertEqual(self.particle.vy, 3.0)
        self.assertEqual(self.particle.radius, 0.5)
        self.assertEqual(self.particle.mass, 1.0)
        self.assertEqual(self.particle.friction_coefficient, 0.1)  # default
    
    def test_gravity_effect(self):
        """Test that gravity affects particle velocity."""
        initial_vy = self.particle.vy
        dt = 0.1
        gravity = -9.8
        
        self.particle.update(dt, gravity, None)
        
        # Velocity should decrease (become more negative) due to gravity
        self.assertLess(self.particle.vy, initial_vy)
        self.assertAlmostEqual(self.particle.vy, initial_vy + gravity * dt, places=5)
    
    def test_position_update(self):
        """Test that position updates based on velocity."""
        initial_x = self.particle.x
        initial_y = self.particle.y
        dt = 0.1
        
        self.particle.update(dt, 0, None)  # No gravity, no bounds
        
        self.assertAlmostEqual(self.particle.x, initial_x + self.particle.vx * dt, places=5)
        self.assertAlmostEqual(self.particle.y, initial_y + self.particle.vy * dt, places=5)
    
    def test_boundary_collision_left(self):
        """Test collision with left boundary."""
        self.particle.x = 0.3  # Close to left boundary
        self.particle.vx = -1.0  # Moving left
        bounds = {'x_min': 0, 'x_max': 10, 'y_min': 0, 'y_max': 10}
        
        self.particle.update(0.1, 0, bounds)
        
        # Should bounce and reverse direction
        self.assertGreater(self.particle.vx, 0)  # Now moving right
        self.assertGreaterEqual(self.particle.x, self.particle.radius)
    
    def test_boundary_collision_bottom(self):
        """Test collision with bottom boundary."""
        self.particle.y = 0.3  # Close to bottom
        self.particle.vy = -1.0  # Moving down
        bounds = {'x_min': 0, 'x_max': 10, 'y_min': 0, 'y_max': 10}
        
        self.particle.update(0.1, 0, bounds)
        
        # Should bounce
        self.assertGreater(self.particle.vy, 0)  # Now moving up
        self.assertGreaterEqual(self.particle.y, self.particle.radius)
    
    def test_friction_effect(self):
        """Test that friction reduces velocity."""
        self.particle.vx = 5.0
        self.particle.vy = 5.0
        self.particle.friction_coefficient = 0.5
        initial_speed = np.sqrt(self.particle.vx**2 + self.particle.vy**2)
        
        dt = 0.1
        self.particle.update(dt, 0, None, surface_friction=0.0)
        
        final_speed = np.sqrt(self.particle.vx**2 + self.particle.vy**2)
        self.assertLess(final_speed, initial_speed)
    
    def test_friction_stops_particle(self):
        """Test that very small velocities are set to zero."""
        self.particle.vx = 0.005  # Very small
        self.particle.vy = 0.005
        self.particle.friction_coefficient = 1.0
        
        # Apply friction multiple times
        for _ in range(10):
            self.particle.update(0.1, 0, None, surface_friction=0.0)
        
        # Should be zero or very close
        self.assertLess(abs(self.particle.vx), 0.02)
        self.assertLess(abs(self.particle.vy), 0.02)
    
    def test_simulation_add_particle(self):
        """Test adding particles to simulation."""
        initial_count = len(self.sim.particles)
        particle = Particle(5, 5, 0, 0)
        self.sim.add_particle(particle)
        
        self.assertEqual(len(self.sim.particles), initial_count + 1)
        self.assertIn(particle, self.sim.particles)


@unittest.skipIf(TapFlowSimulation is None or FluidType is None, "Tap flow classes not available")
class TestTapFlowSimulation(unittest.TestCase):
    """Test cases for tap flow simulation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sim = TapFlowSimulation(tap_height=5.0, tap_area=0.0001, fluid_type='water')
    
    def test_torricelli_law_calculation(self):
        """Test exit velocity calculation using Torricelli's law."""
        # v = sqrt(2gh)
        # For h = 5m, g = 9.81: v = sqrt(2 * 9.81 * 5) ≈ 9.90 m/s
        expected_velocity = np.sqrt(2 * 9.81 * 5.0)
        self.assertAlmostEqual(self.sim.exit_velocity, expected_velocity, places=2)
    
    def test_height_affects_velocity(self):
        """Test that higher tap produces higher velocity."""
        sim_low = TapFlowSimulation(tap_height=1.0, tap_area=0.0001)
        sim_high = TapFlowSimulation(tap_height=10.0, tap_area=0.0001)
        
        self.assertLess(sim_low.exit_velocity, sim_high.exit_velocity)
    
    def test_flow_rate_calculation(self):
        """Test flow rate calculation Q = A * v."""
        expected_flow_rate = self.sim.tap_area * self.sim.exit_velocity
        self.assertAlmostEqual(self.sim.flow_rate, expected_flow_rate, places=6)
    
    def test_area_affects_flow_rate(self):
        """Test that larger area produces higher flow rate."""
        sim_small = TapFlowSimulation(tap_height=5.0, tap_area=0.0001)
        sim_large = TapFlowSimulation(tap_height=5.0, tap_area=0.0002)
        
        # Same velocity, but larger area = larger flow rate
        self.assertAlmostEqual(sim_small.exit_velocity, sim_large.exit_velocity, places=2)
        self.assertLess(sim_small.flow_rate, sim_large.flow_rate)
    
    def test_set_tap_height(self):
        """Test updating tap height."""
        new_height = 10.0
        self.sim.set_tap_height(new_height)
        
        self.assertEqual(self.sim.tap_height, new_height)
        expected_velocity = np.sqrt(2 * 9.81 * new_height)
        self.assertAlmostEqual(self.sim.exit_velocity, expected_velocity, places=2)
    
    def test_set_tap_area(self):
        """Test updating tap area."""
        new_area = 0.0002
        old_velocity = self.sim.exit_velocity
        self.sim.set_tap_area(new_area)
        
        self.assertEqual(self.sim.tap_area, new_area)
        # Velocity shouldn't change
        self.assertEqual(self.sim.exit_velocity, old_velocity)
        # Flow rate should change
        expected_flow_rate = new_area * old_velocity
        self.assertAlmostEqual(self.sim.flow_rate, expected_flow_rate, places=6)
    
    def test_minimum_height(self):
        """Test that height has a minimum value."""
        self.sim.set_tap_height(0.05)  # Try to set below minimum
        self.assertGreaterEqual(self.sim.tap_height, 0.1)
    
    def test_minimum_area(self):
        """Test that area has a minimum value."""
        self.sim.set_tap_area(0.000001)  # Try to set below minimum
        self.assertGreaterEqual(self.sim.tap_area, 0.00001)
    
    def test_fluid_types(self):
        """Test different fluid types."""
        fluids = ['water', 'oil', 'honey', 'alcohol', 'mercury']
        for fluid in fluids:
            sim = TapFlowSimulation(tap_height=5.0, tap_area=0.0001, fluid_type=fluid)
            fluid_props = FluidType.get_fluid(fluid)
            self.assertEqual(sim.fluid['name'], fluid_props['name'])
            self.assertEqual(sim.fluid['density'], fluid_props['density'])
    
    def test_get_info(self):
        """Test getting simulation info."""
        info = self.sim.get_info()
        
        self.assertIn('tap_height', info)
        self.assertIn('tap_area', info)
        self.assertIn('exit_velocity', info)
        self.assertIn('flow_rate', info)
        self.assertIn('flow_rate_lpm', info)
        self.assertIn('fluid_type', info)
        
        # Check flow rate conversion
        expected_lpm = self.sim.flow_rate * 60000
        self.assertAlmostEqual(info['flow_rate_lpm'], expected_lpm, places=2)


@unittest.skipIf(AdvancedFluidSolver is None, "AdvancedFluidSolver class not available")
class TestAdvancedFluidSolver(unittest.TestCase):
    """Test cases for advanced fluid dynamics solver."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.solver = AdvancedFluidSolver(width=50, height=50, viscosity=0.01, dt=0.1)
    
    def test_initialization(self):
        """Test solver initialization."""
        self.assertEqual(self.solver.width, 50)
        self.assertEqual(self.solver.height, 50)
        self.assertEqual(self.solver.viscosity, 0.01)
        self.assertEqual(self.solver.dt, 0.1)
        
        # Check field shapes
        self.assertEqual(self.solver.u.shape, (50, 50))
        self.assertEqual(self.solver.v.shape, (50, 50))
        self.assertEqual(self.solver.density.shape, (50, 50))
        self.assertEqual(self.solver.pressure.shape, (50, 50))
    
    def test_add_density_source(self):
        """Test adding density source."""
        initial_density = self.solver.density[25, 25]
        self.solver.add_density_source(25, 25, 100, radius=5)
        
        # Density should increase at center
        self.assertGreater(self.solver.density[25, 25], initial_density)
    
    def test_add_velocity_source(self):
        """Test adding velocity source."""
        initial_u = self.solver.u[25, 25]
        initial_v = self.solver.v[25, 25]
        
        self.solver.add_velocity_source(25, 25, 2.0, -1.0, radius=5)
        
        # Velocity should change
        self.assertNotEqual(self.solver.u[25, 25], initial_u)
        self.assertNotEqual(self.solver.v[25, 25], initial_v)
    
    def test_boundary_conditions_velocity(self):
        """Test velocity boundary conditions."""
        # Set some internal values
        self.solver.u[10, 10] = 5.0
        self.solver.v[10, 10] = 3.0
        
        # Apply boundary conditions
        self.solver.set_bounds(1, self.solver.u, is_velocity=True)
        self.solver.set_bounds(1, self.solver.v, is_velocity=True)
        
        # Boundaries should be set (no-slip)
        self.assertNotEqual(self.solver.u[0, 10], 0)  # Top boundary
        self.assertNotEqual(self.solver.u[-1, 10], 0)  # Bottom boundary
    
    def test_boundary_conditions_density(self):
        """Test density boundary conditions."""
        # Set some internal values
        self.solver.density[10, 10] = 50.0
        
        # Apply boundary conditions
        self.solver.set_bounds(0, self.solver.density, is_velocity=False)
        
        # Boundaries should mirror adjacent cells
        self.assertEqual(self.solver.density[0, 10], self.solver.density[1, 10])
        self.assertEqual(self.solver.density[-1, 10], self.solver.density[-2, 10])
    
    def test_linear_solve(self):
        """Test linear solver."""
        x = np.zeros((50, 50))
        x0 = np.ones((50, 50)) * 10.0
        
        # Solve with simple parameters
        self.solver.linear_solve(0, x, x0, 1.0, 5.0, iterations=10, is_velocity=False)
        
        # Solution should converge towards x0
        self.assertGreater(np.mean(x), 0)
    
    def test_diffuse(self):
        """Test diffusion step."""
        # Set initial field
        self.solver.u[25, 25] = 10.0
        self.solver.u_prev = self.solver.u.copy()
        
        # Diffuse
        self.solver.diffuse(1, self.solver.u, self.solver.u_prev, 0.01, 0.1, is_velocity=True)
        
        # Value should spread out
        self.assertLess(self.solver.u[25, 25], 10.0)  # Center decreases
        self.assertGreater(self.solver.u[24, 25], 0)  # Neighbors increase
    
    def test_advect(self):
        """Test advection step."""
        # Set up velocity field
        self.solver.u.fill(1.0)  # Constant x-velocity
        self.solver.v.fill(0.0)  # No y-velocity
        
        # Set initial density
        self.solver.density_prev[25, 25] = 100.0
        self.solver.density.fill(0.0)
        
        # Advect
        self.solver.advect(0, self.solver.density, self.solver.density_prev, 
                          self.solver.u, self.solver.v, 0.1, is_velocity=False)
        
        # Density should move in x-direction
        self.assertGreater(self.solver.density[25, 26], 0)  # Moved right
    
    def test_project(self):
        """Test pressure projection."""
        # Set up velocity field with divergence
        self.solver.u[25, 25] = 5.0
        self.solver.u[25, 26] = 5.0
        self.solver.v[25, 25] = 5.0
        self.solver.v[26, 25] = 5.0
        
        p = np.zeros((50, 50))
        div = np.zeros((50, 50))
        
        # Project
        self.solver.project(self.solver.u, self.solver.v, p, div)
        
        # Pressure should be calculated
        self.assertNotEqual(np.sum(p), 0)
    
    def test_step(self):
        """Test full simulation step."""
        # Add some initial conditions
        self.solver.add_density_source(25, 25, 50, radius=3)
        self.solver.add_velocity_source(25, 25, 1.0, -1.0, radius=3)
        
        initial_density_sum = np.sum(self.solver.density)
        
        # Step simulation
        self.solver.step()
        
        # Density should decay
        final_density_sum = np.sum(self.solver.density)
        self.assertLess(final_density_sum, initial_density_sum)
    
    def test_get_velocity_magnitude(self):
        """Test velocity magnitude calculation."""
        self.solver.u[25, 25] = 3.0
        self.solver.v[25, 25] = 4.0
        
        vel_mag = self.solver.get_velocity_magnitude()
        expected = np.sqrt(3.0**2 + 4.0**2)  # Should be 5.0
        
        self.assertAlmostEqual(vel_mag[25, 25], expected, places=5)
    
    def test_zero_viscosity(self):
        """Test that zero viscosity doesn't cause errors."""
        self.solver.viscosity = 0.0
        self.solver.u[25, 25] = 5.0
        self.solver.u_prev = self.solver.u.copy()
        
        # Should not raise error
        self.solver.diffuse(1, self.solver.u, self.solver.u_prev, 0.0, 0.1, is_velocity=True)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with very small values
        self.solver.add_density_source(0, 0, 0.001, radius=1)
        self.solver.add_velocity_source(49, 49, 0.001, 0.001, radius=1)
        
        # Should not raise errors
        self.solver.step()
        
        # Test with very large values
        self.solver.add_density_source(25, 25, 10000, radius=10)
        self.solver.add_velocity_source(25, 25, 100, 100, radius=10)
        
        # Should handle gracefully
        self.solver.step()


class TestIntegration(unittest.TestCase):
    """Integration tests for multiple components."""
    
    def test_particle_in_simulation(self):
        """Test particle behavior in simulation context."""
        sim = PhysicsSimulation(width=10, height=10, gravity=-9.8)
        particle = Particle(5, 9, 0, 0, friction_coefficient=0.1)
        sim.add_particle(particle)
        
        # Update simulation
        sim.update()
        
        # Particle should fall due to gravity
        self.assertLess(particle.y, 9.0)
        self.assertLess(particle.vy, 0)  # Moving down
    
    def test_tap_flow_physics(self):
        """Test tap flow physics calculations."""
        sim = TapFlowSimulation(tap_height=10.0, tap_area=0.0002)
        
        # Verify physics
        expected_v = np.sqrt(2 * 9.81 * 10.0)
        self.assertAlmostEqual(sim.exit_velocity, expected_v, places=2)
        
        expected_q = 0.0002 * expected_v
        self.assertAlmostEqual(sim.flow_rate, expected_q, places=6)


def run_tests():
    """Run all test cases."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestParticlePhysics))
    suite.addTests(loader.loadTestsFromTestCase(TestTapFlowSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedFluidSolver))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
