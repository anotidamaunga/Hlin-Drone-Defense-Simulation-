# Hlin: Drone Defense Simulation

**Hlin** (named after the Norse goddess of protection) is an educational control-theory simulation of a counter-drone intercept system. A threat drone flies an evasive path toward a protected zone, while an interceptor drone tracks and engages it using classical guidance (Proportional Navigation) and optionally reinforcement learning.

## Architecture

The simulation consists of six core modules:

1. **quad_dynamics.py**: 3-DOF quadrotor dynamics model
2. **threat_path.py**: Generates evasive threat trajectories with drift, sway, and jinking
3. **position_controller.py**: PD controller converting desired position to thrust/tilt commands
4. **pn_guidance.py**: Proportional Navigation guidance law with tunable N
5. **tracking.py**: Radar sensor simulation and Kalman filter
6. **run_sim.py**: Main simulation loop with logging and visualization


