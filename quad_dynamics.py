"""
Simplified 3-DOF quadrotor dynamics model.
State: [x, y, z, vx, vy, vz]
Inputs: [T, phi_cmd, theta_cmd] (thrust, roll angle, pitch angle)
"""

import numpy as np
from scipy.integrate import solve_ivp


def quad_dynamics(t, state, inputs, mass=1.0, g=9.81):
    """
    Quadrotor dynamics function for use with solve_ivp.

    Args:
        t: Time (scalar)
        state: [x, y, z, vx, vy, vz]
        inputs: [T, phi_cmd, theta_cmd] or function returning inputs
        mass: Drone mass (kg)
        g: Gravitational acceleration (m/s^2)

    Returns:
        dstate_dt: [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = state

    # Get inputs (allow inputs to be a function for time-varying control)
    if callable(inputs):
        T, phi_cmd, theta_cmd = inputs(t)
    else:
        T, phi_cmd, theta_cmd = inputs

    # Simplified near-hover dynamics
    # Roll (phi) controls lateral acceleration in y
    # Pitch (theta) controls lateral acceleration in x
    # Thrust controls vertical acceleration
    ax = g * theta_cmd  # Pitch -> x acceleration
    ay = -g * phi_cmd  # Roll -> y acceleration
    az = (T - mass * g) / mass  # Thrust -> z acceleration

    return [vx, vy, vz, ax, ay, az]


def integrate_dynamics(state, inputs, dt, **kwargs):
    """
    Integrate quadrotor dynamics using a fixed timestep.

    Args:
        state: Current state [x, y, z, vx, vy, vz]
        inputs: [T, phi_cmd, theta_cmd] or function
        dt: Time step (s)
        **kwargs: Additional arguments for quad_dynamics

    Returns:
        new_state: Updated state after dt seconds
    """
    # Use solve_ivp with a single time step
    t_span = (0, dt)
    t_eval = [dt]

    # Create a wrapper for inputs that uses the current time
    if callable(inputs):
        def wrapped_inputs(t):
            return inputs(t)
    else:
        def wrapped_inputs(t):
            return inputs

    def dynamics(t, state):
        return quad_dynamics(t, state, wrapped_inputs, **kwargs)

    sol = solve_ivp(
        dynamics,
        t_span,
        state,
        t_eval=t_eval,
        method='RK45'
    )

    return sol.y[:, -1]