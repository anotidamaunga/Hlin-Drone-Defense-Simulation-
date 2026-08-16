"""
Position controller: converts desired position to thrust and tilt commands.
Uses PD control with tilt angle clamping for realism.
"""

import numpy as np


class PositionController:
    """
    PD position controller for 3D drone control.
    Converts desired position to [T, phi_cmd, theta_cmd].
    """

    def __init__(self, config):
        """
        Initialize position controller.

        Args:
            config: Configuration dictionary or module with parameters
        """
        self.kp_xy = config.KP_XY
        self.kd_xy = config.KD_XY
        self.kp_z = config.KP_Z
        self.kd_z = config.KD_Z
        self.max_tilt = config.MAX_TILT
        self.max_thrust = getattr(config, 'MAX_THRUST', None)
        self.g = config.G
        self.mass = config.MASS

    def compute_control(self, desired_pos, current_state, desired_vel=None):
        """
        Compute control inputs from desired position and current state.

        Args:
            desired_pos: [x, y, z] desired position
            current_state: [x, y, z, vx, vy, vz] current state
            desired_vel: [vx, vy, vz] desired velocity (optional)
                         If None, velocity error is just -current velocity

        Returns:
            [T, phi_cmd, theta_cmd]: Thrust, roll command, pitch command
        """
        x, y, z, vx, vy, vz = current_state
        pos_error = np.array(desired_pos) - np.array([x, y, z])

        if desired_vel is not None:
            vel_error = np.array(desired_vel) - np.array([vx, vy, vz])
        else:
            # PD control with velocity feedback (derivative on output)
            vel_error = -np.array([vx, vy, vz])

        # Horizontal control (xy)
        accel_xy = self.kp_xy * pos_error[:2] + self.kd_xy * vel_error[:2]

        # Vertical control (z)
        accel_z = self.kp_z * pos_error[2] + self.kd_z * vel_error[2]

        # Convert accelerations to tilt commands
        # ax = g * theta_cmd -> theta_cmd = ax / g
        # ay = -g * phi_cmd -> phi_cmd = -ay / g
        theta_cmd = np.clip(accel_xy[0] / self.g, -self.max_tilt, self.max_tilt)
        phi_cmd = np.clip(-accel_xy[1] / self.g, -self.max_tilt, self.max_tilt)

        # Thrust command for vertical acceleration
        # az = (T - m*g)/m -> T = m*(az + g)
        T = self.mass * (accel_z + self.g)
        T = max(T, 0.1)  # Minimum thrust to avoid negative
        if self.max_thrust is not None:
            T = min(T, self.max_thrust)

        return [T, phi_cmd, theta_cmd]

    def compute_control_from_path(self, path_generator, t, current_state):
        """
        Convenience method: compute control from a path generator.

        Args:
            path_generator: ThreatPathGenerator instance
            t: Current time
            current_state: Current state

        Returns:
            [T, phi_cmd, theta_cmd]: Control inputs
        """
        desired_pos = path_generator.get_desired_position(t)
        desired_vel = path_generator.get_velocity_at_time(t)
        return self.compute_control(desired_pos, current_state, desired_vel)