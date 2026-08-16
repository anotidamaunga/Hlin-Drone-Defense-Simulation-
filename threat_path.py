"""
Threat path generator: produces desired trajectory for the threat drone.
Combines drift, sinusoidal sway, and high-frequency jinking.
"""

import numpy as np


class ThreatPathGenerator:
    """
    Generates a 3D desired trajectory for the threat drone.

    The path combines:
    - Linear drift toward a protected zone
    - Slow sinusoidal lateral weave
    - High-frequency "jinking" evasive maneuvers
    """

    def __init__(self, config):
        """
        Initialize threat path generator.

        Args:
            config: Configuration dictionary or module with parameters
        """
        self.drift_speed = config.THREAT_DRIFT_SPEED
        self.sway_amplitude = config.THREAT_SWAY_AMPLITUDE
        self.sway_frequency = config.THREAT_SWAY_FREQUENCY
        self.jink_amplitude = config.THREAT_JINK_AMPLITUDE
        self.jink_frequency = config.THREAT_JINK_FREQUENCY
        self.target = np.array(config.PROTECTED_ZONE)
        self.initial_pos = np.array(config.THREAT_INITIAL_POS)

    def get_desired_position(self, t):
        """
        Get desired position at time t.

        Args:
            t: Time (s)

        Returns:
            desired_pos: [x, y, z] desired position
        """
        # Linear drift toward target
        # Normalized direction from initial to target
        direction = self.target - self.initial_pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            direction = direction / dist
        else:
            direction = np.array([0, 0, 0])

        # Drift progress: move toward target at constant speed
        # Clamp to prevent overshoot
        drift_pos = self.initial_pos + direction * self.drift_speed * t

        # Add sinusoidal sway in y and z
        sway_y = self.sway_amplitude * np.sin(self.sway_frequency * t)
        sway_z = self.sway_amplitude * 0.5 * np.cos(self.sway_frequency * t * 0.7)

        # Add high-frequency jinking in x and y
        jink_x = self.jink_amplitude * 0.3 * np.sin(self.jink_frequency * t * 2.3)
        jink_y = self.jink_amplitude * np.sin(self.jink_frequency * t * 1.7 + 0.5)
        jink_z = self.jink_amplitude * 0.5 * np.sin(self.jink_frequency * t * 2.1 + 1.2)

        # Combine components
        desired_pos = np.array([
            drift_pos[0] + jink_x,
            drift_pos[1] + sway_y + jink_y,
            drift_pos[2] + sway_z + jink_z
        ])

        return desired_pos

    def _unit_direction(self):
        """Unit vector from initial threat position toward the protected zone."""
        direction = self.target - self.initial_pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            return direction / dist
        return np.zeros(3)

    def get_velocity_at_time(self, t, dt=0.001):
        """
        Analytic velocity at time t (closed-form derivative of get_desired_position).

        Args:
            t: Time (s)
            dt: Unused, kept for backward-compatible call signature.

        Returns:
            velocity: [vx, vy, vz] velocity at time t
        """
        direction = self._unit_direction()
        drift_vel = direction * self.drift_speed

        sway_y_dot = self.sway_amplitude * self.sway_frequency * np.cos(self.sway_frequency * t)
        sway_z_dot = -self.sway_amplitude * 0.5 * (self.sway_frequency * 0.7) * \
            np.sin(self.sway_frequency * t * 0.7)

        jink_x_dot = self.jink_amplitude * 0.3 * (self.jink_frequency * 2.3) * \
            np.cos(self.jink_frequency * t * 2.3)
        jink_y_dot = self.jink_amplitude * (self.jink_frequency * 1.7) * \
            np.cos(self.jink_frequency * t * 1.7 + 0.5)
        jink_z_dot = self.jink_amplitude * 0.5 * (self.jink_frequency * 2.1) * \
            np.cos(self.jink_frequency * t * 2.1 + 1.2)

        return np.array([
            drift_vel[0] + jink_x_dot,
            drift_vel[1] + sway_y_dot + jink_y_dot,
            drift_vel[2] + sway_z_dot + jink_z_dot
        ])

    def get_acceleration_at_time(self, t, dt=0.001):
        """
        Analytic acceleration at time t (closed-form second derivative).

        Args:
            t: Time (s)
            dt: Unused, kept for backward-compatible call signature.

        Returns:
            acceleration: [ax, ay, az] acceleration at time t
        """
        sway_y_ddot = -self.sway_amplitude * self.sway_frequency ** 2 * np.sin(self.sway_frequency * t)
        sway_z_ddot = -self.sway_amplitude * 0.5 * (self.sway_frequency * 0.7) ** 2 * \
            np.cos(self.sway_frequency * t * 0.7)

        jink_x_ddot = -self.jink_amplitude * 0.3 * (self.jink_frequency * 2.3) ** 2 * \
            np.sin(self.jink_frequency * t * 2.3)
        jink_y_ddot = -self.jink_amplitude * (self.jink_frequency * 1.7) ** 2 * \
            np.sin(self.jink_frequency * t * 1.7 + 0.5)
        jink_z_ddot = -self.jink_amplitude * 0.5 * (self.jink_frequency * 2.1) ** 2 * \
            np.sin(self.jink_frequency * t * 2.1 + 1.2)

        return np.array([
            jink_x_ddot,
            sway_y_ddot + jink_y_ddot,
            sway_z_ddot + jink_z_ddot
        ])