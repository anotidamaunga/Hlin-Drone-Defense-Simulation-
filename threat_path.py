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

        # Reactive evasion : steer away
        # from the interceptor's live position once it's within threat awareness radius

        self.evasion_awareness_radius = getattr(
            config, 'THREAT_EVASION_AWARENESS_RADIUS', 25.0)
        self.evasion_amplitude = getattr(
            config, 'THREAT_EVASION_AMPLITUDE', 15.0)

    def get_desired_position(self, t, threat_pos=None, interceptor_pos=None):
        """
        Get desired position at time t.

        Args:
            t: Time (s)
            threat_pos: [x, y, z] the threat's actual current position.
                Together with interceptor_pos, enables reactive evasion —
                without both, this returns the same purely time-scripted
                path as before (backward compatible for callers that don't
                track live interceptor state, e.g. plotting/analysis code).
            interceptor_pos: [x, y, z] the interceptor's actual current
                position, to evade away from.

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

        progress = min(self.drift_speed * t, dist)
        drift_pos = self.initial_pos + direction * progress

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

        # Reactive evasion: push the desired position directly away from the interceptor
        if threat_pos is not None and interceptor_pos is not None:
            away = np.array(threat_pos) - np.array(interceptor_pos)
            range_to_interceptor = np.linalg.norm(away)
            if 1e-6 < range_to_interceptor < self.evasion_awareness_radius:
                evasion_strength = (
                    (self.evasion_awareness_radius - range_to_interceptor)
                    / self.evasion_awareness_radius
                )
                desired_pos = (desired_pos +
                                (away / range_to_interceptor) *
                                evasion_strength * self.evasion_amplitude)

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
        Analytic velocity at time t (closed-form derivative of the
        drift/sway/jink terms in get_desired_position).

        Deliberately excludes the reactive evasion term's derivative: that
        offset depends on the live interceptor position, which isn't a
        smooth function of t alone, so there's no clean closed form. The
        position controller's proportional (pos_error) term still responds
        to the evasion offset via desired position even without a matching
        velocity feedforward here -- it just tracks it slightly less
        crisply, which reads as reasonable "startled reaction" lag rather
        than a bug.

        Args:
            t: Time (s)
            dt: Unused, kept for backward-compatible call signature.

        Returns:
            velocity: [vx, vy, vz] velocity at time t
        """
        direction = self._unit_direction()
        dist = np.linalg.norm(self.target - self.initial_pos)
        drift_vel = direction * self.drift_speed if self.drift_speed * t < dist else np.zeros(3)

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