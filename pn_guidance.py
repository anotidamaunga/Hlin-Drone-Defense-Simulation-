"""
Proportional Navigation guidance law.
Computes commanded lateral acceleration based on LOS rate.
"""

import numpy as np


class ProportionalNavigation:
    """
    Proportional Navigation guidance law.

    Commanded acceleration = N * Vc * LOS_rate
    where N is navigation constant, Vc is closing velocity,
    and LOS_rate is line-of-sight rotation rate.
    """

    def __init__(self, config):
        """
        Initialize PN guidance.

        Args:
            config: Configuration dictionary or module with parameters
        """
        self.N = config.PN_NAVIGATION_CONSTANT
        self.Vc_min = config.PN_CLOSING_VELOCITY_MIN
        self.pursuit_accel_gain = getattr(config, 'PN_PURSUIT_ACCEL_GAIN', 5.0)
        self.g = config.G
        self.mass = config.MASS
        # The interceptor's own (higher) envelope, not the threat's
        # MAX_TILT/MAX_THRUST -- see config.py for why these are separate.
        self.max_tilt = getattr(config, 'INTERCEPTOR_MAX_TILT', config.MAX_TILT)
        self.max_thrust = getattr(config, 'INTERCEPTOR_MAX_THRUST',
                                   getattr(config, 'MAX_THRUST', None))

    def compute_guidance(self, interceptor_pos, interceptor_vel, target_pos, target_vel):
        """
        Compute commanded acceleration from PN guidance.

        Args:
            interceptor_pos: [x, y, z] interceptor position
            interceptor_vel: [vx, vy, vz] interceptor velocity
            target_pos: [x, y, z] target position (estimated)
            target_vel: [vx, vy, vz] target velocity (estimated)

        Returns:
            commanded_accel: [ax, ay, az] desired acceleration in inertial frame
            target_heading: [dx, dy, dz] unit vector toward target
            los_rate: Magnitude of LOS rate
        """
        # Relative position (from interceptor to target)
        r = np.array(target_pos) - np.array(interceptor_pos)
        range_mag = np.linalg.norm(r)

        if range_mag < 1e-6:
            return np.zeros(3), np.zeros(3), 0.0

        # Unit LOS vector
        los_unit = r / range_mag

        # Relative velocity
        v_rel = np.array(target_vel) - np.array(interceptor_vel)

        # Closing velocity (positive when approaching, negative when opening)
        Vc_raw = -np.dot(v_rel, los_unit)

        # LOS rate: rate of change of LOS angle
        # In 3D, LOS rate vector = (r x v_rel) / (r·r)
        # This gives angular velocity perpendicular to LOS
        los_rate_vector = np.cross(r, v_rel) / (range_mag ** 2)
        los_rate_mag = np.linalg.norm(los_rate_vector)

        if Vc_raw < 0:

            return (los_unit * self.pursuit_accel_gain, los_unit, los_rate_mag)

        Vc = max(Vc_raw, self.Vc_min)

        # PN commanded acceleration (perpendicular to LOS)
        # a_cmd = N * Vc * (LOS_rate_vector x LOS_unit)
        # This produces acceleration perpendicular to the LOS
        if los_rate_mag > 1e-6:
            # Direction of commanded acceleration
            accel_direction = np.cross(los_rate_vector / los_rate_mag, los_unit)
            accel_magnitude = self.N * Vc * los_rate_mag
            commanded_accel = accel_magnitude * accel_direction
        else:
            commanded_accel = np.zeros(3)

        return commanded_accel, los_unit, los_rate_mag

    def compute_control_command(self, interceptor_pos, interceptor_vel,
                                target_pos, target_vel, dt=0.1):
        """
        Convert PN's commanded acceleration directly into [T, phi_cmd,
        theta_cmd], the same way rl_interceptor.py's InterceptorEnv.step()
        and HybridGuidanceTrainer._simulate_with_n() already do.

        This used to route the acceleration through PositionController by
        converting it into a kinematically-extrapolated desired_pos/vel
        (desired_pos = pos + vel*dt + ..., desired_vel = vel + accel*dt).
        That fed the interceptor's own current velocity back into
        PositionController's proportional term (pos_error ~= vel*dt), which
        is positive feedback on velocity — the commanded acceleration grew
        with however fast the interceptor was already moving, regardless of
        target position, and it would run away and overshoot the target
        after ~15-20s once real closing speed built up. Converting directly
        to tilt/thrust removes that loop entirely: PN's acceleration command
        maps straight to actuator commands, same as everywhere else in the
        sim, with no dependency on the interceptor's own current velocity.

        Args:
            interceptor_pos: [x, y, z] interceptor position
            interceptor_vel: [vx, vy, vz] interceptor velocity
            target_pos: [x, y, z] target position (estimated)
            target_vel: [vx, vy, vz] target velocity (estimated)
            dt: Unused, kept for call-signature compatibility with the
                previous position/velocity-based version.

        Returns:
            control: [T, phi_cmd, theta_cmd] ready for integrate_dynamics
            commanded_accel: [ax, ay, az] PN commanded acceleration
            metadata: Dict with debugging info. Included so this method's
                return signature matches HybridGuidance.compute_control_command
                exactly — the sim swaps `sim.guidance` between a plain
                ProportionalNavigation instance and a HybridGuidance instance
                (see hybrid_guidance.py), so both must return the same shape.
        """
        # Get PN commanded acceleration
        accel_cmd, los_unit, los_rate = self.compute_guidance(
            interceptor_pos, interceptor_vel, target_pos, target_vel
        )

        theta_cmd = np.clip(accel_cmd[0] / self.g, -self.max_tilt, self.max_tilt)
        phi_cmd = np.clip(-accel_cmd[1] / self.g, -self.max_tilt, self.max_tilt)
        T = max(self.mass * (accel_cmd[2] + self.g), 0.1)
        if self.max_thrust is not None:
            T = min(T, self.max_thrust)
        control = [T, phi_cmd, theta_cmd]

        metadata = {
            'pn_accel': accel_cmd,
            'ai_accel': np.zeros(3),
            'final_accel': accel_cmd,
            'mode': 'pn_only',
            'pn_N': self.N,
            'los_rate': los_rate,
            'miss_dist': np.linalg.norm(np.array(target_pos) - np.array(interceptor_pos))
        }

        return control, accel_cmd, metadata