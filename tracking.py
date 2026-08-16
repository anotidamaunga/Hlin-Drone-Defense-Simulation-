"""
Sensor simulation and Kalman filter for target tracking.
"""

import numpy as np


def radar_sensor(true_pos, noise_std):
    """
    Simulate radar measurements with Gaussian noise.

    Args:
        true_pos: [x, y, z] true position
        noise_std: Standard deviation of position noise

    Returns:
        measurement: [x, y, z] noisy position measurement
    """
    noise = np.random.normal(0, noise_std, 3)
    return true_pos + noise


class KalmanFilter:
    """
    Constant-velocity Kalman filter for tracking a moving target.

    State: [x, y, z, vx, vy, vz]
    Measurement: [x, y, z]
    """

    def __init__(self, dt, q_pos, q_vel, r_pos):
        """
        Initialize Kalman filter.

        Args:
            dt: Time step (s)
            q_pos: Position process noise
            q_vel: Velocity process noise
            r_pos: Position measurement noise
        """
        self.dt = dt

        # State transition matrix F
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix H (position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Process noise covariance Q
        # Based on constant velocity model with acceleration noise
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4

        # Position and velocity coupling terms
        self.Q = np.array([
            [dt4 / 4, 0, 0, dt3 / 2, 0, 0],
            [0, dt4 / 4, 0, 0, dt3 / 2, 0],
            [0, 0, dt4 / 4, 0, 0, dt3 / 2],
            [dt3 / 2, 0, 0, dt2, 0, 0],
            [0, dt3 / 2, 0, 0, dt2, 0],
            [0, 0, dt3 / 2, 0, 0, dt2]
        ]) * q_pos

        # Add velocity process noise. Scaled by dt (variance accumulates
        # proportionally to elapsed time for a random-walk velocity model)
        # instead of being added at full magnitude every step regardless of
        # dt — unscaled, this added q_vel=0.5 of variance on every 10ms tick
        # at the sim's 100Hz rate, making the filter treat its own velocity
        # prediction as nearly worthless and track raw noisy finite
        # differences of position instead of a smoothed estimate.
        self.Q[3:, 3:] += np.eye(3) * q_vel * dt

        # Measurement noise covariance R
        self.R = np.eye(3) * r_pos

        # State estimate and covariance
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1000  # Large initial uncertainty

        self.initialized = False

    def initialize(self, position, velocity=None):
        """
        Initialize filter with first measurement.

        Args:
            position: [x, y, z] initial position
            velocity: [vx, vy, vz] initial velocity (optional)
        """
        self.x[:3] = position
        if velocity is not None:
            self.x[3:] = velocity
        self.initialized = True

    def predict(self):
        """
        Prediction step: propagate state forward in time.
        """
        if not self.initialized:
            return

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """
        Update step: incorporate measurement.

        Args:
            measurement: [x, y, z] position measurement
        """
        if not self.initialized:
            self.initialize(measurement)
            return

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Innovation
        y = measurement - self.H @ self.x

        # Update state and covariance
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def predict_update(self, measurement):
        """
        Convenience method: predict then update.

        Args:
            measurement: [x, y, z] position measurement
        """
        self.predict()
        self.update(measurement)

    def get_estimate(self):
        """
        Get current state estimate.

        Returns:
            state: [x, y, z, vx, vy, vz]
        """
        return self.x.copy()

    def get_position(self):
        """
        Get position estimate.

        Returns:
            position: [x, y, z]
        """
        return self.x[:3].copy()

    def get_velocity(self):
        """
        Get velocity estimate.

        Returns:
            velocity: [vx, vy, vz]
        """
        return self.x[3:].copy()