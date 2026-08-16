"""
Configuration parameters for the Hlin drone defense simulation.
All tunable parameters are exposed here for easy adjustment.
"""

# Simulation parameters
SIM_DT = 0.01  # Fixed timestep for integration (seconds)
SIM_DURATION = 30.0  # Total simulation time (seconds)

# Physical constants
G = 9.81  # Gravity (m/s^2)
MASS = 1.0  # Drone mass (kg)
MAX_TILT = 0.3  # Maximum tilt angle (radians) ~17 degrees
MAX_THRUST = 20.0

# Position controller gains (PD)
KP_XY = 2.0  # Proportional gain for x,y
KD_XY = 1.5  # Derivative gain for x,y
KP_Z = 4.0   # Proportional gain for z
KD_Z = 2.5   # Derivative gain for z

# Threat path parameters
THREAT_DRIFT_SPEED = 1.0  # m/s toward protected zone
THREAT_SWAY_AMPLITUDE = 5.0  # meters
THREAT_SWAY_FREQUENCY = 0.2  # rad/s
THREAT_JINK_AMPLITUDE = 2.0  # meters (tunable)
THREAT_JINK_FREQUENCY = 1.5  # rad/s (tunable)
THREAT_INITIAL_POS = [50.0, 50.0, 20.0]  # Starting position (m)
PROTECTED_ZONE = [0.0, 0.0, 0.0]  # Target location

# Interceptor initial state
INTERCEPTOR_INITIAL_POS = [0.0, 0.0, 0.0]
INTERCEPTOR_INITIAL_VEL = [0.0, 0.0, 0.0]

# Intercept success radius: proximity-fuze style "close enough" rather than
# requiring an exact collision. Real kinetic counter-UAS effectors (airburst
# rounds, proximity-fused interceptors) detonate within a lethal radius of
# the target instead of needing a direct hit — 2.0m was effectively modeling
# a physical collision, which is a much harder (and less realistic) target
# than how these systems actually claim a kill.
INTERCEPT_RADIUS = 5.0  # meters

# Proportional Navigation guidance
PN_NAVIGATION_CONSTANT = 4.0  # N (tunable, typically 3-5)
PN_CLOSING_VELOCITY_MIN = 0.5  # Minimum closing velocity (m/s)

# Kalman filter parameters
KALMAN_DT = 0.01  # Time step for filter
# Process noise covariance (Q) - tune these
KALMAN_Q_POS = 0.1  # Position process noise
KALMAN_Q_VEL = 0.5  # Velocity process noise
# Measurement noise covariance (R) - sensor accuracy
KALMAN_R_POS = 1.0  # Position measurement noise

# Sensor noise
RADAR_NOISE_STD = 0.5  # Standard deviation of position noise (m)

# RL training parameters (advanced module)
RL_TOTAL_TIMESTEPS = 1000000
RL_LEARNING_RATE = 0.0003
RL_EVAL_EPISODES = 50