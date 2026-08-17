"""
Reinforcement Learning interceptor for the Hlin drone defense simulation.
Trains an AI agent to intercept the threat drone using PPO.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch
import os
import json
import shutil
from datetime import datetime

import config
from quad_dynamics import integrate_dynamics
from threat_path import ThreatPathGenerator
from position_controller import PositionController


class InterceptorEnv(gym.Env):
    """
    Custom Gym environment for training an AI interceptor drone.

    The agent learns to control the interceptor to shoot down the threat drone.

    Observation space (17 dimensions):
    - Relative position (3): threat_pos - interceptor_pos
    - Relative velocity (3): threat_vel - interceptor_vel
    - Threat acceleration (3): from path generator
    - Interceptor state (3): current position
    - Range to target (1)
    - Closing velocity (1)
    - Time step normalized (1)
    - Previous action (3): for smoothness

    Action space (3 dimensions):
    - Commanded acceleration in x, y, z (m/s²)

    Reward function:
    - Negative miss distance (primary)
    - Bonus for close approaches
    - Penalty for control effort
    - Terminal bonus for intercept (< config.INTERCEPT_RADIUS)
    - Terminal penalty for threat reaching target
    """

    def __init__(self, config, use_jink=True, render_mode=None):
        super().__init__()

        self.config = config
        self.dt = config.SIM_DT
        self.max_steps = int(config.SIM_DURATION / config.SIM_DT)
        self.use_jink = use_jink

        # Action space: commanded acceleration in x, y, z
        self.action_space = spaces.Box(
            low=-15.0, high=15.0, shape=(3,), dtype=np.float32
        )

        # Observation space: rel_pos(3) + rel_vel(3) + threat_accel(3) +
        # interceptor_pos(3) + range(1) + closing_vel(1) + time_norm(1) +
        # prev_action(3) = 18 values (must match _get_observation()).
        self.observation_space = spaces.Box(
            low=-100, high=100, shape=(18,), dtype=np.float32
        )

        # Initialize components
        self.threat_path = ThreatPathGenerator(config)
        self.position_controller = PositionController(config)

        # State variables
        self.threat_state = None
        self.interceptor_state = None
        self.step_count = 0
        self.prev_action = np.zeros(3)
        self.best_miss_distance = np.inf

        # Tracking
        self.threat_history = []
        self.interceptor_history = []
        self.miss_distances = []

        # Randomize threat parameters for generalization
        self.jink_amplitude = config.THREAT_JINK_AMPLITUDE
        self.jink_frequency = config.THREAT_JINK_FREQUENCY

        # Threat spawn distance band (meters from the protected zone/
        # interceptor). Mutable via set_engagement_range() so a curriculum
        # callback can start episodes close (easy, frequent intercepts to
        # learn from) and widen toward the full operational range over the
        # course of training.
        self.engagement_min = 40.0
        self.engagement_max = 80.0

    def set_engagement_range(self, min_dist, max_dist):
        """Update the threat spawn distance band used by future reset() calls."""
        self.engagement_min = min_dist
        self.engagement_max = max_dist

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Randomize threat initial position within a fixed engagement band
        # around the protected zone (distance 40-80m). Sampling a ring
        # offset from (50, 50) instead could put the threat anywhere from
        # ~0.7m to ~141m from the interceptor/protected zone, which used to
        # trigger an instant terminal condition (miss_dist < 3 or > 100) on
        # the very first step regardless of the agent's action.
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(self.engagement_min, self.engagement_max)
        threat_x = distance * np.cos(angle)
        threat_y = distance * np.sin(angle)
        threat_z = 20 + np.random.uniform(-10, 10)

        self.threat_state = np.array([
            threat_x, threat_y, threat_z,
            0.0, 0.0, 0.0
        ])

        # Re-anchor the drift trajectory to THIS episode's randomized spawn.
        # ThreatPathGenerator.initial_pos otherwise stays at the fixed
        # config.THREAT_INITIAL_POS for the whole training run, so
        # get_desired_position(t) drives the threat's PD controller toward a
        # path anchored at (50, 50, 20) regardless of where the episode says
        # the threat actually starts — for a spawn far from that quadrant
        # this is a 100m+ position error from step zero, and the threat
        # spends the episode fighting to reach a phantom trajectory instead
        # of behaving like it's actually heading for the protected zone from
        # its real position. (HybridGuidanceTrainer.generate_training_data
        # in hybrid_guidance.py already had to work around this same issue
        # locally; this fixes it at the source.)
        self.threat_path.initial_pos = self.threat_state[:3].copy()

        # Randomize interceptor start position near origin
        interceptor_x = np.random.uniform(-5, 5)
        interceptor_y = np.random.uniform(-5, 5)
        interceptor_z = np.random.uniform(0, 5)

        self.interceptor_state = np.array([
            interceptor_x, interceptor_y, interceptor_z,
            0.0, 0.0, 0.0
        ])

        # Randomize threat jinking parameters for better generalization
        if self.use_jink:
            self.jink_amplitude = np.random.uniform(1.0, 4.0)
            self.jink_frequency = np.random.uniform(0.5, 3.0)
            self.threat_path.jink_amplitude = self.jink_amplitude
            self.threat_path.jink_frequency = self.jink_frequency

        self.step_count = 0
        self.prev_action = np.zeros(3)
        self.best_miss_distance = np.inf
        self.threat_history = []
        self.interceptor_history = []
        self.miss_distances = []

        # Initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Execute one step with the given action."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.prev_action = action.copy()


        accel_cmd = np.array(action)
        g = self.config.G
        mass = self.config.MASS
        max_tilt = getattr(self.config, 'INTERCEPTOR_MAX_TILT', self.config.MAX_TILT)

        theta_cmd = np.clip(accel_cmd[0] / g, -max_tilt, max_tilt)
        phi_cmd = np.clip(-accel_cmd[1] / g, -max_tilt, max_tilt)
        T = mass * (accel_cmd[2] + g)
        T = max(T, 0.1)
        max_thrust = getattr(self.config, 'INTERCEPTOR_MAX_THRUST',
                              getattr(self.config, 'MAX_THRUST', None))
        if max_thrust is not None:

            T = min(T, max_thrust)

        self.interceptor_state = integrate_dynamics(
            self.interceptor_state,
            [T, phi_cmd, theta_cmd],
            self.dt,
            mass=mass,
            g=g
        )

        # Clamp position to keep in bounds
        self.interceptor_state[:3] = np.clip(self.interceptor_state[:3], -10, 100)

        # Move threat drone according to path reactive evasion

        t = self.step_count * self.dt
        threat_desired = self.threat_path.get_desired_position(
            t, threat_pos=self.threat_state[:3],
            interceptor_pos=self.interceptor_state[:3]
        )
        threat_vel = self.threat_path.get_velocity_at_time(t)

        # Position controller for threat (follows path)
        threat_control = self.position_controller.compute_control(
            threat_desired,
            self.threat_state,
            desired_vel=threat_vel
        )

        # Integrate threat dynamics
        self.threat_state = integrate_dynamics(
            self.threat_state,
            threat_control,
            self.dt,
            mass=self.config.MASS,
            g=self.config.G
        )

        # Calculate metrics
        miss_dist = np.linalg.norm(self.threat_state[:3] - self.interceptor_state[:3])
        self.miss_distances.append(miss_dist)
        self.best_miss_distance = min(self.best_miss_distance, miss_dist)

        # Store history
        self.threat_history.append(self.threat_state.copy())
        self.interceptor_history.append(self.interceptor_state.copy())

        # Calculate reward
        reward = self._calculate_reward(action, miss_dist)

        # Check termination conditions
        terminated = False
        truncated = False

        # Success: proximity-fuze style "close enough" rather than a literal
        # collision (see config.INTERCEPT_RADIUS)
        if miss_dist < self.config.INTERCEPT_RADIUS:
            # Sized well above the dense per-step distance term's typical
            # accumulated magnitude (~-150 over a full non-intercepting
            # episode, post dt-scaling) so a successful intercept is
            # unambiguously the best outcome even late in an episode,
            # rather than merely "less negative" than not intercepting.
            reward += 300.0  # Bonus for successful intercept
            terminated = True

        # Failure: threat reached protected zone
        threat_to_target = np.linalg.norm(self.threat_state[:3] -
                                          np.array(self.config.PROTECTED_ZONE))
        if threat_to_target < self.config.PROTECTED_ZONE_RADIUS:
            reward -= 50.0  # Penalty for failure
            terminated = True

        # Check if threat is getting too far
        if miss_dist > 100:
            reward -= 20.0
            terminated = True

        # Time limit
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        # Get observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Construct the observation vector."""
        rel_pos = self.threat_state[:3] - self.interceptor_state[:3]
        rel_vel = self.threat_state[3:] - self.interceptor_state[3:]
        range_mag = np.linalg.norm(rel_pos)

        # Threat acceleration (estimated from path)
        try:
            threat_accel = self.threat_path.get_acceleration_at_time(
                self.step_count * self.dt
            )
        except:
            threat_accel = np.zeros(3)

        # Closing velocity (negative when approaching)
        if range_mag > 0:
            closing_vel = -np.dot(rel_vel, rel_pos / range_mag)
        else:
            closing_vel = 0

        # Normalize time
        time_norm = self.step_count / self.max_steps

        # Combine observations
        obs = np.concatenate([
            rel_pos,  # 3
            rel_vel,  # 3
            threat_accel,  # 3
            self.interceptor_state[:3],  # 3
            [range_mag],  # 1
            [closing_vel],  # 1
            [time_norm],  # 1
            self.prev_action  # 3
        ])

        return obs.astype(np.float32)

    def _get_info(self):
        """Get additional information for logging."""
        return {
            'miss_distance': self.miss_distances[-1] if self.miss_distances else 0,
            'best_miss_distance': self.best_miss_distance,
            'step': self.step_count
        }

    def _calculate_reward(self, action, miss_dist):
        """Calculate the reward for the current step."""
        reward = 0.0

        # Primary reward: negative miss distance (closer is better)
        reward -= 0.1 * miss_dist

        # Bonus for reducing miss distance quickly
        if len(self.miss_distances) > 1:
            reduction = self.miss_distances[-2] - miss_dist
            if reduction > 0:
                reward += 0.5 * reduction

        # rel_pos/range_mag computed unconditionally (not just inside the
        # closing-course check below) since the drift-away penalty needs
        # them regardless of episode step count or exact-zero miss_dist.
        rel_pos = self.threat_state[:3] - self.interceptor_state[:3]
        range_mag = np.linalg.norm(rel_pos)

        # Bonus for being on intercept course (range decreasing)
        if len(self.miss_distances) > 1 and miss_dist > 0:
            rel_vel = self.threat_state[3:] - self.interceptor_state[3:]
            if np.dot(rel_pos, rel_vel) < 0:  # Closing
                reward += 0.1

        # Penalty for drifting away from the threat when idle: the
        # interceptor's own velocity component pointing away from the
        # threat (independent of what the threat itself is doing, unlike
        # the closing-course bonus above which uses relative velocity).
        # Without this, a policy trained in an environment where most
        # episodes resolve quickly (fast threat drift + reactive evasion
        # mean most engagements are short) gets little experience with an
        # extended "threat hasn't been caught yet, keep pressing" phase,
        # and observed behavior was to wander off with increasing speed
        # once an engagement ran past a quick resolution -- there was
        # nothing in training to teach it that abandoning pursuit is bad
        # if the reward stops accumulating fast negative miss-distance
        # penalties anyway once it's already far away.
        if range_mag > 1e-6:
            own_radial_vel = np.dot(self.interceptor_state[3:], rel_pos) / range_mag
            if own_radial_vel < 0:
                reward -= 0.2 * abs(own_radial_vel)

        # Control effort penalty (minimize acceleration)
        reward -= 0.001 * np.linalg.norm(action)

        # Smoothness penalty (penalize large changes in action)
        if hasattr(self, 'prev_action') and len(action) > 0:
            reward -= 0.001 * np.linalg.norm(action - self.prev_action)

        # Bonus for being close to threat. Layered (additive, not elif) so
        # the gradient gets progressively steeper the closer the agent gets
        # to the 2m intercept threshold, instead of a single coarse bonus
        # that's roughly flat over the last 10m where precision matters most.
        reward += max(0.0, 20.0 - miss_dist) * 0.1
        reward += max(0.0, 10.0 - miss_dist) * 0.1
        reward += max(0.0, 5.0 - miss_dist) * 0.6

        # Penalty for altitude mismatch
        alt_diff = abs(self.threat_state[2] - self.interceptor_state[2])
        reward -= 0.01 * alt_diff

        # Scale the dense per-step shaping reward to a per-simulated-second
        # basis. With SIM_DT=0.01 there are 100 integration steps per
        # second, so leaving this unscaled let the miss-distance term
        # accumulate ~100x faster than the weights were tuned for, drowning
        # out the +/-50 terminal intercept bonus/penalty over any episode
        # longer than a few hundred steps.
        reward *= self.dt

        return reward

    def render(self, mode='human'):
        """Render the environment."""
        # This would be implemented if using a visualizer
        pass

    def close(self):
        """Clean up."""
        pass


class EngagementCurriculumCallback(BaseCallback):
    """
    Linearly widens the threat spawn distance band (InterceptorEnv's
    engagement_min/max) over the course of training, from a short, easy
    range up to the full operational range.

    A sparse terminal bonus for reaching a rarely-visited state (miss_dist
    < 2m) provides zero learning signal until the policy actually reaches
    that state at least once. Starting episodes close lets the agent
    experience real intercepts early and get gradient from them, then the
    band widens as it gets better, so it isn't stuck training exclusively
    on the hardest version of the task from step one.
    """

    def __init__(self, total_timesteps, start_range=(15.0, 30.0),
                 end_range=(40.0, 80.0), verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_range = start_range
        self.end_range = end_range

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        min_dist = self.start_range[0] + progress * (self.end_range[0] - self.start_range[0])
        max_dist = self.start_range[1] + progress * (self.end_range[1] - self.start_range[1])

        # VecEnv.env_method() would call getattr() on the Monitor wrapper
        # directly, which doesn't forward unknown methods to the wrapped
        # InterceptorEnv (see the .unwrapped note in InterceptorAI.evaluate),
        # so reach into the env list and unwrap explicitly instead.
        venv = self.training_env
        base_venv = venv.venv if hasattr(venv, "venv") else venv
        for env in base_venv.envs:
            env.unwrapped.set_engagement_range(min_dist, max_dist)

        return True


class InterceptorAI:
    """
    AI-powered interceptor using PPO.
    Can be trained and then used for inference.
    """

    def __init__(self, config, model_path=None):
        """
        Initialize the AI interceptor.

        Args:
            config: Configuration object
            model_path: Path to saved model (optional)
        """
        self.config = config
        self.model_path = model_path
        self.model = None
        self.env = None
        self.vec_env = None
        self.vec_normalize = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def create_env(self, use_jink=True):
        """
        Create the training environment.

        Args:
            use_jink: Whether to use jinking threat paths
        """
        env = InterceptorEnv(self.config, use_jink=use_jink)
        env = Monitor(env, "./logs/")
        return env

    def train(self, total_timesteps=200000, save_path="./models/",
              auto_promote=True, models_dir="./models/", promotion_eval_episodes=100):
        """
        Train the PPO agent.

        Args:
            total_timesteps: Number of timesteps to train
            save_path: Directory to save models
            auto_promote: If True (default), evaluate the freshly-trained
                model against whatever is currently in models_dir after
                training completes, and promote it (copy final_model.zip +
                vec_normalize.pkl into models_dir, update the registry) if
                it's actually better. This matters because every other
                module in this codebase (run_sim.py, compare_ai_vs_pn, the
                interactive menu, hybrid_guidance.py's test functions) all
                hardcode models_dir/final_model.zip as the model they load —
                training a better model does nothing for the rest of the
                sim until it's promoted here. Set False for quick
                smoke-test runs you don't want influencing the "best model."
            models_dir: Directory treated as the canonical "best model"
                location (see promote_if_best).
            promotion_eval_episodes: Episode count for the comparison eval.
        """
        print("=" * 60)
        print("Training AI Interceptor with PPO")
        print("=" * 60)

        # Create environment with randomization for generalization
        def make_env():
            return self.create_env(use_jink=True)

        # Vectorized environment for parallel training
        self.vec_env = DummyVecEnv([make_env for _ in range(4)])
        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True)

        # Create logs directory
        os.makedirs(save_path, exist_ok=True)
        os.makedirs("./logs/", exist_ok=True)

        # Evaluation environment (without jink for consistent evaluation)
        eval_env = DummyVecEnv([lambda: self.create_env(use_jink=False)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

        # Callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path="./logs/",
            eval_freq=10000,
            deterministic=True,
            render=False,
            n_eval_episodes=10
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=25000,
            save_path=save_path,
            name_prefix="ppo_interceptor"
        )

        # Curriculum: start with an easy, close-in engagement band on the
        # training env (eval_env stays fixed at the full range set in
        # InterceptorEnv.__init__, so eval numbers stay comparable across
        # training).
        curriculum_callback = EngagementCurriculumCallback(total_timesteps)

        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./logs/",
            verbose=1,
            device='auto'
        )

        print("\nStarting training...")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Using {self.vec_env.num_envs} parallel environments")
        print("-" * 60)

        # Train
        start_time = datetime.now()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback, curriculum_callback],
            progress_bar=True
        )
        training_time = datetime.now() - start_time

        print("-" * 60)
        print(f"Training complete! Time: {training_time}")
        print(f"Model saved to: {save_path}")
        print("=" * 60)

        # Save final model
        self.model.save(os.path.join(save_path, "final_model"))
        self.vec_env.save(os.path.join(save_path, "vec_normalize.pkl"))

        if auto_promote and os.path.abspath(save_path) != os.path.abspath(models_dir):
            promote_if_best(save_path, models_dir=models_dir,
                             n_eval_episodes=promotion_eval_episodes)

        return self.model

    def load_model(self, model_path):
        """
        Load a trained model.

        Args:
            model_path: Path to the saved model
        """
        print(f"Loading model from: {model_path}")
        self.model = PPO.load(model_path, device='auto')
        self.model_path = model_path

        # Training wraps the env in VecNormalize(norm_obs=True), so the
        # policy expects normalized observations. Load the matching running
        # stats (saved alongside the model as vec_normalize.pkl) so
        # predict() can normalize raw observations the same way, instead of
        # handing the policy inputs far outside the distribution it was
        # trained on. Without this, HybridGuidance/AIInterceptorSimulation
        # (i.e. run_sim.py, visualizer_pygame.py, ai_only/blended modes)
        # feed the model unnormalized observations directly -- this is what
        # caused ai_only mode to diverge in a run_sim.py side-by-side test
        # while blended (partially averaged with PN's non-learned control
        # law) and evaluate() (which already normalizes) stayed stable.
        self.vec_normalize = None
        vec_normalize_path = os.path.join(os.path.dirname(model_path) or ".", "vec_normalize.pkl")
        if os.path.exists(vec_normalize_path):
            dummy_env = DummyVecEnv([lambda: self.create_env()])
            self.vec_normalize = VecNormalize.load(vec_normalize_path, dummy_env)
            self.vec_normalize.training = False

        return self.model

    def predict(self, observation, deterministic=True):
        """
        Get action from the trained model.

        Args:
            observation: The observation vector
            deterministic: Whether to use deterministic actions

        Returns:
            action: The action to take
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        obs = np.asarray(observation, dtype=np.float32)
        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(obs)

        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def evaluate(self, n_episodes=100, render=False, use_jink=True):
        """
        Evaluate the trained model.

        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            use_jink: Whether to use jinking threats

        Returns:
            stats: Dictionary of evaluation statistics
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        print(f"\nEvaluating model over {n_episodes} episodes...")

        # Training wraps the env in VecNormalize(norm_obs=True), so the
        # policy was fit on normalized observations. If we hand it raw
        # observations here (as a bare InterceptorEnv would), the network
        # sees inputs far outside the distribution it was trained on and
        # produces meaningless actions. Reload the saved running obs
        # statistics (vec_normalize.pkl, saved next to the model) and apply
        # them here too, with training=False so eval doesn't perturb the
        # stats and norm_reward=False so reported rewards stay on the raw
        # per-episode scale.
        env = DummyVecEnv([lambda: self.create_env(use_jink=use_jink)])
        vec_normalize_path = None
        if self.model_path:
            candidate = os.path.join(os.path.dirname(self.model_path), "vec_normalize.pkl")
            if os.path.exists(candidate):
                vec_normalize_path = candidate

        if vec_normalize_path:
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False
            inner_env = env.venv.envs[0]
        else:
            print("  (No vec_normalize.pkl found next to the model — "
                  "evaluating with raw, unnormalized observations.)")
            inner_env = env.envs[0]

        success_count = 0
        miss_distances = []
        intercept_times = []
        total_rewards = []

        for episode in range(n_episodes):
            obs = env.reset()
            done = [False]
            total_reward = 0.0
            step_count = 0

            info = {}
            while not done[0]:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, infos = env.step(action)
                total_reward += reward[0]
                step_count += 1
                info = infos[0]

            # DummyVecEnv auto-resets the underlying InterceptorEnv the
            # instant an episode ends, so reading best_miss_distance off
            # inner_env here would return the *next* episode's freshly-reset
            # np.inf rather than the episode that just finished. The info
            # dict from the terminal step() call is captured before that
            # reset happens, so pull the value from there instead.
            miss_dist = info['best_miss_distance']
            miss_distances.append(miss_dist)
            total_rewards.append(total_reward)

            if miss_dist < self.config.INTERCEPT_RADIUS:
                success_count += 1
                intercept_times.append(step_count * inner_env.unwrapped.dt)

            # Progress indicator
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Best miss={miss_dist:.2f}m, "
                      f"Success rate={success_count / (episode + 1) * 100:.1f}%")

        # Calculate statistics
        stats = {
            'success_rate': success_count / n_episodes * 100,
            'mean_miss_distance': np.mean(miss_distances),
            'std_miss_distance': np.std(miss_distances),
            'best_miss': np.min(miss_distances),
            'worst_miss': np.max(miss_distances),
            'mean_intercept_time': np.mean(intercept_times) if intercept_times else np.inf,
            'mean_reward': np.mean(total_rewards),
            'success_count': success_count,
            'total_episodes': n_episodes
        }

        print("-" * 60)
        print("Evaluation Results:")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Mean Miss Distance: {stats['mean_miss_distance']:.2f} m")
        print(f"Best Miss: {stats['best_miss']:.2f} m")
        print(f"Mean Intercept Time: {stats['mean_intercept_time']:.2f} s")
        print(f"Mean Reward: {stats['mean_reward']:.1f}")
        print("=" * 60)

        env.close()
        return stats


def promote_if_best(candidate_dir, models_dir="./models/", n_eval_episodes=100,
                     best_miss_weight=0.6):
    """
    Evaluate the model in candidate_dir and, if it beats whatever is
    currently in models_dir, copy it in (best_model.zip + vec_normalize.pkl)
    and record it in models_dir/best_model_registry.json.

    The candidate is candidate_dir/best_model.zip — the checkpoint
    EvalCallback selected during training via periodic eval — rather than
    final_model.zip (whatever the policy happened to be at the very last
    timestep), since training isn't monotonic and the last snapshot isn't
    reliably the best one from that run.

    This is the single mechanism that makes a newly-trained model actually
    used elsewhere: run_sim.py, compare_ai_vs_pn(), the interactive menu,
    and hybrid_guidance.py's test functions all load config.BEST_MODEL_PATH
    (== models_dir/best_model.zip by default) as the model they use —
    training a better model in some other directory has no effect on any of
    them until it's promoted here.

    Ranking is by success_rate first (an actual intercept always wins), then
    by a weighted score combining best_miss and mean_miss_distance:
        score = best_miss_weight * best_miss + (1 - best_miss_weight) * mean_miss_distance
    (lower is better). mean_miss_distance alone is dominated by
    episode-to-episode spawn geometry (a 40-80m randomized engagement band
    means some episodes are just much harder than others) rather than
    actual guidance skill, while success_rate has been 0% across every run
    so far and so can't discriminate at all. best_miss — the closest
    approach achieved in any evaluated episode — is a noisy single-sample
    statistic on its own, but it's a more direct signal of how close the
    policy can actually get under favorable conditions, which is why it's
    weighted more heavily (default 0.6) rather than used alone.

    Evaluation uses run_sim.evaluate_blended() — the full deployment loop
    (sensor noise, Kalman filtering, blended PN+AI guidance) — not
    InterceptorAI.evaluate()'s pure-AI-only loop. This used to use
    evaluate() and it produced a real, costly mistake: a candidate that
    fixed a genuine behavioral bug (wandering away from the target during
    extended engagements) scored *worse* on pure-AI evaluate() (3.3% vs the
    incumbent's 40%) purely because pure-AI performance doesn't predict
    blended performance, while the same candidate scored dramatically
    *better* on evaluate_blended() (94% vs 89%) — the metric that actually
    reflects how the system is used. The gate would have silently rejected
    the better model. Imported lazily (not at module level) to avoid a
    circular import: run_sim imports hybrid_guidance, which imports
    InterceptorAI from this module.

    Both stats come from evaluate_blended(), which runs randomized episodes
    without a fixed seed, so this is a noisy comparison, not a rigorous
    statistical test — treat n_eval_episodes as a knob to trade eval time
    for comparison stability, not a guarantee.

    Returns True if candidate_dir was promoted, False otherwise.
    """
    from run_sim import evaluate_blended

    candidate_model = os.path.join(candidate_dir, "best_model.zip")
    if not os.path.exists(candidate_model):
        print(f"No best_model.zip in {candidate_dir}, skipping promotion.")
        return False

    print(f"\nEvaluating candidate model in {candidate_dir} for promotion "
          f"against {models_dir} (blended, deployment-realistic)...")
    candidate_stats = evaluate_blended(
        config, model_path=candidate_model, n_episodes=n_eval_episodes
    )

    def score(stats):
        # Lower is better. success_rate is compared separately (and always
        # decides first), so this only has to rank models within the same
        # success_rate tier.
        return (best_miss_weight * stats['best_miss'] +
                (1 - best_miss_weight) * stats['mean_miss_distance'])

    registry_path = os.path.join(models_dir, "best_model_registry.json")
    current_model = os.path.join(models_dir, "best_model.zip")

    if os.path.exists(registry_path):
        with open(registry_path) as f:
            best = json.load(f)
        current_success = best.get('success_rate', -1.0)
        current_score = best.get('score')
        if current_score is None:
            # Registry predates the weighted-score criterion (mean-miss-only).
            current_score = best.get('mean_miss_distance', np.inf)
    elif os.path.exists(current_model):
        # A model already lives in models_dir but was never put there by
        # this mechanism (e.g. a direct InterceptorAI.train(save_path=
        # models_dir) call, which skips promote_if_best entirely since
        # candidate_dir would equal models_dir) -- evaluate it once to get
        # a real baseline instead of blindly overwriting it.
        #
        # CAUTION: this assumes vec_normalize.pkl next to current_model
        # actually belongs to it. If models_dir/best_model.zip and
        # models_dir/vec_normalize.pkl ever end up from two different
        # training runs (e.g. something outside this function copied one
        # but not the other), evaluate() will silently score the policy
        # under the wrong observation normalization -- this happened once
        # during development, when an old promotion convention updated
        # vec_normalize.pkl but left an unrelated pre-existing best_model.zip
        # in place, and produced a bogus "4% success rate" baseline. Only
        # ever write these two files together (as the promotion branch below
        # does) to keep that from recurring.
        print(f"No registry found for the existing model in {models_dir}; "
              f"evaluating it as the baseline...")
        current_stats = evaluate_blended(
            config, model_path=current_model, n_episodes=n_eval_episodes
        )
        current_success = current_stats['success_rate']
        current_score = score(current_stats)
    else:
        current_success = -1.0
        current_score = np.inf

    candidate_score = score(candidate_stats)
    is_better = (
        candidate_stats['success_rate'] > current_success or
        (candidate_stats['success_rate'] == current_success and
         candidate_score < current_score)
    )

    if is_better:
        os.makedirs(models_dir, exist_ok=True)
        shutil.copy(candidate_model, os.path.join(models_dir, "best_model.zip"))
        vec_norm_src = os.path.join(candidate_dir, "vec_normalize.pkl")
        if os.path.exists(vec_norm_src):
            shutil.copy(vec_norm_src, os.path.join(models_dir, "vec_normalize.pkl"))

        registry = {
            'source_dir': candidate_dir,
            'evaluation': 'blended (run_sim.evaluate_blended)',
            # evaluate_blended()'s stats are numpy float32/float64 (from
            # np.mean/np.min/np.max), which json.dump can't serialize.
            'success_rate': float(candidate_stats['success_rate']),
            'mean_miss_distance': float(candidate_stats['mean_miss_distance']),
            'best_miss': float(candidate_stats['best_miss']),
            'mean_intercept_time': float(candidate_stats['mean_intercept_time']),
            'score': float(candidate_score),
            'best_miss_weight': best_miss_weight,
            'n_eval_episodes': n_eval_episodes,
            'promoted_at': datetime.now().isoformat()
        }
        # Serialize fully before touching the file, so a bad value can't
        # leave a half-written, unparseable registry (as it did during
        # testing when a numpy float slipped through before the float()
        # casts above were added).
        registry_json = json.dumps(registry, indent=2)
        with open(registry_path, 'w') as f:
            f.write(registry_json)

        print(f"PROMOTED: {candidate_dir} -> {models_dir} "
              f"(success_rate={candidate_stats['success_rate']:.1f}%, "
              f"best_miss={candidate_stats['best_miss']:.2f}m, "
              f"mean_miss={candidate_stats['mean_miss_distance']:.2f}m, "
              f"score={candidate_score:.2f})")
        return True
    else:
        print(f"NOT promoted: {candidate_dir} "
              f"(success_rate={candidate_stats['success_rate']:.1f}%, "
              f"best_miss={candidate_stats['best_miss']:.2f}m, "
              f"mean_miss={candidate_stats['mean_miss_distance']:.2f}m, "
              f"score={candidate_score:.2f}) "
              f"did not beat the current best in {models_dir} "
              f"(success_rate={current_success:.1f}%, score={current_score:.2f})")
        return False


class AIInterceptorSimulation:
    """
    Wrapper to use the AI interceptor in the main Hlin simulation.
    """

    def __init__(self, config, model_path=None):
        """
        Initialize the AI interceptor simulation.

        Args:
            config: Configuration object
            model_path: Path to trained model (optional)
        """
        self.config = config
        self.ai = InterceptorAI(config, model_path)
        self.env = None
        self.obs = None

    def initialize(self):
        """Create the environment and reset."""
        self.env = InterceptorEnv(self.config, use_jink=True)
        self.obs, _ = self.env.reset()
        return self.obs

    def step(self, action=None):
        """
        Take a step in the simulation.

        Args:
            action: Action to take (if None, use AI prediction)

        Returns:
            obs, reward, done, truncated, info
        """
        if self.obs is None:
            self.initialize()

        if action is None and self.ai.model is not None:
            action = self.ai.predict(self.obs, deterministic=True)
        elif action is None:
            # Random action if no model
            action = self.env.action_space.sample()

        self.obs, reward, done, truncated, info = self.env.step(action)
        return self.obs, reward, done, truncated, info

    def get_state(self):
        """Get the current state of the simulation."""
        if self.env is None:
            return None, None
        return self.env.threat_state, self.env.interceptor_state

    def get_best_miss(self):
        """Get the best miss distance so far."""
        return self.env.best_miss_distance if self.env else np.inf


def train_ai_interceptor():
    """Train the AI interceptor."""
    ai = InterceptorAI(config)
    ai.train(total_timesteps=config.RL_TOTAL_TIMESTEPS)
    return ai


def compare_ai_vs_pn():
    """
    Compare AI interceptor performance against PN guidance.
    """
    print("\n" + "=" * 70)
    print("AI vs PN Guidance Comparison")
    print("=" * 70)

    from run_sim import DroneDefenseSimulationHybrid
    from pn_guidance import ProportionalNavigation
    from tracking import KalmanFilter, radar_sensor

    # Test configurations
    test_scenarios = [
        {'jink_amp': 1.0, 'jink_freq': 1.0},
        {'jink_amp': 2.0, 'jink_freq': 1.5},
        {'jink_amp': 3.0, 'jink_freq': 2.0},
        {'jink_amp': 4.0, 'jink_freq': 1.0},
        {'jink_amp': 2.0, 'jink_freq': 3.0},
    ]

    results = []

    for scenario in test_scenarios:
        # Set parameters
        config.THREAT_JINK_AMPLITUDE = scenario['jink_amp']
        config.THREAT_JINK_FREQUENCY = scenario['jink_freq']

        print(f"\nScenario: Jink Amp={scenario['jink_amp']:.1f}m, "
              f"Freq={scenario['jink_freq']:.1f}rad/s")
        print("-" * 40)

        # --- PN Guidance ---
        print("Running PN guidance...")
        sim_pn = DroneDefenseSimulationHybrid(config)
        sim_pn.run()
        miss_pn = sim_pn.miss_distance
        print(f"  PN miss distance: {miss_pn:.2f}m")

        # --- AI Guidance ---
        print("Running AI guidance...")

        # Load trained model
        model_path = config.BEST_MODEL_PATH
        if not os.path.exists(model_path):
            print("  No trained model found. Skipping AI evaluation.")
            miss_ai = np.inf
        else:
            ai = InterceptorAI(config, model_path)

            # Run evaluation
            env = InterceptorEnv(config, use_jink=True)
            obs, _ = env.reset()
            done = False
            truncated = False

            while not (done or truncated):
                action = ai.predict(obs, deterministic=True)
                obs, _, done, truncated, _ = env.step(action)

            miss_ai = env.best_miss_distance
            env.close()
            print(f"  AI miss distance: {miss_ai:.2f}m")

        # Calculate improvement
        if miss_ai < np.inf and miss_pn > 0:
            improvement = (miss_pn - miss_ai) / miss_pn * 100
        else:
            improvement = 0

        results.append({
            'jink_amp': scenario['jink_amp'],
            'jink_freq': scenario['jink_freq'],
            'miss_pn': miss_pn,
            'miss_ai': miss_ai,
            'improvement': improvement
        })

    # Display results table
    print("\n" + "=" * 70)
    print("SUMMARY: AI vs PN Guidance")
    print("=" * 70)
    print(f"{'Jink Amp':>10} {'Jink Freq':>10} {'PN (m)':>12} {'AI (m)':>12} {'Improvement':>12}")
    print("-" * 70)
    for r in results:
        ai_str = f"{r['miss_ai']:.2f}" if r['miss_ai'] < np.inf else "N/A"
        imp_str = f"{r['improvement']:.1f}%" if r['miss_ai'] < np.inf else "N/A"
        print(f"{r['jink_amp']:>10.1f} {r['jink_freq']:>10.2f} "
              f"{r['miss_pn']:>12.2f} {ai_str:>12} {imp_str:>12}")

    return results


def main():
    """Main entry point for AI interceptor."""
    print("=" * 60)
    print("HLIN: AI Interceptor for Drone Defense")
    print("=" * 60)

    import sys
    import os

    # Check if model exists
    model_exists = os.path.exists(config.BEST_MODEL_PATH)

    print("\nOptions:")
    print("1. Train AI interceptor from scratch")
    print("2. Evaluate trained AI interceptor")
    print("3. Compare AI vs PN guidance")
    print("4. Run interactive AI simulation (with Pygame)")

    if not model_exists:
        print("\n⚠️  No trained model found. You should train first (Option 1).")

    choice = input("\nEnter choice (1-4): ")

    if choice == '1':
        train_ai_interceptor()

    elif choice == '2':
        if model_exists:
            ai = InterceptorAI(config, config.BEST_MODEL_PATH)
            ai.evaluate(n_episodes=50)
        else:
            print("No trained model found. Train first with Option 1.")

    elif choice == '3':
        if model_exists:
            compare_ai_vs_pn()
        else:
            print("No trained model found. Train first with Option 1.")

    elif choice == '4':
        # Run with Pygame visualization
        try:
            from visualizer_pygame import HlinPygameSimulation

            # Create AI simulation
            if model_exists:
                ai_sim = AIInterceptorSimulation(config, config.BEST_MODEL_PATH)
            else:
                print("No trained model found. Using random actions.")
                ai_sim = AIInterceptorSimulation(config)

            # This would need to be adapted to work with the visualizer
            # For now, just run a quick demo
            print("Running AI simulation...")
            ai_sim.initialize()

            for i in range(1000):
                obs, reward, done, truncated, info = ai_sim.step()
                if done or truncated:
                    break

            print(f"Best miss distance: {ai_sim.get_best_miss():.2f}m")

        except ImportError:
            print("Pygame visualizer not available. Running console version.")
            ai_sim = AIInterceptorSimulation(config, config.BEST_MODEL_PATH if model_exists else None)
            ai_sim.initialize()

            # Run simulation
            while True:
                obs, reward, done, truncated, info = ai_sim.step()
                if done or truncated:
                    break

            print(f"Best miss distance: {ai_sim.get_best_miss():.2f}m")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()