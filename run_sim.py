"""
Main simulation runner with hybrid guidance support.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse

import config
from quad_dynamics import integrate_dynamics
from threat_path import ThreatPathGenerator
from position_controller import PositionController
from tracking import KalmanFilter, radar_sensor

# Import hybrid guidance
from hybrid_guidance import HybridGuidance, HybridGuidanceTrainer


class DroneDefenseSimulationHybrid:
    """
    Drone defense simulation with hybrid guidance (PN + AI).
    """

    def __init__(self, config, guidance_mode='blended', blend_weight=0.5,
                 model_path=None, use_adaptive=False, guidance=None,
                 randomize_scenario=False, engagement_min=40.0,
                 engagement_max=80.0, use_jink=True):
        """
        Initialize simulation with hybrid guidance.

        Args:
            config: Configuration object
            guidance_mode: 'pn_only', 'ai_only', 'blended', 'adaptive'
            blend_weight: Blend weight for blended mode
            model_path: Path to AI model
            use_adaptive: Use adaptive N mode
            guidance: An already-constructed HybridGuidance (or
                trainer.hybrid) instance to reuse instead of building a new
                one. Pass this when running many episodes back-to-back
                (e.g. evaluate_blended()) so the PPO model isn't reloaded
                from disk every episode; call guidance.reset_episode()
                yourself between runs if you do (run() also calls it).
                When set, guidance_mode/blend_weight/model_path/use_adaptive
                are ignored.
            randomize_scenario: If True, sample the threat's spawn distance/
                angle from the protected zone and its jink parameters the
                same way InterceptorEnv.reset() does for RL training/eval,
                instead of using the fixed config.THREAT_INITIAL_POS
                scenario. Needed to get a representative sample of outcomes
                across many episodes rather than repeating one fixed
                engagement.
            engagement_min/engagement_max: Threat spawn distance band (m)
                when randomize_scenario=True.
            use_jink: Whether to randomize jink amplitude/frequency per
                episode when randomize_scenario=True (mirrors
                InterceptorEnv.reset()).
        """
        self.config = config
        self.dt = config.SIM_DT
        self.duration = config.SIM_DURATION
        self.num_steps = int(self.duration / self.dt)

        # Initialize components
        self.threat_path = ThreatPathGenerator(config)
        self.position_controller = PositionController(config)

        # Initialize hybrid guidance
        if guidance is not None:
            self.guidance = guidance
        elif use_adaptive:
            # Load adaptive N network if trained
            trainer = HybridGuidanceTrainer(config)
            trainer.load_adaptor("./models/n_adaptor.pth")
            self.guidance = trainer.hybrid
            self.guidance.mode = 'adaptive'
            print("Using adaptive N guidance")
        else:
            self.guidance = HybridGuidance(
                config,
                mode=guidance_mode,
                model_path=model_path or config.BEST_MODEL_PATH,
                blend_weight=blend_weight
            )
            print(f"Using {guidance_mode} guidance (blend={blend_weight:.2f})")

        # Kalman filter
        self.kalman = KalmanFilter(
            dt=config.KALMAN_DT,
            q_pos=config.KALMAN_Q_POS,
            q_vel=config.KALMAN_Q_VEL,
            r_pos=config.KALMAN_R_POS
        )

        # Logging
        self.log = {
            'time': [],
            'threat_state': [],
            'interceptor_state': [],
            'threat_desired': [],
            'kalman_estimate': [],
            'measurements': [],
            'miss_distance': [],
            'guidance_metadata': [],
            'pn_accel': [],
            'ai_accel': [],
            'final_accel': []
        }

        # State initialization
        if randomize_scenario:
            # Mirrors InterceptorEnv.reset() (rl_interceptor.py) so
            # evaluate_blended() sees the same distribution of engagements
            # the RL model was trained/evaluated against.
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(engagement_min, engagement_max)
            threat_x = distance * np.cos(angle)
            threat_y = distance * np.sin(angle)
            threat_z = 20 + np.random.uniform(-10, 10)
            self.threat_state = np.array([threat_x, threat_y, threat_z, 0.0, 0.0, 0.0])

            # Re-anchor the drift trajectory to this episode's randomized
            # spawn (see the matching fix/comment in InterceptorEnv.reset())
            # instead of leaving it at the fixed config.THREAT_INITIAL_POS.
            self.threat_path.initial_pos = self.threat_state[:3].copy()

            if use_jink:
                self.threat_path.jink_amplitude = np.random.uniform(1.0, 4.0)
                self.threat_path.jink_frequency = np.random.uniform(0.5, 3.0)

            self.interceptor_state = np.array([
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(0, 5),
                0.0, 0.0, 0.0
            ])
        else:
            self.threat_state = np.array([
                config.THREAT_INITIAL_POS[0],
                config.THREAT_INITIAL_POS[1],
                config.THREAT_INITIAL_POS[2],
                0, 0, 0
            ])

            self.interceptor_state = np.array([
                config.INTERCEPTOR_INITIAL_POS[0],
                config.INTERCEPTOR_INITIAL_POS[1],
                config.INTERCEPTOR_INITIAL_POS[2],
                config.INTERCEPTOR_INITIAL_VEL[0],
                config.INTERCEPTOR_INITIAL_VEL[1],
                config.INTERCEPTOR_INITIAL_VEL[2]
            ])

        # Tracking
        self.miss_distance = np.inf
        self.time_of_closest_approach = 0
        # Set once run() resolves the engagement (mirrors InterceptorEnv's
        # termination conditions): 'intercept', 'threat_reached_target',
        # 'too_far', or 'timeout' (ran the full duration unresolved).
        self.success = False
        self.termination_reason = None

    def run(self, verbose=True):
        """
        Run the simulation until the engagement resolves (intercept,
        failure, or the threat getting too far away) or time runs out.

        Args:
            verbose: Print progress/summary output. Set False for batch
                evaluation (evaluate_blended()) to avoid flooding the
                console over many episodes.
        """
        if verbose:
            print("\nStarting simulation...")

        # In case this sim reuses a HybridGuidance instance across
        # episodes (see the `guidance` constructor arg), clear its
        # step-counter/prev-action memory so this run starts clean.
        self.guidance.reset_episode()

        # Initialize Kalman
        first_measurement = radar_sensor(
            self.threat_state[:3],
            self.config.RADAR_NOISE_STD
        )
        self.kalman.initialize(first_measurement)

        for step in range(self.num_steps):
            t = step * self.dt

            # --- Threat Drone --- (reactive evasion: steers away from the
            # interceptor's actual live position once close enough, on top
            # of the scripted drift/sway/jink)
            threat_desired = self.threat_path.get_desired_position(
                t, threat_pos=self.threat_state[:3],
                interceptor_pos=self.interceptor_state[:3]
            )
            threat_control = self.position_controller.compute_control(
                threat_desired,
                self.threat_state,
                desired_vel=self.threat_path.get_velocity_at_time(t)
            )

            self.threat_state = integrate_dynamics(
                self.threat_state,
                threat_control,
                self.dt,
                mass=self.config.MASS,
                g=self.config.G
            )

            # --- Sensor ---
            measurement = radar_sensor(
                self.threat_state[:3],
                self.config.RADAR_NOISE_STD
            )

            # --- Kalman Filter ---
            self.kalman.predict_update(measurement)
            kalman_estimate = self.kalman.get_estimate()

            # --- Hybrid Guidance ---
            # Guidance acceleration is converted directly to [T, phi_cmd,
            # theta_cmd] (see ProportionalNavigation/HybridGuidance
            # .compute_control_command) instead of being routed through
            # PositionController via a kinematically-extrapolated desired
            # position/velocity — that path fed the interceptor's own
            # velocity back into the position-error term (positive feedback
            # on velocity) and caused it to run away past the target once it
            # built up real closing speed.
            interceptor_control, accel_cmd, metadata = self.guidance.compute_control_command(
                self.interceptor_state[:3],
                self.interceptor_state[3:],
                kalman_estimate[:3],
                kalman_estimate[3:]
            )

            self.interceptor_state = integrate_dynamics(
                self.interceptor_state,
                interceptor_control,
                self.dt,
                mass=self.config.MASS,
                g=self.config.G
            )

            # --- Logging ---
            miss_dist = np.linalg.norm(
                self.threat_state[:3] - self.interceptor_state[:3]
            )

            self.log['time'].append(t)
            self.log['threat_state'].append(self.threat_state.copy())
            self.log['interceptor_state'].append(self.interceptor_state.copy())
            self.log['threat_desired'].append(threat_desired)
            self.log['kalman_estimate'].append(kalman_estimate.copy())
            self.log['measurements'].append(measurement.copy())
            self.log['miss_distance'].append(miss_dist)
            self.log['guidance_metadata'].append(metadata)

            if 'pn_accel' in metadata:
                self.log['pn_accel'].append(metadata['pn_accel'])
            if 'ai_accel' in metadata:
                self.log['ai_accel'].append(metadata['ai_accel'])
            self.log['final_accel'].append(accel_cmd)

            # Track closest approach
            if miss_dist < self.miss_distance:
                self.miss_distance = miss_dist
                self.time_of_closest_approach = t

            # Progress
            if verbose and step % (self.num_steps // 10) == 0:
                progress = step / self.num_steps * 100
                mode = metadata.get('mode', 'unknown')
                print(f"Progress: {progress:.0f}%, Miss: {miss_dist:.2f}m, Mode: {mode}")

            # Resolution check (mirrors InterceptorEnv's termination
            # conditions in rl_interceptor.py, so success here means the
            # same thing it means during RL training/eval).
            if miss_dist < self.config.INTERCEPT_RADIUS:
                self.success = True
                self.termination_reason = 'intercept'
                break
            threat_to_target = np.linalg.norm(
                self.threat_state[:3] - np.array(self.config.PROTECTED_ZONE)
            )
            if threat_to_target < self.config.PROTECTED_ZONE_RADIUS:
                self.termination_reason = 'threat_reached_target'
                break
            if miss_dist > 100:
                self.termination_reason = 'too_far'
                break
        else:
            self.termination_reason = 'timeout'

        if verbose:
            print(f"\nSimulation complete! ({self.termination_reason})")
            print(f"Min miss distance: {self.miss_distance:.2f} m")
            print(f"Time of closest approach: {self.time_of_closest_approach:.2f} s")

        return self.log

    def plot_results(self, log=None):
        """Plot simulation results."""
        if log is None:
            log = self.log

        time = np.array(log['time'])
        threat_states = np.array(log['threat_state'])
        interceptor_states = np.array(log['interceptor_state'])
        miss_distances = np.array(log['miss_distance'])

        fig = plt.figure(figsize=(15, 10))

        # 1. 3D Trajectory
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(threat_states[:, 0], threat_states[:, 1], threat_states[:, 2],
                'r-', label='Threat', linewidth=2)
        ax1.plot(interceptor_states[:, 0], interceptor_states[:, 1], interceptor_states[:, 2],
                'b-', label='Interceptor', linewidth=2)
        ax1.scatter(*self.config.PROTECTED_ZONE, color='green', s=200, marker='*', label='Protected')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectories - Hybrid Guidance')
        ax1.legend()

        # 2. Miss Distance
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(time, miss_distances, 'k-', linewidth=2)
        ax2.axhline(y=self.config.INTERCEPT_RADIUS, color='r', linestyle='--', label='Intercept threshold')
        ax2.scatter(self.time_of_closest_approach, self.miss_distance,
                   color='red', s=100, label=f'Min: {self.miss_distance:.2f}m')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Miss Distance (m)')
        ax2.set_title('Miss Distance')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. XY Trajectory
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(threat_states[:, 0], threat_states[:, 1], 'r-', label='Threat', linewidth=2)
        ax3.plot(interceptor_states[:, 0], interceptor_states[:, 1], 'b-', label='Interceptor', linewidth=2)
        ax3.scatter(*self.config.PROTECTED_ZONE[:2], color='green', s=100, marker='*')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_title('XY Trajectory')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.axis('equal')

        # 4. Guidance Mode Distribution
        ax4 = fig.add_subplot(2, 3, 4)
        if len(log['guidance_metadata']) > 0:
            modes = [m.get('mode', 'unknown') for m in log['guidance_metadata']]
            mode_counts = {}
            for m in modes:
                mode_counts[m] = mode_counts.get(m, 0) + 1
            ax4.pie(mode_counts.values(), labels=mode_counts.keys(), autopct='%1.1f%%')
            ax4.set_title('Guidance Mode Distribution')

        # 5. Acceleration Comparison
        ax5 = fig.add_subplot(2, 3, 5)
        if len(log.get('pn_accel', [])) > 0:
            pn_accel = np.array(log['pn_accel'])
            ai_accel = np.array(log['ai_accel']) if len(log.get('ai_accel', [])) > 0 else None
            final_accel = np.array(log['final_accel'])

            ax5.plot(time[:len(pn_accel)], np.linalg.norm(pn_accel, axis=1),
                    'r-', label='PN', alpha=0.7)
            if ai_accel is not None:
                ax5.plot(time[:len(ai_accel)], np.linalg.norm(ai_accel, axis=1),
                        'g-', label='AI', alpha=0.7)
            ax5.plot(time[:len(final_accel)], np.linalg.norm(final_accel, axis=1),
                    'b-', label='Final', linewidth=2)
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Acceleration Magnitude (m/s²)')
            ax5.set_title('Guidance Accelerations')
            ax5.grid(True, alpha=0.3)
            ax5.legend()

        # 6. Adaptive N (if available)
        ax6 = fig.add_subplot(2, 3, 6)
        if len(log['guidance_metadata']) > 0:
            n_values = [m.get('adapted_N', m.get('pn_N', 4.0))
                       for m in log['guidance_metadata']]
            ax6.plot(time[:len(n_values)], n_values, 'purple', linewidth=2)
            ax6.axhline(y=config.PN_NAVIGATION_CONSTANT, color='r', linestyle='--',
                       label='Default N', alpha=0.7)
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Navigation Constant N')
            ax6.set_title('Adaptive N (if active)')
            ax6.grid(True, alpha=0.3)
            ax6.legend()

        plt.tight_layout()
        plt.show()

        return fig


def evaluate_blended(config, model_path=None, guidance_mode='blended',
                      blend_weight=0.5, n_episodes=50, use_jink=True,
                      engagement_min=40.0, engagement_max=80.0):
    """
    Evaluate a guidance mode over randomized engagement scenarios using the
    full deployment-realistic simulation loop (radar sensor noise, Kalman
    filtering, HybridGuidance's PN+AI blending) rather than InterceptorEnv's
    pure-AI-control training/eval loop.

    InterceptorAI.evaluate() can only measure the AI driving the
    interceptor alone (equivalent to guidance_mode='ai_only' here, minus
    the sensor/Kalman realism) — it can't tell you how the model performs
    in the configuration it's actually meant to be used in, blended with
    PN. This runs that configuration directly, the same way run_sim.py's
    DroneDefenseSimulationHybrid does for a single engagement, repeated
    over many randomized scenarios (mirroring InterceptorEnv.reset()'s
    engagement-band sampling, so results are comparable to
    InterceptorAI.evaluate()'s).

    Args:
        config: Configuration object
        model_path: Path to AI model (defaults to config.BEST_MODEL_PATH)
        guidance_mode: 'pn_only', 'ai_only', or 'blended' (not 'adaptive' —
            that needs a trained N-adaptor network loaded separately)
        blend_weight: Blend weight for blended mode
        n_episodes: Number of randomized episodes to evaluate
        use_jink: Randomize jink amplitude/frequency per episode
        engagement_min/engagement_max: Threat spawn distance band (m)

    Returns:
        stats: dict with success_rate, mean_miss_distance,
            std_miss_distance, best_miss, worst_miss, mean_intercept_time,
            success_count, total_episodes — same shape as
            InterceptorAI.evaluate()'s return value for direct
            comparability.
    """
    print(f"\nEvaluating '{guidance_mode}' guidance (blend={blend_weight:.2f}) "
          f"over {n_episodes} randomized episodes...")

    # Built once and reused across every episode so the PPO model isn't
    # reloaded from disk n_episodes times; run() resets its per-episode
    # state (step_count, prev_ai_accel) via guidance.reset_episode().
    guidance = HybridGuidance(
        config, mode=guidance_mode,
        model_path=model_path or config.BEST_MODEL_PATH,
        blend_weight=blend_weight
    )

    success_count = 0
    miss_distances = []
    intercept_times = []

    for episode in range(n_episodes):
        sim = DroneDefenseSimulationHybrid(
            config, guidance=guidance,
            randomize_scenario=True, engagement_min=engagement_min,
            engagement_max=engagement_max, use_jink=use_jink
        )
        sim.run(verbose=False)

        miss_distances.append(sim.miss_distance)
        if sim.success:
            success_count += 1
            intercept_times.append(sim.time_of_closest_approach)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}: "
                  f"Best miss={sim.miss_distance:.2f}m, "
                  f"Success rate={success_count / (episode + 1) * 100:.1f}%")

    stats = {
        'success_rate': success_count / n_episodes * 100,
        'mean_miss_distance': float(np.mean(miss_distances)),
        'std_miss_distance': float(np.std(miss_distances)),
        'best_miss': float(np.min(miss_distances)),
        'worst_miss': float(np.max(miss_distances)),
        'mean_intercept_time': float(np.mean(intercept_times)) if intercept_times else np.inf,
        'success_count': success_count,
        'total_episodes': n_episodes
    }

    print("-" * 60)
    print(f"Evaluation Results ({guidance_mode}, blend={blend_weight:.2f}):")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    print(f"Mean Miss Distance: {stats['mean_miss_distance']:.2f} m")
    print(f"Best Miss: {stats['best_miss']:.2f} m")
    print(f"Mean Intercept Time: {stats['mean_intercept_time']:.2f} s")
    print("=" * 60)

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Hlin Hybrid Guidance Simulation')
    parser.add_argument('--mode', type=str, default='blended',
                       choices=['pn_only', 'ai_only', 'blended', 'adaptive'],
                       help='Guidance mode')
    parser.add_argument('--weight', type=float, default=0.5,
                       help='Blend weight for blended mode (0-1)')
    parser.add_argument('--model', type=str, default=config.BEST_MODEL_PATH,
                       help='Path to AI model')
    parser.add_argument('--visual', action='store_true',
                       help='Use Pygame visualization')
    parser.add_argument('--train', action='store_true',
                       help='Train adaptive N network first')

    args = parser.parse_args()

    if args.train:
        print("Training adaptive N network...")
        trainer = HybridGuidanceTrainer(config)
        trainer.generate_training_data(n_episodes=1000)
        trainer.train_adaptor(epochs=50)
        trainer.save_adaptor()
        return

    # Run simulation. randomize_scenario=True so repeated manual runs
    # sample a different engagement each time instead of always the same
    # fixed geometry (see evaluate_blended() for the properly-averaged
    # success rate across many randomized scenarios).
    sim = DroneDefenseSimulationHybrid(
        config,
        guidance_mode=args.mode,
        blend_weight=args.weight,
        model_path=args.model,
        use_adaptive=(args.mode == 'adaptive'),
        randomize_scenario=True
    )

    if args.visual:
        try:
            from visualizer_pygame import HlinPygameVisualizer
            # Visualization integration would go here
            print("Visual mode coming soon...")
            sim.run()
            sim.plot_results()
        except ImportError:
            print("Pygame not available. Running console mode.")
            sim.run()
            sim.plot_results()
    else:
        sim.run()
        sim.plot_results()


if __name__ == "__main__":
    main()