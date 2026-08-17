"""
Hybrid Guidance: Combines Proportional Navigation with AI corrections.
The AI learns to adjust PN outputs for better performance against evasive threats.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

import config
from pn_guidance import ProportionalNavigation
from rl_interceptor import InterceptorAI


class HybridGuidance:
    """
    Hybrid guidance that combines PN with AI corrections.

    Three modes:
    1. PN only: Pure proportional navigation
    2. AI only: Pure AI control
    3. Blended: PN + AI correction (default)
    4. Switchable: PN normally, AI when needed
    5. Adaptive: AI adjusts PN parameters
    """

    def __init__(self, config, mode='blended', model_path=None, blend_weight=0.5):
        """
        Initialize hybrid guidance.

        Args:
            config: Configuration object
            mode: 'pn_only', 'ai_only', 'blended', 'switchable', 'adaptive'
            model_path: Path to trained AI model (for AI modes)
            blend_weight: Weight for blending (0=PN only, 1=AI only)
        """
        self.config = config
        self.mode = mode
        self.blend_weight = blend_weight

        # Initialize PN guidance
        self.pn = ProportionalNavigation(config)

        # Initialize AI (if needed)
        self.ai = None
        if mode in ['ai_only', 'blended', 'switchable']:
            self.ai = InterceptorAI(config, model_path)

        # For adaptive mode
        self.n_adaptor = None
        if mode == 'adaptive':
            self.n_adaptor = NNAdaptor(config)

        # For switchable mode
        self.switch_threshold = 10.0  # Miss distance threshold for switching
        self.pn_performance = deque(maxlen=50)  # Track PN performance

        # Statistics
        self.pn_accel = np.zeros(3)
        self.ai_accel = np.zeros(3)
        self.final_accel = np.zeros(3)
        self.adaptation_history = []

        # Step/time tracking and previous-action memory, needed to build an
        # observation vector that matches InterceptorEnv._get_observation()
        # exactly (the AI policy's input layer is sized for that 18-dim
        # observation, so any mismatch here throws a shape error, or if the
        # dims happen to be compatible in size but wrong in meaning, silently
        # feeds the model garbage).
        self.step_count = 0
        self.max_steps = int(config.SIM_DURATION / config.SIM_DT)
        self.prev_ai_accel = np.zeros(3)

    def reset_episode(self):
        """
        Reset per-episode state (step counter, previous-action memory) so a
        single HybridGuidance instance can be reused across multiple
        episodes — e.g. by evaluate_blended() in run_sim.py, to avoid
        reloading the PPO model from disk for every episode — without
        leaking step_count/prev_ai_accel across the episode boundary.
        Leaking step_count in particular would make time_norm in the AI's
        observation start near 1.0 (immediately clipped to the "episode
        almost over" end of its range) for every episode after the first.
        """
        self.step_count = 0
        self.prev_ai_accel = np.zeros(3)

    def compute_guidance(self, interceptor_pos, interceptor_vel,
                         target_pos, target_vel, threat_accel=None):
        """
        Compute guidance command using hybrid approach.

        Args:
            interceptor_pos: [x, y, z] interceptor position
            interceptor_vel: [vx, vy, vz] interceptor velocity
            target_pos: [x, y, z] target position
            target_vel: [vx, vy, vz] target velocity
            threat_accel: [ax, ay, az] threat's current acceleration
                (e.g. from ThreatPathGenerator.get_acceleration_at_time(t)).
                Required for the AI observation to match InterceptorEnv's
                training observation; defaults to zeros if not supplied.

        Returns:
            commanded_accel: [ax, ay, az] commanded acceleration
            metadata: Dict with debugging info
        """
        if threat_accel is None:
            threat_accel = np.zeros(3)

        # 1. Compute PN guidance
        accel_pn, los_unit, los_rate = self.pn.compute_guidance(
            interceptor_pos, interceptor_vel, target_pos, target_vel
        )
        self.pn_accel = accel_pn

        # 2. Get AI guidance (if applicable)
        accel_ai = np.zeros(3)
        if self.ai is not None and self.ai.model is not None:
            # Construct observation for AI. This MUST match
            # InterceptorEnv._get_observation() field-for-field (same 8
            # components, same order, same 18 total dims) since that's the
            # observation space the policy's input layer was trained on.
            rel_pos = np.array(target_pos) - np.array(interceptor_pos)
            rel_vel = np.array(target_vel) - np.array(interceptor_vel)
            range_mag = np.linalg.norm(rel_pos)

            if range_mag > 0:
                closing_vel = -np.dot(rel_vel, rel_pos / range_mag)
            else:
                closing_vel = 0.0

            time_norm = min(self.step_count / self.max_steps, 1.0)

            obs = np.concatenate([
                rel_pos,                    # 3
                rel_vel,                    # 3
                np.array(threat_accel),     # 3
                np.array(interceptor_pos),  # 3
                [range_mag],                # 1
                [closing_vel],               # 1
                [time_norm],                 # 1
                self.prev_ai_accel           # 3
            ]).astype(np.float32)

            # Get AI action (acceleration command)
            ai_action = self.ai.predict(obs, deterministic=True)
            accel_ai = np.array(ai_action)
            self.ai_accel = accel_ai
            self.prev_ai_accel = accel_ai.copy()

        self.step_count += 1

        # 3. Combine based on mode
        if self.mode == 'pn_only':
            final_accel = accel_pn

        elif self.mode == 'ai_only':
            final_accel = accel_ai

        elif self.mode == 'blended':
            # Weighted blend of PN and AI
            final_accel = (1 - self.blend_weight) * accel_pn + self.blend_weight * accel_ai

        elif self.mode == 'adaptive':
            # AI adapts PN's navigation constant N
            adapted_n = self.n_adaptor.predict(interceptor_pos, interceptor_vel,
                                               target_pos, target_vel)
            # Use adapted N in PN
            self.pn.N = adapted_n
            final_accel, _, _ = self.pn.compute_guidance(
                interceptor_pos, interceptor_vel, target_pos, target_vel
            )
            self.adaptation_history.append(adapted_n)

        elif self.mode == 'switchable':
            # Use PN normally, switch to AI if performance degrades
            miss_dist = np.linalg.norm(np.array(target_pos) - np.array(interceptor_pos))
            self.pn_performance.append(miss_dist)

            # Check if PN is struggling (increasing miss distance)
            if len(self.pn_performance) > 10:
                is_degrading = np.mean(self.pn_performance) > self.switch_threshold
                if is_degrading:
                    final_accel = accel_ai  # Switch to AI
                else:
                    final_accel = accel_pn
            else:
                final_accel = accel_pn

        else:
            final_accel = accel_pn

        self.final_accel = final_accel

        # 4. Return with metadata
        metadata = {
            'pn_accel': accel_pn,
            'ai_accel': accel_ai,
            'final_accel': final_accel,
            'mode': self.mode,
            'blend_weight': self.blend_weight,
            'pn_N': self.pn.N if hasattr(self.pn, 'N') else config.PN_NAVIGATION_CONSTANT,
            'los_rate': los_rate,
            'miss_dist': np.linalg.norm(np.array(target_pos) - np.array(interceptor_pos))
        }

        if self.mode == 'adaptive' and hasattr(self, 'adaptation_history'):
            metadata['adapted_N'] = self.adaptation_history[
                -1] if self.adaptation_history else config.PN_NAVIGATION_CONSTANT

        return final_accel, metadata

    def compute_control_command(self, interceptor_pos, interceptor_vel,
                                target_pos, target_vel, dt=0.1, threat_accel=None):
        """
        Convert the blended guidance acceleration directly into [T,
        phi_cmd, theta_cmd], the same way rl_interceptor.py's
        InterceptorEnv.step() already does.

        This used to route the acceleration through PositionController by
        converting it into a kinematically-extrapolated desired_pos/vel
        (desired_pos = pos + vel*dt + ..., desired_vel = vel + accel*dt).
        That fed the interceptor's own current velocity back into
        PositionController's proportional term (pos_error ~= vel*dt), which
        is positive feedback on velocity: the commanded acceleration grew
        with however fast the interceptor was already moving, regardless of
        target position, causing it to run away and overshoot the target
        once real closing speed built up (see pn_guidance.py's version of
        this method for the full derivation). Converting directly to
        tilt/thrust removes that loop entirely.

        Args:
            interceptor_pos: [x, y, z] interceptor position
            interceptor_vel: [vx, vy, vz] interceptor velocity
            target_pos: [x, y, z] target position
            target_vel: [vx, vy, vz] target velocity
            dt: Unused, kept for call-signature compatibility with the
                previous position/velocity-based version.
            threat_accel: [ax, ay, az] threat's current acceleration, passed
                through to compute_guidance for the AI observation

        Returns:
            control: [T, phi_cmd, theta_cmd] ready for integrate_dynamics
            accel_cmd: [ax, ay, az] commanded acceleration
            metadata: Dict with debugging info
        """
        # Get guidance command
        accel_cmd, metadata = self.compute_guidance(
            interceptor_pos, interceptor_vel, target_pos, target_vel, threat_accel
        )

        g = self.config.G
        mass = self.config.MASS
        # The interceptor's own (higher) envelope, not the threat's
        # MAX_TILT/MAX_THRUST -- see config.py for why these are separate.
        max_tilt = getattr(self.config, 'INTERCEPTOR_MAX_TILT', self.config.MAX_TILT)
        max_thrust = getattr(self.config, 'INTERCEPTOR_MAX_THRUST',
                              getattr(self.config, 'MAX_THRUST', None))

        theta_cmd = np.clip(accel_cmd[0] / g, -max_tilt, max_tilt)
        phi_cmd = np.clip(-accel_cmd[1] / g, -max_tilt, max_tilt)
        T = max(mass * (accel_cmd[2] + g), 0.1)
        if max_thrust is not None:
            T = min(T, max_thrust)
        control = [T, phi_cmd, theta_cmd]

        return control, accel_cmd, metadata


class NNAdaptor(nn.Module):
    """
    Neural network that adapts PN's navigation constant N based on the situation.
    """

    def __init__(self, config, hidden_size=128):
        super().__init__()

        self.config = config

        # Input: relative pos/vel, miss distance, closing velocity, etc.
        input_size = 11
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

        # Scale output to reasonable N range (2-8)
        self.n_min = 2.0
        self.n_max = 8.0

        # Training
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.training_data = []

    def forward(self, interceptor_pos, interceptor_vel, target_pos, target_vel):
        """
        Predict optimal N for current situation.

        Returns:
            N: Adapted navigation constant
        """
        # Compute features
        rel_pos = np.array(target_pos) - np.array(interceptor_pos)
        rel_vel = np.array(target_vel) - np.array(interceptor_vel)
        range_mag = np.linalg.norm(rel_pos)

        # Normalize features
        features = np.array([
            rel_pos[0] / 50,  # Normalized X
            rel_pos[1] / 50,  # Normalized Y
            rel_pos[2] / 30,  # Normalized Z
            rel_vel[0] / 10,  # Normalized velocity
            rel_vel[1] / 10,
            rel_vel[2] / 10,
            range_mag / 100,  # Normalized range
            -np.dot(rel_vel, rel_pos / (range_mag + 1e-6)) / 10,  # Closing vel
            interceptor_vel[0] / 10,
            interceptor_vel[1] / 10,
            interceptor_vel[2] / 10
        ])

        # Convert to tensor and predict
        features_tensor = torch.FloatTensor(features).unsqueeze(0)

        with torch.no_grad():
            normalized_n = self.network(features_tensor).item()

        # Scale to N range
        N = self.n_min + normalized_n * (self.n_max - self.n_min)

        return N

    def train_step(self, batch_size=32):
        """Train the adaptor on collected data."""
        if len(self.training_data) < batch_size:
            return

        # Sample batch
        batch = random.sample(self.training_data, batch_size)

        # Prepare data
        features_batch = []
        targets_batch = []

        for data in batch:
            features, optimal_N = data
            features_batch.append(features)
            targets_batch.append([optimal_N])

        features_tensor = torch.FloatTensor(features_batch)
        targets_tensor = torch.FloatTensor(targets_batch)

        # Forward pass
        predictions = self.network(features_tensor)
        loss = nn.MSELoss()(predictions, targets_tensor)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, interceptor_pos, interceptor_vel, target_pos, target_vel):
        """Predict optimal N (wrapper for forward)."""
        return self.forward(interceptor_pos, interceptor_vel, target_pos, target_vel)


class HybridGuidanceTrainer:
    """
    Trainer for the hybrid guidance system.
    """

    def __init__(self, config):
        self.config = config
        self.hybrid = HybridGuidance(config, mode='adaptive')

        # Training environment
        from threat_path import ThreatPathGenerator
        self.threat_path = ThreatPathGenerator(config)

        # Tracking
        self.episode_rewards = []
        self.best_reward = -np.inf

    def generate_training_data(self, n_episodes=1000):
        """
        Generate data for training the adaptive N network.

        For each episode, try different N values and record which performs best.
        """
        print("Generating training data for adaptive N...")

        # Test different N values
        n_values = np.linspace(2, 8, 7)  # 2, 3, 4, 5, 6, 7, 8

        # Fixed initial conditions for evaluation
        np.random.seed(42)

        for episode in range(n_episodes):
            # Randomize threat path
            amp = np.random.uniform(1, 4)
            freq = np.random.uniform(0.5, 3)
            self.threat_path.jink_amplitude = amp
            self.threat_path.jink_frequency = freq

            # Random initial positions
            threat_pos = np.array([
                50 + np.random.uniform(-20, 20),
                50 + np.random.uniform(-20, 20),
                20 + np.random.uniform(-10, 10)
            ])

            interceptor_pos = np.array([
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(0, 5)
            ])

            # Re-anchor the drift trajectory to THIS episode's randomized
            # threat_pos. Without this, get_desired_position(t) keeps
            # drifting from the fixed config.THREAT_INITIAL_POS regardless
            # of where the episode says the threat actually starts, so the
            # "desired" path and the simulated threat position disagree from
            # step zero (and the search for best_N below is comparing miss
            # distances against a target that isn't where it's supposed to
            # be).
            self.threat_path.initial_pos = threat_pos.copy()

            # Evaluate each N value
            best_N = 4.0
            best_miss = np.inf

            for N in n_values:
                # Simulate with this N
                miss = self._simulate_with_n(
                    N, threat_pos, interceptor_pos,
                    amp, freq
                )

                if miss < best_miss:
                    best_miss = miss
                    best_N = N

            # Generate features for this scenario
            rel_pos = np.array(threat_pos) - np.array(interceptor_pos)
            rel_vel = np.array([0, 0, 0]) - np.array([0, 0, 0])
            range_mag = np.linalg.norm(rel_pos)

            features = [
                rel_pos[0] / 50,
                rel_pos[1] / 50,
                rel_pos[2] / 30,
                rel_vel[0] / 10,
                rel_vel[1] / 10,
                rel_vel[2] / 10,
                range_mag / 100,
                -np.dot(rel_vel, rel_pos / (range_mag + 1e-6)) / 10,
                0, 0, 0  # Interceptor velocity (assumed zero at start)
            ]

            # Store training data
            self.hybrid.n_adaptor.training_data.append(
                (features, best_N)
            )

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{n_episodes}, Best N: {best_N:.2f}")

        print(f"Generated {len(self.hybrid.n_adaptor.training_data)} training samples")

    def _simulate_with_n(self, N, threat_pos, interceptor_pos, amp, freq):
        """Simulate a single engagement with given N."""
        # Simplified simulation - just estimate miss distance
        # This is a fast approximation for training data generation

        from quad_dynamics import integrate_dynamics

        dt = 0.01
        steps = 1000

        interceptor_state = np.array([*interceptor_pos, 0, 0, 0])

        pn = ProportionalNavigation(self.config)
        pn.N = N

        g = self.config.G
        mass = self.config.MASS
        # The interceptor's own (higher) envelope -- see config.py.
        max_tilt = getattr(self.config, 'INTERCEPTOR_MAX_TILT', self.config.MAX_TILT)

        miss = np.inf
        for step in range(steps):
            t = step * dt

            # Threat position/velocity from the (now correctly re-anchored)
            # path generator, instead of a hardcoded zero velocity that
            # starved PN guidance of the target's actual motion.
            threat_pos_t = self.threat_path.get_desired_position(t)
            threat_vel_t = self.threat_path.get_velocity_at_time(t)

            # PN guidance
            accel, _, _ = pn.compute_guidance(
                interceptor_state[:3],
                interceptor_state[3:],
                threat_pos_t,
                threat_vel_t
            )

            # Route the commanded acceleration through the same thrust/tilt
            # inversion used everywhere else in the sim, instead of the
            # previous code's unused `accel` variable and a hardcoded
            # constant upward climb that was actually driving the
            # interceptor regardless of N. Without this, every N value
            # produced ~the same trajectory and the "best N" search below
            # was comparing noise.
            theta_cmd = np.clip(accel[0] / g, -max_tilt, max_tilt)
            phi_cmd = np.clip(-accel[1] / g, -max_tilt, max_tilt)
            T = max(mass * (accel[2] + g), 0.1)

            interceptor_state = integrate_dynamics(
                interceptor_state, [T, phi_cmd, theta_cmd], dt,
                mass=mass, g=g
            )

            # Check miss distance
            miss = np.linalg.norm(threat_pos_t - interceptor_state[:3])
            if miss < 0.5:  # Intercept
                return miss

        return miss

    def train_adaptor(self, epochs=100, batch_size=32):
        """Train the adaptive N network."""
        print("\nTraining adaptive N network...")

        for epoch in range(epochs):
            loss = self.hybrid.n_adaptor.train_step(batch_size)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        print("Training complete!")

    def save_adaptor(self, path="./models/n_adaptor.pth"):
        """Save the trained adaptor."""
        import torch
        torch.save(self.hybrid.n_adaptor.state_dict(), path)
        print(f"Adaptor saved to {path}")

    def load_adaptor(self, path="./models/n_adaptor.pth"):
        """Load a trained adaptor."""
        import torch
        self.hybrid.n_adaptor.load_state_dict(torch.load(path))
        print(f"Adaptor loaded from {path}")


def test_hybrid_modes():
    """Test and compare all hybrid modes."""
    print("\n" + "=" * 70)
    print("TESTING HYBRID GUIDANCE MODES")
    print("=" * 70)

    from run_sim import DroneDefenseSimulationHybrid
    import matplotlib.pyplot as plt

    # Test configurations
    modes = [
        ('pn_only', 0.0, "Pure PN"),
        ('blended', 0.3, "PN + 30% AI"),
        ('blended', 0.5, "PN + 50% AI"),
        ('blended', 0.7, "PN + 70% AI"),
        ('ai_only', 1.0, "Pure AI"),
    ]

    results = []

    for mode, weight, label in modes:
        print(f"\nTesting: {label}")
        print("-" * 40)

        # Create simulation with hybrid guidance
        sim = DroneDefenseSimulationHybrid(config)

        # Modify to use hybrid guidance
        from hybrid_guidance import HybridGuidance
        hybrid = HybridGuidance(
            config,
            mode=mode,
            model_path=config.BEST_MODEL_PATH,
            blend_weight=weight
        )

        # Replace PN with hybrid
        sim.pn_guidance = hybrid

        # Run simulation
        sim.run()

        results.append({
            'mode': mode,
            'label': label,
            'miss_distance': sim.miss_distance,
            'time_ca': sim.time_of_closest_approach,
            'success': sim.miss_distance < config.INTERCEPT_RADIUS
        })

    # Display results
    print("\n" + "=" * 70)
    print("HYBRID MODE COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Mode':<25} {'Miss Distance':>15} {'Success':>10} {'Time to CA':>12}")
    print("-" * 70)
    for r in results:
        success_str = "✅" if r['success'] else "❌"
        print(f"{r['label']:<25} {r['miss_distance']:>15.2f} {success_str:>10} {r['time_ca']:>12.2f}")

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    modes = [r['label'] for r in results]
    misses = [r['miss_distance'] for r in results]
    colors = ['red' if m > config.INTERCEPT_RADIUS else 'green' for m in misses]

    bars = ax.bar(modes, misses, color=colors, alpha=0.7)
    ax.axhline(y=config.INTERCEPT_RADIUS, color='k', linestyle='--', label='Intercept threshold', alpha=0.5)
    ax.set_xlabel('Guidance Mode')
    ax.set_ylabel('Miss Distance (m)')
    ax.set_title('Hybrid Guidance Performance Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add value labels
    for bar, miss in zip(bars, misses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f'{miss:.1f}m', ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return results


def test_adaptive_n():
    """Test the adaptive N approach."""
    print("\n" + "=" * 70)
    print("TESTING ADAPTIVE N GUIDANCE")
    print("=" * 70)

    from run_sim import DroneDefenseSimulationHybrid

    # Train adaptor if not already trained
    trainer = HybridGuidanceTrainer(config)
    trainer.generate_training_data(n_episodes=500)
    trainer.train_adaptor(epochs=50)
    trainer.save_adaptor()

    # Test adaptive N
    sim = DroneDefenseSimulationHybrid(config)

    # Replace PN with adaptive hybrid
    from hybrid_guidance import HybridGuidance
    hybrid = HybridGuidance(config, mode='adaptive')
    sim.pn_guidance = hybrid

    # Run simulation
    sim.run()

    print(f"\nAdaptive N Results:")
    print(f"Min Miss Distance: {sim.miss_distance:.2f} m")
    print(f"Time of Closest Approach: {sim.time_of_closest_approach:.2f} s")

    # Plot adaptation history
    import matplotlib.pyplot as plt

    if hasattr(hybrid, 'adaptation_history') and len(hybrid.adaptation_history) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # N adaptation over time
        ax1.plot(hybrid.adaptation_history, 'b-', linewidth=2)
        ax1.axhline(y=config.PN_NAVIGATION_CONSTANT, color='r', linestyle='--',
                    label='Default N', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Navigation Constant N')
        ax1.set_title('Adaptive N over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Miss distance
        miss_dist = np.array(sim.log['miss_distance'])
        ax2.plot(miss_dist, 'g-', linewidth=2)
        ax2.axhline(y=config.INTERCEPT_RADIUS, color='r', linestyle='--', label='Intercept threshold', alpha=0.7)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Miss Distance (m)')
        ax2.set_title('Miss Distance with Adaptive N')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    return sim


def main():
    """Main entry point for hybrid guidance."""
    print("=" * 60)
    print("HLIN: HYBRID GUIDANCE SYSTEM")
    print("Combining PN + AI for Optimal Performance")
    print("=" * 60)

    print("\nAvailable Modes:")
    print("1. Test all hybrid modes (PN, Blended, AI)")
    print("2. Test adaptive N (AI tunes PN parameter)")
    print("3. Train adaptive N network")
    print("4. Run custom hybrid simulation")

    choice = input("\nEnter choice (1-4): ")

    if choice == '1':
        test_hybrid_modes()
    elif choice == '2':
        test_adaptive_n()
    elif choice == '3':
        trainer = HybridGuidanceTrainer(config)
        n_samples = input("Number of training episodes (default: 1000): ")
        n_samples = int(n_samples) if n_samples else 1000
        trainer.generate_training_data(n_episodes=n_samples)
        trainer.train_adaptor(epochs=100)
        trainer.save_adaptor()
    elif choice == '4':
        # Custom simulation
        from run_sim import DroneDefenseSimulation

        mode = input("Mode (pn_only/ai_only/blended/adaptive): ")
        weight = float(input("Blend weight (0-1, only for blended): ") or "0.5")

        hybrid = HybridGuidance(
            config,
            mode=mode,
            model_path=config.BEST_MODEL_PATH,
            blend_weight=weight
        )

        sim = DroneDefenseSimulation(config)
        sim.pn_guidance = hybrid
        sim.run()
        sim.plot_results()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()