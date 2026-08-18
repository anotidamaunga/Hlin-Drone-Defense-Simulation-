# Hlin — Drone Defense Simulation

Hlin is a simulation and research testbed for drone-vs-drone interception. A
"threat" drone flies an evasive trajectory toward a protected zone while an
"interceptor" drone tries to close within intercept range before it arrives.
Guidance for the interceptor can be pure classical Proportional Navigation
(PN), a PPO-trained reinforcement-learning policy, or a blend of the two.

## How it works

- **Threat drone** (`threat_path.py`) follows a scripted trajectory —
  constant drift toward the protected zone, a slow sinusoidal weave, and
  high-frequency jinking — plus reactive evasion that steers it away from the
  interceptor's live position once the interceptor is within an awareness
  radius.
- **Interceptor drone** is steered by one of several guidance strategies:
  - **PN only** (`pn_guidance.py`) — classical Proportional Navigation,
    commanded acceleration = `N * closing_velocity * LOS_rate`.
  - **AI only** (`rl_interceptor.py`) — a PPO policy (via
    [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)) trained
    against a custom Gymnasium environment (`InterceptorEnv`) that rewards
    closing distance and successful intercepts.
  - **Blended / adaptive** (`hybrid_guidance.py`) — combines PN and AI
    accelerations with a tunable blend weight, or lets a small adaptor
    network adjust PN's navigation constant on the fly.
- **Dynamics** (`quad_dynamics.py`) is a simplified 3-DOF near-hover
  quadrotor model integrated with `scipy.integrate.solve_ivp`, driven by
  `[thrust, roll_cmd, pitch_cmd]` from a PD position controller
  (`position_controller.py`).
- **Sensing** (`tracking.py`) simulates a noisy radar measurement of the
  threat's position, fed through a constant-velocity Kalman filter to
  produce the position/velocity estimate the guidance laws actually see.
- **Visualization** — `run_sim.py` produces Matplotlib summary plots
  (3D trajectory, miss distance, guidance-mode breakdown, acceleration
  comparison); `visualizer_pygame.py` provides a real-time Pygame view.

An engagement ends in one of four outcomes: `intercept` (interceptor closes
within `INTERCEPT_RADIUS`), `threat_reached_target` (threat breaches
`PROTECTED_ZONE_RADIUS`), `too_far` (interceptor loses the threat beyond
100 m), or `timeout` (neither happens within `SIM_DURATION`).

## Project layout

| File | Purpose |
|---|---|
| `config.py` | All tunable simulation, physics, controller, and RL parameters |
| `quad_dynamics.py` | 3-DOF quadrotor dynamics + integration |
| `position_controller.py` | PD controller: desired position/velocity → thrust/tilt command |
| `threat_path.py` | Threat drone trajectory generator (drift + sway + jink + evasion) |
| `pn_guidance.py` | Proportional Navigation guidance law |
| `tracking.py` | Radar sensor model + Kalman filter |
| `rl_interceptor.py` | Gymnasium env, PPO training/evaluation for the AI interceptor |
| `hybrid_guidance.py` | Combines PN + AI (blended/switchable/adaptive modes) |
| `run_sim.py` | Main simulation runner, plotting, and batch evaluation |
| `visualizer_pygame.py` | Real-time Pygame visualization |
| `models/`, `models_fixed*/` | Saved PPO checkpoints and `VecNormalize` stats from training runs |
| `logs/`, `logs_test/` | TensorBoard event logs and monitor CSVs from training |

## Requirements

- Python 3.13
- `numpy`, `scipy`, `matplotlib`
- `torch`
- `gymnasium`, `stable-baselines3`
- `pygame` (for real-time visualization)

```bash
pip install -r requirements.txt
```

## Usage

### Run a single engagement

```bash
python run_sim.py --mode blended --weight 0.5
```

`--mode` selects the guidance strategy (`pn_only`, `ai_only`, `blended`,
`adaptive`); `--weight` sets the PN/AI blend weight for `blended` mode;
`--model` overrides the AI model path (defaults to `config.BEST_MODEL_PATH`).
Each run randomizes the threat's spawn distance/angle and prints a summary,
then opens a Matplotlib results window.

### Train the RL interceptor

```bash
python rl_interceptor.py
```

Runs an interactive menu to train a new PPO model, evaluate an existing one,
compare AI vs. PN guidance, or run an interactive Pygame simulation.

### Batch-evaluate a guidance mode

Use `evaluate_blended()` in `run_sim.py` to run many randomized engagements
end-to-end (radar noise, Kalman filtering, PN+AI blending included) and get
aggregate success rate / miss distance statistics — this is the
deployment-realistic counterpart to `InterceptorAI.evaluate()`, which only
exercises the AI in isolation.

## Tuning

Nearly every physical and behavioral parameter — drone mass, thrust/tilt
limits, controller gains, threat evasion behavior, PN navigation constant,
Kalman noise assumptions, RL training length — lives in `config.py`.
