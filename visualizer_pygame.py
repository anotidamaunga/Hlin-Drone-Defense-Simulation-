"""
Pygame visualizer for the Hlin drone defense simulation.
Provides real-time 2D visualization with top-down and side views.
"""

import pygame
import numpy as np
import math
from threading import Thread
import time
from quad_dynamics import integrate_dynamics
from tracking import radar_sensor


class HlinVisualizer:
    """
    Real-time 3D visualization of the drone intercept simulation.
    Displays top-down view, side view, and telemetry data.
    """

    def __init__(self, config, width=1200, height=800):
        """
        Initialize the visualizer.

        Args:
            config: Configuration object with simulation parameters
            width: Window width
            height: Window height
        """
        pygame.init()

        self.config = config
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Hlin: Drone Defense Simulation")

        # Colors
        self.COLORS = {
            'background': (20, 20, 30),
            'grid': (40, 40, 50),
            'threat': (255, 50, 50),
            'threat_path': (200, 50, 50, 50),
            'interceptor': (50, 150, 255),
            'interceptor_path': (50, 150, 255, 50),
            'kalman': (50, 255, 50, 100),
            'protected_zone': (0, 255, 0),
            'miss_distance': (255, 255, 0),
            'text': (200, 200, 200),
            'warning': (255, 200, 0),
            'success': (0, 255, 0)
        }

        # View configuration
        self.view_mode = 'top'  # 'top' or 'side'
        self.zoom = 1.0
        self.follow_interceptor = True

        # Fonts
        self.font_small = pygame.font.SysFont('Arial', 14)
        self.font_medium = pygame.font.SysFont('Arial', 18)
        self.font_large = pygame.font.SysFont('Arial', 24)

        # Data storage for visualization
        self.history_threat = []
        self.history_interceptor = []
        self.history_kalman = []
        self.max_history = 500

        # Simulation state
        self.running = True
        self.paused = False
        self.simulation_speed = 1.0
        self.frame_count = 0


        self.termination_reason = None

        # Performance
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Sets view_rect/panel_rect/minimap_rect, used by world_to_screen()
        # and the grid/panel rendering -- must run before the first render().
        self.setup_views()

    def setup_views(self):
        """Set up the view layout."""
        # Main view area
        self.view_rect = pygame.Rect(0, 0, self.width - 300, self.height)

        # Side panel for telemetry
        self.panel_rect = pygame.Rect(self.width - 300, 0, 300, self.height)

        # Mini-map position
        self.minimap_rect = pygame.Rect(10, self.height - 150, 150, 140)

    def world_to_screen(self, x, y, z=None):
        """
        Convert world coordinates to screen coordinates.

        Args:
            x: World X coordinate
            y: World Y coordinate
            z: World Z coordinate (optional, for 3D projection)

        Returns:
            screen_x, screen_y: Screen coordinates
        """
        # Get the view bounds
        if self.follow_interceptor and len(self.history_interceptor) > 0:
            center_x = self.history_interceptor[-1][0]
            center_y = self.history_interceptor[-1][1]
        else:
            center_x = 0
            center_y = 0

        # Calculate scale based on zoom and view
        scale = 1.0 / self.zoom

        # Convert to screen coordinates
        screen_x = (x - center_x) * scale + self.view_rect.width / 2 + self.view_rect.x
        screen_y = (y - center_y) * scale + self.view_rect.height / 2 + self.view_rect.y

        return int(screen_x), int(screen_y)

    def draw_grid(self, surface):
        """Draw grid lines for spatial reference."""
        if len(self.history_interceptor) > 0:
            center_x = self.history_interceptor[-1][0]
            center_y = self.history_interceptor[-1][1]
        else:
            center_x = 0
            center_y = 0

        scale = 1.0 / self.zoom
        grid_size = 10 * scale

        # Draw vertical lines
        for x in np.arange(-100, 100, grid_size):
            screen_x, _ = self.world_to_screen(x + center_x % grid_size, 0)
            if self.view_rect.x <= screen_x <= self.view_rect.x + self.view_rect.width:
                pygame.draw.line(surface, self.COLORS['grid'],
                                 (screen_x, self.view_rect.y),
                                 (screen_x, self.view_rect.y + self.view_rect.height), 1)

        # Draw horizontal lines
        for y in np.arange(-100, 100, grid_size):
            _, screen_y = self.world_to_screen(0, y + center_y % grid_size)
            if self.view_rect.y <= screen_y <= self.view_rect.y + self.view_rect.height:
                pygame.draw.line(surface, self.COLORS['grid'],
                                 (self.view_rect.x, screen_y),
                                 (self.view_rect.x + self.view_rect.width, screen_y), 1)

    def draw_paths(self, surface):
        """Draw the historical paths of both drones."""
        # Threat path
        if len(self.history_threat) > 2:
            points = [self.world_to_screen(p[0], p[1]) for p in self.history_threat]
            if len(points) > 1:
                pygame.draw.lines(surface, self.COLORS['threat_path'], False, points, 2)

        # Interceptor path
        if len(self.history_interceptor) > 2:
            points = [self.world_to_screen(p[0], p[1]) for p in self.history_interceptor]
            if len(points) > 1:
                pygame.draw.lines(surface, self.COLORS['interceptor_path'], False, points, 2)

        # Kalman estimate path
        if len(self.history_kalman) > 2:
            points = [self.world_to_screen(p[0], p[1]) for p in self.history_kalman]
            if len(points) > 1:
                pygame.draw.lines(surface, self.COLORS['kalman'], False, points, 1)

    def draw_drones(self, surface):
        """Draw the current drone positions."""
        if len(self.history_threat) > 0:
            threat = self.history_threat[-1]
            screen_x, screen_y = self.world_to_screen(threat[0], threat[1])

            # Draw threat drone with glow effect
            for radius in [20, 15, 10]:
                alpha = 50 if radius == 20 else 100 if radius == 15 else 255
                color = (*self.COLORS['threat'][:3], alpha)
                pygame.draw.circle(surface, self.COLORS['threat'],
                                   (screen_x, screen_y), radius, 2 if radius == 10 else 1)
            pygame.draw.circle(surface, self.COLORS['threat'],
                               (screen_x, screen_y), 8)

            # Draw "X" through threat
            pygame.draw.line(surface, (255, 255, 255),
                             (screen_x - 6, screen_y - 6),
                             (screen_x + 6, screen_y + 6), 2)
            pygame.draw.line(surface, (255, 255, 255),
                             (screen_x + 6, screen_y - 6),
                             (screen_x - 6, screen_y + 6), 2)

        if len(self.history_interceptor) > 0:
            interceptor = self.history_interceptor[-1]
            screen_x, screen_y = self.world_to_screen(interceptor[0], interceptor[1])

            # Draw interceptor drone
            # Main body
            pygame.draw.circle(surface, self.COLORS['interceptor'],
                               (screen_x, screen_y), 10)
            # Outer ring (rotor effect)
            for i in range(4):
                angle = pygame.time.get_ticks() / 500 + i * np.pi / 2
                dx = 14 * np.cos(angle)
                dy = 14 * np.sin(angle)
                pygame.draw.circle(surface, self.COLORS['interceptor'],
                                   (screen_x + int(dx), screen_y + int(dy)), 4)
            # Center dot
            pygame.draw.circle(surface, (255, 255, 255),
                               (screen_x, screen_y), 4)

        # Draw Kalman estimate
        if len(self.history_kalman) > 0:
            kalman = self.history_kalman[-1]
            screen_x, screen_y = self.world_to_screen(kalman[0], kalman[1])
            pygame.draw.circle(surface, self.COLORS['kalman'],
                               (screen_x, screen_y), 6, 2)

    def draw_protected_zone(self, surface):
        """Draw the protected zone."""
        px, py = self.config.PROTECTED_ZONE[:2]
        screen_x, screen_y = self.world_to_screen(px, py)

        # Pulsing circle
        pulse = 5 * (1 + 0.3 * np.sin(pygame.time.get_ticks() / 500))
        pygame.draw.circle(surface, self.COLORS['protected_zone'],
                           (screen_x, screen_y), int(15 + pulse), 3)
        pygame.draw.circle(surface, self.COLORS['protected_zone'],
                           (screen_x, screen_y), 5)

        # Label
        label = self.font_small.render("Protected Zone", True, self.COLORS['protected_zone'])
        surface.blit(label, (screen_x - 30, screen_y - 30))

    def draw_telemetry(self, surface):
        """Draw telemetry data in the side panel."""
        # Clear panel
        pygame.draw.rect(surface, (30, 30, 40), self.panel_rect)
        pygame.draw.line(surface, (60, 60, 70),
                         (self.panel_rect.x, self.panel_rect.y),
                         (self.panel_rect.x, self.panel_rect.y + self.panel_rect.height), 2)

        y_offset = 10

        # Title
        title = self.font_large.render("TELEMETRY", True, self.COLORS['text'])
        surface.blit(title, (self.panel_rect.x + 10, y_offset))
        y_offset += 40


        banner_by_reason = {
            'intercept': ("THREAT NEUTRALIZED", self.COLORS['success'], (10, 40, 10)),
            'threat_reached_target': ("PROTECTED ZONE BREACHED", self.COLORS['threat'], (40, 10, 10)),
            'too_far': ("THREAT ESCAPED", self.COLORS['warning'], (40, 35, 10)),
        }
        if self.termination_reason in banner_by_reason:
            text, color, bg = banner_by_reason[self.termination_reason]
            banner = self.font_large.render(text, True, color)
            pygame.draw.rect(surface, bg,
                             (self.panel_rect.x + 5, y_offset - 5, self.panel_rect.width - 10, 34))
            surface.blit(banner, (self.panel_rect.x + 10, y_offset))
            y_offset += 45

        # Speed indicator
        if len(self.history_threat) > 0 and len(self.history_interceptor) > 0:
            # Miss distance (full 3D — this used to drop the z component,
            # which could show a smaller/misleading number than the real
            # miss distance used to actually decide intercept success)
            threat = self.history_threat[-1]
            interceptor = self.history_interceptor[-1]
            miss_dist = np.linalg.norm(np.array(threat) - np.array(interceptor))
            intercept_radius = getattr(self.config, 'INTERCEPT_RADIUS', 2.0)

            # Status
            status_text = "STATUS: "
            if self.termination_reason in banner_by_reason:
                status_text += banner_by_reason[self.termination_reason][0]
                color = banner_by_reason[self.termination_reason][1]
            elif miss_dist < intercept_radius:
                status_text += "INTERCEPT!"
                color = self.COLORS['success']
            elif miss_dist < 2 * intercept_radius:
                status_text += "ENGAGING"
                color = self.COLORS['warning']
            else:
                status_text += "TRACKING"
                color = self.COLORS['text']

            status = self.font_medium.render(status_text, True, color)
            surface.blit(status, (self.panel_rect.x + 10, y_offset))
            y_offset += 35

            # Miss distance
            dist_text = f"MISS DIST: {miss_dist:.2f} m"
            dist = self.font_medium.render(dist_text, True, self.COLORS['miss_distance'])
            surface.blit(dist, (self.panel_rect.x + 10, y_offset))
            y_offset += 30

            # Threat position
            threat_pos = f"THREAT: ({threat[0]:.1f}, {threat[1]:.1f}, {threat[2]:.1f})"
            threat_text = self.font_small.render(threat_pos, True, self.COLORS['threat'])
            surface.blit(threat_text, (self.panel_rect.x + 10, y_offset))
            y_offset += 25

            # Interceptor position
            inter_pos = f"INTERCEPT: ({interceptor[0]:.1f}, {interceptor[1]:.1f}, {interceptor[2]:.1f})"
            inter_text = self.font_small.render(inter_pos, True, self.COLORS['interceptor'])
            surface.blit(inter_text, (self.panel_rect.x + 10, y_offset))
            y_offset += 25

            # Kalman estimate
            if len(self.history_kalman) > 0:
                kalman = self.history_kalman[-1]
                kalman_pos = f"KALMAN: ({kalman[0]:.1f}, {kalman[1]:.1f}, {kalman[2]:.1f})"
                kalman_text = self.font_small.render(kalman_pos, True, self.COLORS['kalman'])
                surface.blit(kalman_text, (self.panel_rect.x + 10, y_offset))
                y_offset += 25

            # Speed
            if len(self.history_threat) > 1:
                # Approximate velocities
                dt = self.dt if hasattr(self, 'dt') else 0.01
                if len(self.history_threat) >= 2:
                    v_threat = np.linalg.norm(
                        np.array(self.history_threat[-1][:3]) -
                        np.array(self.history_threat[-2][:3])
                    ) / max(dt, 0.001)
                    v_inter = np.linalg.norm(
                        np.array(self.history_interceptor[-1][:3]) -
                        np.array(self.history_interceptor[-2][:3])
                    ) / max(dt, 0.001)

                    speed_text = f"THREAT SPEED: {v_threat:.1f} m/s"
                    surface.blit(self.font_small.render(speed_text, True, self.COLORS['text']),
                                 (self.panel_rect.x + 10, y_offset))
                    y_offset += 20

                    speed_text = f"INTERCEPT SPEED: {v_inter:.1f} m/s"
                    surface.blit(self.font_small.render(speed_text, True, self.COLORS['text']),
                                 (self.panel_rect.x + 10, y_offset))
                    y_offset += 25

        # Controls info
        y_offset += 10
        controls = [
            "CONTROLS:",
            "SPACE: Pause",
            "R: Reset View",
            "Z: Zoom In",
            "X: Zoom Out",
            "F: Follow Toggle",
            "V: View Mode",
            "ESC: Quit"
        ]
        for c in controls:
            text = self.font_small.render(c, True, (150, 150, 150))
            surface.blit(text, (self.panel_rect.x + 10, y_offset))
            y_offset += 20

    def draw_minimap(self, surface):
        """Draw a minimap showing the overall engagement."""
        # Clear minimap area
        pygame.draw.rect(surface, (20, 20, 30), self.minimap_rect)
        pygame.draw.rect(surface, (60, 60, 70), self.minimap_rect, 2)

        if len(self.history_threat) > 0 and len(self.history_interceptor) > 0:
            # Get bounds
            all_x = [p[0] for p in self.history_threat] + [p[0] for p in self.history_interceptor]
            all_y = [p[1] for p in self.history_threat] + [p[1] for p in self.history_interceptor]

            if len(all_x) > 0 and len(all_y) > 0:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                range_x = max(max_x - min_x, 1)
                range_y = max(max_y - min_y, 1)

                def map_to_minimap(x, y):
                    map_x = self.minimap_rect.x + 5 + (x - min_x) / range_x * (self.minimap_rect.width - 10)
                    map_y = self.minimap_rect.y + 5 + (y - min_y) / range_y * (self.minimap_rect.height - 10)
                    return int(map_x), int(map_y)

                # Draw paths
                if len(self.history_threat) > 2:
                    points = [map_to_minimap(p[0], p[1]) for p in self.history_threat]
                    pygame.draw.lines(surface, self.COLORS['threat_path'], False, points, 1)

                if len(self.history_interceptor) > 2:
                    points = [map_to_minimap(p[0], p[1]) for p in self.history_interceptor]
                    pygame.draw.lines(surface, self.COLORS['interceptor_path'], False, points, 1)

                # Draw current positions
                threat = self.history_threat[-1]
                inter = self.history_interceptor[-1]

                tx, ty = map_to_minimap(threat[0], threat[1])
                ix, iy = map_to_minimap(inter[0], inter[1])

                pygame.draw.circle(surface, self.COLORS['threat'], (tx, ty), 4)
                pygame.draw.circle(surface, self.COLORS['interceptor'], (ix, iy), 4)

    def update(self, threat_state, interceptor_state, kalman_estimate, dt=0.01,
               termination_reason=None):
        """
        Update the visualization with new simulation data.

        Args:
            threat_state: [x, y, z, vx, vy, vz] threat state
            interceptor_state: [x, y, z, vx, vy, vz] interceptor state
            kalman_estimate: [x, y, z, vx, vy, vz] Kalman estimate
            dt: Time step
            termination_reason: None while the engagement is ongoing, else
                one of 'intercept', 'threat_reached_target', 'too_far'
                (mirrors sim.termination_reason, which is itself sticky for
                the run once set, so this only ever latches from None to a
                reason, never back)
        """
        self.dt = dt
        if termination_reason is not None:
            self.termination_reason = termination_reason

        # Store history
        self.history_threat.append(threat_state[:3])
        self.history_interceptor.append(interceptor_state[:3])
        self.history_kalman.append(kalman_estimate[:3])

        # Trim history
        if len(self.history_threat) > self.max_history:
            self.history_threat.pop(0)
        if len(self.history_interceptor) > self.max_history:
            self.history_interceptor.pop(0)
        if len(self.history_kalman) > self.max_history:
            self.history_kalman.pop(0)

    def render(self):
        """Render the visualization."""
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.zoom = 1.0
                elif event.key == pygame.K_z:
                    self.zoom *= 1.2
                elif event.key == pygame.K_x:
                    self.zoom /= 1.2
                elif event.key == pygame.K_f:
                    self.follow_interceptor = not self.follow_interceptor
                elif event.key == pygame.K_v:
                    self.view_mode = 'side' if self.view_mode == 'top' else 'top'
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.zoom *= 1.1
                else:
                    self.zoom /= 1.1

        if self.paused:
            return

        # Clear screen
        self.screen.fill(self.COLORS['background'])

        # Draw main view
        self.draw_grid(self.screen)
        self.draw_paths(self.screen)
        self.draw_protected_zone(self.screen)
        self.draw_drones(self.screen)

        # Draw telemetry panel
        self.draw_telemetry(self.screen)

        # Draw minimap
        self.draw_minimap(self.screen)

        # Draw FPS and speed indicator
        fps_text = self.font_small.render(f"FPS: {int(self.clock.get_fps())}", True, (150, 150, 150))
        self.screen.blit(fps_text, (10, 10))

        speed_text = self.font_small.render(f"SPEED: {self.simulation_speed:.1f}x", True, (150, 150, 150))
        self.screen.blit(speed_text, (10, 30))

        # Draw view mode
        view_text = self.font_small.render(f"VIEW: {self.view_mode.upper()}", True, (150, 150, 150))
        self.screen.blit(view_text, (10, 50))

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()


class HlinPygameSimulation:
    """
    Wrapper that runs the Hlin simulation with Pygame visualization.
    """

    def __init__(self, config, sim_class):
        """
        Initialize the simulation with visualization.

        Args:
            config: Configuration object
            sim_class: The simulation class (DroneDefenseSimulation)
        """
        self.config = config
        self.sim_class = sim_class
        self.visualizer = HlinVisualizer(config)
        self.sim = None

    def run(self):
        """Run the simulation with real-time visualization."""
        # Initialize simulation. randomize_scenario=True so repeated runs
        # sample a different engagement each time (spawn distance/angle,
        # jink parameters) instead of always the same fixed geometry --
        # otherwise a handful of demo runs all look the same and give a
        # misleading impression of how the model performs in general (see
        # evaluate_blended() in run_sim.py for the properly-averaged number).
        self.sim = self.sim_class(self.config, randomize_scenario=True)

        # Override the run method to step one frame at a time
        print("Starting Hlin simulation with Pygame visualization...")
        print("Controls: SPACE=pause, Z/X=zoom, F=follow, V=view mode, ESC=quit")

        # Initialize Kalman filter
        from tracking import radar_sensor
        first_measurement = radar_sensor(
            self.sim.threat_state[:3],
            self.config.RADAR_NOISE_STD
        )
        self.sim.kalman.initialize(first_measurement)


        step = 0
        reported = False
        while self.visualizer.running and step < self.sim.num_steps:
            if not self.visualizer.paused and self.sim.termination_reason is None:
                t = step * self.sim.dt

                # Run one simulation step
                self._simulation_step(t)

                # Update visualization
                self.visualizer.update(
                    self.sim.threat_state,
                    self.sim.interceptor_state,
                    self.sim.kalman.get_estimate(),
                    self.sim.dt,
                    termination_reason=self.sim.termination_reason
                )

                step += 1

            if self.sim.termination_reason is not None and not reported:
                print(f"\nEngagement resolved: {self.sim.termination_reason}")
                print(f"Minimum miss distance: {self.sim.miss_distance:.2f} m")
                print(f"Time of closest approach: {self.sim.time_of_closest_approach:.2f} s")
                reported = True

            # Render
            self.visualizer.render()

        # Print final results (only if the loop exited via step count/quit
        # without ever resolving — the resolved case already printed above)
        if not reported:
            print(f"\nSimulation complete!")
            print(f"Minimum miss distance: {self.sim.miss_distance:.2f} m")
            print(f"Time of closest approach: {self.sim.time_of_closest_approach:.2f} s")

        # Keep window open for a moment
        while self.visualizer.running:
            self.visualizer.render()

        self.visualizer.close()

    def _simulation_step(self, t):
        """Execute one step of the simulation."""

        threat_desired = self.sim.threat_path.get_desired_position(
            t, threat_pos=self.sim.threat_state[:3],
            interceptor_pos=self.sim.interceptor_state[:3]
        )
        threat_control = self.sim.position_controller.compute_control(
            threat_desired,
            self.sim.threat_state,
            desired_vel=self.sim.threat_path.get_velocity_at_time(t)
        )

        self.sim.threat_state = integrate_dynamics(
            self.sim.threat_state,
            threat_control,
            self.sim.dt,
            mass=self.config.MASS,
            g=self.config.G
        )

        # Sensor measurement
        measurement = radar_sensor(
            self.sim.threat_state[:3],
            self.config.RADAR_NOISE_STD
        )

        # Kalman filter
        self.sim.kalman.predict_update(measurement)
        kalman_estimate = self.sim.kalman.get_estimate()


        interceptor_control, accel_cmd, guidance_metadata = self.sim.guidance.compute_control_command(
            self.sim.interceptor_state[:3],
            self.sim.interceptor_state[3:],
            kalman_estimate[:3],
            kalman_estimate[3:]
        )

        self.sim.interceptor_state = integrate_dynamics(
            self.sim.interceptor_state,
            interceptor_control,
            self.sim.dt,
            mass=self.config.MASS,
            g=self.config.G
        )

        # Logging
        self.sim.log['time'].append(t)
        self.sim.log['threat_state'].append(self.sim.threat_state.copy())
        self.sim.log['interceptor_state'].append(self.sim.interceptor_state.copy())
        self.sim.log['threat_desired'].append(threat_desired)
        self.sim.log['kalman_estimate'].append(kalman_estimate.copy())
        self.sim.log['measurements'].append(measurement.copy())
        self.sim.log['guidance_metadata'].append(guidance_metadata)
        if 'pn_accel' in guidance_metadata:
            self.sim.log['pn_accel'].append(guidance_metadata['pn_accel'])
        if 'ai_accel' in guidance_metadata:
            self.sim.log['ai_accel'].append(guidance_metadata['ai_accel'])
        self.sim.log['final_accel'].append(accel_cmd)

        # Miss distance
        miss_dist = np.linalg.norm(
            self.sim.threat_state[:3] - self.sim.interceptor_state[:3]
        )
        self.sim.log['miss_distance'].append(miss_dist)

        if miss_dist < self.sim.miss_distance:
            self.sim.miss_distance = miss_dist
            self.sim.time_of_closest_approach = t


        if self.sim.termination_reason is None:
            if miss_dist < self.config.INTERCEPT_RADIUS:
                self.sim.success = True
                self.sim.termination_reason = 'intercept'
            else:
                threat_to_target = np.linalg.norm(
                    self.sim.threat_state[:3] - np.array(self.config.PROTECTED_ZONE)
                )
                if threat_to_target < 3.0:
                    self.sim.termination_reason = 'threat_reached_target'
                elif miss_dist > 100:
                    self.sim.termination_reason = 'too_far'


def main():
    """Main entry point for Pygame visualization."""
    import config
    from run_sim import DroneDefenseSimulationHybrid

    # Check if pygame is available
    try:
        import pygame
    except ImportError:
        print("Pygame not installed. Please install with: pip install pygame")
        return

    # Run the simulation with visualization
    sim_vis = HlinPygameSimulation(config, DroneDefenseSimulationHybrid)
    sim_vis.run()


if __name__ == "__main__":
    main()