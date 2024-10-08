import collections
import numpy as np
import gymnasium as gym
import tensorflow as tf
from typing import Optional
from pygame import gfxdraw
import pygame
from os import path
import pandas as pd

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

output_min = [-1, -1, -8]
output_max = [1, 1, 8]

class Pendulum_Custom_Environment_DensityAdjustedReward(gym.Env):
    
    steps = 0
    model = None

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, kde, render_mode: Optional[str] = None, g=10.0, model_filename="model.keras", window_size=4, reward_function_id = 0, initial_state=0):
        self.model = tf.keras.models.load_model(model_filename, compile=False)
        self.window_size = window_size

        self.reward_function_id = reward_function_id
        self.initial_state = initial_state
        self.kde = kde
        # FIFO-buffer to store state history
        self.stateBuffer = collections.deque(maxlen=window_size)
        

        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = gym.spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        th, thdot = self.state  # th := theta

        self.steps += 1

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        #######potentielle optionen für den adjusted reward:
        # künstliches Terminieren der episode ab zu geringer dichte (bzw out of bounds)
        # harter cut vs term z.b. expontentiell?
        # grenzen des zustandsraum beachten

        ##costs werden hier auf dem alten state brechnet, wir haben auf dem neuen state berechnet
        ######hier jetzt den adjusted reward reinbringen

        if(self.reward_function_id == 0):
            ##standard cost function
            costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        else:
            #jetzt dichte des states berechnen, damit dann adjusted reward function
            state_features = pd.DataFrame([[np.cos(th), np.sin(th), thdot]], columns=["cosAngle", "sinAngle", "angVel"])
            log_density = self.kde.score_samples(state_features)
            state_density = np.exp(log_density)

            #hier jetzt verschieden reward Funktionen, die die Dichte jeweils anders bewerten
            ####Density Adjusted Reward 1
            if(self.reward_function_id == 1):
                #Dichte wird mit einem Faktor reingebracht, hier ist der reward im besten Fall dann jedoch etwa bei 0.5 und kann nie null werden
                #theoretisch könnten wir den faktor auch noch als optionalen parameter mitgeben
                costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2) + 1/(10*state_density)
            ####Density Adjusted Reward 2
            elif(self.reward_function_id == 2):
                # diese reward funktion soll States mit einer Dichte < x generell mit kosten von 2 bestrafen
                # Werte darüber werden nicht bestraft
                if(state_density < 0.01):
                    costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2) + 30
                else: 
                    costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
            ####Density Adjusted Reward 3
            elif(self.reward_function_id == 3):
                # reward funktion bestraft states mit einer geringen Dichte pauschal stark und zusätzlich wird die Episode auf done gesetzt
                if(state_density < 0.01):
                    costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2) + 100
                else:
                    costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
            else:
                print("no reward function with the given id %d specified yet" % self.reward_function_id)
        
        
        #ab hier alles wie im normalen custom environment

        # append last state + new action to state buffer
        netInput = [np.cos(th), np.sin(th), thdot, u]
        self.stateBuffer.append(np.float64(netInput))


        state = np.array([list(self.stateBuffer)])

        # NN recall
        netOutput = self.model.predict(np.float64(state), verbose=0)[0]

        # Begrenzung der Ausgabe auf begrenzenden Datenbereich
        netOutput = np.clip(netOutput, output_min, output_max)
                
        # retrieve new state
        #self.state = np.float64([netOutput[0], netOutput[1], netOutput[2]])

        newth = np.arctan2(netOutput[1], netOutput[0])  # angle from sin, cos
        newthdot = netOutput[2]

        # Compute done condition
        done = False
        if self.steps >= 200 or (self.reward_function_id == 3 and state_density < 0.01):  # Episode ends after 200 steps
            done = True


        #newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        #newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        #newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        
        return self._get_obs(), -costs, done, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])

        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        
        if(self.initial_state == 1):
            # Theta soll im unteren Bereich starten (theta < -2pi/3 oder theta > 2pi/3)
            # Wir definieren einen Bereich um die untere Position herum
            lower_bound_theta = 2 * np.pi / 3
            upper_bound_theta = -2 * np.pi / 3

            # Randomly choose theta in the lower section: either < -2pi/3 or > 2pi/3
            if np.random.rand() > 0.5:
                theta = np.random.uniform(lower_bound_theta, np.pi)
            else:
                theta = np.random.uniform(-np.pi, upper_bound_theta)

            thetadot = np.random.uniform(-1, 1)  # Initial angular velocity (you can adjust this range)
            self.state = np.array([theta, thetadot])

        if(self.initial_state == 2):
            # Starte immer ganz unten in Ruhe
            thetadot = 0
            theta = np.pi
            self.state = np.array([theta, thetadot])


        self.last_u = None

        self.steps = 0

        # Fill the buffer with the same initial state
        for i in range(self.window_size):
            action_sample = self.action_space.sample()
            self.stateBuffer.append(np.append(self._get_obs(), action_sample))

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi