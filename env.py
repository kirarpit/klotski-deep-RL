import numpy as np
from gym.spaces import Box, Discrete
from board import Board
from utils import setup_logger, merge_dicts
import gym
import time

NUM_ACTION_PER_PIECE = 4

log = setup_logger("env")

ENV_DEFAULT_CONFIG = {
    "max_steps": 5000,
    "rewards": {
        "max_steps": 0,
        "per_step_penalty": -0.01,
        "novel_state": 0.1,
        "invalid_move": 0,
        "solved": 10
    },
}


class KlotskiEnv(gym.Env):
    def __init__(self, config):
        # check if all env_config keys are in default config
        if not all(key in ENV_DEFAULT_CONFIG for key in config.keys()):
            raise KeyError(
                "Custom environment configuration not found in default "
                "configuration.")
        self.config = merge_dicts(ENV_DEFAULT_CONFIG, config)

        self.board = Board()
        num_pieces = len(self.board.pieces)
        self.action_space = Discrete(num_pieces * NUM_ACTION_PER_PIECE)
        self.observation_space = Box(low=0, high=num_pieces,
                                     shape=(
                                         self.board.height, self.board.width),
                                     dtype=np.int)
        self.viewer = None
        self.reset()

    def step(self, action: int):
        info = {}
        done = False
        reward = self.config["rewards"]["per_step_penalty"]
        piece_id = action // NUM_ACTION_PER_PIECE
        action_direction = action - piece_id * NUM_ACTION_PER_PIECE

        state, is_valid = self.board.step(piece_id, action_direction)

        # check if terminal condition and set reward
        if not is_valid:
            reward += self.config["rewards"]["invalid_move"]
        elif self.board.solved:
            reward += self.config["rewards"]["solved"]
            done = True

        if (not self.board.solved and
                self.board.step_cnt >= self.config["max_steps"]):
            reward += self.config["rewards"]["max_steps"]
            done = True

        log.debug(
            "action {}, Game state {}, reward {}, is_terminal {}".format(action,
                                                                         state,
                                                                         reward,
                                                                         done))
        return state, reward, done, info

    def reset(self):
        return self.board.reset()

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        sf = 100
        margin = 0.05

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.board.width * sf,
                                           self.board.height * sf)

        for _, piece in self.board.pieces.items():
            color = piece.color

            # add margins to vertices
            margins = [(margin, margin), (margin, -margin), (-margin, -margin),
                       (-margin, margin)]
            vertices = piece.vertices
            vertices = [(vertex[0] + margin[0], vertex[1] + margin[1]) for
                        vertex, margin in list(zip(vertices, margins))]

            # scale and flip
            vertices = [(v[1] * sf, (self.board.height - v[0]) * sf) for v in
                        vertices]
            self.viewer.draw_polygon(vertices, color=color)

        if mode == 'human':
            time.sleep(1 / 100)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
