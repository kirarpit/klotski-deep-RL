import numpy as np
from piece import Piece, EMPTY_CELL_ID, DAUGHTER_PIECE_IDS
from gym.spaces import Box, Discrete
from utils import setup_logger
import gym
import time

HEIGHT = 5
WIDTH = 4
NUM_PIECES = 10
NUM_ACTION_PER_PIECE = 4
MAX_STEPS = 5000

log = setup_logger("env")


class KlotskiEnv(gym.Env):
    REWARDS = {"max_steps": 0,
               "novel_state": 0.1,
               "invalid_move": 0,
               "solved": 10}

    def __init__(self):
        self.state = None
        self.pieces = None
        self.action_space = Discrete(NUM_PIECES*NUM_ACTION_PER_PIECE)
        self.observation_space = Box(0, NUM_PIECES, (HEIGHT*WIDTH, ), np.int)
        self._step_cnt = None
        self.is_over = None
        self.viewer = None
        self.visited_states = None

    def step(self, action):
        self._step_cnt += 1
        reward = 0
        done = False
        piece_id = action//NUM_ACTION_PER_PIECE
        action_direction = action - piece_id*NUM_ACTION_PER_PIECE

        # check if valid move
        is_valid_action, new_delta_cells, old_delta_cells = self.is_valid_action(action)

        # update game state
        if is_valid_action:
            self.pieces[piece_id].step(action_direction)
            self.mark_cells(new_delta_cells, piece_id)
            self.mark_cells(old_delta_cells, EMPTY_CELL_ID)

        # check if terminal condition and set reward
        if self._step_cnt >= MAX_STEPS:
            reward = self.REWARDS["max_steps"]
            done = True
        else:
            if not is_valid_action:
                reward = self.REWARDS["invalid_move"]
            elif (self.state[3][1] == DAUGHTER_PIECE_IDS[0] and self.state[3][2] == DAUGHTER_PIECE_IDS[0] and
                  self.state[4][1] == DAUGHTER_PIECE_IDS[0] and self.state[4][2] == DAUGHTER_PIECE_IDS[0]):
                reward = self.REWARDS["solved"]
                done = True

        log.debug("action {}, Game state {}, reward {}, is_terminal {}".format(action, self.state, reward, done))

        # Check if previously visited. Reward if not.
        if self.get_state_id() not in self.visited_states:
            self.visited_states.add(self.get_state_id())
            reward += self.REWARDS["novel_state"]

        self.is_over = done
        return self.get_state(), reward, done, {}

    def reset(self):
        self.state = np.zeros((HEIGHT, WIDTH), dtype=np.int)
        self.pieces = {}
        self._step_cnt = 0
        self.is_over = False
        self.visited_states = set()
        for piece_id in range(NUM_PIECES):
            piece = Piece.init_piece(piece_id)
            self.pieces[piece_id] = piece
            self.mark_cells(piece.get_occupied_cells(), piece_id)

        # mark empty cells
        self.mark_cells(Piece.get_empty_cells(), EMPTY_CELL_ID)

        return self.get_state()

    def is_valid_action(self, action):
        piece_id = action//NUM_ACTION_PER_PIECE
        action_direction = action - piece_id*NUM_ACTION_PER_PIECE
        current_cells = self.pieces[piece_id].get_occupied_cells()
        new_cells = self.pieces[piece_id].get_new_occupied_cells(action_direction)
        new_delta_cells = new_cells - current_cells
        old_delta_cells = current_cells - new_cells
        is_valid_action = len(new_delta_cells) == sum([int(0 <= cell[0] < HEIGHT and
                                                           0 <= cell[1] < WIDTH and
                                                           self.state[cell[0]][cell[1]] == EMPTY_CELL_ID)
                                                       for cell in new_delta_cells])
        return is_valid_action, new_delta_cells, old_delta_cells

    def get_state(self):
        return np.ravel(self.state)

    def get_simple_state(self):
        state = []
        for piece_id in self.get_state():
            if piece_id != EMPTY_CELL_ID:
                state.append(self.pieces[piece_id].get_piece_type_id())
            else:
                state.append(piece_id)
        return str(state)

    def mark_cells(self, cells, piece_id):
        for cell in cells:
            self.state[cell[0]][cell[1]] = piece_id

    def play(self, piece_id, action):
        self.step(piece_id*NUM_ACTION_PER_PIECE + action)

    def get_state_id(self):
        return str(self.get_state())

    def get_valid_actions(self):
        valid_actions = []
        for action in range(NUM_PIECES*NUM_ACTION_PER_PIECE):
            if self.is_valid_action(action)[0]:
                valid_actions.append(action)
        return valid_actions

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        sf = 100
        margin = 0.05

        if self.viewer is None:
            self.viewer = rendering.Viewer(WIDTH*sf, HEIGHT*sf)

        for piece_id in range(NUM_PIECES):
            cells = list(self.pieces[piece_id].get_occupied_cells())
            cells.sort()
            cell = cells[0]
            color = self.pieces[piece_id].get_color()
            vertices = self.pieces[piece_id].get_vertices()
            margins = [(margin, margin), (margin, -margin), (-margin, -margin), (-margin, margin)]
            vertices = [(vertex[0]+margin[0], vertex[1]+margin[1]) for vertex, margin in list(zip(vertices, margins))]
            vertices = [(cell[0] + vertex[0], cell[1] + vertex[1]) for vertex in vertices]
            vertices = [(v[1]*sf, (HEIGHT-v[0])*sf) for v in vertices]
            self.viewer.draw_polygon(vertices, color=color)

        if mode == 'human':
            time.sleep(1/50)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
