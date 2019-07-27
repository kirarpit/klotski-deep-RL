import numpy as np
from piece import Piece
from gym.spaces import Box, Discrete
from utils import setup_logger
import gym

HEIGHT = 5
WIDTH = 4
NUM_PIECES = 10
NUM_ACTION_PER_PIECE = 4
MAX_STEPS = 1000

log = setup_logger("env")


class KlotskiEnv(gym.Env):
    def __init__(self):
        self.state = None
        self.pieces = None
        self.action_space = Discrete(NUM_PIECES*NUM_ACTION_PER_PIECE)
        self.observation_space = Box(0, NUM_PIECES, (HEIGHT*WIDTH, ), np.int)
        self._step_cnt = 0

    def step(self, action):
        self._step_cnt += 1
        reward = 2
        done = False
        piece_id = action//NUM_ACTION_PER_PIECE
        action_direction = action - piece_id*NUM_ACTION_PER_PIECE

        # check if valid move
        current_cells = self.pieces[piece_id].get_occupied_cells()
        new_cells = self.pieces[piece_id].get_new_occupied_cells(action_direction)
        new_delta_cells = new_cells - current_cells
        old_delta_cells = current_cells - new_cells
        is_valid_move = len(new_delta_cells) == sum([int(0 <= cell[0] < HEIGHT and
                                                         0 <= cell[1] < WIDTH and
                                                         self.state[cell[0]][cell[1]] == Piece.EMPTY_CELL_ID)
                                                     for cell in new_delta_cells])

        # update game state
        if is_valid_move:
            self.pieces[piece_id].step(action_direction)
            self.mark_cells(new_delta_cells, piece_id)
            self.mark_cells(old_delta_cells, Piece.EMPTY_CELL_ID)

        # check if terminal condition and set reward
        if self._step_cnt >= MAX_STEPS:
            reward = -1
            done = True
        else:
            if not is_valid_move:
                reward = -1
            elif (self.state[3][1] == Piece.DAUGHTER_PIECE_ID and self.state[3][2] == Piece.DAUGHTER_PIECE_ID and
                  self.state[4][1] == Piece.DAUGHTER_PIECE_ID and self.state[4][2] == Piece.DAUGHTER_PIECE_ID):
                reward = 10
                done = True

        log.debug("action {}, Game state {}, reward {}, is_terminal {}".format(action, self.state, reward, done))
        return self.get_state(), reward, done, {}

    def reset(self):
        self.state = np.zeros((HEIGHT, WIDTH), dtype=np.int)
        self.pieces = {}
        for piece_id in range(NUM_PIECES):
            piece = Piece.init_piece(piece_id)
            self.pieces[piece_id] = piece
            self.mark_cells(piece.get_occupied_cells(), piece_id)

        # mark empty cells
        self.mark_cells(Piece.EMPTY_CELLS, Piece.EMPTY_CELL_ID)

        return self.get_state()

    def get_state(self):
        return np.ravel(self.state)

    def mark_cells(self, cells, piece_id):
        for cell in cells:
            self.state[cell[0]][cell[1]] = piece_id

    def play(self, piece_id, action):
        self.step(piece_id*NUM_ACTION_PER_PIECE + action)

    def close(self):
        pass

    def render(self, mode='human'):
        pass
