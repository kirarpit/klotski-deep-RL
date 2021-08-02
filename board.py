import numpy as np
from piece import PieceFactory


def get_islands(grid):
    def get_neighs(cell):
        neighs = []
        deltas = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for delta in deltas:
            neigh = cell[0] + delta[0], cell[1] + delta[1]
            if 0 <= neigh[0] < len(grid) and 0 <= neigh[1] < len(grid[0]):
                neighs.append(neigh)
        return neighs

    def find_island_cells(cell, value, island, visited):
        visited.add(cell)
        island.add(cell)

        for neigh in get_neighs(cell):
            if grid[neigh[0]][neigh[1]] == value and neigh not in visited:
                find_island_cells(neigh, grid[neigh[0]][neigh[1]], island,
                                  visited)

    islands = []
    visited = set()
    for row_id in range(len(grid)):
        for col_id in range(len(grid[0])):
            if (row_id, col_id) not in visited:
                island = set()
                find_island_cells((row_id, col_id), grid[row_id][col_id],
                                  island, visited)
                islands.append((island, grid[row_id][col_id]))

    return islands


class Board:
    def __init__(self):
        # grid
        self.height = None
        self.width = None

        self._step_cnt = 0
        self._is_over = False

        self.piece_factory = None
        self.pieces = {}
        self.goal_piece = None

        self._layout = None
        self.goal_cells = set()
        self.load_layout()

    def __str__(self):
        print(self.state)
        return ""

    @property
    def step_cnt(self):
        return self._step_cnt

    @property
    def solved(self):
        return self._is_over

    def load_layout(self, filename="default.kl"):
        # read file and get grid
        with open(filename) as f:
            lines = [list(line) for line in f.read().splitlines()]
            grid = np.array(lines)
        self._layout = grid
        self.reset()

    def reset(self):
        self._step_cnt = 0
        self._is_over = False
        self.goal_piece = None
        self.height, self.width = self._layout.shape
        self.goal_cells = {(self.height - 1, self.width // 2 - 1),
                           (self.height - 2, self.width // 2 - 1),
                           (self.height - 1, self.width // 2),
                           (self.height - 2, self.width // 2)}

        # get islands and initialize pieces
        self.piece_factory = PieceFactory()
        self.pieces = {}
        for island, value in get_islands(self._layout):
            if value != ".":
                piece = self.piece_factory.new(island)
                self.pieces[piece.uid] = piece
                if value == "*":
                    self.goal_piece = piece

        assert self.goal_piece is not None, \
            "goal piece must be defined using '*' character"
        return self.state

    @property
    def empty_cells(self):
        rows, cols = np.where(self.state == 0)
        return set(list(zip(rows, cols)))

    @property
    def state(self):
        state = np.zeros((self.height, self.width), dtype=int)
        for uid, piece in self.pieces.items():
            state[tuple(zip(*piece.cells))] = uid
        return state

    def step(self, piece_uid, direction):
        if self._is_over:
            raise ValueError("Game is already over!")
        self._step_cnt += 1

        # check if valid action
        current_cells = self.pieces[piece_uid].cells
        new_cells = self.pieces[piece_uid].get_new_occupied_cells(direction)
        assert type(current_cells) == type(new_cells)
        assert isinstance(current_cells, set)

        new_delta_cells = new_cells - current_cells
        empty_cells = self.empty_cells
        is_valid_action = all([int(0 <= cell[0] < self.height and
                                   0 <= cell[1] < self.width and
                                   cell in empty_cells)
                               for cell in new_delta_cells])

        if is_valid_action:
            self.pieces[piece_uid].step(direction)

        # check if solved
        if self.goal_piece.cells == self.goal_cells:
            self._is_over = True

        return self.state, is_valid_action

    def save_board(self, filename=None):
        pass