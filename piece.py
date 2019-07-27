class Piece:
    EMPTY_CELL_ID = 10
    DAUGHTER_PIECE_ID = 8
    EMPTY_CELLS = [(4, 1), (4, 2)]
    ACTION_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (0, -1), 3: (1, 0)}

    def __init__(self, cells):
        self._occupied_cells = set(cells)

    @staticmethod
    def init_piece(piece_id):
        """
        7 8 8 9
        7 8 8 9
        4 5 5 6
        4 1 2 6
        0     3

        :param piece_id: ID of the piece as shown above
        :return: A Piece object
        """
        if piece_id == 0:
            return Piece([(4, 0)])
        elif piece_id == 1:
            return Piece([(3, 1)])
        elif piece_id == 2:
            return Piece([(3, 2)])
        elif piece_id == 3:
            return Piece([(4, 3)])
        elif piece_id == 4:
            return Piece([(2, 0), (3, 0)])
        elif piece_id == 5:
            return Piece([(2, 1), (2, 2)])
        elif piece_id == 6:
            return Piece([(2, 3), (3, 3)])
        elif piece_id == 7:
            return Piece([(0, 0), (1, 0)])
        elif piece_id == 8:
            return Piece([(0, 1), (0, 2), (1, 1), (1, 2)])
        elif piece_id == 9:
            return Piece([(0, 3), (1, 3)])
        else:
            raise ValueError("Undefined piece ID!")

    def step(self, action):
        """
        The direction to move this piece towards

        :param action: 0->N, 1->E, 2->W, 3->S
        :return: None
        """
        cells = self.get_new_occupied_cells(action)
        self._occupied_cells = cells

    def get_occupied_cells(self):
        return self._occupied_cells

    def get_new_occupied_cells(self, action):
        dx, dy = self.ACTION_DELTAS[action]
        return set([(cell[0]+dx, cell[1]+dy) for cell in self._occupied_cells])
