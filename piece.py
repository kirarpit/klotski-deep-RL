# Piece encoding
EMPTY_CELL_ID = 10
SERVANTS_PIECE_IDS = [0, 1, 2, 3]
HORIZONTAL_PIECE_IDS = [5]
VERTICAL_PIECE_IDS = [4, 6, 7, 9]
DAUGHTER_PIECE_IDS = [8]

# All types of pieces and their color and vertices mapping
PIECE_TYPES = [SERVANTS_PIECE_IDS, HORIZONTAL_PIECE_IDS, VERTICAL_PIECE_IDS, DAUGHTER_PIECE_IDS]
PIECE_COLORS = [(0, 255, 0), (139, 0, 139), (0, 0, 255), (255, 0, 0)]
PIECE_VERTICES = [[(0, 0), (0, 1), (1, 1), (1, 0)],
                  [(0, 0), (0, 2), (1, 2), (1, 0)],
                  [(0, 0), (0, 1), (2, 1), (2, 0)],
                  [(0, 0), (0, 2), (2, 2), (2, 0)]]


class Piece:
    ACTION_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (0, -1), 3: (1, 0)}

    def __init__(self, cells, color, vertices, piece_type_id):
        self._occupied_cells = set(cells)
        self.color = color
        self.vertices = vertices
        self.piece_type_id = piece_type_id

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

        # get occupied cells
        if piece_id == 0:
            cells = [(4, 0)]
        elif piece_id == 1:
            cells = [(3, 1)]
        elif piece_id == 2:
            cells = [(3, 2)]
        elif piece_id == 3:
            cells = [(4, 3)]
        elif piece_id == 4:
            cells = [(2, 0), (3, 0)]
        elif piece_id == 5:
            cells = [(2, 1), (2, 2)]
        elif piece_id == 6:
            cells = [(2, 3), (3, 3)]
        elif piece_id == 7:
            cells = [(0, 0), (1, 0)]
        elif piece_id == 8:
            cells = [(0, 1), (0, 2), (1, 1), (1, 2)]
        elif piece_id == 9:
            cells = [(0, 3), (1, 3)]
        else:
            raise ValueError("Undefined piece ID!")

        # get color
        color = None
        vertices = None
        piece_type_id = None
        for piece_type in PIECE_TYPES:
            if piece_id in piece_type:
                piece_type_id = PIECE_TYPES.index(piece_type)
                color = PIECE_COLORS[piece_type_id]
                vertices = PIECE_VERTICES[piece_type_id]
                break

        return Piece(cells, color, vertices, piece_type_id)

    @staticmethod
    def get_empty_cells():
        return [(4, 1), (4, 2)]

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

    def get_color(self):
        return self.color

    def get_vertices(self):
        return self.vertices

    def get_piece_type_id(self):
        return self.piece_type_id

    def get_new_occupied_cells(self, action):
        dx, dy = self.ACTION_DELTAS[action]
        return set([(cell[0]+dx, cell[1]+dy) for cell in self._occupied_cells])
