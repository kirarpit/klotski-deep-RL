import matplotlib.pyplot as plt


class PieceFactory:
    def __init__(self):
        self.piece_cnt = 0
        self.piece_type_cnt = 0
        self.piece_shape_type_map = {}
        self.cmap = plt.get_cmap("tab10")

    def new(self, cells):
        # assign unique ID
        self.piece_cnt += 1
        piece_uid = self.piece_cnt

        # get piece type
        piece_type = self.get_piece_type(cells)

        # set color
        piece_color = self.cmap(piece_type)[:-1]

        return Piece(cells, piece_uid, piece_type, piece_color)

    def get_piece_type(self, cells):
        row_ids, col_ids = zip(*cells)
        shape = max(row_ids) - min(row_ids), max(col_ids) - min(col_ids)
        if shape not in self.piece_shape_type_map:
            self.piece_shape_type_map[shape] = self.piece_type_cnt
            self.piece_type_cnt += 1
        return self.piece_shape_type_map[shape]


class Piece:
    ACTION_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def __init__(self, cells, uid, type_id, color):
        self._occupied_cells = set(cells)
        self.uid = uid
        self.type = type_id
        self.color = color

    @property
    def cells(self):
        return self._occupied_cells

    @property
    def vertices(self):
        row_ids, col_ids = zip(*self._occupied_cells)
        return [(min(row_ids), min(col_ids)),
                (min(row_ids), max(col_ids) + 1),
                (max(row_ids) + 1, max(col_ids) + 1),
                (max(row_ids) + 1, min(col_ids))]

    def step(self, action):
        """
        The direction to move this piece towards

        :param action: 0->N, 1->E, 2->W, 3->S
        :return: None
        """
        self._occupied_cells = self.get_new_occupied_cells(action)

    def get_new_occupied_cells(self, action):
        dx, dy = self.ACTION_DELTAS[action]
        return set(
            [(cell[0] + dx, cell[1] + dy) for cell in self._occupied_cells])
