import collections
import itertools

Coords = collections.namedtuple('Coords', ['x', 'y', 'z'])
Dims = collections.namedtuple('Dims', ['dim1', 'dim2', 'dim3'])


class Box:
    idx_gen = itertools.count(start=0, step=1)

    def __init__(self, dims, weight=0, idx=None, pos=Coords(0, 0, 0),
                 orientation=Dims(0, 0, 0), name='N/A'):
        if idx is not None:
            self.idx = idx
        else:
            self.idx = next(Box.idx_gen)
        self.name = name
        self.is_packed = False
        self.dims = Dims(*dims)
        self.weight = weight
        self.pos = pos
        self.orientation = orientation
        self.vol = 1
        for dim in self.dims:
            self.vol *= dim

    def __eq__(self, other):
        return self.idx == other.idx

    def __repr__(self):
        repr_str = f'Box(idx={self.idx}, dims={self.dims}, weight={self.weight}'
        if self.pos != Coords(0, 0, 0):
            repr_str += f', pos={self.pos}'
        if self.orientation != Dims(0, 0, 0):
            repr_str += f', orientation={self.orientation}'
        repr_str += ')'
        return repr_str

    def __str__(self):
        output = f'Box({self.dims.dim1}, {self.dims.dim2}, {self.dims.dim3}'
        if self.weight != 0:
            output += f', weight={self.weight}'
        output += ')'
        return output
