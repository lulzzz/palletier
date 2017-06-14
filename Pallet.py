from collections import namedtuple
import itertools

Coords = namedtuple('Coords', ['x', 'y', 'z'])
Dims = namedtuple('Dims', ['dim1', 'dim2', 'dim3'])


class Pallet:
    idx_gen = itertools.count(start=0, step=1)

    def __init__(self, dims, idx=None, max_weight=20, name='N/A', ptype=1,
                 orientation=Dims(0, 0, 0)):
        if idx is not None:
            self.idx = idx
        else:
            self.idx = next(Pallet.idx_gen)
        self.name = name
        self.type = ptype
        self.dims = Dims(*dims)
        self.orientation = orientation
        self.weight = 0
        self.max_weight = max_weight
        self.vol = 1
        for dim in self.dims:
            self.vol *= dim

    def __repr__(self):
        repr_str = (f'Pallet(idx={self.idx}, dims={self.dims},' +
                    f' max_weight={self.max_weight}, name={self.name}, type={self.type})')
        return repr_str