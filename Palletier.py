# coding: utf-8

# # Palletier
# Palletier is a Python implementation of the solution for the distributer's
# pallet packing problem presented by Erhan Baltacioğlu in his thesis
#   The distributer's three-dimensional pallet-packing problem: a human
#   intelligence-based heuristic approach.
import collections
from itertools import permutations
from copy import copy, deepcopy

from Box import Box
from Packer import Packer
from Pallet import Pallet
from PackedPallet import PackedPallet

Dims = collections.namedtuple('Dims', ['dim1', 'dim2', 'dim3'])

class Solver:
    """The volume optimization solver"""
    def __init__(self):
        self.pallets = []
        self.boxes = []
        self.total_num_boxes = 0
        self.total_boxes_vol = 0
        self.packed_pallets = []

    def initialize_from_file(self, filename):
        filestring = f'inputs/{filename}.txt'
        with open(filestring, 'r', encoding='utf-8') as input_file:
            pallet_num = int(input_file.readline())
            for i in range(pallet_num):
                pallet_line = input_file.readline()
                pallet_dims = list(int(dim.strip())
                                   for dim in pallet_line.strip().split(','))
                self.pallets.append(Pallet(pallet_dims))
            for line in input_file:
                *dims, num = (int(dim.strip()) for dim in line.strip().split(','))
                for i in range(num):
                    self.boxes.append(Box(dims))
        self.total_boxes_vol = sum(box.vol for box in self.boxes)
        self.total_num_boxes = len(self.boxes)

    def pack(self):
        remaining_boxes = self.boxes
        while len(remaining_boxes) != 0:  # All boxes need to be packed
            single_solutions = []  # A solution for each pallet type
            for pallet in self.pallets:
                packer = Packer(remaining_boxes, pallet)
                pallet_ori, packed, unpacked, vol_util = packer.iterations()
                single_solutions.append((pallet_ori, packed, unpacked, vol_util))
                pallet.weight = 0  # Reset weight for next iteration
            # Get the best solution by utilization percentage
            best_pallet, best_packed, best_unpacked, vol_util = max(single_solutions,
                                                                key=lambda x: x[3])
            # Make this a test
            # The boxes we sent to pack do not fit into any pallets
            if len(best_unpacked) == len(remaining_boxes):
                for box in best_unpacked:
                    box.orientation = box.dims
                    self.packed_pallets.append(PackedPallet(
                        Pallet(dims=box.dims, name='BOX', ptype=0,
                               orientation=box.dims),
                        [box],
                    ))
                break
            else:
                self.packed_pallets.append(PackedPallet(copy(best_pallet),
                                                        deepcopy(best_packed)))
                remaining_boxes = best_unpacked

    def print_solution(self):
        for packed in self.packed_pallets:
            dims = packed.pallet.orientation
            print('Packed Pallet #{0} with utilization of {1}'.format(
                    packed.idx, packed.utilization))
            print('Using Pallet #{0} with dims ({1}, {2}, {3})'.format(
                    packed.pallet.idx, (*dims)))
            print('With {0} boxes:'.format(packed.num_boxes))
            for box in packed.boxes:
                print('Box #{0} with dims ({1}, {2}, {3})'.format(box.idx,
                                                            (*box.orientation)),
                      end=' ')
                print('located at ({0}, {1}, {2})'.format((*box.pos)))

def main():
    test_case = input('Enter test case: ')
    if not test_case:
        test_case = '6'
    print('Solving test case {}'.format(test_case))
    solver = Solver()
    solver.initialize_from_file(test_case)
    solver.pack()
    solver.print_solution()



if __name__ == '__main__':
    main()
