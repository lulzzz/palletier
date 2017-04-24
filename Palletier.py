# coding: utf-8

# # Palletier
# Palletier is a Python implementation of the solution for the distributer's
# pallet packing problem presented by Erhan Baltacioğlu in his thesis
#   The distributer's three-dimensional pallet-packing problem: a human
#   intelligence-based heuristic approach.
import collections
from itertools import permutations
import numpy as np
# import pdb
import q

Coords = collections.namedtuple('Coords', ['x', 'y', 'z'])
Dims = collections.namedtuple('Dims', ['dim1', 'dim2', 'dim3'])
Layer = collections.namedtuple('Layer', ['width', 'value'])


class Corner:
    def __init__(self, x, z):
        self.x = x
        self.z = z

    def __repr__(self):
        return "Corner(x={0}, z={1})".format(self.x, self.z)


class Topology:
    def __init__(self, pallet_x, pallet_z):
        self.corners = [Corner(pallet_x, 0)]
        self.max_x = pallet_x
        self.max_z = pallet_z

    @property
    def smallest(self):
        return min(self.corners, key=lambda x: x.z)

    @smallest.deleter
    def smallest(self):
        self.corners.remove(self.smallest)

    @property
    def smallest_index(self):
        return self.corners.index(self.smallest)

    @property
    def smallest_next(self):
        """The next corner after the smallest gap"""
        return self.corners[self.smallest_index + 1]

    @smallest_next.deleter
    def smallest_next(self):
        self.corners.remove(self.smallest_next)

    @property
    def smallest_prev(self):
        """The corner just before the smallest_gap"""
        return self.corners[self.smallest_index - 1]

    @smallest_prev.deleter
    def smallest_prev(self):
        self.corners.remove(self.smallest_prev)

    def update(self, dims, situation):
        """
        Update the topology according to the current situation
        and the dimensions of the box that was packed.

        Return the x coordinate at which the box was packed.
        """
        x_coord = None
        if situation == 1:
            x_coord = 0
            if dims.dim1 == self.smallest.x:
                self.smallest.z += dims.dim3
            else:
                new_corner = Corner(dims.dim1, self.smallest.z + dims.dim3)
                self.corners.insert(0, new_corner)

        elif situation == 2:
            if dims.dim1 == self.smallest.x:
                x_coord = 0
                if self.smallest.z + dims.dim3 == self.smallest_next.z:
                    del self.smallest
                else:
                    self.smallest.z += dims.dim3
            else:
                x_coord = self.smallest.x - dims.dim1
                if self.smallest.z + dims.dim3 == self.smallest_next.z:
                    self.smallest.x -= dims.dim1
                else:
                    new_corner = Corner(self.smallest.x,
                                        self.smallest.z + dims.dim3)
                    self.corners.insert(1, new_corner)
                    self.smallest.x -= dims.dim1

        elif situation == 3:
            x_coord = self.smallest_prev.x
            if dims.dim1 == self.smallest.x - self.smallest_prev.x:
                if self.smallest.z + dims.dim3 == self.smallest_prev.z:
                    self.smallest_prev.x = self.smallest.x
                    del self.smallest
                else:
                    self.smallest.z += dims.dim3
            else:
                if self.smallest.z + dims.dim3 == self.smallest_prev.z:
                    self.smallest_prev.x += dims.dim1
                else:
                    new_corner = Corner(self.smallest_prev.x + dims.dim1,
                                        self.smallest.z + dims.dim3)
                    self.corners.insert(self.smallest_index, new_corner)
        elif situation == 4.1:
            if dims.dim1 == self.smallest.x - self.smallest_prev.x:
                x_coord = self.smallest_prev.x
                if self.smallest.z + dims.dim3 == self.smallest_next.z:
                    self.smallest_prev.x = self.smallest_next.x
                    del self.smallest
                else:
                    self.smallest.z += dims.dim3
            elif self.smallest.smallest_prev.x < (self.max_x - self.smallest.x):
                if self.smallest.z + dims.dim3 == self.smallest_prev.z:
                    x_coord = self.smallest.x - dims.dim1
                    self.smallest.x -= dims.dim1
                else:
                    x_coord = self.smallest_prev.x
                    new_corner = Corner(self.smallest_prev.x + dims.dim1,
                                        self.smallest.z + dims.dim3)
                    self.corners.insert(self.smallest_index, new_corner)
            else:
                if self.smallest.z + dims.dim3 == self.smallest_prev.z:
                    x_coord = self.smallest_prev.x
                    self.smallest_prev.x += dims.dim1
                else:
                    x_coord = self.smallest.x - dims.dim1
                    new_corner = Corner(self.smallest.x,
                                        self.smallest.z + dims.dim3)
                    self.corners.insert(self.smallest_index + 1, new_corner)
                    self.smallest.x -= dims.dim1

        elif situation == 4.2:
            x_coord = self.smallest_prev.x
            if dims.dim1 == self.smallest.x - self.smallest_prev.x:
                if self.smallest.z + dims.dim3 == self.smallest_prev.z:
                    self.smallest_prev.x = self.smallest.x
                    del self.smallest
                else:
                    self.smallest.z += dims.dim3
            else:
                if self.smallest.z + dims.dim3 == self.smallest_prev.z:
                    self.smallest_prev.x += dims.dim1
                elif self.smallest.z + dims.dim3 == self.smallest_next.z:
                    x_coord = self.smallest.x - dims.dim1
                    self.smallest.x -= dims.dim1
                else:
                    new_corner = Corner(self.smallest_prev.x + dims.dim1,
                                        self.smallest.z + dims.dim3)
                    self.corners.insert(self.smallest_index - 1, new_corner)
        return x_coord

    def even(self):
        """
        Even out the topology, leaving only one corner at the lower right
        corner available
        """
        pass


class Box:
    def __init__(self, dims, pos=Coords(0, 0, 0), orientation=Dims(0, 0, 0)):
        self.status = False
        self.dims = Dims(*dims)
        self.pos = pos
        self.orientation = orientation
        self.vol = np.prod(self.dims)

    def __eq__(self, other):
        return self.dims == other.dims

    def __repr__(self):
        repr_str = 'Box(dims={0}, pos={1}, orientation={2})'
        return repr_str.format(self.dims, self.pos, self.orientation)


class Solver:
    """The volume optimization solver"""
    @staticmethod
    def initialize(filename):
        print('Initializing')
        boxes = []
        with open('inputs/{0}.txt'.format(filename), 'r', encoding='utf-8') as input_file:
            pallet_line = input_file.readline()
            pallet_dims = list(int(dim.strip()) for dim in pallet_line.strip().split(','))
            for line in input_file:
                dim1, dim2, dim3, num = (int(dim.strip()) for dim in line.strip().split(','))
                for i in range(num):
                    boxes.append(Box((dim1, dim2, dim3)))
        return pallet_dims, boxes

    def reset_boxes(self):
        for box in self.boxes:
            box.status = False

    @staticmethod
    def get_candidate_layers(boxes, pallet_orientation):
        candidate_layers = []
        for box in boxes:
            # We only want (dim1, dim2, dim3), (dim2, dim1, dim3) and (dim3, dim1, dim2)
            for orientation in list(permutations(box.dims))[::2]:
                ex_dim, dim2, dim3 = orientation
                if (ex_dim > pallet_orientation[1] or
                        ((dim2 > pallet_orientation[0] or dim3 > pallet_orientation[2]) and
                         (dim2 > pallet_orientation[2] or dim3 > pallet_orientation[0]))):
                    continue
                same_width = False
                for layer in candidate_layers:
                    if layer.width == ex_dim:
                        same_width = True
                        break
                if same_width:
                    continue
                layer_value = sum(min(abs(ex_dim - dim)
                                      for dim in box2.dims)
                                  for box2 in boxes if box2 is not box)
                layer = Layer(width=ex_dim, value=layer_value)
                candidate_layers.append(layer)
        return candidate_layers

    def pack_box(self, box, coords, orientation):
        self.boxes[box].status = True
        self.boxes[box].pos = coords
        self.boxes[box].orientation = orientation
        print('{0}'.format(self.boxes[box]))
        self.packed_vol += self.boxes[box].vol
        self.num_packed += 1

    def get_box(self, max_len_x, gap_len_y, max_len_y, gap_len_z, max_len_z):
        min_y_diff = min_x_diff = min_z_diff = 99999
        other_y_diff = other_x_diff = other_z_diff = 99999
        # Best box in the best orientation
        best_match = (None, None)
        other_best_match = (None, None)
        checked = []
        for idx, box in enumerate(self.boxes):
            if box.status:
                continue
            if box in checked:
                continue
            else:
                checked.append(box)
            for orientation in set(permutations(box.dims)):
                dim1, dim2, dim3 = orientation
                if dim1 <= max_len_x and dim2 <= max_len_y and dim3 <= max_len_z:
                    if dim2 <= gap_len_y:
                        y_diff = gap_len_y - dim2
                        x_diff = max_len_x - dim1
                        z_diff = abs(gap_len_z - dim3)
                        if (y_diff, x_diff, z_diff) < (min_y_diff, min_x_diff, min_z_diff):
                            min_y_diff = y_diff
                            min_x_diff = x_diff
                            min_z_diff = z_diff
                            best_match = (idx, Dims(*orientation))
                    # The box doesn't quite fit the layer thickness
                    else:
                        y_diff = dim2 - gap_len_y
                        x_diff = max_len_x - dim1
                        z_diff = abs(gap_len_z - dim3)
                        if (y_diff, x_diff, z_diff) < (other_y_diff, other_x_diff, other_z_diff):
                            other_y_diff = y_diff
                            other_x_diff = x_diff
                            other_z_diff = z_diff
                            other_best_match = (idx, Dims(*orientation))

        return best_match, other_best_match


    def check_boxes(self, match, other_match, edge):
        self.edge_is_even = False
        if any(match):
            return match[0], match[1], None
        else:
            if any(other_match) and (self.layer_in_layer or len(edge.corners) == 1):
                if not self.layer_in_layer:
                    self.prev_layer = self.layer_thickness
                    self.lilz = edge.smallest.z
                self.layer_in_layer += (other_match[1][1] - self.layer_thickness)
                new_thickness = other_match[1][1]  # Match -> Orientation -> Y dimension
                return other_match[0], other_match[1], new_thickness
            else:
                #q.d()
                if len(edge.corners) == 1:
                    self.layer_finished = True
                else:
                    self.edge_is_even = True
                    if edge.smallest_index == 0:
                        del edge.smallest
                    elif edge.smallest is edge.corners[-1]:
                        edge.smallest_prev.x = edge.smallest.x
                        del edge.smallest
                    else:
                        if edge.smallest_prev.z == edge.smallest_next.z:
                            del self.smallest_prev
                            del self.smallest
                        else:
                            if edge.smallest_prev.z < edge.smallest_next.z:
                                self.smallest_prev.x = edge.smallest.x
                            del edge.smallest
                return None, None, None

    def get_layer(self, pallet_orientation, remaining_y):
        eval_value = 99999999
        layer_thickness = 0
        pallet_x, pallet_y, pallet_z = pallet_orientation
        for box in self.boxes:
            if box.status:
                continue
            for orientation in list(permutations(box.dims))[::2]:
                ex_dim, dim2, dim3 = orientation
                layer_eval = 0
                if (ex_dim <= remaining_y and
                        (dim2 <= pallet_x and dim3 <= pallet_z) or
                        (dim3 <= pallet_x and dim2 <= pallet_z)):
                    layer_eval = sum(min(abs(ex_dim - box_dim)
                                         for box_dim in box2.dims)
                                     for box2 in self.boxes
                                     if not box.status and box2 is not box)  # Different boxes
                    if layer_eval < eval_value:
                        eval_value = layer_eval
                        layer_thickness = ex_dim
        if layer_thickness == 0 or layer_thickness > remaining_y:
            self.packing = False
        return layer_thickness

    def pack_layer(self, layer_thickness, pallet_orientation,
                   remaining_y, remaining_z, packed_y):
        if layer_thickness == 0:
            self.packing = False
            return
        pallet_x = pallet_orientation[0]
        pallet_z = pallet_orientation[2]
        edge = Topology(pallet_x, pallet_z)
        while True:
            y_coord = packed_y
            z_coord = edge.smallest.z

            if len(edge.corners) == 1:  # Situation 1: No Box on sides of gap
                len_x = edge.smallest.x
                lpz = remaining_z - edge.smallest.z
                # Find candidate boxes
                match, other_match = self.get_box(len_x, layer_thickness,
                                                  remaining_y, lpz, lpz)
                # Get the best box to fit
                box, orientation, new_thickness = self.check_boxes(match,
                                                                   other_match,
                                                                   edge)
                if self.layer_finished:
                    break
                if self.edge_is_even:
                    continue
                if new_thickness:
                    self.layer_thickness = new_thickness

                x_coord = edge.update(orientation, situation=1)
                coords = Coords(x_coord, y_coord, z_coord)
                self.pack_box(box, coords, orientation)

            elif edge.smallest_index == 0:  # Situation 2: No Box on left side of gap
                len_x = edge.smallest.x
                len_z = edge.smallest_next.z - edge.smallest.z
                lpz = remaining_z - edge.smallest.z
                # Find candidate boxes
                match, other_match = self.get_box(len_x, layer_thickness,
                                                  remaining_y, len_z, lpz)
                # Get the best box to fit
                box, orientation, new_thickness = self.check_boxes(match,
                                                                   other_match,
                                                                   edge)
                if self.layer_finished:
                    break
                if self.edge_is_even:
                    continue
                if new_thickness:
                    self.layer_thickness = new_thickness

                x_coord = edge.update(orientation, situation=2)
                coords = Coords(x_coord, y_coord, z_coord)
                self.pack_box(box, coords, orientation)

            elif edge.smallest is edge.corners[-1]:  # Situation 3: No Box on right side of gap
                len_x = edge.smallest.x - edge.smallest_prev.x
                len_z = edge.smallest_prev.z - edge.smallest.z
                lpz = remaining_z - edge.smallest.z
                # Find candidate boxes
                match, other_match = self.get_box(len_x, layer_thickness,
                                                  remaining_y, len_z, lpz)
                box, orientation, new_thickness = self.check_boxes(match,
                                                                   other_match,
                                                                   edge)
                if self.layer_finished:
                    break
                if self.edge_is_even:
                    continue
                if new_thickness:
                    self.layer_thickness = new_thickness

                x_coord = edge.update(orientation, situation=3)
                coords = Coords(x_coord, y_coord, z_coord)
                self.pack_box(box, coords, orientation)

            elif edge.smallest_prev.z == edge.smallest_next.z:  # Siuation 4A: Z dims of the gap are the same on both sides
                len_x = edge.smallest.x - edge.smallest_prev.x
                len_z = edge.smallest_prev.z - edge.smallest.z
                lpz = remaining_z - edge.smallest.z
                match, other_match = self.get_box(len_x, layer_thickness,
                                                  remaining_y, len_z, lpz)
                box, orientation, new_thickness = self.check_boxes(match,
                                                                   other_match,
                                                                   edge)
                if self.layer_finished:
                    break
                if self.edge_is_even:
                    continue
                if new_thickness:
                    self.layer_thickness = new_thickness

                x_coord = edge.update(orientation, situation=4.1)
                coords = Coords(x_coord, y_coord, z_coord)
                self.pack_box(box, coords, orientation)
            else:  # Situation 4B: z dims of the gap are different on the sides
                len_x = edge.smallest.x - edge.smallest_prev.x
                len_z = edge.smallest_prev.z - edge.smallest.z
                lpz = remaining_z - edge.smallest.z
                match, other_match = self.get_box(len_x, layer_thickness, remaining_y, len_z, lpz)
                box, orientation, new_thickness = self.check_boxes(match, other_match,
                                                                   edge)
                if self.layer_finished:
                    break
                if self.edge_is_even:
                    continue
                if new_thickness:
                    self.layer_thickness = new_thickness

                x_coord = edge.update(orientation, 4.2)
                coords = Coords(x_coord, y_coord, z_coord)
                self.pack_box(box, coords, orientation)

    def iterations(self):
        unique_permutations = set(perm for perm in permutations(self.pallet_dims))
        # pdb.set_trace()
        for variant, pallet_orientation in enumerate(unique_permutations):
            candidate_layers = self.get_candidate_layers(self.boxes, pallet_orientation)
            layers = sorted(candidate_layers, key=lambda x: x.value)
            print("Num Layers:{}".format(len(layers)))
            for iteration, layer in enumerate(layers):
                print('Variant: {0} Iteration: {1} Past: {2:0.2f}% Best: {3:0.2f}%'.format(
                    variant, iteration,
                    (solver.packed_vol / solver.pallet_vol) * 100,
                    (solver.best_vol / solver.pallet_vol) * 100
                ))
                self.reset_boxes()
                self.layer_thickness = layer.width
                self.packed_vol = 0
                packed_y = 0
                remaining_y = pallet_orientation[1]
                remaining_z = pallet_orientation[2]
                self.num_packed = 0
                self.packing = True
                while self.packing:
                    self.layer_in_layer = 0
                    self.layer_finished = False
                    self.pack_layer(self.layer_thickness, pallet_orientation,
                                    remaining_y, remaining_z, packed_y)
                    packed_y += self.layer_thickness
                    remaining_y = pallet_orientation[1] - packed_y

                    if self.layer_in_layer != 0:
                        prev_packed_y = packed_y
                        prev_remaining_y = remaining_y
                        remaining_y = packed_y - self.layer_thickness + self.prev_layer
                        remaining_z = self.lilz  # What?
                        self.layer_finished = False
                        self.pack_layer(self.layer_thickness, pallet_orientation,
                                        remaining_y, remaining_z, packed_y)
                        packed_y = prev_packed_y
                        remaining_y = prev_remaining_y
                        remaining_z = self.pallet_dims.dim3

                    self.layer_thickness = self.get_layer(pallet_orientation, remaining_y)
                if self.packed_vol > self.best_vol:
                    self.best_vol = self.packed_vol
                    self.best_orientation = pallet_orientation
                    self.best_layer = layer
                    self.best_num_packed = self.num_packed
                if (self.best_vol == self.pallet_vol or
                        self.best_num_packed == self.total_boxes):
                    return
                print()
            if (self.best_vol == self.pallet_vol or
                    self.best_num_packed == self.total_boxes):
                return
            print()

    def __init__(self, filename):
        pallet_dims, self.boxes = self.initialize(filename)
        self.pallet_dims = Dims(*pallet_dims)
        self.total_boxes = len(self.boxes)
        self.pallet_vol = np.prod(self.pallet_dims)
        self.total_box_vol = sum(box.vol for box in self.boxes)
        self.packed_vol = 0
        self.best_vol = 0
        self.best_orientation = None
        self.best_layer = None
        self.num_packed = 0
        self.best_num_packed = None
        self.packed_y = 0
        self.packing = True
        self.prev_layer = 0
        self.layer_in_layer = 0
        self.layer_finished = False
        self.edge_is_even = False
        self.layer_thickness = 0
        self.lilz = 0


#import pdb; pdb.set_trace()

solver = Solver('lego-1big-1square-1one')
print("Pallet Dimensions: {0}".format(solver.pallet_dims))
print("Number of Boxes: {0}".format(len(solver.boxes)))
print("Pallet Volume: {0}".format(solver.pallet_vol))
print("Total Volume: {0}".format(solver.total_box_vol))
solver.iterations()
print("Best Solution: {0:0.2f}%".format((solver.best_vol / solver.pallet_vol) * 100))
