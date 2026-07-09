from itertools import combinations, product

import numpy as np
import twogroup_linalg as lin

from . import flint_wrappers as fl
from .core import GateFinder

class TIGateFinder:
    """Helper class for analyzing translation-invariant codes in n dimensions.

    Attributes:
        dimension: spatial dimension of translation-invariant CSS code
        nr_qubits: number of qubits *per unit cell*
        checks: X checks
        other_checks: Z checks (optional)
        gates: Ansatz gates
        Format for checks, other_checks, gates is the same as for GateFinder, just that a qubit number is replaced by a pair of (1) shift coordinate (int tuple) and (2) unit-cell-internal qubit number
    """
    def __init__(self, nr_qubits, dimension, checks = None, gates = None, other_checks = None):
        self.nr_qubits = nr_qubits
        self.dimension = dimension
        self.checks = []
        if checks is not None:
            self.add_checks(checks)
        self.other_checks = []
        if other_checks is not None:
            self.add_other_checks(other_checks)
        self.gates = []
        if gates is not None:
            self.extend_gate_orders(len(gates))
            for l in range(len(gates)):
                self.add_gates(gates[l], l)

    def add_checks(self, checks):
        self.check_qubits_valid(checks)
        self.checks += checks
    def add_other_checks(self, other_checks):
        self.check_qubits_valid(other_checks)
        self.other_checks += other_checks

    def extend_gate_orders(self, l):
        """Reserve space for gates of orders up to 2^(l+1)."""
        if len(self.gates) <= l:
            self.gates += [[] for _ in range(l - len(self.gates) + 1)]

    def add_gates(self, gates, l):
        self.check_qubits_valid(gates)
        self.gates[l] += gates

    def check_qubits_valid(self, listlist):
        for mlist in listlist:
            for coord, intern in mlist:
                if len(coord) != self.dimension:
                    raise ValueError(f"Coordinate {coord} has wrong number of entries (should be {self.dimension})")
                if intern < 0 or intern >= self.nr_qubits:
                    raise ValueError(f"Internal qubit number {intern} given but must be between 0<={intern}<{self.nr_qubits}")

    def add_all_single_qubit_gates(self, l):
        self.extend_gate_orders(l)
        for intern in range(self.nr_qubits):
            self.gates[l].append([((0,)*self.dimension, intern)])

    def add_gates_in_groups(self, groups, l, k):
        """Add all gate locations of size k and phase-level l within each group in "groups".

        Each group in groups is a list of pairs of coord tuple and internal qubit nr.
        """
        self.extend_gate_orders(l)
        loc_set = set()
        for group in groups:
            loc_set.update(set(map(frozenset, map(normalize_ti_gate, combinations(group, k)))))
        self.gates[l] += list(map(set, loc_set))

    def add_gates_in_coord_group(self, group, l, k):
        """Add all gate locations of size k and phase-level l within each subset of qubits supported on the set of coordinates.
        """
        self.extend_gate_orders(l)
        qubits = [(coord, intern) for coord in group for intern in range(self.nr_qubits)]
        loc_set = set(map(frozenset, map(normalize_ti_gate, combinations(qubits, k))))
        self.gates[l] += list(map(set, loc_set))


    def consolidate_gates(self):
        """Normalize and deduplicate the ansatz gates.

        (1) shift each gate so its lexicographically smallest qubit sits at coordinate 0
            (via normalize_ti_gate),
        (2) within each order keep only one copy of identical gates, and
        (3) drop any gate from self.gates[i] that also occurs in self.gates[j] for some j>i.

        A gate that appears at several orders is therefore kept only at the highest one.
        """
        # canonical, hashable form for every gate
        normalized = [[frozenset(normalize_ti_gate(gate)) for gate in lgates]
                      for lgates in self.gates]

        new_gates = []
        for i, lgates in enumerate(normalized):
            # all gates occurring at some strictly higher order
            higher = set().union(*normalized[i+1:]) if i + 1 < len(normalized) else set()
            seen = set()
            kept = []
            for gate in lgates:
                if gate in seen or gate in higher:
                    continue
                seen.add(gate)
                kept.append(set(gate))
            new_gates.append(kept)
        self.gates = new_gates

    def as_finite_code(self, lattice):
        """Transform into a regular code by putting it on a finite lattice with twisted boundary conditions.

        Arguments:
            lattice: Matrix describing twisted periodic boundary conditions, used to construct a finite code from the unit-cell data. Numpy array whose *rows* are the vectors that are identified with the origin.
        """
        lattice_hnf = fl.hnf(lattice)[0] # row-operation hnf
        periods = lattice_hnf.diagonal()
        total_dim = np.prod(periods)
        cum_dims = np.cumprod(periods)
        cum_dims = np.insert(cum_dims, 0, 1)[:-1]

        tgf = GateFinder(total_dim * self.nr_qubits)
        tgf.add_checks(TIGateFinder.generate_ti_list(lattice_hnf, cum_dims, total_dim, periods, self.checks))
        for l in range(len(self.gates)):
            tgf.add_gates(TIGateFinder.generate_ti_list(lattice_hnf, cum_dims, total_dim, periods, self.gates[l]), l)
        tgf.add_other_checks(TIGateFinder.generate_ti_list(lattice_hnf, cum_dims, total_dim, periods, self.other_checks))

        return tgf

    @staticmethod
    def reduce_coordinate(coord, hnf):
        """Reduce a coordinate vector to lie within the parallelogram defined by the hnf."""
        #print("coords before\n", coords)
        for d in range(len(coord)):
            offs = coord[d] // hnf[d,d]
            coord -= offs * hnf[d, :]
        return coord

    @staticmethod
    def nr_to_coord(nr, cum_dims, periods):
        return np.array(nr)[None] // cum_dims % periods

    @staticmethod
    def coord_to_qubit(lattice_hnf, cum_dims, total_dim, coord, internal):
        """Take a pair of (coordinate vector modulo PBC lattice, internal qubit number) and transform it into a global qubit number.

        Arguments:
            lattice_hnf: pbc lattice in hermite normal form
            cum_dims: cumulative product of diagonal of lattice_hnf starting from 1
            coord: unit cell coordinate (not necessarily reduced to the standard box)
            internal: internal qubit number
            output: qubit number
        """
        for d in range(len(coord)):
            offs = coord[d] // lattice_hnf[d,d]
            coord -= offs * lattice_hnf[d, :]

        return int(np.dot(coord, cum_dims) + internal * total_dim)

    @staticmethod
    def generate_ti_list(lattice_hnf, cum_dims, total_dim, periods, qubit_listlist):
        """Take a list of lists of coordinates + internal qubit nr's and turn it into a list of list of qubits."""
        output_listlist = []
        for i in range(total_dim):
            shift_coord = TIGateFinder.nr_to_coord(i, cum_dims, periods)
            for qubit_list in qubit_listlist:
                output_listlist.append([TIGateFinder.coord_to_qubit(lattice_hnf, cum_dims, total_dim, coord+shift_coord, intern) for coord, intern in qubit_list])

        return output_listlist


    def __add__(self, other):
        if self.dimension != other.dimension:
            raise ValueError("Space(time) dimensions of added codes must agree.")
        res = TIGateFinder(self.nr_qubits + other.nr_qubits, self.dimension)
        res.checks= self.checks + shift_ti_loc_list(other.checks, self.nr_qubits)
        res.other_checks = self.other_checks + shift_ti_loc_list(other.other_checks, self.nr_qubits)

        maxlev = max(len(self.gates), len(other.gates))
        res.gates = self.gates
        res.extend_gate_orders(maxlev)
        for l in range(len(other.gates)):
            res.gates[l] += shift_ti_loc_list(other.gates[l], self.nr_qubits)
        return res

    def inverted_coordinates(self):
        """
        Invert all spatial coordinates
        """
        def invert_coords(mlist):
            return [[((-np.array(coord)).tolist(),i) for coord, i in entry] for entry in mlist]
        return TIGateFinder(self.nr_qubits,
                                              self.dimension,
                                              checks = invert_coords(self.checks),
                                              other_checks = invert_coords(self.other_checks),
                                              gates = [invert_coords(gts) for gts in self.gates])

def shift_ti_loc_list(loc_list, shift):
    """Shift the internal qubit number (needed for addition of codes)."""
    return [[(coord, bit+shift) for coord, bit in loc] for loc in loc_list]


def transpose_ti_map(loc_listlist, nr_out):
    """Transpose a translation-invariant Z2-linear map given in sparse form.

    The input ``loc_listlist`` has one list per internal input number ``j`` (the basis
    element ``(0, j)`` at the origin). Each such list holds the pairs ``(coord, i)`` of
    output basis elements that ``(0, j)`` maps to; by translation invariance ``(c, j)``
    maps to ``{(c + coord, i)}``.

    The transpose sends output basis elements to input basis elements. Since
    ``(0, j) -> (coord, i)`` means ``M[(coord, i), (0, j)] = 1``, the transpose has
    ``(0, i) -> (-coord, j)``.

    Arguments:
        loc_listlist: the map to transpose, as a list of lists of (coord, internal) pairs.
        nr_out: number of internal output basis elements of the map, i.e. the length of
            the transposed list. Must be larger than the maximum internal number
            appearing in the entries of ``loc_listlist``.

    Returns:
        The transposed map in the same sparse format, a list of ``nr_out`` lists.
    """
    transposed = [[] for _ in range(nr_out)]
    for j, loc_list in enumerate(loc_listlist):
        for coord, i in loc_list:
            transposed[i].append((tuple(-c for c in coord), j))
    return transposed



def normalize_ti_gate(gate):
    """Shift a translation-invariant gate location to a standard coordinate, where the "smallest" coordinate is zero.

    For "smallest" we first compare the coordinates lexicographically and then the internal qubit nr.
    """
    # normalize coord type to tuple first, so the min-comparison never mixes list/tuple
    # coordinates (both occur in this codebase) and the output is always hashable.
    gate = [(tuple(coord), intern) for coord, intern in gate]
    coordmin, _ = min(gate)
    gate_normalized = set()
    for gt_coord, gt_intern in gate:
        gt_coord_normalized = tuple(a-b for a, b in zip(gt_coord, coordmin))
        gate_normalized.add((gt_coord_normalized, gt_intern))
    return gate_normalized


# this is only copy+paste so far
def ti_pullback_homomorphism(nr_qubits, checks, gates):
    """Compute the pullback homomorphism from physical gates to check phase functions.

    gates is a list of pairs (l,loc). l is the power of the gate, i.e. there's a 1/2^(l+1) prefactor. loc is list of qubits where the gate acts.
    checks is a list of list (pure-python sparse binary matrix)
    """
    # transpose check matrix in sparse form - list of checks involving each qubit.
    checks_reverse = [[] for _ in range(nr_qubits)]
    for j, check in enumerate(checks):
        for i in check:
            checks_reverse[i].append(j)

    def weak_compositions(Lambda, k):
        """Enumerate all possible divisions of Lambda as a sum of k positive integers."""
        if k == 1:
            yield (Lambda,)
        else:
            for x in range(Lambda + 1):
                for rest in weak_compositions(Lambda - x, k - 1):
                    yield (x, *rest)

    # for each gate order l, and each physical gate location lgate, create a dictionary corresponding to the pullback of that gate location
    pullback_columns = []
    for l, lgates in enumerate(gates):
        pullback_columns_l = []
        for lgate in lgates:
            k = len(lgate)
            pullback_column = {}
            for Lambda in range(l+1): #
                for Lambdaxs in weak_compositions(Lambda, k):
                    Lxss = [combinations(Aix, Lambdax+1) for Aix, Lambdax in zip([checks_reverse[g] for g in lgate], Lambdaxs)]
                    #print("Lxss", [set(xx) for xx in Lxss])
                    #print("test2", set(Lxss[0]))
                    #for xxxx in Lxss[0]:
                    #    print("test", xxxx)
                    #print("productLxss", list(product(*Lxss)))
                    for Lxs in product(*Lxss):
                        Lx_union = frozenset(set().union(*(set(Lx) for Lx in Lxs)))
                        val = int((-2)**Lambda)
                        if Lx_union in pullback_column:
                            pullback_column[Lx_union] += val
                        else:
                            pullback_column[Lx_union] = val
            pullback_columns_l.append(pullback_column)
        pullback_columns.append(pullback_columns_l)

    # generate dict for all the maximum levels for each qubit tuple with a check gate
    check_gate_levels = {}
    for l, lcols in enumerate(pullback_columns):
        for col in lcols:
            for (gate_loc, val) in col.items():
                largest_pow_2 = l+1 if val==0 else (val & -val).bit_length() - 1
                lev = l - largest_pow_2
                if lev >= 0:
                    if gate_loc in check_gate_levels:
                        check_gate_levels[gate_loc] = max(check_gate_levels[gate_loc], lev)
                    else: check_gate_levels[gate_loc] = lev

    # rescale all values according to the level
    for l, lcols in enumerate(pullback_columns):
        for i, col in enumerate(lcols):
            col_rescaled = {}
            for (gate_loc, val) in col.items():
                if gate_loc in check_gate_levels: # if val is zero mod l for gate_loc everywhere, then gate_loc is not in check_gate_levels
                    lev = check_gate_levels[gate_loc]
                    if lev >= l:
                        val_rescaled = val * 2**(lev-l)
                    else:
                        val_rescaled = val // 2**(l-lev)
                    val_rescaled %= 2**(lev+1)
                    col_rescaled[gate_loc] = val_rescaled
            lcols[i] = col_rescaled

    # count number of check gate locations for each level
    max_level = -1 if len(check_gate_levels)==0 else max(check_gate_levels.values())
    check_dims = [0] * (max_level + 1)
    # internal index numbers for all check gate locations of given level l
    check_gate_indices = [{} for _ in range(max_level+1)]
    check_gate_locs = [[] for _ in range(max_level+1)] # list of list of check gate locations
    for check_gate, lev in check_gate_levels.items():
        check_gate_indices[lev][check_gate] = check_dims[lev]
        check_gate_locs[lev].append(check_gate)
        check_dims[lev] += 1

    gate_dims = [len(lgates) for lgates in gates]

    # assemble into dense matrix
    pullback = lin.Hom.zeros(check_dims, gate_dims)
    for l, lcols in enumerate(pullback_columns):
        for i, lcol in enumerate(lcols):
            for gate_loc, val in lcol.items():
                gate_lev = check_gate_levels[gate_loc]
                gate_ind = check_gate_indices[gate_lev][gate_loc]
                pullback[gate_lev, l][gate_ind, i] = val

    return pullback, check_gate_locs


def twobga_code(dimension, poly1, poly2):
    """Creates an "infinite" abelian 2-block group-algebra code from two polynomials. After compactification with as_finite_code() this is a true 2-block group-algebra code."""
    code = TIGateFinder(2, dimension)
    code.add_checks([[(coord,0) for coord in poly1] + [(coord,1) for coord in poly2]])
    code.add_other_checks([[((-np.array(coord)).tolist(),1) for coord in poly1] + [((-np.array(coord)).tolist(),0) for coord in poly2]])
    return code