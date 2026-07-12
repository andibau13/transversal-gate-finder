"""Finding diagonal transversal gates on finite CSS codes.

The main entry point is the GateFinder class. The underlying structures are:

- PhaseLocs: supports of the generating phase functions of a 2-group of phase
  functions on a Z2 module (used for the ansatz gates of a GateFinder).
- Z2Hom: a Z2-linear map in sparse column form (used for checks and logicals).
- TwoGroupHom: a homomorphism between finite abelian 2-groups (wrapping a
  twogroup_linalg.Hom), with optional PhaseLocs attached to source and target.

Conventions follow twogroup_linalg: index 0 refers to the target (rows) and
index 1 to the source (columns) of a homomorphism, so phase_locs0 describes the
target generators and phase_locs1 the source generators of a TwoGroupHom.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Callable, Hashable, Iterable, Optional, Sequence, Union

import numpy as np
import twogroup_linalg as lin


class PhaseLocs:
    """Supports of the generating phase functions of a 2-group of phase functions on a Z2 module.

    A generating phase function of level l with support "loc" assigns the phase
    val/2^(l+1) * prod_{i in loc} x_i to a Z2 vector x, where val is the
    coefficient of the generator. In QEC terms, an entry of locs[l] is the set
    of qubits participating in one ansatz gate of order 2^(l+1).

    Attributes:
        dim: dimension of the underlying Z2 module (number of qubits)
        locs: locs[l][i] is the support of the i-th generator of level l, a set of int
    """

    def __init__(self, dim: int, locs: Optional[Sequence[Sequence[Iterable[int]]]] = None):
        self.dim = dim
        self.locs: list[list[set[int]]] = []
        if locs is not None:
            for l, llocs in enumerate(locs):
                self.add_locs(llocs, l)

    @property
    def dims(self) -> list[int]:
        """Number of generators per level, the dimension of the 2-group of phase functions."""
        return [len(llocs) for llocs in self.locs]

    def extend_levels(self, l: int) -> None:
        """Reserve space for generators of levels up to l."""
        if len(self.locs) <= l:
            self.locs += [[] for _ in range(l - len(self.locs) + 1)]

    def add_locs(self, locs: Iterable[Iterable[int]], l: int) -> None:
        """Append generator supports at level l."""
        self.extend_levels(l)
        for loc in locs:
            loc = set(loc)
            for i in loc:
                if not 0 <= i < self.dim:
                    raise ValueError(f"Support index {i} out of range 0..{self.dim - 1}")
            self.locs[l].append(loc)

    def consolidate(self) -> None:
        """Remove duplicate supports, keeping a support only at the highest level where it occurs."""
        current = set()
        for l in reversed(range(len(self.locs))):
            llocs = set(map(frozenset, self.locs[l])) - current
            current |= llocs
            self.locs[l] = list(map(set, llocs))

    def elem_to_string(self, elem: lin.Elem) -> str:
        """Write an element of the 2-group of phase functions as a string."""
        terms = []
        for li in range(len(elem.dim)):
            for i in range(elem.dim[li]):
                if elem[li][i] != 0:
                    terms.append(f"{elem[li][i]}/{2 ** (li + 1)}*{sorted(self.locs[li][i])}")
        return ", ".join(terms)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PhaseLocs):
            return NotImplemented
        return self.dim == other.dim and list(map(lambda ls: list(map(frozenset, ls)), self.locs)) \
            == list(map(lambda ls: list(map(frozenset, ls)), other.locs))


class Z2Hom:
    """Z2-linear map in sparse column form.

    Attributes:
        h: h[j] is the sorted list of non-zero row indices of column j
        dim0: target dimension (number of rows)
        dim1: source dimension (number of columns, equal to len(h))
    """

    def __init__(self, dim0: int, columns: Optional[Iterable[Iterable[int]]] = None):
        self.dim0 = dim0
        self.h: list[list[int]] = []
        if columns is not None:
            self.add_columns(columns)

    @property
    def dim1(self) -> int:
        return len(self.h)

    def add_columns(self, columns: Iterable[Iterable[int]]) -> None:
        if isinstance(columns, Z2Hom):
            columns = columns.h
        for col in columns:
            col = sorted(set(col))
            if col and not (0 <= col[0] and col[-1] < self.dim0):
                raise ValueError(f"Column entry out of range 0..{self.dim0 - 1}: {col}")
            self.h.append(col)

    def to_array(self) -> np.ndarray:
        """Dense 0/1 coefficient matrix of shape (dim0, dim1)."""
        return matrix_from_list(self.dim0, self.h)

    @staticmethod
    def from_array(matrix: np.ndarray) -> "Z2Hom":
        return Z2Hom(matrix.shape[0], list_from_matrix(matrix))

    def transpose(self) -> "Z2Hom":
        transposed = [[] for _ in range(self.dim0)]
        for j, col in enumerate(self.h):
            for i in col:
                transposed[i].append(j)
        return Z2Hom(self.dim1, transposed)

    def __matmul__(self, other: "Z2Hom") -> "Z2Hom":
        if not isinstance(other, Z2Hom):
            return NotImplemented
        if other.dim0 != self.dim1:
            raise ValueError("Dimension mismatch in Z2Hom composition")
        return Z2Hom.from_array((self.to_array() @ other.to_array()) % 2)

    def is_zero(self) -> bool:
        return all(len(col) == 0 for col in self.h)

    def phase_pullback(self, phase_locs: PhaseLocs) -> "TwoGroupHom":
        """Pull back phase functions along this Z2-linear map.

        A phase function on the target module of self (with support given by
        phase_locs) pulls back to a phase function on the source module. In QEC
        terms, self is the check map (columns = checks, rows = qubits), the
        input phase functions are the ansatz gates, and the output phase
        functions live on the checks; the kernel of the returned homomorphism
        corresponds to the code-space preserving (transversal) gates.

        Returns:
            TwoGroupHom with phase_locs1 set to the input phase_locs and
            phase_locs0 set to the supports of the target phase functions
            (combinations of columns of self with non-trivial phase factor).
        """
        if phase_locs.dim != self.dim0:
            raise ValueError("phase_locs must live on the target module of self")
        reverse = self.transpose().h  # for each row of self, the columns containing it
        columns = [[pullback_column([reverse[i] for i in loc], l) for loc in llocs]
                   for l, llocs in enumerate(phase_locs.locs)]
        h, target_keys = assemble_pullback(columns, phase_locs.dims)
        target_locs = PhaseLocs(self.dim1, [[set(key) for key in lkeys] for lkeys in target_keys])
        return TwoGroupHom(h, phase_locs0=target_locs, phase_locs1=phase_locs)


def _locs_match(a, b) -> bool:
    """Test whether two optional phase-locs objects describe the same 2-group generators."""
    return a is b or a == b


class TwoGroupHom:
    """Homomorphism between finite abelian 2-groups, with optional phase function supports.

    Attributes:
        h: lin.Hom object holding the block coefficient matrix
        phase_locs0: PhaseLocs (or TIPhaseLocs) describing the supports of the target
            generators (parallel to h.dim0), or None if the target is an abstract 2-group
        phase_locs1: like phase_locs0 for the source generators (parallel to h.dim1)
    """

    def __init__(self, h: lin.Hom, phase_locs0=None, phase_locs1=None):
        self.h = h
        self.phase_locs0 = phase_locs0
        self.phase_locs1 = phase_locs1

    @property
    def dim0(self) -> list[int]:
        return self.h.dim0

    @property
    def dim1(self) -> list[int]:
        return self.h.dim1

    def __matmul__(self, other: Union["TwoGroupHom", lin.Elem]):
        """Composition with another TwoGroupHom, or application to an Elem."""
        if isinstance(other, lin.Elem):
            return self.h @ other
        if isinstance(other, TwoGroupHom):
            if not _locs_match(other.phase_locs0, self.phase_locs1):
                raise ValueError("Composition requires other.phase_locs0 == self.phase_locs1")
            return TwoGroupHom(self.h @ other.h, self.phase_locs0, other.phase_locs1)
        return NotImplemented

    def kernel(self, return_solve_helper: bool = False):
        """Kernel as a TwoGroupHom into the source 2-group of self (phase_locs0 = self.phase_locs1)."""
        if return_solve_helper:
            k, helper = self.h.kernel(return_solve_helper=True)
            return TwoGroupHom(k, phase_locs0=self.phase_locs1), helper
        return TwoGroupHom(self.h.kernel(), phase_locs0=self.phase_locs1)

    def epi_mono(self) -> tuple["TwoGroupHom", "TwoGroupHom"]:
        """Factor self = mono @ epi through the image. Returns (mono, epi)."""
        mono, epi = self.h.epi_mono()
        return (TwoGroupHom(mono, phase_locs0=self.phase_locs0),
                TwoGroupHom(epi, phase_locs1=self.phase_locs1))

    def solve_with_helper(self, b: lin.Elem, helper) -> lin.Elem:
        return self.h.solve_with_helper(b, helper)

    def is_zero(self) -> bool:
        return self.h.is_zero()

    def transpose(self) -> "TwoGroupHom":
        """Transpose (dual) homomorphism, see transpose_hom."""
        return TwoGroupHom(transpose_hom(self.h), phase_locs0=self.phase_locs1, phase_locs1=self.phase_locs0)

    def to_string(self) -> str:
        """Write the images of all source generators as a string, one line per generator.

        Target generators are labeled by their phase_locs0 support if given, else by (level, index).
        """
        out = ""
        for lj in range(len(self.h.dim1)):
            out += f"order {2 ** (lj + 1)}\n"
            for j in range(self.h.dim1[lj]):
                terms = []
                for li in range(len(self.h.dim0)):
                    for i in range(self.h.dim0[li]):
                        c = self.h[li, lj][i, j]
                        if c != 0:
                            if self.phase_locs0 is not None:
                                label = sorted(self.phase_locs0.locs[li][i])
                            else:
                                label = (li, i)
                            terms.append(f"{c}/{2 ** (min(li, lj) + 1)}*{label}")
                out += ", ".join(terms) + "\n"
        return out


class GateFinder:
    """
    Helper class for finding diagonal transversal gates on a given CSS code.

    Attributes:
        nr_qubits: number of qubits in the CSS code
        checks: X checks as a Z2Hom (columns = checks, rows = qubits)
        logicals: X logicals as a Z2Hom
        other_checks: Z checks (optional) as a Z2Hom
        gates: Ansatz gates from which the physical transversal gates will consist,
            as a PhaseLocs object. gates.locs[l] lists all ansatz gates of order 2^(l+1),
            that is, with prefactor 1/2^(l+1) (for example T is l=2, CS or S are l=1);
            each gate is the set of participating qubits.
    """

    def __init__(self, nr_qubits: int, checks=None, gates=None, logicals=None, other_checks=None):
        self.nr_qubits = nr_qubits
        self.checks = Z2Hom(nr_qubits)
        self.logicals = Z2Hom(nr_qubits)
        self.other_checks = Z2Hom(nr_qubits)
        self.gates = PhaseLocs(nr_qubits)
        if checks is not None:
            self.add_checks(checks)
        if logicals is not None:
            self.add_logicals(logicals)
        if other_checks is not None:
            self.add_other_checks(other_checks)
        if gates is not None:
            for l, lgates in enumerate(gates):
                self.add_gates(lgates, l)

    def add_checks(self, checks) -> None:
        self.checks.add_columns(checks)

    def add_logicals(self, logicals) -> None:
        self.logicals.add_columns(logicals)

    def add_other_checks(self, other_checks) -> None:
        self.other_checks.add_columns(other_checks)

    def extend_gate_orders(self, l: int) -> None:
        """Reserve space to include gates of orders up to l."""
        self.gates.extend_levels(l)

    def consolidate_gates(self) -> None:
        """Remove duplicate gates, and remove gate locations of order l if they are already present at a higher order."""
        self.gates.consolidate()

    def add_gates(self, gates: Iterable[Iterable[int]], l: int) -> None:
        self.gates.add_locs(gates, l)

    def add_all_singlequbit_gates(self, l: int) -> None:
        """Add single-qubit ansatz gates of order 2^(l+1) at all qubits."""
        self.gates.add_locs([{i} for i in range(self.nr_qubits)], l)

    def add_gates_in_groups(self, groups, l: int, k: int) -> None:
        """Add all ansatz gates involving k qubits of order 2^(l+1) within each group in "groups"."""
        if isinstance(groups, Z2Hom):
            groups = groups.h
        loc_set = set()
        for group in groups:
            loc_set.update(set(map(frozenset, combinations(group, k))))
        self.gates.add_locs(map(set, loc_set), l)

    def add_gates_in_checks(self, l: int, k: int) -> None:
        """
        Add all ansatz gates involving k qubits of order 2^(l+1) within each X check
        """
        self.add_gates_in_groups(self.checks, l, k)

    def remove_redundant_checks(self) -> None:
        """Remove X checks that are Z2-linear combinations of the other X checks."""
        pivots = lin.get_pivots(lin.z2lin.rref(self.checks.to_array()))
        self.checks = Z2Hom(self.nr_qubits, [self.checks.h[i] for i in pivots])

    def remove_redundant_logicals(self) -> None:
        """Remove X logicals that are Z2-linear combinations of the other X logicals and the X checks.

        Logicals are only defined modulo the X checks (stabilizers), so a logical equal to another logical times a product of checks is considered redundant.
        """
        _, logical_pivots = lin.remove_image(self.checks.to_array(), self.logicals.to_array())
        self.logicals = Z2Hom(self.nr_qubits, [self.logicals.h[i] for i in logical_pivots])

    def logicals_from_other_checks(self) -> None:
        """
        Initialize X logicals from X checks and Z logicals (self.other_checks, if given)
        """
        zker = lin.z2lin.kernel(self.other_checks.to_array().T)
        ind_xchecks, log_nrs = lin.remove_image(self.checks.to_array(), zker)
        self.logicals = Z2Hom.from_array(zker[:, log_nrs])

    def pullback_checks(self) -> TwoGroupHom:
        return self.checks.phase_pullback(self.gates)

    def pullback_logicals(self) -> TwoGroupHom:
        return self.logicals.phase_pullback(self.gates)

    def find_gates(self) -> None:
        """
        Compute the space of all transversal logical gates and the physical stabilizers.

        Sets the following attibutes of self:
            transphys_allphys: All transversal physical gates: 2-group homomorphism from the group of all code-space preserving physical gates to the group of all physical gates formed by the ansatz gates
            translog_alllog: All transversal logical gates: 2-group homomorphism from the group of logicals with a transversal implementation to the group of all logicals
            rep_find_helper: allows the method find_phys_rep to quickly find a physical representative for a given transversal logical
            stabphys_allphys: All transversal stabilizers: 2-group homomorphism from the group of all physical gates preserving the code space to the group of all physical gates
        """

        # allphys: group of all physical diagonal gates at the gate locations
        # transphys: group of all physical diagonal gates that preserve the code space, including ones with trivial logical action
        # allcheck: group of all phase functions on the checks
        # alllog: group of all diagonal logical gates
        # translog: group of all logical gates with transversal physical implementation
        # stabphys: group of physical transversal gates with trivial logical action

        allphys_allcheck = self.pullback_checks() # map all physical -> all check
        self.transphys_allphys = allphys_allcheck.kernel() # map transversal physical -> all physical
        self.find_logical_action()

    def find_logical_action(self) -> None:
        allphys_alllog = self.pullback_logicals() # map all physical -> all logical
        self.alllog_locs = allphys_alllog.phase_locs0 # supports of the logical phase functions
        transphys_alllog = allphys_alllog @ self.transphys_allphys # map transversal physical -> all logical
        self.translog_alllog, self.transphys_translog = transphys_alllog.epi_mono() # map transversal logical -> all logical
        _, self.log_find_helper = self.translog_alllog.kernel(return_solve_helper = True) # allows test_if_implemented to solve for a preimage in the transversal logicals
        stabphys_transphys, self.rep_find_helper = self.transphys_translog.kernel(return_solve_helper = True) # stabilizer physical -> transversal physical
        self.stabphys_allphys = self.transphys_allphys @ stabphys_transphys # stabilizer physical -> all physical

    def print_transversal_logicals(self) -> None:
        """
        Can be called after find_gates()
        """
        print(self.translog_alllog.to_string())

    def print_physical_stabilizers(self) -> None:
        """
        Can be called after find_gates()
        """
        print(self.stabphys_allphys.to_string())

    def find_phys_rep(self, logic_gate) -> lin.Elem:
        """
        Can be called after find_gates().
        Find physical representative for transversal logical gate.

        Parameters:
            logic_gate: coefficient list, linear combination of generator transversal logicals
        """
        logic_elem = lin.Elem(np.array(logic_gate), self.translog_alllog.dim1)
        transphys_rep = self.transphys_translog.solve_with_helper(logic_elem, self.rep_find_helper)
        phys_rep = self.transphys_allphys @ transphys_rep
        return phys_rep

    def print_phys_rep(self, logic_gate) -> None:
        rep = self.find_phys_rep(logic_gate)
        print(self.gates.elem_to_string(rep))

    def test_if_implemented(self, gates, coeffs) -> Optional[lin.Elem]:
        """
        Can be called after find_gates().
        Test whether a given diagonal logical gate has a transversal implementation.

        Parameters:
            gates: Gate locations in the same format as GateFinder.gates.locs, except that the qubit numbers refer to the logical qubits of the CSS code
            coeffs: Int list with one coefficient per gate location, flattened over all orders (an Elem coefficient vector for "gates")

        Returns:
            None if the logical gate is not implemented by any transversal gate. Otherwise an Elem over the abstract 2-group of transversal logicals (the source of self.translog_alllog), which can be passed to find_phys_rep to obtain a physical implementation.
        """
        loc_levels = {frozenset(loc): (lev, i)
                      for lev, llocs in enumerate(self.alllog_locs.locs) for i, loc in enumerate(llocs)}

        target = lin.Elem.zeros(self.translog_alllog.dim0)
        pos = 0
        for l, lgates in enumerate(gates):
            for gate in lgates:
                c = int(coeffs[pos]) % 2**(l+1)
                pos += 1
                if c == 0:
                    continue
                entry = loc_levels.get(frozenset(gate))
                if entry is None:
                    return None
                lev, i = entry
                if lev >= l:
                    val = c * 2**(lev-l)
                else:
                    # the location only supports logical gates of order 2^(lev+1)
                    if c % 2**(l-lev) != 0:
                        return None
                    val = c // 2**(l-lev)
                target[lev][i] = (int(target[lev][i]) + val) % 2**(lev+1)

        try:
            return self.translog_alllog.solve_with_helper(target, self.log_find_helper)
        except ValueError:
            return None

    def find_phys_rep_free(self, gates, coeffs) -> Optional[lin.Elem]:
        """
        Can be called after find_gates().
        Find a physical representative for a logical gate given in free form (gate locations on the logical qubits plus coefficients, see test_if_implemented), or None if the gate has no transversal implementation.
        """
        translog = self.test_if_implemented(gates, coeffs)
        if translog is None:
            return None
        return self.find_phys_rep(translog.v)

    def print_phys_rep_free(self, gates, coeffs) -> None:
        rep = self.find_phys_rep_free(gates, coeffs)
        if rep is None:
            print("no transversal implementation")
        else:
            print(self.gates.elem_to_string(rep))

    def test_commutation(self) -> None:
        """Test if z checks commute with x checks and x logicals."""
        print("X checks and Z checks commute:", (self.checks.transpose() @ self.other_checks).is_zero())
        print("X logicals and Z checks commute:", (self.logicals.transpose() @ self.other_checks).is_zero())

    def __add__(self, other: "GateFinder") -> "GateFinder":
        """
        Combine two GateFinder objects into one.
        Stacks the two CSS codes, with independent ansatz gates
        """
        res = GateFinder(self.nr_qubits + other.nr_qubits)
        for attr in ("checks", "logicals", "other_checks"):
            getattr(res, attr).add_columns(getattr(self, attr).h)
            getattr(res, attr).add_columns(shift_loc_list(getattr(other, attr).h, self.nr_qubits))
        for l, llocs in enumerate(self.gates.locs):
            res.gates.add_locs(llocs, l)
        for l, llocs in enumerate(other.gates.locs):
            res.gates.add_locs(shift_loc_list(llocs, self.nr_qubits), l)
        return res


def weak_compositions(Lambda: int, k: int):
    """Enumerate all possible divisions of Lambda as a sum of k positive integers."""
    if k == 1:
        yield (Lambda,)
    else:
        for x in range(Lambda + 1):
            for rest in weak_compositions(Lambda - x, k - 1):
                yield (x, *rest)

def pullback_column(check_lists: Sequence[Sequence[Hashable]], l: int) -> dict:
    """
    Compute the pullback of a single phase function generator (ansatz gate location) of level l.

    Parameters:
        check_lists: For each index in the generator support (qubit in the gate location),
            the list of target generators (checks) containing that index
        l: The level of the generator (gate order 2^(l+1))

    Returns:
        Dict mapping target combinations (frozensets of entries of check_lists) to the integer numerator of their phase factor, relative to the denominator 2^(l+1)
    """
    k = len(check_lists)
    column = {}
    for Lambda in range(l+1):
        for Lambdaxs in weak_compositions(Lambda, k):
            Lxss = [combinations(Aix, Lambdax+1) for Aix, Lambdax in zip(check_lists, Lambdaxs)]
            for Lxs in product(*Lxss):
                Lx_union = frozenset(set().union(*(set(Lx) for Lx in Lxs)))
                val = int((-2)**Lambda)
                if Lx_union in column:
                    column[Lx_union] += val
                else:
                    column[Lx_union] = val
    return column

def assemble_pullback(pullback_columns: Sequence[Sequence[dict]], source_dims: Sequence[int],
                      level_key: Optional[Callable[[Hashable], Hashable]] = None
                      ) -> tuple[lin.Hom, list[list[Hashable]]]:
    """
    Assemble pullback columns into a Hom object.

    Each target generator key is assigned a 2-group level based on the 2-adic
    valuation of the raw values it receives, maximized over all columns.

    Parameters:
        pullback_columns: pullback_columns[l][i] is the pullback of the ith source
            generator of level l, a dict as returned by pullback_column
        source_dims: Number of source generators at each level
        level_key: optional function grouping target keys into classes that are forced
            to share a common level (used in the translation-invariant case, where all
            translates of a check combination must sit at the same level so they can
            later be summed by TITwoGroupHom.ti_sum). Defaults to each key being its
            own class.

    Returns:
        pullback: Hom object with one row for each target key with non-trivial phase factor
        target_keys: List of target keys (the dict keys of the columns) for each level
    """
    if level_key is None:
        level_key = lambda key: key

    # find the maximum level for each class of target keys
    class_levels: dict = {}
    for l, lcols in enumerate(pullback_columns):
        for col in lcols:
            for (key, val) in col.items():
                largest_pow_2 = l+1 if val==0 else (val & -val).bit_length() - 1
                lev = l - largest_pow_2
                if lev >= 0:
                    cls = level_key(key)
                    class_levels[cls] = max(class_levels.get(cls, -1), lev)

    # rescale all values according to the level
    rescaled_columns = []
    for l, lcols in enumerate(pullback_columns):
        rescaled_l = []
        for col in lcols:
            col_rescaled = {}
            for (key, val) in col.items():
                # if val is zero mod 2^(l+1) for the class everywhere, then it has no level
                lev = class_levels.get(level_key(key))
                if lev is None:
                    continue
                if lev < l:
                    val = val // 2**(l-lev)
                # Hom coefficients between Z_{2^(l+1)} and Z_{2^(lev+1)} are valued in Z_{2^(min(lev,l)+1)}
                col_rescaled[key] = val % 2**(min(lev, l)+1)
            rescaled_l.append(col_rescaled)
        rescaled_columns.append(rescaled_l)

    # enumerate the target generators of each level, in order of first appearance
    max_level = max(class_levels.values(), default=-1)
    target_dims = [0] * (max_level + 1)
    key_indices: list[dict] = [{} for _ in range(max_level+1)]
    target_keys: list[list] = [[] for _ in range(max_level+1)]
    for lcols in rescaled_columns:
        for col in lcols:
            for key in col:
                lev = class_levels[level_key(key)]
                if key not in key_indices[lev]:
                    key_indices[lev][key] = target_dims[lev]
                    target_keys[lev].append(key)
                    target_dims[lev] += 1

    # assemble into dense matrix
    pullback = lin.Hom.zeros(target_dims, source_dims)
    for l, lcols in enumerate(rescaled_columns):
        for i, col in enumerate(lcols):
            for key, val in col.items():
                lev = class_levels[level_key(key)]
                pullback[lev, l][key_indices[lev][key], i] = val

    return pullback, target_keys

def transpose_hom(X: lin.Hom) -> lin.Hom:
    """Transpose (dual) of a 2-group homomorphism.

    Under the perfect pairing <x, y> = sum_i x_i*y_i / 2^(l_i+1), every finite abelian 2-group is its own dual, and the dual of a homomorphism is given by the plain block-wise transpose of its coefficient matrix, with source and target 2-groups interchanged. (The stored block coefficients are unchanged since the enhancement factor 2^max(0, i-j) turns into 2^max(0, j-i) under dualization, and the value group Z_{2^(min(i,j)+1)} is symmetric.)
    """
    return lin.Hom(X.M.T.copy(), X.dim1, X.dim0)

def quotient_image_by_image(K: lin.Hom, P: lin.Hom, K_solve_helper = None) -> lin.Hom:
    """Quotient the image of an injective homomorphism K: T -> G by the image of a homomorphism P: S -> G, given the promise im(P) is a subgroup of im(K).

    Works by (1) solving K f = P for f: S -> T column-wise (unique since K is injective), and (2) computing the cokernel of f as the transpose of the kernel of the transpose (kernel and cokernel are exchanged under the self-duality of finite abelian 2-groups).

    Parameters:
        K: injective Hom T -> G
        P: Hom S -> G with im(P) contained in im(K)
        K_solve_helper: optional solve helper for K, as returned by K.kernel(return_solve_helper=True); computed on the fly if not given

    Returns:
        The quotient projection q: T -> Q, a surjective Hom onto the quotient 2-group Q = im(K)/im(P), whose kernel is the preimage of im(P) under K
    """
    if K_solve_helper is None:
        _, K_solve_helper = K.kernel(return_solve_helper = True)

    # solve K f = P, one column (generator of the source of P) at a time
    f = lin.Hom.zeros(K.dim1, P.dim1)
    for l in range(len(P.dim1)):
        for j in range(P.dim1[l]):
            gen = lin.Elem.zeros(P.dim1)
            gen[l][j] = 1
            x = K.solve_with_helper(P @ gen, K_solve_helper)
            for i in range(len(K.dim1)):
                if i <= l:
                    f[i, l][:, j] = x[i]
                else:
                    # image of the generator at level i is c * 2^(i-l) with c the stored block coefficient;
                    # divisibility is guaranteed since 2^(l+1) * x = 0 by injectivity of K
                    assert np.all(x[i] % 2**(i-l) == 0)
                    f[i, l][:, j] = x[i] // 2**(i-l)

    return transpose_hom(transpose_hom(f).kernel())

def matrix_from_list(nr_rows: int, index_list) -> np.ndarray:
    """Generate matrix from a list of index sets - each list entry corresponds to one column, and the integers in the entry are the rows where the column is non-zero."""
    check_list = []
    for check_num in index_list:
        check = np.zeros((nr_rows,), dtype=int)
        check[list(check_num)] = 1
        check_list.append(check)
    if not check_list:
        return np.zeros((nr_rows, 0), dtype=int)
    return np.array(check_list).T

def list_from_matrix(matrix: np.ndarray) -> list[set[int]]:
    return [set((np.nonzero(matrix[:, i])[0]).tolist()) for i in range(matrix.shape[1])]

def shift_loc_list(loc_list, shift: int) -> list[set[int]]:
    return [{bit+shift for bit in loc} for loc in loc_list]
