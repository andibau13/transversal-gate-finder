"""Finding diagonal transversal gates on finite CSS codes.

The main entry point is the GateFinder class. The underlying structures are:

- PhaseLocs: supports of the generating phase functions of a 2-group of phase
  functions on a Z2 module (used for the ansatz gates of a GateFinder).
- Z2Hom: a Z2-linear map in sparse column form (used for checks and logicals).
- TwoGroupHom: a homomorphism between finite abelian 2-groups (wrapping a
  twogroup_linalg.Hom), with optional PhaseLocs attached to source and target.
- TwoGroupElem: an element of a finite abelian 2-group (wrapping a
  twogroup_linalg.Elem), with an optional PhaseLocs.

Conventions follow twogroup_linalg: index 0 refers to the target (rows) and
index 1 to the source (columns) of a homomorphism, so phase_locs0 describes the
target generators and phase_locs1 the source generators of a TwoGroupHom.

Two homomorphisms/elements compose only if their phase_locs objects are the
*same* Python object (checked with "is"); phase_locs objects are always passed
by reference, never copied.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Callable, Hashable, Iterable, Optional, Sequence, Union

import numpy as np
import twogroup_linalg as lin


def _check_phase_locs_dims(phase_locs, dims: Sequence[int], name: str) -> None:
    """Check that a phase-locs object has exactly dims[l] generators at each level l."""
    pdims = phase_locs.dims
    for l, d in enumerate(dims):
        count = pdims[l] if l < len(pdims) else 0
        if count != d:
            raise ValueError(f"{name} has {count} generators at level {l} but the homomorphism expects {d}")
    for l in range(len(dims), len(pdims)):
        if pdims[l] != 0:
            raise ValueError(f"{name} has generators at level {l} beyond the homomorphism's target/source")


class PhaseLocs:
    """Supports of the generating phase functions of a 2-group of phase functions on a Z2 module.

    A generating phase function of level l with support "loc" assigns the phase
    val/2^(l+1) * prod_{i in loc} x_i to a Z2 vector x, where val is the
    coefficient of the generator. In QEC terms, an entry of locs[l] is the set
    of qubits participating in one ansatz gate of order 2^(l+1).

    The locs are always kept consolidated: every support is stored as a frozenset
    and appears at exactly one level, the highest one it was added at (see add_loc).

    Attributes:
        dim: dimension of the underlying Z2 module (number of qubits)
        locs: locs[l][i] is the support of the i-th generator of level l, a frozenset of int
    """

    def __init__(self, dim: int, locs: Optional[Sequence[Sequence[Iterable[int]]]] = None):
        self.dim = dim
        self.locs: list[list[frozenset[int]]] = []
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

    def add_loc(self, qubits: Iterable[int], l: int) -> None:
        """Add a single generator support at level l, keeping the locs consolidated.

        Does nothing if the support already occurs at level l or higher; if it occurs
        at a lower level it is moved up to l (a support is kept only at its highest level).
        """
        loc = frozenset(qubits)
        for i in loc:
            if not 0 <= i < self.dim:
                raise ValueError(f"Support index {i} out of range 0..{self.dim - 1}")
        self.extend_levels(l)
        if any(loc in self.locs[higher] for higher in range(l, len(self.locs))):
            return
        for lower in range(l):
            if loc in self.locs[lower]:
                self.locs[lower].remove(loc)
        self.locs[l].append(loc)

    def add_locs(self, locs: Iterable[Iterable[int]], l: int) -> None:
        """Add generator supports at level l (see add_loc)."""
        for loc in locs:
            self.add_loc(loc, l)

    def add_all_single_locs(self, l: int) -> None:
        """Add all singleton supports {i} at level l (single-qubit ansatz gates of order 2^(l+1))."""
        self.add_locs([{i} for i in range(self.dim)], l)

    def add_locs_in_groups(self, groups, l: int, k: int) -> None:
        """Add all k-subset supports at level l within each group in "groups".

        Each group is a collection of indices (a set of qubits); groups may also be
        given as a Z2Hom (e.g. a GateFinder's checks), in which case its columns are used.
        """
        if isinstance(groups, Z2Hom):
            groups = groups.h
        for group in groups:
            self.add_locs(combinations(group, k), l)


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
            # Z2 reduction: a row index survives iff it appears an odd number of times.
            # (Repeats occur e.g. when compactifying a translation-invariant check onto a
            # small torus where two cells map onto the same physical qubit; they must cancel.)
            parity: dict[int, int] = {}
            for i in col:
                if not 0 <= i < self.dim0:
                    raise ValueError(f"Column entry out of range 0..{self.dim0 - 1}: {i}")
                parity[i] = parity.get(i, 0) ^ 1
            self.h.append(sorted(i for i, p in parity.items() if p))

    def to_array(self) -> np.ndarray:
        """Dense 0/1 coefficient matrix of shape (dim0, dim1). Each column has ones at its stored row indices."""
        matrix = np.zeros((self.dim0, len(self.h)), dtype=int)
        for j, col in enumerate(self.h):
            matrix[list(col), j] = 1
        return matrix

    @staticmethod
    def from_array(matrix: np.ndarray) -> "Z2Hom":
        """Build a Z2Hom from a dense 0/1 matrix; each column becomes the set of its non-zero rows."""
        columns = [np.nonzero(matrix[:, j])[0].tolist() for j in range(matrix.shape[1])]
        return Z2Hom(matrix.shape[0], columns)

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

    def remove_redundant_columns(self, relative_to: Optional["Z2Hom"] = None) -> None:
        """Drop columns that are Z2-linear combinations of the earlier columns.

        In place: keeps only a maximal Z2-independent subset of the columns (the first
        one of each dependency). If relative_to is given, columns already contained in
        the span of relative_to's columns are also dropped (used e.g. to reduce logicals
        modulo the checks; then relative_to must have the same number of rows as self).
        """
        if relative_to is None:
            pivots = lin.get_pivots(lin.z2lin.rref(self.to_array()))
        else:
            _, pivots = lin.remove_image(relative_to.to_array(), self.to_array())
        self.h = [self.h[i] for i in pivots]

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


class TwoGroupHom:
    """Homomorphism between finite abelian 2-groups, with optional phase function supports.

    Attributes:
        h: lin.Hom object holding the block coefficient matrix
        phase_locs0: PhaseLocs (or TIPhaseLocs) describing the supports of the target
            generators (parallel to h.dim0), or None if the target is an abstract 2-group
        phase_locs1: like phase_locs0 for the source generators (parallel to h.dim1)
    """

    def __init__(self, h: lin.Hom, phase_locs0=None, phase_locs1=None):
        if phase_locs0 is not None:
            _check_phase_locs_dims(phase_locs0, h.dim0, "phase_locs0")
        if phase_locs1 is not None:
            _check_phase_locs_dims(phase_locs1, h.dim1, "phase_locs1")
        self.h = h
        self.phase_locs0 = phase_locs0
        self.phase_locs1 = phase_locs1

    @property
    def dim0(self) -> list[int]:
        return self.h.dim0

    @property
    def dim1(self) -> list[int]:
        return self.h.dim1

    def __matmul__(self, other: Union["TwoGroupHom", "TwoGroupElem", lin.Elem]):
        """Composition with another TwoGroupHom, or application to a (TwoGroup)Elem.

        Composition/application is allowed only if the shared phase_locs are the *same*
        object (checked with "is"); a raw lin.Elem carries no phase_locs and is accepted.
        """
        if isinstance(other, TwoGroupElem):
            if other.phase_locs is not None and other.phase_locs is not self.phase_locs1:
                raise ValueError("Application requires elem.phase_locs is self.phase_locs1")
            return TwoGroupElem(self.h @ other.e, phase_locs=self.phase_locs0)
        if isinstance(other, lin.Elem):
            return TwoGroupElem(self.h @ other, phase_locs=self.phase_locs0)
        if isinstance(other, TwoGroupHom):
            if other.phase_locs0 is not self.phase_locs1:
                raise ValueError("Composition requires other.phase_locs0 is self.phase_locs1")
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
        """Transpose (dual) homomorphism, see lin.Hom.transpose."""
        return TwoGroupHom(self.h.transpose(), phase_locs0=self.phase_locs1, phase_locs1=self.phase_locs0)

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


class TwoGroupElem:
    """Element of a finite abelian 2-group, with optional phase function supports.

    Attributes:
        e: lin.Elem holding the coefficient vector
        phase_locs: PhaseLocs (or TIPhaseLocs) describing the supports of the generators
            (parallel to e.dim), or None if the group is abstract
    """

    def __init__(self, e: lin.Elem, phase_locs=None):
        if phase_locs is not None:
            _check_phase_locs_dims(phase_locs, e.dim, "phase_locs")
        self.e = e
        self.phase_locs = phase_locs

    @property
    def dim(self) -> list[int]:
        return self.e.dim

    @property
    def v(self) -> np.ndarray:
        return self.e.v

    def to_string(self) -> str:
        """Write the element as a string; generators labeled by their phase_locs support if given, else (level, index)."""
        terms = []
        for li in range(len(self.e.dim)):
            for i in range(self.e.dim[li]):
                c = self.e[li][i]
                if c != 0:
                    label = sorted(self.phase_locs.locs[li][i]) if self.phase_locs is not None else (li, i)
                    terms.append(f"{c}/{2 ** (li + 1)}*{label}")
        return ", ".join(terms)


class TransversalGateQueries:
    """Mixin: query physical representatives of non-trivial transversal gates.

    Mixed into GateFinder and TIGateFinder. All methods rely on the attributes computed by
    GateFinder.find_logical_action (translog_alllog, transphys_translog, log_find_helper,
    rep_find_helper, stabphys_allphys) together with transphys_allphys. On a TIGateFinder these
    are populated by find_gates_compactify (find_gates_nonlocal only sets transphys_translog,
    rep_find_helper and transphys_allphys, so on it only find_phys_rep / print_phys_rep apply).

    Throughout, "non-trivial transversal gate" means an element of the 2-group
    self.transphys_translog.dim0: the transversal logical gates for a GateFinder or a
    TIGateFinder after find_gates_compactify, or the classes modulo local transversal gates
    for a TIGateFinder after find_gates_nonlocal.
    """

    def print_transversal_logicals(self) -> None:
        """Print the transversal logical gates (needs translog_alllog; not set by find_gates_nonlocal)."""
        print(self.translog_alllog.to_string())

    def print_physical_stabilizers(self) -> None:
        """Print the physical transversal stabilizers (needs stabphys_allphys; not set by find_gates_nonlocal)."""
        print(self.stabphys_allphys.to_string())

    def find_phys_rep(self, nontrivial_gate) -> "TwoGroupElem":
        """
        Find a physical representative for a non-trivial transversal gate.

        Parameters:
            nontrivial_gate: coefficient list for an Elem over the group of non-trivial
                transversal gates (self.transphys_translog.dim0)

        Returns:
            TwoGroupElem over the group of physical ansatz gate configurations
            (phase_locs = self.transphys_allphys.phase_locs0)
        """
        elem = lin.Elem(np.array(nontrivial_gate), self.transphys_translog.dim0)
        transphys_rep = self.transphys_translog.solve_with_helper(elem, self.rep_find_helper)
        return self.transphys_allphys @ transphys_rep

    def print_phys_rep(self, nontrivial_gate) -> None:
        print(self.find_phys_rep(nontrivial_gate).to_string())

    def test_if_implemented(self, gates) -> Optional[lin.Elem]:
        """
        Test whether a given diagonal logical gate has a transversal implementation
        (needs translog_alllog / log_find_helper; not set by find_gates_nonlocal).

        Parameters:
            gates: Dict mapping (qubit set, gate level) to coefficient, where the qubit
                set is a set of logical qubit numbers of the CSS code and the gate level l
                corresponds to order 2^(l+1); the coefficient is the numerator of the phase
                factor relative to the denominator 2^(l+1).

        Returns:
            None if the logical gate is not implemented by any transversal gate. Otherwise an Elem over the abstract 2-group of transversal logicals (the source of self.translog_alllog), which can be passed to find_phys_rep to obtain a physical implementation.
        """
        loc_levels = {frozenset(loc): (lev, i)
                      for lev, llocs in enumerate(self.translog_alllog.phase_locs0.locs) for i, loc in enumerate(llocs)}

        target = lin.Elem.zeros(self.translog_alllog.dim0)
        for (gate, l), coeff in gates.items():
            c = int(coeff) % 2**(l+1)
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

    def find_phys_rep_free(self, gates) -> Optional["TwoGroupElem"]:
        """
        Find a physical representative for a logical gate given in free form (a dict of
        {(logical qubit set, gate level): coefficient}, see test_if_implemented), or None
        if the gate has no transversal implementation.
        """
        translog = self.test_if_implemented(gates)
        if translog is None:
            return None
        return self.find_phys_rep(translog.v)

    def print_phys_rep_free(self, gates) -> None:
        rep = self.find_phys_rep_free(gates)
        if rep is None:
            print("no transversal implementation")
        else:
            print(rep.to_string())


class GateFinder(TransversalGateQueries):
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
            each gate is the set of participating qubits. Add gates via gates.add_locs,
            gates.add_all_single_locs, gates.add_locs_in_groups, etc.

    find_gates() additionally sets (see its docstring for details):
        transphys_allphys: all transversal physical gates -> all physical gates
        translog_alllog: all transversal logical gates -> all logical gates
        transphys_translog: transversal physical gates -> transversal logical gates
        stabphys_allphys: all transversal stabilizers -> all physical gates
        log_find_helper, rep_find_helper: solve helpers used by test_if_implemented
            and find_phys_rep
    """

    def __init__(self, nr_qubits: int, checks=None, gates=None, logicals=None, other_checks=None):
        self.nr_qubits = nr_qubits
        self.checks = Z2Hom(nr_qubits, checks)
        self.logicals = Z2Hom(nr_qubits, logicals)
        self.other_checks = Z2Hom(nr_qubits, other_checks)
        self.gates = PhaseLocs(nr_qubits, gates)

    def logicals_from_other_checks(self) -> None:
        """
        Initialize X logicals from X checks and Z logicals (self.other_checks, if given)
        """
        zker = lin.z2lin.kernel(self.other_checks.to_array().T)
        ind_xchecks, log_nrs = lin.remove_image(self.checks.to_array(), zker)
        self.logicals = Z2Hom.from_array(zker[:, log_nrs])

    def find_gates(self) -> None:
        """
        Compute the space of all transversal logical gates and the physical stabilizers.

        Sets the following attibutes of self:
            transphys_allphys: All transversal physical gates: 2-group homomorphism from the group of all code-space preserving physical gates to the group of all physical gates formed by the ansatz gates
            translog_alllog: All transversal logical gates: 2-group homomorphism from the group of logicals with a transversal implementation to the group of all logicals
            rep_find_helper: allows the method find_phys_rep to quickly find a physical representative for a given transversal logical
            stabphys_allphys: All transversal stabilizers: 2-group homomorphism from the group of all physical gates preserving the code space to the group of all physical gates

        Naming convention for the different TwoGroupHoms: A Hom "x_y" is a hom from teh 2-group x to the 2-group y. the following 2-groups for x and y are possible:
        - allphys: group of all physical diagonal gates at the gate locations
        - transphys: group of all physical diagonal gates that preserve the code space, including ones with trivial logical action
        - allcheck: group of all phase functions on the checks
        - alllog: group of all diagonal logical gates
        - translog: group of all logical gates with transversal physical implementation
        - stabphys: group of physical transversal gates with trivial logical action
        """

        allphys_allcheck = self.checks.phase_pullback(self.gates) # map all physical -> all check
        self.transphys_allphys = allphys_allcheck.kernel() # map transversal physical -> all physical
        self.find_logical_action()

    def find_logical_action(self) -> None:
        allphys_alllog = self.logicals.phase_pullback(self.gates) # map all physical -> all logical
        transphys_alllog = allphys_alllog @ self.transphys_allphys # map transversal physical -> all logical
        self.translog_alllog, self.transphys_translog = transphys_alllog.epi_mono() # map transversal logical -> all logical
        _, self.log_find_helper = self.translog_alllog.kernel(return_solve_helper = True) # allows test_if_implemented to solve for a preimage in the transversal logicals
        stabphys_transphys, self.rep_find_helper = self.transphys_translog.kernel(return_solve_helper = True) # stabilizer physical -> transversal physical
        self.stabphys_allphys = self.transphys_allphys @ stabphys_transphys # stabilizer physical -> all physical

    def test_commutation(self) -> None:
        """Test if z checks commute with x checks and x logicals."""
        print("X checks and Z checks commute:", (self.checks.transpose() @ self.other_checks).is_zero())
        print("X logicals and Z checks commute:", (self.logicals.transpose() @ self.other_checks).is_zero())

    def __add__(self, other: "GateFinder") -> "GateFinder":
        """
        Combine two GateFinder objects into one.
        Stacks the two CSS codes, with independent ansatz gates
        """
        def shift_loc_list(loc_list, shift: int) -> list[set[int]]:
            return [{bit+shift for bit in loc} for loc in loc_list]
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
    """Enumerate all possible decompositions of Lambda as a sum of k positive integers."""
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
