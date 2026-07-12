import twogroup_linalg as lin
#import scipy.linalg
import numpy as np
from itertools import combinations, product


class GateFinder:
    """
    Helper class for finding diagonal transversal gates on a given CSS code.

    Attributes:
        nr_qubits: number of qubits in the CSS code
        checks: List of X checks, each specified by a set of qubit numbers
        logicals: List of X logicals, each specified by a set of qubit numbers
        other_checks: List of Z checks (optional), each specified by a set of qubit numbers
        gates: Ansatz gates from which the physical transversal gates will consist. List; gates[l] is a list of all ansatz gates or order 2^(l+1), that is, with prefactor 1/2^(l+1) (for example T is l=2, CS or S are l=1). gates[l][i] is the set of qubits participating in the gate, a set of int.
    """
    def __init__(self, nr_qubits, checks = None, gates = None, logicals = None, other_checks = None):
        self.nr_qubits = nr_qubits
        self.checks = []
        if checks is not None:
            self.add_checks(checks)
        self.logicals = []
        if logicals is not None:
            self.add_logicals(logicals)
        self.other_checks = []
        if other_checks is not None:
            self.add_other_checks(other_checks)
        self.gates = []
        if gates is not None:
            for l, lgates in enumerate(gates):
                self.add_gates(lgates, l)

    def add_checks(self, checks):
        self.checks += [set(check) for check in checks]
    def add_logicals(self, logicals):
        self.logicals += [set(logical) for logical in logicals]
    def add_other_checks(self, other_checks):
        self.other_checks += [set(check) for check in other_checks]


    def extend_gate_orders(self, l):
        """Reserve space to include gates of orders up to l."""
        if len(self.gates) <= l:
            self.gates += [[] for _ in range(l - len(self.gates) + 1)]

    def consolidate_gates(self):
        """Remove duplicate gates, and remove gate locations of order l if they are already present at a higher order."""
        current_gatesset = set()
        for l in reversed(range(len(self.gates))):
            lgates_set = set(map(frozenset, self.gates[l])) - current_gatesset
            current_gatesset |= lgates_set
            self.gates[l] = list(map(set, lgates_set))


    def add_gates(self, gates, l):
        self.extend_gate_orders(l)
        self.gates[l] += [set(gate) for gate in gates]

    def add_all_singlequbit_gates(self, l):
        """Add single-qubit ansatz gates of order 2^(l+1) at all qubits."""
        self.extend_gate_orders(l)
        self.gates[l] += [{i} for i in range(self.nr_qubits)]

    def add_gates_in_groups(self, groups, l, k):
        """Add all ansatz gates involving k qubits of order 2^(l+1) within each group in "groups"."""
        self.extend_gate_orders(l)
        loc_set = set()
        for group in groups:
            loc_set.update(set(map(frozenset, combinations(group, k))))
        self.gates[l] += list(map(set, loc_set))

    def add_gates_in_checks(self, l, k):
        """
        Add all ansatz gates involving k qubits of order 2^(l+1) within each X check
        """
        self.add_gates_in_groups(self.checks, l, k)


    def remove_redundant_checks(self):
        """Remove X checks that are Z2-linear combinations of the other X checks."""
        check_array = matrix_from_list(self.nr_qubits, self.checks)
        pivots = lin.get_pivots(lin.z2lin.rref(check_array))
        self.checks = [self.checks[i] for i in pivots]

    def remove_redundant_logicals(self):
        """Remove X logicals that are Z2-linear combinations of the other X logicals and the X checks.

        Logicals are only defined modulo the X checks (stabilizers), so a logical equal to another logical times a product of checks is considered redundant.
        """
        check_array = matrix_from_list(self.nr_qubits, self.checks)
        logical_array = matrix_from_list(self.nr_qubits, self.logicals)
        _, logical_pivots = lin.remove_image(check_array, logical_array)
        self.logicals = [self.logicals[i] for i in logical_pivots]

    def logicals_from_other_checks(self):
        """
        Initialize X logicals from X checks and Z logicals (self.other_checks, if given)
        """
        #print("test x and z stabilizers commute:", ~np.any(self.checks.T @ zchecks % 2 == 1))
        #self.test_commutation()
        check_array = matrix_from_list(self.nr_qubits, self.checks)
        other_check_array = matrix_from_list(self.nr_qubits, self.other_checks)

        zker = lin.z2lin.kernel(other_check_array.T)
        #print("zker", zker)
        ind_xchecks, log_nrs = lin.remove_image(check_array, zker)
        #print("independent x check numbers:", ind_xchecks)
        logical_array = zker[:, log_nrs]
        #print("logarr", logical_array)
        self.logicals = list_from_matrix(logical_array)

        #print(f"found {len(self.logicals)} logical x operators\n")

    def pullback_checks(self):
        return pullback_homomorphism(self.nr_qubits, self.checks, self.gates)[0]

    def pullback_logicals(self):
        return pullback_homomorphism(self.nr_qubits, self.logicals, self.gates)

    def find_gates(self):
        """
        Compute the space of all transversal logical gates and the physical stabilizers.

        Sets the following attibutes of self:
            transphys_allphys: All transversal physical gates: 2-group homomorphism from the group of all code-space preserving physical gates to the group of all physical gates formed by the ansatz gates
            translog_alllog: All transversal logical gates: 2-group homomorphism from the group of logicals with a transversal implementation to the group of all logicals
            rep_find_helper: allows the method find_phys_rep to quickly find a physical representative for a given transversal logical
            stabphys_allphys: All transversal stabilizers: 2-group homomorphism from the group of all physical gates preserving the code space to the group of all physical gates
        """

        # allphys: group of all physical diagonal gates at the gate_locs
        # transphys: group of all physical diagonal gates that preserve the code space, including ones with trivial logical action
        # allcheck: group of all phase functions on the checks
        # alllog: group of all diagonal logical gates
        # translog: group of all logical gates with transversal physical implementation
        # stabphys: group of physical transversal gates with trivial logical action
        
        allphys_allcheck = self.pullback_checks() # map all physical -> all check
        self.transphys_allphys = allphys_allcheck.kernel() # map transversal physical -> all physical
        self.find_logical_action()

    def find_logical_action(self):
        allphys_alllog, self.alllog_locs = self.pullback_logicals() # map all physical -> all logical
        transphys_alllog = allphys_alllog @ self.transphys_allphys # map transversal physical -> all logical
        self.translog_alllog, self.transphys_translog = transphys_alllog.epi_mono() # map transversal logical -> all logical
        _, self.log_find_helper = self.translog_alllog.kernel(return_solve_helper = True) # allows test_if_implemented to solve for a preimage in the transversal logicals
        stabphys_transphys, self.rep_find_helper = self.transphys_translog.kernel(return_solve_helper = True) # stabilizer physical -> transversal physical
        self.stabphys_allphys = self.transphys_allphys @ stabphys_transphys # stabilizer physical -> all physical


    def print_transversal_logicals(self):
        """
        Can be called after find_gates()
        """
        print(gates_string(self.translog_alllog, self.alllog_locs))

    def print_physical_stabilizers(self):
        """
        Can be called after find_gates()
        """
        print(gates_string(self.stabphys_allphys, self.gates))

    def find_phys_rep(self, logic_gate):
        """
        Can be called after find_gates().
        Find physical representative for transversal logical gate.

        Parameters:
            logic_gate: Elem object, linear combination of generator transversal logicals
        """
        logic_elem = lin.Elem(np.array(logic_gate), self.translog_alllog.dim1)
        transphys_rep = self.transphys_translog.solve_with_helper(logic_elem, self.rep_find_helper)
        phys_rep = self.transphys_allphys @ transphys_rep
        return phys_rep

    def print_phys_rep(self, logic_gate):
        rep = self.find_phys_rep(logic_gate)
        print(gate_string(rep, self.gates))

    def test_if_implemented(self, gates, coeffs):
        """
        Can be called after find_gates().
        Test whether a given diagonal logical gate has a transversal implementation.

        Parameters:
            gates: Gate locations in the same format as self.gates, except that the qubit numbers refer to the logical qubits of the CSS code
            coeffs: Int list with one coefficient per gate location, flattened over all orders (an Elem coefficient vector for "gates")

        Returns:
            None if the logical gate is not implemented by any transversal gate. Otherwise an Elem over the abstract 2-group of transversal logicals (the source of self.translog_alllog), which can be passed to find_phys_rep to obtain a physical implementation.
        """
        loc_levels = {frozenset(loc): (lev, i)
                      for lev, llocs in enumerate(self.alllog_locs) for i, loc in enumerate(llocs)}

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

    def find_phys_rep_free(self, gates, coeffs):
        """
        Can be called after find_gates().
        Find a physical representative for a logical gate given in free form (gate locations on the logical qubits plus coefficients, see test_if_implemented), or None if the gate has no transversal implementation.
        """
        translog = self.test_if_implemented(gates, coeffs)
        if translog is None:
            return None
        return self.find_phys_rep(translog.v)

    def print_phys_rep_free(self, gates, coeffs):
        rep = self.find_phys_rep_free(gates, coeffs)
        if rep is None:
            print("no transversal implementation")
        else:
            print(gate_string(rep, self.gates))

    def test_commutation(self):
        """Test if z checks commute with x checks and x logicals."""
        check_array = matrix_from_list(self.nr_qubits, self.checks)
        logical_array = matrix_from_list(self.nr_qubits, self.logicals)
        other_check_array = matrix_from_list(self.nr_qubits, self.other_checks)
        print("X checks and Z checks commute:", ~np.any((check_array.T @ other_check_array)%2))
        print("X logicals and Z checks commute:", ~np.any((logical_array.T @ other_check_array)%2))


    def __add__(self, other):
        """
        Combine two GateFinder objects into one.
        Stacks the two CSS codes, with independent ansatz gates
        """
        res = GateFinder(self.nr_qubits + other.nr_qubits)
        res.checks = [set(check) for check in self.checks] + shift_loc_list(other.checks, self.nr_qubits)
        res.logicals = [set(logical) for logical in self.logicals] + shift_loc_list(other.logicals, self.nr_qubits)
        res.other_checks = [set(check) for check in self.other_checks] + shift_loc_list(other.other_checks, self.nr_qubits)
        maxlev = max(len(self.gates), len(other.gates))
        res.gates = [[set(gate) for gate in lgates] for lgates in self.gates]
        res.extend_gate_orders(maxlev)
        for l in range(len(other.gates)):
            res.gates[l] += shift_loc_list(other.gates[l], self.nr_qubits)
        return res


def weak_compositions(Lambda, k):
    """Enumerate all possible divisions of Lambda as a sum of k positive integers."""
    if k == 1:
        yield (Lambda,)
    else:
        for x in range(Lambda + 1):
            for rest in weak_compositions(Lambda - x, k - 1):
                yield (x, *rest)

def pullback_column(check_lists, l):
    """
    Compute the pullback of a single ansatz gate location of order 2^(l+1).

    Parameters:
        check_lists: For each qubit in the gate location, the list of checks containing that qubit
        l: The order of the ansatz gate is 2^(l+1)

    Returns:
        Dict mapping check combinations (frozensets of entries of check_lists) to the integer numerator of their phase factor, relative to the denominator 2^(l+1)
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

def pullback_homomorphism(nr_qubits, checks, gates):
    """
    Compute the pullback homomorphism from qubit phase functions (combinations of ansatz gates) to check phase functions.

    Parameters:
        nr_qubits: Number of qubits of the CSS code on which the gate locations are specified
        checks: X checks of the CSS code (list of qubit sets, like GateFinder.checks)
        gates: List of ansatz gates (like GateFinder.gates)
    """
    # transpose check matrix in sparse form - list of checks involving each qubit.
    checks_reverse = [[] for _ in range(nr_qubits)]
    for j, check in enumerate(checks):
        for i in check:
            checks_reverse[i].append(j)

    # for each gate order l, and each physical gate location lgate, create a dictionary corresponding to the pullback of that gate location
    pullback_columns = [[pullback_column([checks_reverse[g] for g in lgate], l) for lgate in lgates]
                        for l, lgates in enumerate(gates)]

    return assemble_pullback(pullback_columns, [len(lgates) for lgates in gates])

def assemble_pullback(pullback_columns, gate_dims):
    """
    Assemble pullback columns into a Hom object.

    Parameters:
        pullback_columns: pullback_columns[l][i] is the pullback of the ith ansatz gate of order 2^(l+1), a dict as returned by pullback_column
        gate_dims: Number of ansatz gates at each order

    Returns:
        pullback: Hom object with one row for each check combination with non-trivial phase factor
        check_gate_locs: List of check gate locations (the dict keys of the columns) for each level
    """
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
                    if lev < l:
                        val = val // 2**(l-lev)
                    # Hom coefficients between Z_{2^(l+1)} and Z_{2^(lev+1)} are valued in Z_{2^(min(lev,l)+1)}
                    col_rescaled[gate_loc] = val % 2**(min(lev, l)+1)
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

    # assemble into dense matrix
    pullback = lin.Hom.zeros(check_dims, gate_dims)
    for l, lcols in enumerate(pullback_columns):
        for i, lcol in enumerate(lcols):
            for gate_loc, val in lcol.items():
                gate_lev = check_gate_levels[gate_loc]
                gate_ind = check_gate_indices[gate_lev][gate_loc]
                pullback[gate_lev, l][gate_ind, i] = val

    return pullback, check_gate_locs

def transpose_hom(X):
    """Transpose (dual) of a 2-group homomorphism.

    Under the perfect pairing <x, y> = sum_i x_i*y_i / 2^(l_i+1), every finite abelian 2-group is its own dual, and the dual of a homomorphism is given by the plain block-wise transpose of its coefficient matrix, with source and target 2-groups interchanged. (The stored block coefficients are unchanged since the enhancement factor 2^max(0, i-j) turns into 2^max(0, j-i) under dualization, and the value group Z_{2^(min(i,j)+1)} is symmetric.)
    """
    return lin.Hom(X.M.T.copy(), X.dim1, X.dim0)

def quotient_image_by_image(K, P, K_solve_helper = None):
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

def gates_string(gates, gate_locs):
    """
    Write a set of gates as a string.

    Parameters:
        gates is: Hom object mapping from some abstract 2-group to the group of physical gates
        gate_locs: Ansatz gates (like GateFinder.gates)
    """
    gate_string = ""
    for lj in range(len(gates.dim1)):
        gate_string += f"order {2**(lj+1)}\n"
        for j in range(gates.dim1[lj]):
            for li in range(len(gates.dim0)):
                for i in range(gates.dim0[li]):
                    if gates[li, lj][i, j] != 0:
                        gate_string += f"{gates[li, lj][i, j]}/{2**(min(li, lj)+1)}*{sorted(gate_locs[li][i])}, "
            gate_string = gate_string[:-2] + "\n"
    return gate_string

def gate_string(gate, gate_locs):
    """
    Write an individual gate as a string.

    Parameters:
        gate: Elem object, representing a physical gate
        gate_locs: Like GateFinder.gates
    """
    gate_string = ""
    for li in range(len(gate.dim)):
        for i in range(gate.dim[li]):
            if gate[li][i] != 0:
                gate_string += f"{gate[li][i]}/{2**(li+1)}*{sorted(gate_locs[li][i])}, "
    return gate_string[:-2]

def matrix_from_list(nr_rows, index_list):
    """Generate matrix from a list of index sets - each list entry corresponds to one column, and the integers in the entry are the rows where the column is non-zero."""
    check_list = []
    for check_num in index_list:
        check = np.zeros((nr_rows,), dtype=int)
        check[list(check_num)] = 1
        check_list.append(check)
    if not check_list:
        return np.zeros((nr_rows, 0), dtype=int)
    return np.array(check_list).T

def list_from_matrix(matrix):
    return [set((np.nonzero(matrix[:, i])[0]).tolist()) for i in range(matrix.shape[1])]

def shift_loc_list(loc_list, shift):
    return [{bit+shift for bit in loc} for loc in loc_list]
