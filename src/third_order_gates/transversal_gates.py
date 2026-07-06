from . import twogroup_linalg as lin
from . import flint_wrappers as fl
#import scipy.linalg
import numpy as np
#import itertools
from itertools import combinations, product


class transversal_gate_finder:
    # 
    def __init__(self, nr_qubits, checks = None, gates = None, logicals = None, other_checks = None):
        self.nr_qubits = nr_qubits
        # List of X checks. Each X check is a list of qubit numbers involved.
        self.checks = checks if checks is not None else []
        # List of X logicals
        self.logicals = logicals if logicals is not None else []
        # List of Z logicals (optional)
        self.other_checks = other_checks if other_checks is not None else []

        # possible gate locations for diagonal clifford-hierarchy gates, list of lists of frozensets of int
        # the l-th entry of self.gates is a list of all order-l gate generators A with a prefactor 1/2^(l+1)
        # A is a list of qubits participating in the gate
        self.gates = gates if gates is not None else []

    

    def add_checks(self, checks):
        self.checks += checks
    def add_logicals(self, logicals):
        self.logicals += logicals
    def add_other_checks(self, other_checks):
        self.other_checks += other_checks


    # add empty sets for gates of orders up to l
    def extend_gate_orders(self, l):
        if len(self.gates) <= l:
            self.gates += [[] for _ in range(l - len(self.gates) + 1)]
    
    # remove duplicate gates, and remove gate locations of order l if they are already present at a higher order
    def consolidate_gates(self):
        current_gatesset = set()
        for l in reversed(range(len(self.gates))):
            lgates_set = set(map(frozenset, self.gates[l])) - current_gatesset
            current_gatesset |= lgates_set
            self.gates[l] = list(map(set, lgates_set))


    def add_gates(self, gates, l):
        self.extend_gate_orders(l)
        self.gates[l] += gates

    # add single-qubit gate locations at all qubits
    def add_all_singlequbit_gates(self, l):
        self.extend_gate_orders(l)
        self.gates[l] += [{i} for i in range(self.nr_qubits)]
    
    # adds all gate locations of size k and phase-level l within each group in "groups".
    def add_gates_in_groups(self, groups, l, k):
        self.extend_gate_orders(l)
        loc_set = set()
        for group in groups:
            loc_set.update(set(map(frozenset, combinations(group, k))))
        self.gates[l] += list(map(set, loc_set))

    def add_gates_in_checks(self, l, k):
        self.add_gates_in_groups(self.checks, l, k)

 
    def logicals_from_other_checks(self):
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
        # allphys: group of all physical diagonal gates at the gate_locs
        # transphys: group of all physical diagonal gates that preserve the code space, including ones with trivial logical action
        # allcheck: group of all phase functions on the checks
        # alllog: group of all diagonal logical gates
        # translog: group of all logical gates with transversal physical implementation
        # stabphys: group of physical transversal gates with trivial logical action

        allphys_allcheck = self.pullback_checks() # map all physical -> all check
        self.transphys_allphys = allphys_allcheck.kernel() # map transversal physical -> all physical
        allphys_alllog, self.alllog_locs = self.pullback_logicals() # map all physical -> all logical
        transphys_alllog = allphys_alllog @ self.transphys_allphys # map transversal physical -> all logical
        self.translog_alllog, self.transphys_translog = transphys_alllog.epi_mono() # map transversal logical -> all logical
        stabphys_transphys, self.rep_find_helper = self.transphys_translog.kernel(return_solve_helper = True) # stabilizer physical -> transversal physical
        self.stabphys_allphys = self.transphys_allphys @ stabphys_transphys # stabilizer physical -> all physical

    def print_transversal_logicals(self):
        print(gates_string(self.translog_alllog, self.alllog_locs))

    def print_physical_stabilizers(self):
        print(gates_string(self.stabphys_allphys, self.gates))

    # finds physical representative for transversal logical gate
    # logic_gate: z248_elem object, linear combination of generator transversal logicals
    def find_phys_rep(self, logic_gate):
        logic_elem = lin.elem(np.array(logic_gate), self.translog_alllog.dim1)
        transphys_rep = self.transphys_translog.solve_with_helper(logic_elem, self.rep_find_helper)
        phys_rep = self.transphys_allphys @ transphys_rep
        return phys_rep
    
    def print_phys_rep(self, logic_gate):
        rep = self.find_phys_rep(logic_gate)
        print(gate_string(rep, self.gates))

    # tests if z checks commute with x checks and x logicals
    def test_commutation(self):
        check_array = matrix_from_list(self.nr_qubits, self.checks)
        other_check_array = matrix_from_list(self.nr_qubits, self.other_checks)
        print("X checks and Z checks commute:", ~np.any((check_array.T @ other_check_array)%2))
        #print("X logicals and Z checks commute:", ~np.any((self.logicals.T @ z_checks)%2))

    
    def __add__(self, other):
        res = transversal_gate_finder(self.nr_qubits + other.nr_qubits)
        res.checks= self.checks + shift_loc_list(other.checks, self.nr_qubits)
        res.logicals = self.logicals + shift_loc_list(other.logicals, self.nr_qubits)
        res.other_checks = self.other_checks + shift_loc_list(other.other_checks, self.nr_qubits)
        maxlev = max(len(self.gates), len(other.gates))
        res.gates = self.gates
        res.extend_gate_orders(maxlev)
        for l in range(len(other.gates)):
            res.gates[l] += shift_loc_list(other.gates[l], self.nr_qubits)
        return res                       


# gates is a list of pairs (l,loc). l is the power of the gate, i.e. there's a 1/2^(l+1) prefactor. loc is list of qubits where the gate acts.
# checks is a list of list (pure-python sparse binary matrix)
def pullback_homomorphism(nr_qubits, checks, gates):
    # transpose check matrix in sparse form - list of checks involving each qubit.
    checks_reverse = [[] for _ in range(nr_qubits)]
    for j, check in enumerate(checks):
        for i in check:
            checks_reverse[i].append(j)
    
    # enumerate all possible divisions of Lambda as a sum of k positive integers
    def weak_compositions(Lambda, k):
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
    pullback = lin.hom.zeros(check_dims, gate_dims)
    for l, lcols in enumerate(pullback_columns):
        for i, lcol in enumerate(lcols):
            for gate_loc, val in lcol.items():
                gate_lev = check_gate_levels[gate_loc]
                gate_ind = check_gate_indices[gate_lev][gate_loc]
                pullback[gate_lev, l][gate_ind, i] = val

    return pullback, check_gate_locs

# writes a set of gates as string
# gates is the hom mapping to the group of physical gates
# gate_locs is list of list of sets
def gates_string(gates, gate_locs):
    gate_string = ""
    for lj in range(len(gates.dim1)):
        gate_string += f"order {2**(lj+1)}\n"
        for j in range(gates.dim1[lj]):
            for li in range(len(gates.dim0)):
                for i in range(gates.dim0[li]):
                    if gates[li, lj][i, j] != 0:
                        gate_string += f"{gates[li, lj][i, j]}/{2**(min(li, lj)+1)}*{list(gate_locs[li][i])}, "
            gate_string = gate_string[:-2] + "\n"
    return gate_string

# writes a gate as string
# gate is a elem, corresponding to an element of the group of physical gates
# gate_locs is list of list of sets
def gate_string(gate, gate_locs):
    gate_string = ""
    for li in range(len(gate.dim)):
        for i in range(gate.dim[li]):
            if gate[li][i] != 0:
                gate_string += f"{gate[li][i]}/{2**(li+1)}*{list(gate_locs[li][i])}, "
    return gate_string[:-2]

# generate matrix from a list of index lists - each list entry corresponds to one column, and the integers in the entry are the rows where the column is non-zero
def matrix_from_list(nr_rows, index_list):
    check_list = []
    #print(index_list)
    for check_num in index_list:
        check = np.zeros((nr_rows,), dtype=int)
        check[check_num] = 1
        check_list.append(check)
    return np.array(check_list).T

def list_from_matrix(matrix):
    return [(np.nonzero(matrix[:, i])[0]).tolist() for i in range(matrix.shape[1])]

# helper class for analyzing translation-invariant codes in n dimensions
# checks, gates, other_checks are the same as for transversal_gate_finder, just that a qubit number is replaced by a pair of (1) shift coordinate (int tuple) and (2) unit-cell-internal qubit number
class ti_transversal_gate_finder:
    def __init__(self, nr_qubits, dimension, checks = None, gates = None, other_checks = None):
        self.nr_qubits = nr_qubits # number of qubits per unit cell
        self.dimension = dimension
        # checks: list of checks per unit cell
        # format: each check is a list of pairs (coordinate (numpy vector), internal qubit number)
        self.checks = []
        if checks is not None:
            self.add_checks(checks)
        self.other_checks = []
        if other_checks is not None:
            self.add_other_checks(other_checks)
        self.gates = []
        if gates is not None:
            self.extend_levels(len(gates))
            for l in range(len(gates)):
                self.add_gates(gates[l], l)

    def add_checks(self, checks):
        self.check_qubits_valid(checks)
        self.checks += checks
    def add_other_checks(self, other_checks):
        self.check_qubits_valid(other_checks)
        self.other_checks += other_checks

    # add empty sets for gates of orders up to l
    def extend_gate_orders(self, l):
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
        for intern in range(self.nr_qubits):
            self.gates[l].append([((0,)*self.dimension, intern)])

    # adds all gate locations of size k and phase-level l within each group in "groups".
    # each group in groups is a list of pairs of coord tuple and internal qubit nr
    def add_gates_in_groups(self, groups, l, k):
        self.extend_gate_orders(l)
        loc_set = set()
        for group in groups:
            loc_set.update(set(map(frozenset, map(normalize_ti_gate, combinations(group, k)))))
        self.gates[l] += list(map(set, loc_set))


    # transform into a regular code by putting it on a finite lattice with twisted boundary conditions
    # lattice: numpy array containing basis vectors. the rows are the vectors that are identified with the origin.
    def as_finite_code(self, lattice):
        lattice_hnf = fl.hnf(lattice)[0] # row-operation hnf
        #print("hnf\n", lattice_hnf)
        periods = lattice_hnf.diagonal()
        total_dim = np.prod(periods)
        cum_dims = np.cumprod(periods)
        cum_dims = np.insert(cum_dims, 0, 1)[:-1]

        tgf = transversal_gate_finder(total_dim * self.nr_qubits)
        tgf.add_checks(ti_transversal_gate_finder.generate_ti_list(lattice_hnf, cum_dims, total_dim, periods, self.checks))
        for l in range(len(self.gates)):
            tgf.add_gates(ti_transversal_gate_finder.generate_ti_list(lattice_hnf, cum_dims, total_dim, periods, self.gates[l]), l)
        tgf.add_other_checks(ti_transversal_gate_finder.generate_ti_list(lattice_hnf, cum_dims, total_dim, periods, self.other_checks))

        return tgf

    # reduces a coordinate vector to lie within the parallelogram defined by the hnf
    @staticmethod
    def reduce_coordinate(coord, hnf):
        #print("coords before\n", coords)
        for d in range(len(coord)):
            offs = coord[d] // hnf[d,d]
            coord -= offs * hnf[d, :]
        return coord

    @staticmethod
    def nr_to_coord(nr, cum_dims, periods):
        return np.array(nr)[None] // cum_dims % periods
    
    # takes a pair of (coordinate vector modulo PBC lattice, internal qubit number) and transforms it into a global qubit number
    # lattice_hnf: pbc lattice in hermite normal form, 
    # cum_dims: cumulative product of diagonal of lattice_hnf starting from 1
    # coord: unit cell coordinate (not necessarily reduced to the standard box)
    # internal: internal qubit number
    # output: qubit number
    @staticmethod
    def coord_to_qubit(lattice_hnf, cum_dims, total_dim, coord, internal):
        for d in range(len(coord)):
            offs = coord[d] // lattice_hnf[d,d]
            coord -= offs * lattice_hnf[d, :]

        return int(np.dot(coord, cum_dims) + internal * total_dim)
    
    # takes a list of lists of coordinates + internal qubit nr's and turns it into a list of list of qubits
    @staticmethod
    def generate_ti_list(lattice_hnf, cum_dims, total_dim, periods, qubit_listlist):
        output_listlist = []
        for i in range(total_dim):
            shift_coord = ti_transversal_gate_finder.nr_to_coord(i, cum_dims, periods)
            for qubit_list in qubit_listlist:
                output_listlist.append([ti_transversal_gate_finder.coord_to_qubit(lattice_hnf, cum_dims, total_dim, coord+shift_coord, intern) for coord, intern in qubit_list])
                
        return output_listlist
    

    def __add__(self, other):
        if self.dimension != other.dimension:
            raise ValueError("Space(time) dimensions of added codes must agree.")
        res = ti_transversal_gate_finder(self.nr_qubits + other.nr_qubits, self.dimension)
        res.checks= self.checks + shift_ti_loc_list(other.checks, self.nr_qubits)
        res.other_checks = self.other_checks + shift_ti_loc_list(other.other_checks, self.nr_qubits)

        maxlev = max(len(self.gates), len(other.gates))
        res.gates = self.gates
        res.extend_gate_orders(maxlev)
        for l in range(len(other.gates)):
            res.gates[l] += shift_ti_loc_list(other.gates[l], self.nr_qubits)  
        return res        
        
    def inverted_coordinates(self):
        def invert_coords(mlist):
            return [[((-np.array(coord)).tolist(),i) for coord, i in entry] for entry in mlist]
        return ti_transversal_gate_finder(self.nr_qubits,
                                              self.dimension,
                                              checks = invert_coords(self.checks),
                                              other_checks = invert_coords(self.other_checks),
                                              gates = [invert_coords(gts) for gts in self.gates])
    
# shifting the internal qubit number (needed for addition of codes)
def shift_ti_loc_list(loc_list, shift):
    return [[(coord, bit+shift) for coord, bit in loc] for loc in loc_list]

def shift_loc_list(loc_list, shift):
    return [[bit+shift for bit in loc] for loc in loc_list]

# shifts a translation-invariant gate location to a standard coordinate, where the "smallest" coordinate is zero. for "smallest" we first compare the coordinates lexicographically and then the internal qubit nr
def normalize_ti_gate(gate):
    coordmin, _ = min(gate)
    gate_normalized = set()
    for gt_coord, gt_intern in gate:
        gt_coord_normalized = tuple(a-b for a, b in zip(gt_coord, coordmin))
        gate_normalized.add((gt_coord_normalized, gt_intern))
    return gate_normalized


# gates is a list of pairs (l,loc). l is the power of the gate, i.e. there's a 1/2^(l+1) prefactor. loc is list of qubits where the gate acts.
# checks is a list of list (pure-python sparse binary matrix)
def ti_pullback_homomorphism(nr_qubits, checks, gates):
    # transpose check matrix in sparse form - list of checks involving each qubit.
    checks_reverse = [[] for _ in range(nr_qubits)]
    for j, check in enumerate(checks):
        for i in check:
            checks_reverse[i].append(j)
    
    # enumerate all possible divisions of Lambda as a sum of k positive integers
    def weak_compositions(Lambda, k):
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
    pullback = lin.hom.zeros(check_dims, gate_dims)
    for l, lcols in enumerate(pullback_columns):
        for i, lcol in enumerate(lcols):
            for gate_loc, val in lcol.items():
                gate_lev = check_gate_levels[gate_loc]
                gate_ind = check_gate_indices[gate_lev][gate_loc]
                pullback[gate_lev, l][gate_ind, i] = val

    return pullback, check_gate_locs
