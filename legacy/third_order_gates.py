from . import z248_linalg as lin
from . import flint_wrappers as fl
#import scipy.linalg
import numpy as np
#import itertools


class transversal_gate_finder:
    # 
    def __init__(self, nr_qubits, checks = None, t_gates = None, cs_gates = None, ccz_gates = None, logicals = None, other_checks = None):
        self.nr_qubits = nr_qubits
        # List of X checks. Each X check is a list of qubit numbers involved.
        self.checks = checks if checks is not None else []
        # List of X logicals
        self.logicals = logicals if logicals is not None else []
        # List of Z logicals (optional)
        self.other_checks = other_checks if other_checks is not None else []

        self.t_gates = t_gates if t_gates is not None else []
        self.cs_gates = cs_gates if cs_gates is not None else []
        self.ccz_gates = ccz_gates if ccz_gates is not None else []

    def add_checks(self, checks):
        self.checks += checks
    def add_logicals(self, logicals):
        self.logicals += logicals
    def add_other_checks(self, other_checks):
        self.other_checks += other_checks
    def add_t_gates(self, t_gates):
        self.t_gates += t_gates
    def add_cs_gates(self, cs_gates):
        self.cs_gates += cs_gates
    def add_ccz_gates(self, ccz_gates):
        self.ccz_gates += ccz_gates

    # add T gate locations on all qubits
    def add_t_gates_all(self):
        self.t_gates = [(i,) for i in range(self.nr_qubits)]

    # add CS gate locations between all pairs of qubits inside a common group
    def add_cs_gates_groups(self, groups):
        loc_set = set()
        for group in groups:
            for i in range(len(group)):
                for j in range(i):
                    loc_set.add(tuple(sorted([group[i], group[j]])))
        self.cs_gates += list(loc_set)

    def add_cs_gates_checks(self):
        self.add_cs_gates_groups(self.checks)

    # add CZZ gate locations between all triples of qubits that are part of a group
    def add_ccz_gates_groups(self, groups):
        loc_set = set()
        for group in groups:
            for i in range(len(group)):
                for j in range(i):
                    for k in range(j):
                        loc_set.add(tuple(sorted([group[i], group[j], group[k]])))
        self.CCZ_gates += list(loc_set)

    def add_ccz_gates_checks(self):
        self.add_ccz_gates_groups(self.checks)

    # "gate_locs[i]" is a numpy array, whose first dimension is i, and whose second dimension is the number of i-tuples where a 3rd order diagonal gate may be supported.
    # checks is a numpy matrix (columns are the checks)
    @staticmethod
    def pullback_homomorphism(gate_locs, checks):
        nr_qubits, nr_checks = checks.shape
        check_locs = transversal_gate_finder.locs_all(nr_checks)
        # print(check_locs)
        # print(gate_locs)
        def check_at(gate_level, check_level, gate_nr, check_nr):
            return checks[gate_locs[gate_level][gate_nr], :][:, check_locs[check_level][check_nr]].T
        z8z8_map = check_at(0,0,0,0)
        z8z4_map = check_at(1,0,0,0) * check_at(1,0,1,0)
        z8z2_map = check_at(2,0,0,0) * check_at(2,0,1,0) * check_at(2,0,2,0)
        z4z8_map = 3 * check_at(0,1,0,0) * check_at(0,1,0,1)
        z4z4_map = (check_at(1,1,0,0) * check_at(1,1,1,1) + check_at(1,1,0,1) * check_at(1,1,1,0)
                    + 2 * check_at(1,1,0,0) * check_at(1,1,0,1) * check_at(1,1,1,0) + 2 * check_at(1,1,0,0) * check_at(1,1,0,1) * check_at(1,1,1,1)
                    + 2 * check_at(1,1,0,0) * check_at(1,1,1,0) * check_at(1,1,1,1) + 2 * check_at(1,1,0,1) * check_at(1,1,1,0) * check_at(1,1,1,1))
        z4z2_map = (check_at(2,1,0,0) * check_at(2,1,1,0) * check_at(2,1,2,1) + check_at(2,1,0,0) * check_at(2,1,1,1) * check_at(2,1,2,0) + check_at(2,1,0,0) * check_at(2,1,1,1) * check_at(2,1,2,1)
                    + check_at(2,1,0,1) * check_at(2,1,1,0) * check_at(2,1,2,0) + check_at(2,1,0,1) * check_at(2,1,1,0) * check_at(2,1,2,1) + check_at(2,1,0,1) * check_at(2,1,1,1) * check_at(2,1,2,0))
        z2z8_map = check_at(0,2,0,0) * check_at(0,2,0,1) * check_at(0,2,0,2)
        z2z4_map = (check_at(1,2,0,0) * check_at(1,2,0,1) * check_at(1,2,1,2) + check_at(1,2,0,0) * check_at(1,2,1,1) * check_at(1,2,0,2) + check_at(1,2,0,0) * check_at(1,2,1,1) * check_at(1,2,1,2)
                    + check_at(1,2,1,0) * check_at(1,2,0,1) * check_at(1,2,0,2) + check_at(1,2,1,0) * check_at(1,2,0,1) * check_at(1,2,1,2) + check_at(1,2,1,0) * check_at(1,2,1,1) * check_at(1,2,0,2))
        z2z2_map = (check_at(2,2,0,0) * check_at(2,2,1,1) * check_at(2,2,2,2) + check_at(2,2,0,0) * check_at(2,2,1,2) * check_at(2,2,2,1)
                    + check_at(2,2,0,1) * check_at(2,2,1,0) * check_at(2,2,2,2) + check_at(2,2,0,1) * check_at(2,2,1,2) * check_at(2,2,2,0)
                    + check_at(2,2,0,2) * check_at(2,2,1,0) * check_at(2,2,2,1) + check_at(2,2,0,2) * check_at(2,2,1,1) * check_at(2,2,2,0))
        full_map = np.vstack([np.hstack([z2z2_map, z2z4_map, z2z8_map]), np.hstack([z4z2_map, z4z4_map, z4z8_map]), np.hstack([z8z2_map, z8z4_map, z8z8_map])])
        dim0 = [check_locs[2-i].shape[1] for i in range(3)]
        dim1 = [gate_locs[2-i].shape[1] for i in range(3)]
        return lin.z248_hom(full_map, dim0, dim1)

    # generate matrix from a list of index lists - each list entry corresponds to one column, and the integers in the entry are the rows where the column is non-zero
    @staticmethod
    def matrix_from_list(nr_rows, index_list):
        check_list = []
        #print(index_list)
        for check_num in index_list:
            check = np.zeros((nr_rows,), dtype=int)
            check[check_num] = 1
            check_list.append(check)
        return np.array(check_list).T
    
    @staticmethod
    def locs_all(nr):
        all_locs = []
        all_locs.append(np.array([(i,) for i in range(nr)]).T)
        all_locs.append(np.array([(i,j) for i in range(nr) for j in range(i+1,nr)],dtype=int).reshape(nr * (nr - 1)//2, 2).T)
        all_locs.append(np.array([(i,j,k) for i in range(nr) for j in range(i+1,nr) for k in range(j+1,nr)],dtype=int).reshape(nr * (nr - 1) * (nr - 2)//6, 3).T)
        return all_locs
    
    @staticmethod
    def list_from_matrix(matrix):
        return [(np.nonzero(matrix[:, i])[0]).tolist() for i in range(matrix.shape[1])]
    
    # def set_checks_from_list(self, check_indices):
    #     self.checks = transversal_gate_finder.matrix_from_list(self.nr_qubits, check_indices)

    # def set_logicals_from_list(self, logical_indices):
    #     self.logicals = transversal_gate_finder.matrix_from_list(self.nr_qubits, logical_indices)

    def logicals_from_other_checks(self):
        #print("test x and z stabilizers commute:", ~np.any(self.checks.T @ zchecks % 2 == 1))
        #self.test_commutation()
        check_array = transversal_gate_finder.matrix_from_list(self.nr_qubits, self.checks)
        other_check_array = transversal_gate_finder.matrix_from_list(self.nr_qubits, self.other_checks)

        zker = lin.z2lin.kernel(other_check_array.T)
        #print("zker", zker)
        ind_xchecks, log_nrs = lin.remove_image(check_array, zker)
        #print("independent x check numbers:", ind_xchecks)
        logical_array = zker[:, log_nrs]
        #print("logarr", logical_array)
        self.logicals = transversal_gate_finder.list_from_matrix(logical_array)

        #print(f"found {len(self.logicals)} logical x operators\n")

    @staticmethod
    def gates_string(gates, gate_locs):
        gates_enhance = gates.enhanced()
        gate_string = ""
        for vec in gates_enhance.T:
            for i in range(gates.dim0[0]):
                entry = vec[i]
                if entry != 0:
                    gate_string += "CCZ^"+str(entry)+"["+ str(gate_locs[2][0,i]) + ", " + str(gate_locs[2][1,i]) + ", " + str(gate_locs[2][2,i]) +"] * "
            for i in range(gates.dim0[1]):
                entry = vec[gates.dim0[0]+i]
                if entry != 0:
                    gate_string += "CS^"+str(entry)+"[" + str(gate_locs[1][0,i]) + ", " + str(gate_locs[1][1,i]) + "] * "
            for i in range(gates.dim0[2]):
                entry = vec[gates.dim0[0]+gates.dim0[1]+i]
                if entry != 0:
                    gate_string += "T^"+str(entry)+"["+ str(gate_locs[0][0,i]) + "] * "
            gate_string += "\n"
        return gate_string
    
    @staticmethod
    def gates_string(gates, gate_locs):
        gates_enhance = gates.enhanced()
        gate_string = ""
        for vec in gates_enhance.T:
            for i in range(gates.dim0[0]):
                entry = vec[i]
                if entry != 0:
                    gate_string += "CCZ^"+str(entry)+"["+ str(gate_locs[2][0,i]) + ", " + str(gate_locs[2][1,i]) + ", " + str(gate_locs[2][2,i]) +"] * "
            for i in range(gates.dim0[1]):
                entry = vec[gates.dim0[0]+i]
                if entry != 0:
                    gate_string += "CS^"+str(entry)+"[" + str(gate_locs[1][0,i]) + ", " + str(gate_locs[1][1,i]) + "] * "
            for i in range(gates.dim0[2]):
                entry = vec[gates.dim0[0]+gates.dim0[1]+i]
                if entry != 0:
                    gate_string += "T^"+str(entry)+"["+ str(gate_locs[0][0,i]) + "] * "
            gate_string += "\n"
        return gate_string
    
    @staticmethod
    def remove_zero_rows(hom):
        non_zero_rows = np.any(hom.M, axis=1)
        new_dim0 = (int(non_zero_rows[:hom.dim0[0]].sum()), int(non_zero_rows[hom.dim0[0]:hom.dim0[0]+hom.dim0[1]].sum()), int(non_zero_rows[hom.dim0[0]+hom.dim0[1]:].sum()))
        new_M = hom.M[non_zero_rows,:]
        return lin.z248_hom(new_M, new_dim0, hom.dim1)

    def find_gates(self):
        check_array = transversal_gate_finder.matrix_from_list(self.nr_qubits, self.checks)
        self.gate_loc_arrays = []
        self.gate_loc_arrays.append(np.array(self.t_gates, dtype=int).T.reshape(1,len(self.t_gates)))
        self.gate_loc_arrays.append(np.array(self.cs_gates, dtype=int).T.reshape(2,len(self.cs_gates)))
        self.gate_loc_arrays.append(np.array(self.ccz_gates, dtype=int).T.reshape(3,len(self.ccz_gates)))

        # allphys: group of all physical diagonal gates at the gate_locs
        # transphys: group of all physical diagonal gates that preserve the code space, including ones with trivial logical action
        # allcheck: group of all phase functions on the checks
        # alllog: group of all diagonal logical gates
        # translog: group of all logical gates with transversal physical implementation
        # stabphys: group of physical transversal gates with trivial logical action

        allphys_allcheck_full = transversal_gate_finder.pullback_homomorphism(self.gate_loc_arrays, check_array)
        allphys_allcheck = transversal_gate_finder.remove_zero_rows(allphys_allcheck_full) # map all physical -> all check
        self.transphys_allphys = allphys_allcheck.kernel() # map transversal physical -> all physical

        logical_array = transversal_gate_finder.matrix_from_list(self.nr_qubits, self.logicals)
        allphys_alllog = transversal_gate_finder.pullback_homomorphism(self.gate_loc_arrays, logical_array) # map all physical -> all logical
        transphys_alllog = allphys_alllog @ self.transphys_allphys # map transversal physical -> all logical
        self.translog_alllog, self.transphys_translog = lin.z248_epi_mono(transphys_alllog) # map transversal logical -> all logical
        stabphys_transphys, self.rep_find_helper = self.transphys_translog.kernel(return_solve_helper = True) # stabilizer physical -> transversal physical
        self.stabphys_allphys = self.transphys_allphys @ stabphys_transphys # stabilizer physical -> all physical

    def print_transversal_logicals(self):
        print(transversal_gate_finder.gates_string(self.translog_alllog, transversal_gate_finder.locs_all(len(self.logicals))))

    def print_physical_stabilizers(self):
        print("Physical stabilizers:\n" + transversal_gate_finder.gates_string(self.stabphys_allphys, self.gate_loc_arrays))

    # finds physical representative for transversal logical gate
    # logic_gate: z248_elem object, linear combination of generator transversal logicals
    def find_phys_rep(self, logic_gate):
        logic_elem = lin.z248_elem(np.array(logic_gate), self.translog_alllog.dim1)
        transphys_rep = lin.z248_solve_with_helper(self.transphys_translog, logic_elem, self.rep_find_helper)
        phys_rep = self.transphys_allphys @ transphys_rep
        return phys_rep
        

    # tests if z checks commute with x checks and x logicals
    def test_commutation(self):
        check_array = transversal_gate_finder.matrix_from_list(self.nr_qubits, self.checks)
        other_check_array = transversal_gate_finder.matrix_from_list(self.nr_qubits, self.other_checks)
        print("X checks and Z checks commute:", ~np.any((check_array.T @ other_check_array)%2))
        #print("X logicals and Z checks commute:", ~np.any((self.logicals.T @ z_checks)%2))

    
    def __add__(self, other):
        return transversal_gate_finder(self.nr_qubits + other.nr_qubits,
                                       checks= self.checks + shift_loc_list(other.checks, self.nr_qubits),
                                       t_gates= self.t_gates + shift_loc_list(other.t_gates, self.nr_qubits),
                                       cs_gates= self.cs_gates + shift_loc_list(other.cs_gates, self.nr_qubits),
                                       ccz_gates= self.ccz_gates + shift_loc_list(other.ccz_gates, self.nr_qubits),
                                       logicals = self.logicals + shift_loc_list(other.logicals, self.nr_qubits),
                                       other_checks = self.other_checks + shift_loc_list(other.other_checks, self.nr_qubits))




# helper class for analyzing translation-invariant codes in n dimensions
class ti_transversal_gate_finder:
    def __init__(self, nr_qubits, dimension, checks = None, t_gates = None, cs_gates = None, ccz_gates = None, other_checks = None):
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
        self.t_gates = []
        if t_gates is not None:
            self.add_t_gates(t_gates)
        self.cs_gates = []
        if cs_gates is not None:
            self.add_cs_gates(cs_gates)
        self.ccz_gates = []
        if ccz_gates is not None:
            self.add_ccz_gates(ccz_gates)

    def add_checks(self, checks):
        self.check_qubits_valid(checks)
        self.checks += checks
    def add_other_checks(self, other_checks):
        self.check_qubits_valid(other_checks)
        self.other_checks += other_checks
    def add_t_gates(self, t_gates):
        self.check_qubits_valid(t_gates)
        self.t_gates += t_gates
    def add_cs_gates(self, cs_gates):
        self.check_qubits_valid(cs_gates)
        self.cs_gates += cs_gates
    def add_ccz_gates(self, ccz_gates):
        self.check_qubits_valid(ccz_gates)
        self.ccz_gates += ccz_gates

    def check_qubits_valid(self, listlist):
        for mlist in listlist:
            for coord, intern in mlist:
                if len(coord) != self.dimension:
                    raise ValueError(f"Coordinate {coord} has wrong number of entries (should be {self.dimension})")
                if intern < 0 or intern >= self.nr_qubits:
                    raise ValueError(f"Internal qubit number {intern} given but must be between 0<={intern}<{self.nr_qubits}")
                
    def add_all_t(self):
        for intern in range(self.nr_qubits):
            self.t_gates.append([([0]*self.dimension, intern)])


    def add_all_cs_onsite(self):
        zerocoord = [0]*self.dimension
        for intern0 in range(self.nr_qubits):
            for intern1 in range(intern0+1, self.nr_qubits):
                self.cs_gates.append([(zerocoord, intern0), (zerocoord, intern1)])

    # adds CS gate locations between all qubits at the origin and all qubits at coord
    def add_all_cs_coord(self, coord):
        zerocoord = [0]*len(coord)
        for intern1 in range(self.nr_qubits):
            for intern2 in range(self.nr_qubits):
                self.cs_gates.append([(zerocoord, intern1), (list(coord), intern2)])

    def add_all_cs_block(self, block_size):
        total_dim = np.prod(block_size)
        cum_dims = np.cumprod(block_size)
        cum_dims = np.insert(cum_dims, 0, 1)[:-1]
        
        self.add_all_cs_onsite()
        for coord_nr in range(1, total_dim):
            self.add_all_cs_coord(ti_transversal_gate_finder.nr_to_coord(coord_nr, cum_dims, block_size).tolist())

    def add_all_ccz_onsite(self):
        zerocoord = [0]*self.dimension
        for intern0 in range(self.nr_qubits):
            for intern1 in range(intern0+1, self.nr_qubits):
                for intern2 in range(intern1+1, self.nr_qubits):
                    self.ccz_gates.append([(zerocoord, intern0), (zerocoord, intern1), (zerocoord, intern2)])

    def add_all_ccz_onecoord(self, coord):
        zerocoord = [0]*self.dimension
        for intern0 in range(self.nr_qubits):
            for intern1 in range(self.nr_qubits):
                for intern2 in range(intern1+1, self.nr_qubits):
                    self.ccz_gates.append([(zerocoord, intern0), (coord, intern1), (coord, intern2)])

    def add_all_ccz_twocoord(self, coord0, coord1):
        zerocoord = [0]*self.dimension
        for intern0 in range(self.nr_qubits):
            for intern1 in range(self.nr_qubits):
                for intern2 in range(self.nr_qubits):
                    self.ccz_gates.append([(zerocoord, intern0), (coord0, intern1), (coord1, intern2)])

    def add_all_ccz_block(self, block_size):
        total_dim = np.prod(block_size)
        cum_dims = np.cumprod(block_size)
        cum_dims = np.insert(cum_dims, 0, 1)[:-1]

        self.add_all_ccz_onsite()
        for coord_nr in range(1, total_dim):
            self.add_all_ccz_onecoord(ti_transversal_gate_finder.nr_to_coord(coord_nr, cum_dims, block_size).tolist())
        for coord_nr0 in range(1, total_dim):
            for coord_nr1 in range(coord_nr0+1, total_dim):
                self.add_all_ccz_twocoord(ti_transversal_gate_finder.nr_to_coord(coord_nr0, cum_dims, block_size).tolist(), ti_transversal_gate_finder.nr_to_coord(coord_nr1, cum_dims, block_size).tolist())


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
        tgf.add_t_gates(ti_transversal_gate_finder.generate_ti_list(lattice_hnf, cum_dims, total_dim, periods, self.t_gates))
        tgf.add_cs_gates(ti_transversal_gate_finder.generate_ti_list(lattice_hnf, cum_dims, total_dim, periods, self.cs_gates))
        tgf.add_ccz_gates(ti_transversal_gate_finder.generate_ti_list(lattice_hnf, cum_dims, total_dim, periods, self.ccz_gates))
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
        return ti_transversal_gate_finder(self.nr_qubits + other.nr_qubits,
                                       self.dimension,
                                       checks= self.checks + shift_ti_loc_list(other.checks, self.nr_qubits),
                                       t_gates= self.t_gates + shift_ti_loc_list(other.t_gates, self.nr_qubits),
                                       cs_gates= self.cs_gates + shift_ti_loc_list(other.cs_gates, self.nr_qubits),
                                       ccz_gates= self.ccz_gates + shift_ti_loc_list(other.ccz_gates, self.nr_qubits),
                                       other_checks = self.other_checks + shift_ti_loc_list(other.other_checks, self.nr_qubits))
    def inverted_coordinates(self):
        def invert_coords(mlist):
            return [[((-np.array(coord)).tolist(),i) for coord, i in entry] for entry in mlist]
        return ti_transversal_gate_finder(self.nr_qubits,
                                              self.dimension,
                                              checks = invert_coords(self.checks),
                                              other_checks = invert_coords(self.other_checks),
                                              t_gates = invert_coords(self.t_gates),
                                              cs_gates = invert_coords(self.cs_gates),
                                              ccz_gates = invert_coords(self.ccz_gates))
    
def shift_ti_loc_list(loc_list, shift):
    return [[(coord, bit+shift) for coord, bit in loc] for loc in loc_list]

def shift_loc_list(loc_list, shift):
    return [[bit+shift for bit in loc] for loc in loc_list]



# constructs the pullback homomorphism for a 2D translation-invariant CSS code
# call a triple (x-coord, y-coord, internal-nr) a *qubit triple*
# xcheck_list: List of list of qubit triples. Each list is one column of the X parity check matrix; the full parity check matrix is obtain from considering these columns shifted by all (x-coord, y-coord) pairs
# gate_loc_list[0]: List of qubit triples.
# gate_loc_list[1]: List of pairs or qubit triples
# gate_loc_list[2]: List of triples of qubit triples
def ti_2d_pullback(gate_loc_list, xcheck_list):
    pass

# computes all singlets, pairs, and triples of checks that may be in the support of the pullback map
# singlet is only in if there is a gate location supported entirely on the check
# pair is only in if there's a gate location which has overlap
# unfinished
# needed? can this be more efficient than remove_zero_rows?
def check_locs(gate_locs, checks):
    singlet_mask = np.zeros(checks.shape[1], dtype=bool)
    for i in gate_locs[0].shape[1]:
        gate_loc = gate_locs[0][:,i]
        check_contains = None