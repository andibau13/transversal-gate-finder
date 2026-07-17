# transversal-gate-finder

This package provides functions and classes for finding transversal gates in a given CSS code.
The gates are diagonal gates at any level of the Clifford hierarchy (Z, S, T, CZ, CS, CCZ, ...) and are composed of a given set of local ansatz gates.

The method is based on the observation that a diagonal gate configuration preserves the code space if and only if the pullback of its phase function under the X-check matrix vanishes. Finding all transversal gates thus reduces to computing the kernel of a homomorphism between abelian 2-groups, which can be done efficiently (in polynomial time). The 2-group linear algebra routines are based on a "filtration" method and implemented in the repository [twogroup-linalg](https://github.com/andibau13/twogroup-linalg). The $\mathbb Z_2$ subroutines are implemented using the package [bitgauss](https://github.com/akissinger/bitgauss).

For more information on the theory, have a look at the accompanying the paper *Efficiently finding (spacetime) diagonal transversal gates in CSS codes*.

## Functionality

- **`GateFinder`**: Helper class for finding transversal gates in finite CSS codes:
  - Attributes: `checks` (X checks), `logicals` (X logicals), `other_checks` (optional Z checks), `gates` (list of ansatz gates).
  - `checks`, `logicals`, `other_checks` are `Z2Hom` objects, add entries with `Z2Hom.add_columns`.
  - `gates` is a `PhaseLocs` object:
    - `gates.add_locs(locs, l)`: add gates of order $2^{l+1}$ (e.g. `l=0` for CZ, `l=2` for T) at all qubit sets in `locs`
    - `gates.add_all_single_locs(l)`: add a single-qubit gate of order $2^{l+1}$ at every qubit
    - `gates.add_locs_in_groups(groups, l, k)`: add a `k`-qubit gate of order $2^{l+1}$ for all each set of `k` qubits within each group in `groups`
  - `logicals_from_other_checks()`: if X and Z checks are given, set `logicals` to some arbitrary logical basis (possibly not the one you want to work with)
  - `+`: stack codes, to search for entangling gates like CZ or CCZ between multiple code blocks.
  - `find_gates()` computes all combinations of ansatz gates that preserve the code space, as the kernel of the pullback homomorphism, and returns a **`GateFinderResults`** object.

- **`GateFinderResults`**: Contains the results of the gate-finding algorithm and provides inspection methods:
  - `print_logical_dimension()`: Show number of independent $\mathbb Z_2$, $\mathbb Z_4$, $\mathbb Z_8$, ... generators of logical action.
  - `print_nontrivial_logicals()`: Shows generators for the logical action of all found transversal gates (transversal gates with trivial action are quotiented out). Gates are printed as sums like `7/8*[0], 1/8*[2, 5]`, meaning a `exp(2πi·7/8)` gate ($T^*$ gate) on qubit 0 times a `exp(2πi/8)` phase gate ($CT$ gate) on qubits 2 and 5.
  - `print_nontrivial_physicals()`: prints one physical representative for each generating logical action listed by `print_nontrivial_logicals`
  - `print_trivial_physicals()`: Lists physical gates with trivial logical action, i.e. stabilizers
  - `print_nontrivial_physical(...)`: Shows a physical representative for a linear combination of the logical actions given by `print_nontrivial_logicals()`
  - `print_physical_from_logical(...)`: Asserts whether a given logical action has a transversal implementation and returns physical representative if yes. The input is of the a list of `(qubit-set, level, coefficient)` triples, e.g. `[({0}, 1, 1), ({0,1}, 0, 1)]` represents an S on qubit 1 times a CZ on qubits 0 and 1.
  
- **`TIGateFinder`**: Analogous helper class for translation-invariant (crystalline) CSS codes on an infinite ℤ^d lattice with a finite unit cell.
  - Qubits/checks/gates are addressed as `((x,y,...), i)` with a lattice coordinate `(x,y,...)` and an internal index `i`. Methods for adding checks and gates etc are analogous to `GateFinder` apart from replacing each qubit index with a `((x,y,...), i)` specifier.
  - There are three methods for finding gates, listed below:
  - `find_gates_nonlocal(local_gates)`: Finds non-trivial transversal gates, where "non-trivial" means that it is not a sum of translates of locally-supported transversal gates (stabilizers). `local_gates` bounds the support of what counts as locally-supported, by default a list of hypercube side lengths (e.g. `[3, 3]`). Locally supported gates must have trivial logical action on any finite compactification, but a non-locally-supported gate can still have trivial action on some (or all) compactification. As there is no logical basis involved, the resulting `GateFinderResults` object does not support ` print_nontrivial_logicals` or `print_physical_from_logical`.
  - `find_gates_compactify(lattice)`: Finds non-trivial transversal gates, where "non-trivial" means non-trivial logical action after a given finite-torus `lattice` compactifcation.
  - `compactify(lattice)`: This turns the infinite translation-invariant code into a finite code after a finite-torus `lattice` compactification. You can then call `find_gates()` on the finite code. This will search for all gates, including ones that are not translation invariant. That is, the `gates` of the finite code consist of one copy of the `gates` of the infinite code per unit cell, but the different unit cells need not have the same coefficient.


### Low-level methods
The `PhaseLocs` class stores a subgroup of phase functions on configurations of $n$ variables (qubits/checks/...). The `TwoGroupHom` class stores a homomorphism between 2-groups, possibly with an annotation `phase_locs0`/`phase_locs1` that identifies the source/target 2-group with a subgroup of phase functions on $n$ variables. The central method is `Z2Hom.phase_pullback`, which calculates the pullback of a $\mathbb Z_2$ homomorphism, mapping (a given `PhaseLocs` subgroup of) phase functions on the target variables to phase function on the source variables. The result is thus a `TwoGroupHom` object with corresponding `phase_locs0/1` annotations.

The subgroup of all transversal gates is the `kernel` of the pullback `TwoGroupHom`. Quotienting by trivial logical action is done via an `epi_mono` decomposition, and physical representatives are found using a `solve` method.

Methods for the translation-invariant case are analogous

## Installation

**Option 1**: pip install

```bash
pip install git+https://github.com/andibau13/transversal-gate-finder.git
```

**Option 2**: clone, install into a venv in editable mode, to play with the examples notebook or look into source

```bash
git clone https://github.com/andibau13/transversal-gate-finder.git
cd transversal-gate-finder
python3 -m venv .venv
.venv/bin/pip install -e .
```

To open the examples notebook:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name my-venv --display-name "gate-finder-test-venv"
jupyter notebook
```
In jupyter, open `examples/examples.ipynb` and select kernel `gate-finder-test-venv`.

## Quick start

Find the transversal T gate of the [[15,1,3]] quantum Reed-Muller code (tetrahedral 3D color code):

```python
import transversal_gate_finder as tg

# define the code by its X checks and X logicals
reed_muller = tg.GateFinder(15)
reed_muller.checks.add_columns([[0,14,4,13,6,12,9,10], [1,14,4,13,5,11,8,10],
                                [2,14,5,11,7,12,6,13], [3,14,9,10,8,11,7,12]])
reed_muller.logicals.add_columns([[0,1,2,4,5,6,13]])

# ansatz: an arbitrary power of T on every qubit
reed_muller.gates.add_all_single_locs(2)

# compute all ansatz configurations preserving the code space
res = reed_muller.find_gates()
res.print_nontrivial_logicals()   # -> 7/8*[0]   (a logical T^7 = T^dagger gate)

# print a physical implementation of the logical T gate
res.print_physical_from_logical([({0}, 2, 1)])
```

For a translation-invariant example, here is the CZ gate between two stacked 2D toric codes, found on a finite 4×4 torus:

```python
tc_2d = tg.TIGateFinder(2, 2)  # 2 qubits per unit cell, 2 dimensions
tc_2d.checks.add_columns([[((0,0),0), ((0,0),1), ((1,0),1), ((0,1),0)]])
tc_2d.other_checks.add_columns([[((0,0),0), ((0,0),1), ((-1,0),0), ((0,-1),1)]])
tc_2d_x2 = tc_2d + tc_2d  # stack two copies
tc_2d_x2.gates.add_all_single_locs(0)  # single-qubit Z ansatz gates
tc_2d_x2.gates.add_locs([[((0,0),0), ((1,0),3)], [((0,0),1), ((0,1),2)]], 0)  # CZ ansatz gates

# pathway 2: logical action on a 4x4 twisted compactification
res = tc_2d_x2.find_gates_compactify([[4,0],[1,4]])
res.print_nontrivial_logicals()   # logical CZs between the two copies, as well as Pauli-Z logicals
res.print_nontrivial_physicals()
```

## Examples

The notebook [`examples/examples.ipynb`](examples/examples.ipynb) contains many more examples, including:

- S in the [[7,1,3]] Steane code and CZ in stacked 2D surface codes,
- T in the [[15,1,3]] Reed-Muller code and √T in the [[31,1,3]] 4D simplicial color code,
- physical S / logical CZ in the 2D color code on a torus and CZ / CCZ in stacked 2D / 3D toric codes, worked through all three translation-invariant pathways,
- physical T / logical CCZ in the 3D color code and a third-order (non-Clifford) gate in the "dual" 3D color code,
- CZ in stacked bivariate bicycle codes, a fold-transversal gate in the [[98,6,10]] BB code, and Cliffords in Haah's cubic code.

## Citation

If you use this code in your research, feel free to cite the accompanying paper:

> Andreas Bauer, *Efficiently finding (spacetime) diagonal transversal gates in CSS codes*.

## License

MIT, see [LICENSE](LICENSE).
</content>
</invoke>
