# transversal-gate-finder

This package provides functions and classes for finding transversal gates in a given CSS code.
The gates are diagonal gates at any level of the Clifford hierarchy (Z, S, T, CZ, CS, CCZ, ...) and are composed of a given set of local ansatz gates.

The method is based on the observation that a diagonal gate configuration preserves the code space if and only if the pullback of its phase function under the X-check matrix vanishes. Finding all transversal gates thus reduces to computing the kernel of a homomorphism between abelian 2-groups, which can be done efficiently (in polynomial time). This is the code accompanying the paper *Efficiently finding (spacetime) diagonal transversal gates in CSS codes*.

## Functionality

- **`GateFinder`** — transversal gates in finite CSS codes:
  - Define a code via its X checks (`add_checks`), X logicals (`add_logicals`), and optionally Z checks (`add_other_checks`, from which logicals can be generated automatically via `logicals_from_other_checks`).
  - Specify the ansatz gate locations: `add_all_singlequbit_gates(l)` for single-qubit gates at level `l` (e.g. `l=1` for S, `l=2` for T), `add_gates(gates, l)` for arbitrary multi-qubit diagonal gates such as CZ or CCZ, or `add_gates_in_groups`/`add_gates_in_checks` for all `k`-qubit gates supported inside given groups of qubits.
  - `find_gates()` computes all combinations of ansatz gates that preserve the code space, as the kernel of the pullback homomorphism.
  - Inspect the result: `print_transversal_logicals()` (logical action of the found gates), `print_physical_stabilizers()` (gates acting trivially), and `find_phys_rep`/`print_phys_rep` (a physical gate configuration implementing a given diagonal logical gate).
  - Stack codes with `+` to search for entangling gates like CZ or CCZ between multiple code blocks.
- **`TIGateFinder`** — the same for translation-invariant (crystalline) codes on an infinite ℤ^d lattice with a finite unit cell, using Laurent-polynomial methods:
  - Qubits are addressed as `((x, y, ...), i)` with a lattice coordinate and an internal index; checks and gates are given per unit cell.
  - `find_gates()` finds all translation-invariant transversal gate configurations.
  - `as_finite_code(lattice)` compactifies the infinite code onto a torus given by an integer lattice basis, returning a `GateFinder`, so translation-invariant solutions can be transferred to concrete finite codes with (twisted) periodic boundary conditions.
  - `twobga_code(dimension, poly1, poly2)` constructs two-block group-algebra codes (e.g. bivariate bicycle codes) from a pair of polynomials.
- **Pullback homomorphisms** — the lower-level entry points `pullback_homomorphism` (finite), `ti_pullback_homomorphism` (translation-invariant), and `ti_local_pullback` (local gates on a translation-invariant code) return the pullback map itself, whose kernel is the group of transversal gates.

Throughout, level `l` refers to diagonal gates of order `2^(l+1)`: CCZ/CZ/Z are level 0, CS/S level 1, T level 2, and so on. Gate configurations are printed as sums like `7/8*[0], 1/8*[2, 5]`, meaning a phase `exp(2πi·7/8)` conditioned on qubit 0 (a T-like gate) times a phase `exp(2πi/8)` conditioned on qubits 2 and 5.

## Installation

Clone the repository and install it in editable mode into a virtual environment:

```bash
git clone https://github.com/andibau13/transversal-gate-finder.git
cd transversal-gate-finder
python3 -m venv .venv
.venv/bin/pip install -e .
```

The dependencies `numpy`, `python-flint`, and [`twogroup-linalg`](https://github.com/andibau13/twogroup-linalg) (linear algebra over abelian 2-groups, installed from GitHub) are pulled in automatically. Python ≥ 3.9 is required.

## Quick start

Find the transversal T gate of the [[15,1,3]] quantum Reed-Muller code (tetrahedral 3D color code):

```python
import transversal_gate_finder as tg

# define the code by its X checks and X logicals
reed_muller = tg.GateFinder(15)
reed_muller.add_checks([[0,14,4,13,6,12,9,10], [1,14,4,13,5,11,8,10],
                        [2,14,5,11,7,12,6,13], [3,14,9,10,8,11,7,12]])
reed_muller.add_logicals([[0,1,2,4,5,6,13]])

# ansatz: an arbitrary power of T on every qubit
reed_muller.add_all_singlequbit_gates(2)

# compute all ansatz configurations preserving the code space,
# and print their action on the logical qubit
reed_muller.find_gates()
reed_muller.print_transversal_logicals()
# -> 7/8*[0]   (a logical T^7 = T^dagger gate)

# print a physical implementation of the logical T gate
reed_muller.print_phys_rep([7])
```

For a translation-invariant example, here is the CZ gate between two stacked 2D toric codes, transferred to a finite 3×3 torus:

```python
tc_2d = tg.TIGateFinder(2, 2)  # 2 qubits per unit cell, 2 dimensions
tc_2d.add_checks([[((0,0),0), ((0,0),1), ((1,0),1), ((0,1),0)]])
tc_2d.add_other_checks([[((0,0),0), ((0,0),1), ((-1,0),0), ((0,-1),1)]])
tc_2d_x2 = tc_2d + tc_2d  # stack two copies
tc_2d_x2.add_all_single_qubit_gates(0)  # Z ansatz gates
tc_2d_x2.add_gates([[((0,0),0), ((1,0),3)], [((0,0),1), ((0,1),2)]], 0)  # CZ ansatz gates
tc_2d_x2.find_gates()

tc_finite = tc_2d_x2.as_finite_code([[3,0],[0,3]])  # compactify onto a 3x3 torus
tc_finite.find_logical_action()
tc_finite.print_transversal_logicals()
# -> 1/2*[0, 3], 1/2*[1, 2]   (logical CZs between the two copies)
```

## Examples

The notebook [`examples/transversal_gates_examples.ipynb`](examples/transversal_gates_examples.ipynb) contains many worked examples, including:

- S in the [[7,1,3]] Steane code and CZ in stacked 2D surface codes,
- T in the [[15,1,3]] Reed-Muller code and physical T / logical CCZ in the 3D color code,
- CZ / CCZ in stacked 2D / 3D toric codes on tori,
- CZ in stacked bivariate bicycle codes and a fold-transversal gate in the [[98,8,10]] BB code,
- CS / CCZ-type gates in two copies of Haah's cubic code,
- a third-order (non-Clifford) gate in the "dual" 3D color code.

## Tests

`tests/test_pullback.py` validates the pullback computation against brute-force enumeration on small codes and against known results:

```bash
.venv/bin/python tests/test_pullback.py
```

## Citation

If you use this code in your research, please cite the accompanying paper:

> Andreas Bauer, *Efficiently finding (spacetime) diagonal transversal gates in CSS codes*.

## License

MIT, see [LICENSE](LICENSE).
