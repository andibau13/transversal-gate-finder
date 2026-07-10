"""Brute-force tests for pullback_homomorphism and ti_pullback_homomorphism.

Runs with pytest, or directly as a script: python tests/test_pullback.py
"""
from itertools import product

import numpy as np
import twogroup_linalg as lin

import transversal_gate_finder as tg
from transversal_gate_finder.core import pullback_homomorphism, matrix_from_list
from transversal_gate_finder.translation_invariance import ti_pullback_homomorphism, ti_local_pullback


def phase_function(gates, coeffs, a):
    """Evaluate S_c(a) mod 1 for gate locations given by qubit lists."""
    tot = 0.0
    for l, lgates in enumerate(gates):
        for i, loc in enumerate(lgates):
            prod_a = 1
            for q in loc:
                prod_a *= a[q]
            tot += int(coeffs[l][i]) * prod_a / 2 ** (l + 1)
    return tot % 1.0


def rand_code_and_gates(rng):
    """Random small CSS code with ansatz gates at all three levels."""
    n = int(rng.integers(4, 7))
    m = int(rng.integers(2, 4))
    checks = [sorted(rng.choice(n, size=rng.integers(2, n + 1), replace=False).tolist()) for _ in range(m)]
    gates = [[set(rng.choice(n, size=rng.integers(1, 4), replace=False).tolist())
              for _ in range(rng.integers(1, 4))] for l in range(3)]
    return n, m, checks, gates


def test_pullback_matches_direct_evaluation():
    """pullback @ c evaluated on the checks must equal S_c(A alpha) for all alpha."""
    rng = np.random.default_rng(0)
    for trial in range(10):
        n, m, checks, gates = rand_code_and_gates(rng)
        pullback, check_gate_locs = pullback_homomorphism(n, checks, gates)

        coeffs = [rng.integers(0, 2 ** (l + 1), size=len(gates[l])) for l in range(3)]
        c_elem = lin.Elem(np.concatenate(coeffs), pullback.dim1)
        image = pullback @ c_elem

        A = matrix_from_list(n, checks)
        for alpha in product([0, 1], repeat=m):
            lhs = phase_function(gates, coeffs, (A @ np.array(alpha)) % 2)
            rhs = phase_function(check_gate_locs, [image[li] for li in range(len(image.dim))], alpha)
            assert np.isclose((lhs - rhs) % 1.0, 0.0) or np.isclose((lhs - rhs) % 1.0, 1.0)


def test_kernel_matches_brute_force():
    """img(pullback.kernel()) must be exactly the set of code-space preserving gates.

    Uses a small mixed-level ansatz (one CCZ-type and two T-type gates), which exercises
    the sub-diagonal Hom blocks (low-order gates contributing to high-level check rows).
    """
    rng = np.random.default_rng(1)
    for trial in range(8):
        n = int(rng.integers(3, 6))
        m = int(rng.integers(2, 4))
        checks = [sorted(rng.choice(n, size=rng.integers(2, n + 1), replace=False).tolist()) for _ in range(m)]
        gates = [[set(rng.choice(n, size=min(3, n), replace=False).tolist())], [],
                 [{int(rng.integers(0, n))}, {int(rng.integers(0, n))}]]

        pullback, _ = pullback_homomorphism(n, checks, gates)
        K = pullback.kernel()
        A = matrix_from_list(n, checks)

        def is_transversal(coeffs):
            return all(np.isclose(phase_function(gates, coeffs, (A @ np.array(alpha)) % 2), 0.0)
                       or np.isclose(phase_function(gates, coeffs, (A @ np.array(alpha)) % 2), 1.0)
                       for alpha in product([0, 1], repeat=m))

        # brute-force set of transversal coefficient vectors
        brute_force = set()
        for cvals in product(range(2), range(8), range(8)):
            if is_transversal([[cvals[0]], [], [cvals[1], cvals[2]]]):
                brute_force.add(cvals)

        # image of the kernel isomorphism
        img = set()
        for svals in product(*[range(2 ** (l + 1)) for l in range(len(K.dim1)) for _ in range(K.dim1[l])]):
            k = K @ lin.Elem(np.array(svals, dtype=np.uint8), K.dim1)
            img.add(tuple(int(x) for x in k.v))

        assert img == brute_force


def rand_ti_loc(rng, dim, nq, size):
    """Random TI location: set of (coord, intern) pairs with coords in [-1,1]^dim."""
    loc = set()
    while len(loc) < size:
        loc.add((tuple(int(x) for x in rng.integers(-1, 2, size=dim)), int(rng.integers(0, nq))))
    return loc


def rand_ti_code_and_gates(rng):
    dim = int(rng.integers(1, 3))
    nq = int(rng.integers(1, 3))
    checks = [sorted(rand_ti_loc(rng, dim, nq, int(rng.integers(2, 4)))) for _ in range(rng.integers(1, 3))]
    gates = [[rand_ti_loc(rng, dim, nq, int(rng.integers(1, 4)))] for l in range(3)]
    return dim, nq, checks, gates


def sum_translates(locs, vals, config):
    """Evaluate a TI phase function (locations with values) on a finitely-supported configuration."""
    tot = 0.0
    for loc, val in zip(locs, vals):
        if val == 0:
            continue
        loc = list(loc)
        (u0, j0) = loc[0]
        # a contributing translate must map the anchor (u0, j0) onto a set point of config
        for (s, j) in config:
            if j != j0:
                continue
            t = tuple(sc - uc for sc, uc in zip(s, u0))
            if all(config.get((tuple(uc + tc for uc, tc in zip(ucoord, t)), jj), 0) for ucoord, jj in loc):
                tot += val
    return tot


def test_ti_pullback_matches_direct_evaluation():
    """TI pullback image evaluated on the checks must equal S_c(A alpha) on the infinite lattice."""
    rng = np.random.default_rng(7)
    for trial in range(8):
        dim, nq, checks, gates = rand_ti_code_and_gates(rng)
        pullback, check_gate_locs = ti_pullback_homomorphism(nq, checks, gates)

        coeffs = [rng.integers(0, 2 ** (l + 1), size=len(gates[l])) for l in range(3)]
        image = pullback @ lin.Elem(np.concatenate(coeffs), pullback.dim1)

        for rep in range(5):
            # random check configuration alpha supported on a box of cells
            alpha = {(cell, j): 1 for cell in product(range(3), repeat=dim)
                     for j in range(len(checks)) if rng.random() < 0.4}
            # qubit configuration a = A alpha
            a = {}
            for (cell, j) in alpha:
                for coord, i in checks[j]:
                    q = (tuple(c + u for c, u in zip(coord, cell)), i)
                    a[q] = a.get(q, 0) ^ 1
            a = {q: v for q, v in a.items() if v}

            lhs = sum(sum_translates(gates[l], [int(c) / 2 ** (l + 1) for c in coeffs[l]], a) for l in range(3))
            rhs = sum(sum_translates(check_gate_locs[li], [int(v) / 2 ** (li + 1) for v in image[li]], alpha)
                      for li in range(len(image.dim)))
            assert np.isclose((lhs - rhs) % 1.0, 0.0) or np.isclose((lhs - rhs) % 1.0, 1.0)


def test_ti_local_pullback_matches_direct_evaluation():
    """Local pullback image evaluated on the checks must equal S_c(A alpha), where the
    gates sit only at the prescribed active locations (no translates)."""
    rng = np.random.default_rng(3)
    for trial in range(8):
        dim, nq, checks, gates = rand_ti_code_and_gates(rng)
        # random active locations for each gate type
        active_gates = [[(tuple(int(x) for x in rng.integers(-2, 3, size=dim)), int(rng.integers(0, len(gates[l]))))
                         for _ in range(int(rng.integers(1, 4)))] for l in range(3)]

        pullback, check_gate_locs = ti_local_pullback(nq, checks, gates, active_gates)

        coeffs = [rng.integers(0, 2 ** (l + 1), size=len(active_gates[l])) for l in range(3)]
        image = pullback @ lin.Elem(np.concatenate(coeffs), pullback.dim1)

        for rep in range(5):
            alpha = {(cell, j): 1 for cell in product(range(-2, 3), repeat=dim)
                     for j in range(len(checks)) if rng.random() < 0.4}
            a = {}
            for (cell, j) in alpha:
                for coord, i in checks[j]:
                    q = (tuple(c + u for c, u in zip(coord, cell)), i)
                    a[q] = a.get(q, 0) ^ 1

            lhs = 0.0
            for l, lactive in enumerate(active_gates):
                for gi, (shift, gate_nr) in enumerate(lactive):
                    if all(a.get((tuple(sc + qc for sc, qc in zip(shift, qcoord)), i), 0)
                           for qcoord, i in gates[l][gate_nr]):
                        lhs += int(coeffs[l][gi]) / 2 ** (l + 1)
            rhs = 0.0
            for li in range(len(image.dim)):
                for ri, loc in enumerate(check_gate_locs[li]):
                    if all(alpha.get((ucoord, j), 0) for ucoord, j in loc):
                        rhs += int(image[li][ri]) / 2 ** (li + 1)
            assert np.isclose((lhs - rhs) % 1.0, 0.0) or np.isclose((lhs - rhs) % 1.0, 1.0)


def test_ti_kernel_unfolds_to_finite_kernel():
    """Unfolding a TI transversal gate onto a compactified code must give a transversal gate."""
    rng = np.random.default_rng(11)
    for trial in range(5):
        dim, nq, checks, gates = rand_ti_code_and_gates(rng)
        code = tg.TIGateFinder(nq, dim, checks=[list(c) for c in checks],
                               gates=[[set(g) for g in lg] for lg in gates])
        pullback, _ = ti_pullback_homomorphism(nq, checks, gates)
        K = pullback.kernel()
        assert (pullback @ K).is_zero()

        L = 4
        fin = code.as_finite_code(np.diag([L] * dim))
        fin_pullback, _ = pullback_homomorphism(fin.nr_qubits, fin.checks, fin.gates)

        for li in range(len(K.dim1)):
            for g in range(K.dim1[li]):
                gen = lin.Elem.zeros(K.dim1)
                gen[li][g] = 1
                c_ti = K @ gen
                # same coefficient at all translates; finite gates are ordered cell-major
                v_fin = np.concatenate([np.tile(np.asarray(c_ti[l]), L ** dim) for l in range(3)])
                assert (fin_pullback @ lin.Elem(v_fin, fin_pullback.dim1)).is_zero()


def test_ti_ccz_3d_toric():
    """The TI kernel of 3 copies of the 3D toric code with the 6 CCZ ansatz locations
    is exactly one Z2 generator: the known transversal CCZ (coefficient 1 everywhere)."""
    tc_3d = tg.TIGateFinder(3, 3)
    tc_3d.add_checks([[((0,0,0), 0), ((0,0,0), 1), ((0,0,0), 2),
                       ((-1,0,0), 0), ((0,-1,0), 1), ((0,0,-1), 2)]])
    tc_3d_x3 = tc_3d + tc_3d + tc_3d
    tc_3d_x3.add_gates([[((0,0,0),0), ((1,0,0),4), ((1,1,0),8)],
                        [((0,0,0),0), ((1,0,0),5), ((1,0,1),7)],
                        [((0,0,0),1), ((0,1,0),3), ((1,1,0),8)],
                        [((0,0,0),1), ((0,1,0),5), ((0,1,1),6)],
                        [((0,0,0),2), ((0,0,1),3), ((1,0,1),7)],
                        [((0,0,0),2), ((0,0,1),4), ((0,1,1),6)]], 0)

    pullback, _ = ti_pullback_homomorphism(tc_3d_x3.nr_qubits, tc_3d_x3.checks, tc_3d_x3.gates)
    K = pullback.kernel()
    assert K.dim1 == [1, 0]
    assert np.all(K.M == 1)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(name, "...", flush=True)
            fn()
            print("  OK")
