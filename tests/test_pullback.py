"""Brute-force tests for the phase_pullback methods and the TI hom classes.

Runs with pytest, or directly as a script: python tests/test_pullback.py
"""
from itertools import product

import numpy as np
import twogroup_linalg as lin

import transversal_gate_finder as tg


def check_matrix(n, checks):
    """Dense 0/1 check matrix (columns = checks, rows = qubits)."""
    return tg.Z2Hom(n, checks).to_array()


def finite_pullback(n, checks, gates):
    """Standard pullback for a finite code given as raw lists; returns (Hom, target locs)."""
    pb = tg.Z2Hom(n, checks).phase_pullback(tg.PhaseLocs(n, gates))
    return pb.h, pb.phase_locs0.locs


def ti_pullback(nq, dim, checks, gates):
    """Translation-invariant (summed) pullback from raw lists; returns (Hom, target locs)."""
    pb = tg.TIZ2Hom(nq, dim, checks).phase_pullback(tg.TIPhaseLocs(nq, dim, gates)).ti_sum()
    return pb.h, pb.phase_locs0.locs


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
        # PhaseLocs consolidates duplicate/cross-level supports, so use its canonical locs
        gates = [[set(loc) for loc in llocs] for llocs in tg.PhaseLocs(n, gates).locs]
        pullback, check_gate_locs = finite_pullback(n, checks, gates)

        coeffs = [rng.integers(0, 2 ** (l + 1), size=len(gates[l])) for l in range(3)]
        c_elem = lin.Elem(np.concatenate(coeffs), pullback.dim1)
        image = pullback @ c_elem

        A = check_matrix(n, checks)
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
        # two distinct level-2 singletons so PhaseLocs consolidation leaves the gate count unchanged
        q2 = rng.choice(n, size=2, replace=False)
        gates = [[set(rng.choice(n, size=min(3, n), replace=False).tolist())], [],
                 [{int(q2[0])}, {int(q2[1])}]]

        pullback, _ = finite_pullback(n, checks, gates)
        K = pullback.kernel()
        A = check_matrix(n, checks)

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
        pullback, check_gate_locs = ti_pullback(nq, dim, checks, gates)
        # normalized, consolidated gate locations as used by the pullback
        ngates = tg.TIPhaseLocs(nq, dim, gates).locs

        coeffs = [rng.integers(0, 2 ** (l + 1), size=len(ngates[l])) for l in range(3)]
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

            lhs = sum(sum_translates(ngates[l], [int(c) / 2 ** (l + 1) for c in coeffs[l]], a) for l in range(3))
            rhs = sum(sum_translates(check_gate_locs[li], [int(v) / 2 ** (li + 1) for v in image[li]], alpha)
                      for li in range(len(image.dim)))
            assert np.isclose((lhs - rhs) % 1.0, 0.0) or np.isclose((lhs - rhs) % 1.0, 1.0)


def test_local_pullback_matches_direct_evaluation():
    """The composition phase_pullback @ local_gates evaluated on the checks must equal
    S_c(A alpha), where the gates sit only at the prescribed active locations (no translates)."""
    rng = np.random.default_rng(3)
    for trial in range(8):
        dim, nq, checks, gates = rand_ti_code_and_gates(rng)
        code = tg.TIGateFinder(nq, dim, checks=[list(c) for c in checks],
                               gates=[[set(g) for g in lg] for lg in gates])
        # random active locations for each gate type (indexing the consolidated code.gates)
        gdims = code.gates.dims
        active_gates = [[(tuple(int(x) for x in rng.integers(-2, 3, size=dim)), int(rng.integers(0, gdims[l])))
                         for _ in range(int(rng.integers(1, 4)))] if gdims[l] else [] for l in range(3)]
        code.set_local_gates(active_gates)

        local_pb = code.checks.phase_pullback(code.gates) @ code.local_gates
        # absolute check gate locations of the target generators
        abs_locs = [[{(tuple(c + s for c, s in zip(coord, shift)), j)
                      for coord, j in local_pb.phase_locs0.locs[li][g]}
                     for shift, g in local_pb.ti_support[li]]
                    for li in range(len(local_pb.h.dim0))]

        coeffs = [rng.integers(0, 2 ** (l + 1), size=len(active_gates[l])) for l in range(3)]
        image = local_pb.h @ lin.Elem(np.concatenate(coeffs), local_pb.h.dim1)

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
                           for qcoord, i in code.gates.locs[l][gate_nr]):
                        lhs += int(coeffs[l][gi]) / 2 ** (l + 1)
            rhs = 0.0
            for li in range(len(image.dim)):
                for ri, loc in enumerate(abs_locs[li]):
                    if all(alpha.get((ucoord, j), 0) for ucoord, j in loc):
                        rhs += int(image[li][ri]) / 2 ** (li + 1)
            assert np.isclose((lhs - rhs) % 1.0, 0.0) or np.isclose((lhs - rhs) % 1.0, 1.0)


def test_ti_kernel_unfolds_to_finite_kernel():
    """Transporting a TI transversal gate onto a compactified code (via TIPhaseLocs.compactify's
    induced homomorphism) must give a transversal gate."""
    from transversal_gate_finder.translation_invariance import extract_compactification_data
    rng = np.random.default_rng(11)
    for trial in range(5):
        dim, nq, checks, gates = rand_ti_code_and_gates(rng)
        code = tg.TIGateFinder(nq, dim, checks=[list(c) for c in checks],
                               gates=[[set(g) for g in lg] for lg in gates])
        code.find_gates()
        K = code.transphys_allphys
        ti_pb = code.checks.phase_pullback(code.gates).ti_sum()
        assert (ti_pb @ code.transphys_allphys).is_zero()

        L = 4
        cd = extract_compactification_data(np.diag([L] * dim))
        fin = code.as_finite_code(np.diag([L] * dim), auto_logicals=False)
        fin.gates, gates_compactify = code.gates.compactify(cd)
        transphys = gates_compactify @ code.transphys_allphys
        fin_pullback = fin.checks.phase_pullback(fin.gates)

        # the transported transversal gates lie in the finite kernel, and no transversal
        # generator was lost (source dimension is preserved)
        assert transphys.dim1 == K.dim1
        assert (fin_pullback @ transphys).is_zero()

        # every individual transported transversal generator is code-space preserving
        for li in range(len(K.dim1)):
            for g in range(K.dim1[li]):
                gen = lin.Elem.zeros(K.dim1)
                gen[li][g] = 1
                assert (fin_pullback.h @ (transphys.h @ gen)).is_zero()


def test_ti_ccz_3d_toric():
    """The TI kernel of 3 copies of the 3D toric code with the 6 CCZ ansatz locations
    is exactly one Z2 generator: the known transversal CCZ (coefficient 1 everywhere)."""
    tc_3d = tg.TIGateFinder(3, 3)
    tc_3d.checks.add_columns([[((0,0,0), 0), ((0,0,0), 1), ((0,0,0), 2),
                              ((-1,0,0), 0), ((0,-1,0), 1), ((0,0,-1), 2)]])
    tc_3d_x3 = tc_3d + tc_3d + tc_3d
    tc_3d_x3.gates.add_locs([[((0,0,0),0), ((1,0,0),4), ((1,1,0),8)],
                        [((0,0,0),0), ((1,0,0),5), ((1,0,1),7)],
                        [((0,0,0),1), ((0,1,0),3), ((1,1,0),8)],
                        [((0,0,0),1), ((0,1,0),5), ((0,1,1),6)],
                        [((0,0,0),2), ((0,0,1),3), ((1,0,1),7)],
                        [((0,0,0),2), ((0,0,1),4), ((0,1,1),6)]], 0)

    tc_3d_x3.find_gates()
    K = tc_3d_x3.transphys_allphys
    assert K.dim1[0] == 1 and all(d == 0 for d in K.dim1[1:])
    assert np.all(K.h.M == 1)


def test_steane_end_to_end():
    """S gates on all qubits of the Steane code give exactly one order-4 transversal logical (S dagger); T is not transversal."""
    steane = tg.GateFinder(7, checks=[[0,3,4,6], [1,4,5,6], [2,3,5,6]], logicals=[[0,2,3]])
    steane.gates.add_all_single_locs(1)
    steane.find_gates()
    assert steane.translog_alllog.dim1 == [0, 1]
    # logical S is transversally implemented, logical T is not
    rep = steane.find_phys_rep_free({(frozenset({0}), 1): 1})
    assert rep is not None
    assert steane.test_if_implemented({(frozenset({0}), 2): 1}) is None
    # physical S-type stabilizers: one order-2 generator per X check
    assert steane.stabphys_allphys.dim1 == [3, 0]


def test_find_gates_nonlocal_cc2d():
    """find_gates_nonlocal on the 2D color code: the quotient map must be surjective and
    kill exactly the sums of translates of local transversal gates."""
    cc_2d = tg.TIGateFinder(2, 2)
    cc_2d.checks.add_columns([[((0,0),0), ((0,0),1), ((1,0),0), ((0,1),1), ((0,1),0), ((-1,1),1)]])
    cc_2d.other_checks = cc_2d.checks
    cc_2d.gates.add_all_single_qubit_gates(1)
    cc_2d.gates.add_gates_in_groups([[((0,0),0),((0,0),1),((-1,1),1),((-1,0),1)]], 0, 2)
    cc_2d.set_local_gates([[], [((0,0),0), ((0,0),1)]])
    cc_2d.find_gates_nonlocal()

    q = cc_2d.transphys_translog
    # surjective: the dual is injective
    assert sum(q.h.transpose().kernel().dim1) == 0

    # q kills the translate-sum of every local transversal gate
    local_pb = cc_2d.checks.phase_pullback(cc_2d.gates) @ cc_2d.local_gates
    local_transversal = local_pb.kernel()
    P = cc_2d.local_gates.ti_sum() @ local_transversal
    for l in range(len(P.dim1)):
        for j in range(P.dim1[l]):
            gen = lin.Elem.zeros(P.dim1)
            gen[l][j] = 1
            x = cc_2d.transphys_allphys.h.solve_with_helper(P @ gen, cc_2d.transphys_solve_helper)
            assert (q.h @ x).is_zero()


def test_z2_column_reduction_on_small_torus():
    """When compactifying onto a torus smaller than a check/logical, coincident qubits must
    cancel in Z2 (not merge). Phase-function (gate) supports instead collapse idempotently."""
    # a Z2Hom column with a repeated row index cancels it
    assert tg.Z2Hom(3, [[0, 1, 0]]).h == [[1]]

    # a 1D check spanning coords 0 and 2, compactified onto a period-2 torus: both coincide -> empty
    code = tg.TIGateFinder(1, 1)
    code.checks.add_columns([[((0,), 0), ((2,), 0)]])
    fin = code.as_finite_code([[2]], auto_logicals=False,
                              manual_logicals=[[((0,), 0), ((2,), 0)]],
                              manual_gates=[[], [{((0,), 0), ((2,), 0)}]])
    assert fin.checks.h == [[], []]          # X checks cancel
    assert fin.logicals.h == [[]]            # manual logical cancels
    assert fin.gates.locs[1] == [frozenset({0})]  # gate support stays weight 1 (idempotent)


def test_ti_z2hom_transpose_and_composition():
    """X and Z checks of the 2D toric code commute: checks^T @ other_checks == 0."""
    tc_2d = tg.TIGateFinder(2, 2)
    tc_2d.checks.add_columns([[((0,0),0),((0,0),1),((1,0),1),((0,1),0)]])
    tc_2d.other_checks.add_columns([[((0,0),0),((0,0),1),((-1,0),0),((0,-1),1)]])
    assert (tc_2d.checks.transpose() @ tc_2d.other_checks).is_zero()
    # transpose is an involution
    assert tc_2d.checks.transpose().transpose().h == [sorted(col) for col in tc_2d.checks.h]


def test_quotient_image_by_image():
    """quotient_image_by_image(K, K @ R) must be the cokernel of R: it kills exactly im(R), is surjective, and has the complementary size."""
    rng = np.random.default_rng(5)
    for trial in range(10):
        dims_G = [int(rng.integers(0, 3)) for _ in range(3)]
        dims_A = [int(rng.integers(0, 3)) for _ in range(3)]
        A = lin.Hom.rand(dims_A, dims_G)
        K = A.kernel()  # injective T -> G
        dims_S = [int(rng.integers(0, 3)) for _ in range(3)]
        R = lin.Hom.rand(K.dim1, dims_S)
        P = K @ R  # im(P) is contained in im(K) by construction

        # transpose is functorial (contravariant), i.e. really is the duality
        assert np.array_equal(P.transpose().M, (R.transpose() @ K.transpose()).M)

        q = K.quotient_image_by_image(P)
        # ker(q) contains im(R) (f = R is the unique solution of K f = P)
        assert (q @ R).is_zero()
        # q is surjective: its dual is injective
        assert sum(q.transpose().kernel().dim1) == 0
        # log2 sizes: |T| = |im R| * |Q|, so ker(q) is not larger than im(R)
        log2size = lambda dims: sum((l + 1) * n for l, n in enumerate(dims))
        im_dims = R.epi_mono()[0].dim1
        assert log2size(K.dim1) == log2size(im_dims) + log2size(q.dim0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(name, "...", flush=True)
            fn()
            print("  OK")
