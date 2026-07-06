from fpylll import IntegerMatrix, LLL
import numpy as np
import itertools

# computes the L1 systole of an integer lattice (defining PBC of code)
def l1_systole(B, radius):
    A = IntegerMatrix.from_matrix(B.tolist())
    LLL.reduction(A)

    B_red = np.array([[A[i, j] for j in range(A.ncols)] for i in range(A.nrows)], dtype=int)

    best_norm = np.inf
    best_vec = None

    # Search all coefficient vectors c in [-radius, radius]^n.
    for coeffs_tuple in itertools.product(range(-radius, radius + 1), repeat=B.shape[0]):
        coeffs = np.array(coeffs_tuple, dtype=int)

        if np.all(coeffs == 0):
            continue

        v = coeffs @ B_red

        norm = int(np.sum(np.abs(v)))

        if norm < best_norm:
            best_norm = norm
            best_vec = v.copy()

    return best_norm, best_vec

