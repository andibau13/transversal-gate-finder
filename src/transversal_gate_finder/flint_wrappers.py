# numpy wrappers for flint integer-matrix functions
from flint import fmpz_mat
import numpy as np

def flint_to_numpy(A):
    return np.array([[A[i, j] for j in range(A.ncols())] for i in range(A.nrows())], dtype=int).reshape(A.nrows(),A.ncols())
def flint_to_numpy_mod8(A):
    return np.array([[A[i, j] % 8 for j in range(A.ncols())] for i in range(A.nrows())], dtype=int).reshape(A.nrows(),A.ncols())
def numpy_to_flint(A):
    return fmpz_mat(A.tolist())

# hermite normal form
def hnf(A):
    A_flint = numpy_to_flint(A)
    hnf_flint, transform_flint = A_flint.hnf(transform=True)
    A_hnf = flint_to_numpy(hnf_flint)
    transform = flint_to_numpy(transform_flint)
    return A_hnf, transform

# integer matrix inverse
def int_inv(A): 
    A_flint = numpy_to_flint(A)
    Ainv_flint = A_flint.inv(integer=True)
    Ainv = flint_to_numpy(Ainv_flint)
    if A.shape[0]>0: # flint sometimes gives -inv instead of inv. We fix it below
        Ainv = Ainv * np.dot(A[0,:], Ainv[:,0])
    return Ainv

# compute hnf and then take result mod 8 (important for numpy integers to not overflow - flint uses infinite integers that get very large very quickly)
def hnf_mod8(A):
    A_flint = numpy_to_flint(A)
    hnf_flint, transform_flint = A_flint.hnf(transform=True)
    A_hnf = flint_to_numpy_mod8(hnf_flint)
    transform = flint_to_numpy_mod8(transform_flint)
    return A_hnf, transform

# compute kernel of integer matrix (using hnf)
def int_kernel(A):
    hnfT, transformT = hnf(A.T)
    A_hnf, transform = hnfT.T, transformT.T
    zero_cols = ~A_hnf.any(axis=0)
    nr_nonzero_cols = zero_cols.argmax() if zero_cols.any() else len(zero_cols)
    return transform[:,nr_nonzero_cols:]

# compute integer kernel and give result mod 8 (does this work?)
def int_kernel_mod8(A):
    hnfT, transformT = hnf_mod8(A.T)
    A_hnf, transform = hnfT.T, transformT.T
    zero_cols = ~hnf.any(axis=0)
    nr_nonzero_cols = zero_cols.argmax() if zero_cols.any() else len(zero_cols)
    return transform[:,nr_nonzero_cols:]