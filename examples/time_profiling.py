from time import perf_counter
import third_order_gates.z248_linalg as lin
#import scipy.linalg
import numpy as np
import third_order_gates.third_order_gates as tg


import sys
from pathlib import Path
project_root = Path.cwd().parents[0]
utilsdir = project_root / "utils"
if str(utilsdir) not in sys.path:
    sys.path.insert(0, str(utilsdir))

import utils

def twobga_code(dimension, poly1, poly2):
    code = tg.ti_transversal_gate_finder(2, dimension)
    code.add_checks([[(coord,0) for coord in poly1] + [(coord,1) for coord in poly2]])
    code.add_other_checks([[((-np.array(coord)).tolist(),1) for coord in poly1] + [((-np.array(coord)).tolist(),0) for coord in poly2]])
    return code

haah_cubic = twobga_code(3, [[0,0,0],[0,0,1],[0,1,0],[1,0,0]], [[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
haah_cubic.add_all_t()
haah_cubic.add_all_cs_onsite()
haah_cubic.add_all_cs_coord([1,0,0])
haah_cubic.add_all_cs_coord([0,1,0])
haah_cubic.add_all_cs_coord([0,0,1])
haah_cubic.add_all_ccz_onecoord([1,0,0])
haah_cubic.add_all_ccz_onecoord([0,1,0])
haah_cubic.add_all_ccz_onecoord([0,0,1])
print(haah_cubic.t_gates)
print(haah_cubic.cs_gates)
print(haah_cubic.ccz_gates)
haah_cubic_finite = haah_cubic.as_finite_code(np.array([[8,0,0],[0,8,0],[3,3,-1]]))
#haah_cubic_finite = haah_cubic.as_finite_code(np.array([[2,0,0],[0,2,0],[3,3,-1]]))
haah_cubic_finite.logicals_from_other_checks()
print(len(haah_cubic_finite.logicals))

t0 = perf_counter()
haah_cubic_finite.find_gates()
t1 = perf_counter()

print(f"obj.A(x): {t1 - t0:.3f} s")

