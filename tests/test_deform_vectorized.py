#!/usr/bin/env python3
"""deform_nucleus must give the nuclei deform_nucleus_scalar gives, bit for bit.

The scalar routine is the original implementation (one nucleon at a time);
the vectorised one replaced it for speed only.  This draws nuclei the way the
sampler does, deforms each with both, and compares the float64 positions for
exact equality -- before the float32 rounding of the output file, so nothing
can hide.  It also checks that libm's pow agrees with what the `**` operator
gives on scalars, which the vectorised code relies on.

    python3 tests/test_deform_vectorized.py [number of nuclei per shape]
"""
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "exec"))
import build_isobars as bi  # noqa: E402


def potentials(R, a):
    rmin, rmax = R / 10, 3 * R
    out = []
    for l in (2, 3, 4):
        res = solve_ivp(fun=lambda t, y: bi.diff_eq(t, y, R, a, l),
                        y0=[rmax, 1, 1, 0], t_span=[rmax, rmin],
                        rtol=1e-10, atol=1e-10)
        k = res.y[1, -1] / res.y[3, -1]
        out.append(interp1d(res.t, res.y[0] - k * res.y[2]))
        out.append(interp1d(res.t, res.y[1] - k * res.y[3]))
    return out


# (A, R_ws, a_ws, R_step, w_gauss, beta20, beta22, beta3, beta4)
SHAPES = [
    (96, 5.327780175, 0.5880870723, 5.4816, 1.0678, 0.19, -0.12, 0.38, 0.0),
    (96, 5.02, 0.52, 5.15, 0.95, 0.06, 0.0, 0.20, 0.0),
    (96, 5.09, 0.46, 5.2, 0.85, -0.15, 0.10, 0.0, 0.05),
    (208, 6.62, 0.546, 6.75, 0.99, 0.0, 0.0, 0.11, 0.0),
    (208, 6.62, 0.546, 6.75, 0.99, 0.25, 0.08, -0.3, -0.1),
    (238, 6.81, 0.55, 6.95, 1.0, 0.28, 0.0, 0.0, 0.09),
    (20, 2.8, 0.57, 2.9, 1.0, 0.45, -0.2, 0.3, 0.2),   # light and extreme
]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(20261117)
    x = rng.uniform(-1, 1, 200000).tolist() + rng.uniform(0, 30, 200000).tolist()
    for power in (2.0, 3.0, 4.0):
        import math
        assert all(math.pow(v, power) == v**int(power) for v in x)
        assert all(math.pow(v, power) == float(np.float64(v)**int(power))
                   for v in x[:50000])
    print("libm pow == the ** operator on Python and numpy scalars")

    bad = 0
    t_scalar = t_vector = 0.0
    for shape in SHAPES:
        A, R, a, R_step, w = shape[:5]
        betas = shape[5:]
        f = potentials(R, a)
        addresses = rng.integers(0, 2**32, n)
        seeds = bi.generate_seed_rows(8675309, addresses, A)
        same = 0
        for row in seeds:
            nucleus = np.empty((A, 3))
            for i in range(A):
                nucleus[i, :] = bi.place_nucleon(R_step, w, row[i])
            t0 = time.perf_counter()
            ref = bi.deform_nucleus_scalar(nucleus.copy(), R, *betas, *f)
            t1 = time.perf_counter()
            new = bi.deform_nucleus(nucleus.copy(), R, *betas, *f)
            t2 = time.perf_counter()
            t_scalar += t1 - t0
            t_vector += t2 - t1
            same += np.array_equal(ref, new)
        bad += n - same
        print("A={:3d} R={:.2f} betas={}: {} of {} nuclei bit-identical".format(
            A, R, betas, same, n))
    print("scalar {:.1f} ms, vectorised {:.2f} ms per nucleus ({:.0f}x)".format(
        1e3 * t_scalar / (n * len(SHAPES)), 1e3 * t_vector / (n * len(SHAPES)),
        t_scalar / t_vector))
    if bad:
        sys.exit("FAIL: {} nuclei differ".format(bad))
    print("PASS")


if __name__ == "__main__":
    main()
