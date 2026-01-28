"""Core MPFIT math functions: A matrix, b vector, solid harmonics."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import sph_harm_y


def _convert_flat_to_hierarchical(
    flat_multipoles: NDArray[np.float64], num_sites: int, max_rank: int
) -> NDArray[np.float64]:
    """
    Convert flat multipole array to hierarchical format.

    Parameters
    ----------
    flat_multipoles : ndarray
        Array with shape (num_sites, N) where N is the number of flattened
        multipole components
    num_sites : int
        Number of multipole sites
    max_rank : int
        Maximum multipole rank (e.g., 4 for hexadecapole)

    Returns
    -------
    hierarchical_multipoles : ndarray
        Array with shape (num_sites, max_rank+1, max_rank+1, 2)
        Format: [site, rank, component, real/imag]
    """
    # Initialize output array
    mm = np.zeros((num_sites, max_rank + 1, max_rank + 1, 2))

    for i in range(num_sites):
        flat_site = flat_multipoles[i]
        idx = 0

        # Monopole (rank 0)
        mm[i, 0, 0, 0] = flat_site[idx]
        idx += 1

        # Higher ranks
        for l in range(1, max_rank + 1):
            # m=0 component (only real part)
            mm[i, l, 0, 0] = flat_site[idx]
            idx += 1

            # m>0 components (real and imaginary parts)
            for m in range(1, l + 1):
                mm[i, l, m, 0] = flat_site[idx]  # Real part
                idx += 1
                mm[i, l, m, 1] = flat_site[idx]  # Imaginary part
                idx += 1

    return mm


def _regular_solid_harmonic_depr(
    l: int, m: int, cs: int, x: float, y: float, z: float
) -> float:
    """Evaluate regular solid harmonics using scipy (scalar, deprecated)."""
    r = np.sqrt(x * x + y * y + z * z)
    if r < 1e-10:
        return 1.0 if (l == 0 and m == 0 and cs == 0) else 0.0
    if l == 4:
        # Initialize array for regular solid harmonic values
        rsharray = np.zeros((5, 5, 2))

        # l=4 (hexadecapole)
        rsharray[4, 0, 0] = 0.125 * (
            8.0 * z**4
            - 24.0 * (x**2 + y**2) * z**2
            + 3.0 * (x**4 + 2.0 * x**2 * y**2 + y**4)
        )
        rsharray[4, 1, 0] = (
            0.25 * np.sqrt(10.0) * (4.0 * x * z**3 - 3.0 * x * z * (x**2 + y**2))
        )
        rsharray[4, 1, 1] = (
            0.25 * np.sqrt(10.0) * (4.0 * y * z**3 - 3.0 * y * z * (x**2 + y**2))
        )
        rsharray[4, 2, 0] = (
            0.25 * np.sqrt(5.0) * (x**2 - y**2) * (6.0 * z**2 - x**2 - y**2)
        )
        rsharray[4, 2, 1] = 0.25 * np.sqrt(5.0) * x * y * (6.0 * z**2 - x**2 - y**2)
        rsharray[4, 3, 0] = 0.25 * np.sqrt(70.0) * z * (x**3 - 3.0 * x * y**2)
        rsharray[4, 3, 1] = 0.25 * np.sqrt(70.0) * z * (3.0 * x**2 * y - y**3)
        rsharray[4, 4, 0] = 0.125 * np.sqrt(35.0) * (x**4 - 6.0 * x**2 * y**2 + y**4)
        rsharray[4, 4, 1] = 0.125 * np.sqrt(35.0) * x * y * (x**2 - y**2)

        return rsharray[l, m, cs]

    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    Y = sph_harm_y(l, m, theta, phi)

    # 'Normalization' factor to remove from the built-in Y_l^m:
    norm = np.sqrt(4.0 * np.pi / (2.0 * l + 1.0))

    if m == 0:
        return norm * r**l * Y.real
    return np.sqrt(2.0) * (-1.0) ** m * norm * r**l * (Y.real if cs == 0 else Y.imag)


def _regular_solid_harmonic(
    l: int, m: int, cs: int,
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Vectorized regular solid harmonic.

    Evaluates R_l^m(cs) at arrays of coordinates, returning an array of values.

    Parameters
    ----------
    l, m : int
        Angular momentum quantum numbers (0 <= m <= l)
    cs : int
        0 for cosine (real) part, 1 for sine (imaginary) part
    x, y, z : ndarray
        Cartesian displacement coordinates (same shape)

    Returns
    -------
    ndarray
        RSH values, same shape as x/y/z
    """
    r = np.sqrt(x * x + y * y + z * z)
    result = np.zeros_like(r)
    nonzero = r > 1e-10

    # r ≈ 0: RSH = 1 only for (l=0, m=0, cs=0)
    if l == 0 and m == 0 and cs == 0:
        result[~nonzero] = 1.0

    if not np.any(nonzero):
        return result

    xn, yn, zn, rn = x[nonzero], y[nonzero], z[nonzero], r[nonzero]

    if l == 4:
        # Explicit Cartesian formulas for hexadecapoles (avoids trig)
        if m == 0:
            val = 0.125 * (8.0 * zn**4 - 24.0 * (xn**2 + yn**2) * zn**2
                           + 3.0 * (xn**4 + 2.0 * xn**2 * yn**2 + yn**4))
        elif m == 1 and cs == 0:
            val = 0.25 * np.sqrt(10.0) * (4.0 * xn * zn**3 - 3.0 * xn * zn * (xn**2 + yn**2))
        elif m == 1 and cs == 1:
            val = 0.25 * np.sqrt(10.0) * (4.0 * yn * zn**3 - 3.0 * yn * zn * (xn**2 + yn**2))
        elif m == 2 and cs == 0:
            val = 0.25 * np.sqrt(5.0) * (xn**2 - yn**2) * (6.0 * zn**2 - xn**2 - yn**2)
        elif m == 2 and cs == 1:
            val = 0.25 * np.sqrt(5.0) * xn * yn * (6.0 * zn**2 - xn**2 - yn**2)
        elif m == 3 and cs == 0:
            val = 0.25 * np.sqrt(70.0) * zn * (xn**3 - 3.0 * xn * yn**2)
        elif m == 3 and cs == 1:
            val = 0.25 * np.sqrt(70.0) * zn * (3.0 * xn**2 * yn - yn**3)
        elif m == 4 and cs == 0:
            val = 0.125 * np.sqrt(35.0) * (xn**4 - 6.0 * xn**2 * yn**2 + yn**4)
        elif m == 4 and cs == 1:
            val = 0.125 * np.sqrt(35.0) * xn * yn * (xn**2 - yn**2)
        else:
            val = np.zeros_like(rn)
        result[nonzero] = val
        return result

    # l < 4: use scipy spherical harmonics
    theta = np.arccos(zn / rn)
    phi = np.arctan2(yn, xn)
    Y = sph_harm_y(l, m, theta, phi)
    norm = np.sqrt(4.0 * np.pi / (2.0 * l + 1.0))

    if m == 0:
        result[nonzero] = norm * rn**l * Y.real
    else:
        factor = np.sqrt(2.0) * ((-1.0) ** m) * norm * rn**l
        result[nonzero] = factor * (Y.real if cs == 0 else Y.imag)

    return result


def build_A_matrix(
    nsite: int,
    xyzmult: NDArray[np.float64],
    xyzcharge: NDArray[np.float64],
    r1: float,
    r2: float,
    maxl: int,
    A: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Construct A matrix as in J. Comp. Chem. Vol. 12, No. 8, 913-917 (1991).

    Returns 3D array A(i,j,k) where i stands for the specific multipole,
    j,k for the charges.
    """
    ncharge = xyzcharge.shape[0]  # or len(xyzcharge)

    W = np.zeros(maxl + 1)
    # compute W integration factor
    for i in range(maxl + 1):
        W[i] = (1.0 / (1.0 - 2.0 * i)) * (r2 ** (1 - 2 * i) - r1 ** (1 - 2 * i))

    for j in range(ncharge):
        # Position relative to multipole site
        xj = xyzcharge[j, 0] - xyzmult[nsite, 0]
        yj = xyzcharge[j, 1] - xyzmult[nsite, 1]
        zj = xyzcharge[j, 2] - xyzmult[nsite, 2]
        for k in range(ncharge):
            # Position relative to multipole site
            xk = xyzcharge[k, 0] - xyzmult[nsite, 0]
            yk = xyzcharge[k, 1] - xyzmult[nsite, 1]
            zk = xyzcharge[k, 2] - xyzmult[nsite, 2]

            _sum = 0.0
            for l in range(maxl + 1):
                if l == 0:
                    _sum = (
                        (1.0 / (2.0 * l + 1.0))
                        * W[0]
                        * _regular_solid_harmonic(0, 0, 0, xj, yj, zj)
                        * _regular_solid_harmonic(0, 0, 0, xk, yk, zk)
                    )
                else:
                    for m in range(l + 1):
                        if m == 0:
                            _sum += (
                                (1.0 / (2.0 * l + 1.0))
                                * W[l]
                                * (
                                    _regular_solid_harmonic(l, 0, 0, xj, yj, zj)
                                    * _regular_solid_harmonic(l, 0, 0, xk, yk, zk)
                                )
                            )
                        else:
                            # For m>0, include both real and imaginary parts
                            _sum += (
                                (1.0 / (2.0 * l + 1.0))
                                * W[l]
                                * (
                                    _regular_solid_harmonic(l, m, 0, xj, yj, zj)
                                    * _regular_solid_harmonic(l, m, 0, xk, yk, zk)
                                    + _regular_solid_harmonic(l, m, 1, xj, yj, zj)
                                    * _regular_solid_harmonic(l, m, 1, xk, yk, zk)
                                )
                            )
            A[j, k] = _sum
    return A


def build_b_vector(
    nsite: int,
    xyzmult: NDArray[np.float64],
    xyzcharge: NDArray[np.float64],
    r1: float,
    r2: float,
    maxl: int,
    multipoles: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Construct b vector as in J. Comp. Chem. Vol. 12, No. 8, 913-917 (1991)."""
    ncharge = xyzcharge.shape[0]

    W = np.zeros(maxl + 1, dtype=np.float64)
    for i in range(maxl + 1):
        W[i] = (1.0 / (1.0 - 2.0 * i)) * (r2 ** (1 - 2 * i) - r1 ** (1 - 2 * i))
    for k in range(ncharge):
        # Compute relative coordinates
        xk = xyzcharge[k, 0] - xyzmult[nsite, 0]
        yk = xyzcharge[k, 1] - xyzmult[nsite, 1]
        zk = xyzcharge[k, 2] - xyzmult[nsite, 2]

        _sum = 0.0
        for l in range(maxl + 1):
            if l == 0:
                # Special case for l = 0
                _sum = (
                    (1.0 / (2.0 * l + 1.0))
                    * W[0]
                    * multipoles[nsite, 0, 0, 0]
                    * _regular_solid_harmonic(0, 0, 0, xk, yk, zk)
                )
            else:
                for m in range(l + 1):
                    if m == 0:
                        # m = 0 case
                        _sum += (
                            (1.0 / (2.0 * l + 1.0))
                            * W[l]
                            * multipoles[nsite, l, 0, 0]
                            * _regular_solid_harmonic(l, 0, 0, xk, yk, zk)
                        )
                    else:
                        # m > 0 case
                        _sum += (
                            (1.0 / (2.0 * l + 1.0))
                            * W[l]
                            * (
                                multipoles[nsite, l, m, 0]
                                * _regular_solid_harmonic(l, m, 0, xk, yk, zk)
                                + multipoles[nsite, l, m, 1]
                                * _regular_solid_harmonic(l, m, 1, xk, yk, zk)
                            )
                        )
        b[k] = _sum
    return b
