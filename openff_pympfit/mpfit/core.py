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


def _regular_solid_harmonic(
    l: int,
    m: int,
    cs: int,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    xp=np,
) -> NDArray[np.float64]:
    """Evaluate regular solid harmonics using scipy or JAX."""
    x = xp.asarray(x, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)
    z = xp.asarray(z, dtype=xp.float64)

    r = xp.sqrt(x * x + y * y + z * z)
    nonzero = r > 1e-10

    origin_value = 1.0 if (l == 0 and m == 0 and cs == 0) else 0.0

    if xp is np and not xp.any(nonzero):
        return xp.full_like(r, origin_value)

    r_safe = xp.where(nonzero, r, 1.0)

    if l == 4:
        if m == 0:
            val = 0.125 * (
                8.0 * z**4
                - 24.0 * (x**2 + y**2) * z**2
                + 3.0 * (x**4 + 2.0 * x**2 * y**2 + y**4)
            )
        elif m == 1 and cs == 0:
            val = (
                0.25
                * xp.sqrt(10.0)
                * (4.0 * x * z**3 - 3.0 * x * z * (x**2 + y**2))
            )
        elif m == 1 and cs == 1:
            val = (
                0.25
                * xp.sqrt(10.0)
                * (4.0 * y * z**3 - 3.0 * y * z * (x**2 + y**2))
            )
        elif m == 2 and cs == 0:
            val = 0.25 * xp.sqrt(5.0) * (x**2 - y**2) * (6.0 * z**2 - x**2 - y**2)
        elif m == 2 and cs == 1:
            val = 0.25 * xp.sqrt(5.0) * x * y * (6.0 * z**2 - x**2 - y**2)
        elif m == 3 and cs == 0:
            val = 0.25 * xp.sqrt(70.0) * z * (x**3 - 3.0 * x * y**2)
        elif m == 3 and cs == 1:
            val = 0.25 * xp.sqrt(70.0) * z * (3.0 * x**2 * y - y**3)
        elif m == 4 and cs == 0:
            val = 0.125 * xp.sqrt(35.0) * (x**4 - 6.0 * x**2 * y**2 + y**4)
        elif m == 4 and cs == 1:
            val = 0.125 * xp.sqrt(35.0) * x * y * (x**2 - y**2)
        else:
            val = xp.zeros_like(r)
        return xp.where(nonzero, val, origin_value)

    theta = xp.arccos(z / r_safe)
    phi = xp.arctan2(y, x)

    if xp is np:
        Y = sph_harm_y(l, m, theta, phi)
    else:
        from jax.scipy.special import sph_harm_y as jax_sph_harm_y
        theta_1d = xp.atleast_1d(theta)
        phi_1d = xp.atleast_1d(phi)
        n_arr = xp.full_like(theta_1d, l, dtype=xp.int32)
        m_arr = xp.full_like(theta_1d, m, dtype=xp.int32)
        Y = jax_sph_harm_y(n_arr, m_arr, theta_1d, phi_1d, n_max=l)
        Y = Y.reshape(theta.shape)

    norm = xp.sqrt(4.0 * xp.pi / (2.0 * l + 1.0))

    if m == 0:
        val = norm * r_safe**l * Y.real
    else:
        factor = xp.sqrt(2.0) * ((-1.0) ** m) * norm * r_safe**l
        val = factor * (Y.real if cs == 0 else Y.imag)

    return xp.where(nonzero, val, origin_value)


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
    # Displacement vectors from multipole site to all charge sites
    dx = xyzcharge[:, 0] - xyzmult[nsite, 0]
    dy = xyzcharge[:, 1] - xyzmult[nsite, 1]
    dz = xyzcharge[:, 2] - xyzmult[nsite, 2]

    # integration factor, W
    W = np.zeros(maxl + 1)
    for i in range(maxl + 1):
        W[i] = (1.0 / (1.0 - 2.0 * i)) * (r2 ** (1 - 2 * i) - r1 ** (1 - 2 * i))

    A[:] = 0.0
    for l in range(maxl + 1):
        weight = W[l] / (2.0 * l + 1.0)
        for m in range(l + 1):
            cs_range = [0] if m == 0 else [0, 1]
            for cs in cs_range:
                rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz)
                A += weight * np.outer(rsh_vals, rsh_vals)

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
    # Displacement vectors from multipole site to all charge sites
    dx = xyzcharge[:, 0] - xyzmult[nsite, 0]
    dy = xyzcharge[:, 1] - xyzmult[nsite, 1]
    dz = xyzcharge[:, 2] - xyzmult[nsite, 2]

    W = np.zeros(maxl + 1, dtype=np.float64)
    for i in range(maxl + 1):
        W[i] = (1.0 / (1.0 - 2.0 * i)) * (r2 ** (1 - 2 * i) - r1 ** (1 - 2 * i))

    b[:] = 0.0
    for l in range(maxl + 1):
        weight = W[l] / (2.0 * l + 1.0)
        for m in range(l + 1):
            cs_range = [0] if m == 0 else [0, 1]
            for cs in cs_range:
                rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz)
                b += weight * multipoles[nsite, l, m, cs] * rsh_vals

    return b
