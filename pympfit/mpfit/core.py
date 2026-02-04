"""Core MPFIT math functions: A matrix, b vector, solid harmonics."""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.special import sph_harm_y

if TYPE_CHECKING:
    import torch


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
) -> NDArray[np.float64]:
    """Evaluate regular solid harmonics using scipy."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    r = np.sqrt(x * x + y * y + z * z)
    result = np.zeros_like(r)
    nonzero = r > 1e-10

    if l == 0 and m == 0 and cs == 0:
        result[~nonzero] = 1.0

    if not np.any(nonzero):
        return result

    xn, yn, zn, rn = x[nonzero], y[nonzero], z[nonzero], r[nonzero]

    if l == 4:
        if m == 0:
            val = 0.125 * (
                8.0 * zn**4
                - 24.0 * (xn**2 + yn**2) * zn**2
                + 3.0 * (xn**4 + 2.0 * xn**2 * yn**2 + yn**4)
            )
        elif m == 1 and cs == 0:
            val = (
                0.25
                * np.sqrt(10.0)
                * (4.0 * xn * zn**3 - 3.0 * xn * zn * (xn**2 + yn**2))
            )
        elif m == 1 and cs == 1:
            val = (
                0.25
                * np.sqrt(10.0)
                * (4.0 * yn * zn**3 - 3.0 * yn * zn * (xn**2 + yn**2))
            )
        elif m == 2 and cs == 0:
            val = 0.25 * np.sqrt(5.0) * (xn**2 - yn**2) * (6.0 * zn**2 - xn**2 - yn**2)
        elif m == 2 and cs == 1:
            val = 0.50 * np.sqrt(5.0) * xn * yn * (6.0 * zn**2 - xn**2 - yn**2)
        elif m == 3 and cs == 0:
            val = 0.25 * np.sqrt(70.0) * zn * (xn**3 - 3.0 * xn * yn**2)
        elif m == 3 and cs == 1:
            val = 0.25 * np.sqrt(70.0) * zn * (3.0 * xn**2 * yn - yn**3)
        elif m == 4 and cs == 0:
            val = 0.125 * np.sqrt(35.0) * (xn**4 - 6.0 * xn**2 * yn**2 + yn**4)
        elif m == 4 and cs == 1:
            val = 0.500 * np.sqrt(35.0) * xn * yn * (xn**2 - yn**2)
        else:
            val = np.zeros_like(rn)
        result[nonzero] = val
        return result

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


def build_A_matrix_torch(
    nsite: int,
    xyzmult: "torch.Tensor",
    xyzcharge: "torch.Tensor",
    r1: float,
    r2: float,
    maxl: int,
) -> "torch.Tensor":
    """Build A matrix using PyTorch + sphericart (fully differentiable).

    Equivalent to `build_A_matrix` but supports autograd for Bayesian inference.
    """
    import sphericart.torch as sph_torch
    import torch

    n_charges = xyzcharge.shape[0]

    # Displacement vectors from multipole site to charges
    displacements = xyzcharge - xyzmult[nsite : nsite + 1, :]

    solid_calc = sph_torch.SolidHarmonics(l_max=maxl)
    Y_solid = solid_calc.compute(displacements)  # (n_charges, (maxl+1)^2)

    W = torch.zeros(maxl + 1, dtype=torch.float64)
    for l in range(maxl + 1):
        if l == 0:
            W[l] = r2 - r1
        else:
            W[l] = (1.0 / (1.0 - 2 * l)) * (r2 ** (1 - 2 * l) - r1 ** (1 - 2 * l))

    # Build A matrix by summing weighted outer products
    A = torch.zeros((n_charges, n_charges), dtype=torch.float64)
    idx = 0
    for l in range(maxl + 1):
        norm = np.sqrt(4.0 * np.pi / (2.0 * l + 1.0))
        weight = W[l] / (2.0 * l + 1.0)

        for m in range(-l, l + 1):
            Y_lm = Y_solid[:, idx] * norm
            A = A + weight * torch.outer(Y_lm, Y_lm)
            idx += 1

    return A
