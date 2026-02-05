"""Transform Cartesian multipoles to spherical harmonics representation.

This module provides functions to convert MBIS Cartesian multipoles to
spherical harmonic representation for compatibility with GDMA-style output.

MBIS produces Cartesian multipoles with the following shapes:
- q (charge): (N, 1)
- mu (dipole): (N, 3)
- theta (quadrupole): (N, 3, 3)
- omega (octupole): (N, 3, 3, 3)

The spherical harmonic representation uses (2l+1) components per rank:
- l=0 (monopole): 1 component  -> Q00
- l=1 (dipole): 3 components   -> Q10, Q11c, Q11s
- l=2 (quadrupole): 5 components -> Q20, Q21c, Q21s, Q22c, Q22s
- l=3 (octupole): 7 components -> Q30, Q31c, Q31s, Q32c, Q32s, Q33c, Q33s

References
----------
The transformations are based on the standard relationships between
Cartesian and spherical tensor components. See Stone, "The Theory of
Intermolecular Forces", 2nd ed., Oxford University Press, 2013.
"""

import numpy as np
from numpy.typing import NDArray


def cartesian_to_spherical_dipole(
    mu: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert Cartesian dipole to spherical harmonics.

    Parameters
    ----------
    mu
        Cartesian dipole array of shape (N, 3) where columns are [x, y, z].

    Returns
    -------
    NDArray[np.float64]
        Spherical harmonic dipole array of shape (N, 3).
        Order: [Q10, Q11c, Q11s] = [z, x, y]
    """
    n_atoms = mu.shape[0]
    spherical = np.zeros((n_atoms, 3))

    # μz = Q10
    # μx = Q11c
    # μy = Q11s
    spherical[:, 0] = mu[:, 2]  # Q10 = μz
    spherical[:, 1] = mu[:, 0]  # Q11c = μx
    spherical[:, 2] = mu[:, 1]  # Q11s = μy

    return spherical


def cartesian_to_spherical_quadrupole(
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert Cartesian quadrupole to spherical harmonics.

    The transformation inverts:
        Theta_xx = -1/2 Q20 + 1/2 sqrt(3) Q22c
        Theta_yy = -1/2 Q20 - 1/2 sqrt(3) Q22c
        Theta_zz = Q20
        Theta_xy = 1/(2*sqrt(3)) Q22s
        Theta_xz = 1/(2*sqrt(3)) Q21c
        Theta_yz = 1/(2*sqrt(3)) Q21s

    Parameters
    ----------
    theta
        Cartesian quadrupole tensor of shape (N, 3, 3).

    Returns
    -------
    NDArray[np.float64]
        Spherical harmonic quadrupole array of shape (N, 5).
        Order: [Q20, Q21c, Q21s, Q22c, Q22s]
    """
    n_atoms = theta.shape[0]
    spherical = np.zeros((n_atoms, 5))

    sqrt3 = np.sqrt(3.0)

    for i in range(n_atoms):
        xx = theta[i, 0, 0]
        yy = theta[i, 1, 1]
        zz = theta[i, 2, 2]
        xy = theta[i, 0, 1]
        xz = theta[i, 0, 2]
        yz = theta[i, 1, 2]

        # Q20 = Theta_zz
        spherical[i, 0] = zz

        # Theta_xz = 1/(2*sqrt(3)) Q21c  =>  Q21c = 2*sqrt(3) * Theta_xz
        spherical[i, 1] = 2.0 * sqrt3 * xz

        # Theta_yz = 1/(2*sqrt(3)) Q21s  =>  Q21s = 2*sqrt(3) * Theta_yz
        spherical[i, 2] = 2.0 * sqrt3 * yz

        # From Theta_xx = -1/2 Q20 + 1/2 sqrt(3) Q22c
        # and  Theta_yy = -1/2 Q20 - 1/2 sqrt(3) Q22c
        # => Theta_xx - Theta_yy = sqrt(3) Q22c
        # => Q22c = (Theta_xx - Theta_yy) / sqrt(3)
        spherical[i, 3] = (xx - yy) / sqrt3

        # Theta_xy = 1/(2*sqrt(3)) Q22s  =>  Q22s = 2*sqrt(3) * Theta_xy
        spherical[i, 4] = 2.0 * sqrt3 * xy

    return spherical


def cartesian_to_spherical_octupole(
    omega: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert Cartesian octupole to spherical harmonics.

    The transformation inverts the relationships:
        Omega_xxx = sqrt(5/8) Q33c - sqrt(3/8) Q31c
        Omega_xxy = sqrt(5/8) Q33s - sqrt(1/24) Q31s
        Omega_xyy = -sqrt(5/8) Q33c - sqrt(1/24) Q31c
        Omega_yyy = -sqrt(5/8) Q33s - sqrt(3/8) Q31s
        Omega_xxz = sqrt(5/12) Q32c - 1/2 Q30
        Omega_xyz = sqrt(5/12) Q32s
        Omega_yyz = -sqrt(5/12) Q32c - 1/2 Q30
        Omega_xzz = sqrt(2/3) Q31c
        Omega_yzz = sqrt(2/3) Q31s
        Omega_zzz = Q30

    Parameters
    ----------
    omega
        Cartesian octupole tensor of shape (N, 3, 3, 3).

    Returns
    -------
    NDArray[np.float64]
        Spherical harmonic octupole array of shape (N, 7).
        Order: [Q30, Q31c, Q31s, Q32c, Q32s, Q33c, Q33s]
    """
    n_atoms = omega.shape[0]
    spherical = np.zeros((n_atoms, 7))

    sqrt_2_3 = np.sqrt(2.0 / 3.0)
    sqrt_5_12 = np.sqrt(5.0 / 12.0)
    sqrt_5_8 = np.sqrt(5.0 / 8.0)
    sqrt_3_8 = np.sqrt(3.0 / 8.0)

    for i in range(n_atoms):
        # Extract Cartesian components
        xxx = omega[i, 0, 0, 0]
        xxz = omega[i, 0, 0, 2]
        xyz = omega[i, 0, 1, 2]
        xzz = omega[i, 0, 2, 2]
        yyy = omega[i, 1, 1, 1]
        yyz = omega[i, 1, 1, 2]
        yzz = omega[i, 1, 2, 2]
        zzz = omega[i, 2, 2, 2]

        # Q30 = Omega_zzz
        q30 = zzz
        spherical[i, 0] = q30

        # Omega_xzz = sqrt(2/3) Q31c  =>  Q31c = Omega_xzz / sqrt(2/3)
        q31c = xzz / sqrt_2_3
        spherical[i, 1] = q31c

        # Omega_yzz = sqrt(2/3) Q31s  =>  Q31s = Omega_yzz / sqrt(2/3)
        q31s = yzz / sqrt_2_3
        spherical[i, 2] = q31s

        # Omega_xxz = sqrt(5/12) Q32c - 1/2 Q30
        # Omega_yyz = -sqrt(5/12) Q32c - 1/2 Q30
        # => Omega_xxz - Omega_yyz = 2*sqrt(5/12) Q32c
        # => Q32c = (Omega_xxz - Omega_yyz) / (2*sqrt(5/12))
        q32c = (xxz - yyz) / (2.0 * sqrt_5_12)
        spherical[i, 3] = q32c

        # Omega_xyz = sqrt(5/12) Q32s  =>  Q32s = Omega_xyz / sqrt(5/12)
        q32s = xyz / sqrt_5_12
        spherical[i, 4] = q32s

        # From Omega_xxx = sqrt(5/8) Q33c - sqrt(3/8) Q31c, solve for Q33c:
        # Q33c = (Omega_xxx + sqrt(3/8) Q31c) / sqrt(5/8)
        q33c = (xxx + sqrt_3_8 * q31c) / sqrt_5_8
        spherical[i, 5] = q33c

        # From Omega_yyy = -sqrt(5/8) Q33s - sqrt(3/8) Q31s, solve for Q33s:
        # Q33s = -(Omega_yyy + sqrt(3/8) Q31s) / sqrt(5/8)
        q33s = -(yyy + sqrt_3_8 * q31s) / sqrt_5_8
        spherical[i, 6] = q33s

    return spherical


def cartesian_to_spherical_multipoles(
    charges: NDArray[np.float64],
    dipoles: NDArray[np.float64] | None = None,
    quadrupoles: NDArray[np.float64] | None = None,
    octupoles: NDArray[np.float64] | None = None,
    max_moment: int = 4,
) -> NDArray[np.float64]:
    """Convert MBIS Cartesian multipoles to spherical harmonic representation.

    Parameters
    ----------
    charges
        MBIS charges of shape (N,) or (N, 1).
    dipoles
        MBIS dipoles of shape (N, 3), or None.
    quadrupoles
        MBIS quadrupoles of shape (N, 3, 3), or None.
    octupoles
        MBIS octupoles of shape (N, 3, 3, 3), or None.
    max_moment
        Maximum multipole moment to include (1-4).

    Returns
    -------
    NDArray[np.float64]
        Combined spherical harmonic multipole array of shape (N, n_components)
        where n_components = (max_moment + 1)^2.
        Components are ordered as:
        - l=0: Q00 (index 0)
        - l=1: Q10, Q11c, Q11s (indices 1-3)
        - l=2: Q20, Q21c, Q21s, Q22c, Q22s (indices 4-8)
        - l=3: Q30, Q31c, Q31s, Q32c, Q32s, Q33c, Q33s (indices 9-15)
    """
    charges = np.atleast_1d(charges.flatten())
    n_atoms = len(charges)
    n_components = (max_moment + 1) ** 2
    multipoles = np.zeros((n_atoms, n_components))

    # l=0: Monopole (charge)
    multipoles[:, 0] = charges

    # l=1: Dipole
    if max_moment >= 1 and dipoles is not None:
        spherical_dipoles = cartesian_to_spherical_dipole(dipoles)
        multipoles[:, 1:4] = spherical_dipoles

    # l=2: Quadrupole
    if max_moment >= 2 and quadrupoles is not None:
        spherical_quadrupoles = cartesian_to_spherical_quadrupole(quadrupoles)
        multipoles[:, 4:9] = spherical_quadrupoles

    # l=3: Octupole
    if max_moment >= 3 and octupoles is not None:
        spherical_octupoles = cartesian_to_spherical_octupole(octupoles)
        multipoles[:, 9:16] = spherical_octupoles

    return multipoles


def spherical_to_cartesian_dipole(
    mu_sph: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert spherical harmonic dipole to Cartesian.

    Parameters
    ----------
    mu_sph
        Spherical harmonic dipole array of shape (N, 3).
        Order: [Q10, Q11c, Q11s] = [z, x, y]

    Returns
    -------
    NDArray[np.float64]
        Cartesian dipole array of shape (N, 3) where columns are [x, y, z].
    """
    n_atoms = mu_sph.shape[0]
    cartesian = np.zeros((n_atoms, 3))

    # Q10 = μz  =>  μz = Q10
    # Q11c = μx =>  μx = Q11c
    # Q11s = μy =>  μy = Q11s
    cartesian[:, 0] = mu_sph[:, 1]  # μx = Q11c
    cartesian[:, 1] = mu_sph[:, 2]  # μy = Q11s
    cartesian[:, 2] = mu_sph[:, 0]  # μz = Q10

    return cartesian


def spherical_to_cartesian_quadrupole(
    q_sph: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert spherical harmonic quadrupole to Cartesian.

    The transformation uses:
        Theta_xx = -1/2 Q20 + 1/2 sqrt(3) Q22c
        Theta_yy = -1/2 Q20 - 1/2 sqrt(3) Q22c
        Theta_zz = Q20
        Theta_xy = 1/(2*sqrt(3)) Q22s
        Theta_xz = 1/(2*sqrt(3)) Q21c
        Theta_yz = 1/(2*sqrt(3)) Q21s

    Parameters
    ----------
    q_sph
        Spherical harmonic quadrupole array of shape (N, 5).
        Order: [Q20, Q21c, Q21s, Q22c, Q22s]

    Returns
    -------
    NDArray[np.float64]
        Cartesian quadrupole tensor of shape (N, 3, 3).
    """
    n_atoms = q_sph.shape[0]
    theta = np.zeros((n_atoms, 3, 3))

    sqrt3 = np.sqrt(3.0)

    for i in range(n_atoms):
        q20 = q_sph[i, 0]
        q21c = q_sph[i, 1]
        q21s = q_sph[i, 2]
        q22c = q_sph[i, 3]
        q22s = q_sph[i, 4]

        # Theta_zz = Q20
        zz = q20

        # Theta_xz = 1/(2*sqrt(3)) Q21c
        xz = q21c / (2.0 * sqrt3)

        # Theta_yz = 1/(2*sqrt(3)) Q21s
        yz = q21s / (2.0 * sqrt3)

        # Theta_xy = 1/(2*sqrt(3)) Q22s
        xy = q22s / (2.0 * sqrt3)

        # Theta_xx = -1/2 Q20 + 1/2 sqrt(3) Q22c
        xx = -0.5 * q20 + 0.5 * sqrt3 * q22c

        # Theta_yy = -1/2 Q20 - 1/2 sqrt(3) Q22c
        yy = -0.5 * q20 - 0.5 * sqrt3 * q22c

        theta[i, 0, 0] = xx
        theta[i, 1, 1] = yy
        theta[i, 2, 2] = zz
        theta[i, 0, 1] = theta[i, 1, 0] = xy
        theta[i, 0, 2] = theta[i, 2, 0] = xz
        theta[i, 1, 2] = theta[i, 2, 1] = yz

    return theta


def spherical_to_cartesian_octupole(
    o_sph: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert spherical harmonic octupole to Cartesian.

    The transformation uses:
        Omega_xxx = sqrt(5/8) Q33c - sqrt(3/8) Q31c
        Omega_xxy = sqrt(5/8) Q33s - sqrt(1/24) Q31s
        Omega_xyy = -sqrt(5/8) Q33c - sqrt(1/24) Q31c
        Omega_yyy = -sqrt(5/8) Q33s - sqrt(3/8) Q31s
        Omega_xxz = sqrt(5/12) Q32c - 1/2 Q30
        Omega_xyz = sqrt(5/12) Q32s
        Omega_yyz = -sqrt(5/12) Q32c - 1/2 Q30
        Omega_xzz = sqrt(2/3) Q31c
        Omega_yzz = sqrt(2/3) Q31s
        Omega_zzz = Q30

    Parameters
    ----------
    o_sph
        Spherical harmonic octupole array of shape (N, 7).
        Order: [Q30, Q31c, Q31s, Q32c, Q32s, Q33c, Q33s]

    Returns
    -------
    NDArray[np.float64]
        Cartesian octupole tensor of shape (N, 3, 3, 3).
    """
    n_atoms = o_sph.shape[0]
    omega = np.zeros((n_atoms, 3, 3, 3))

    sqrt_2_3 = np.sqrt(2.0 / 3.0)
    sqrt_5_12 = np.sqrt(5.0 / 12.0)
    sqrt_5_8 = np.sqrt(5.0 / 8.0)
    sqrt_3_8 = np.sqrt(3.0 / 8.0)
    sqrt_1_24 = np.sqrt(1.0 / 24.0)

    for i in range(n_atoms):
        q30 = o_sph[i, 0]
        q31c = o_sph[i, 1]
        q31s = o_sph[i, 2]
        q32c = o_sph[i, 3]
        q32s = o_sph[i, 4]
        q33c = o_sph[i, 5]
        q33s = o_sph[i, 6]

        # Omega_zzz = Q30
        zzz = q30

        # Omega_xzz = sqrt(2/3) Q31c
        xzz = sqrt_2_3 * q31c

        # Omega_yzz = sqrt(2/3) Q31s
        yzz = sqrt_2_3 * q31s

        # Omega_xxz = sqrt(5/12) Q32c - 1/2 Q30
        xxz = sqrt_5_12 * q32c - 0.5 * q30

        # Omega_yyz = -sqrt(5/12) Q32c - 1/2 Q30
        yyz = -sqrt_5_12 * q32c - 0.5 * q30

        # Omega_xyz = sqrt(5/12) Q32s
        xyz = sqrt_5_12 * q32s

        # Omega_xxx = sqrt(5/8) Q33c - sqrt(3/8) Q31c
        xxx = sqrt_5_8 * q33c - sqrt_3_8 * q31c

        # Omega_yyy = -sqrt(5/8) Q33s - sqrt(3/8) Q31s
        yyy = -sqrt_5_8 * q33s - sqrt_3_8 * q31s

        # Omega_xxy = sqrt(5/8) Q33s - sqrt(1/24) Q31s
        xxy = sqrt_5_8 * q33s - sqrt_1_24 * q31s

        # Omega_xyy = -sqrt(5/8) Q33c - sqrt(1/24) Q31c
        xyy = -sqrt_5_8 * q33c - sqrt_1_24 * q31c

        # Fill in the symmetric tensor
        omega[i, 0, 0, 0] = xxx
        omega[i, 2, 2, 2] = zzz
        omega[i, 1, 1, 1] = yyy

        # xxy and permutations
        omega[i, 0, 0, 1] = omega[i, 0, 1, 0] = omega[i, 1, 0, 0] = xxy

        # xxz and permutations
        omega[i, 0, 0, 2] = omega[i, 0, 2, 0] = omega[i, 2, 0, 0] = xxz

        # xyy and permutations
        omega[i, 0, 1, 1] = omega[i, 1, 0, 1] = omega[i, 1, 1, 0] = xyy

        # xyz and permutations
        omega[i, 0, 1, 2] = omega[i, 0, 2, 1] = omega[i, 1, 0, 2] = xyz
        omega[i, 1, 2, 0] = omega[i, 2, 0, 1] = omega[i, 2, 1, 0] = xyz

        # xzz and permutations
        omega[i, 0, 2, 2] = omega[i, 2, 0, 2] = omega[i, 2, 2, 0] = xzz

        # yyz and permutations
        omega[i, 1, 1, 2] = omega[i, 1, 2, 1] = omega[i, 2, 1, 1] = yyz

        # yzz and permutations
        omega[i, 1, 2, 2] = omega[i, 2, 1, 2] = omega[i, 2, 2, 1] = yzz

    return omega


def spherical_to_cartesian_multipoles(
    multipoles: NDArray[np.float64],
    max_moment: int = 4,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    """Convert spherical harmonic multipoles to Cartesian representation.

    Parameters
    ----------
    multipoles
        Spherical harmonic multipole array of shape (N, n_components)
        where n_components = (max_moment + 1)^2.
        Components are ordered as:
        - l=0: Q00 (index 0)
        - l=1: Q10, Q11c, Q11s (indices 1-3)
        - l=2: Q20, Q21c, Q21s, Q22c, Q22s (indices 4-8)
        - l=3: Q30, Q31c, Q31s, Q32c, Q32s, Q33c, Q33s (indices 9-15)
    max_moment
        Maximum multipole moment included in the input (1-4).

    Returns
    -------
    tuple
        A tuple of (charges, dipoles, quadrupoles, octupoles) where:
        - charges: array of shape (N,)
        - dipoles: array of shape (N, 3) or None if max_moment < 1
        - quadrupoles: array of shape (N, 3, 3) or None if max_moment < 2
        - octupoles: array of shape (N, 3, 3, 3) or None if max_moment < 3
    """
    n_atoms = multipoles.shape[0]

    # l=0: Monopole (charge)
    charges = multipoles[:, 0]

    # l=1: Dipole
    dipoles = None
    if max_moment >= 1:
        spherical_dipoles = multipoles[:, 1:4]
        dipoles = spherical_to_cartesian_dipole(spherical_dipoles)

    # l=2: Quadrupole
    quadrupoles = None
    if max_moment >= 2:
        spherical_quadrupoles = multipoles[:, 4:9]
        quadrupoles = spherical_to_cartesian_quadrupole(spherical_quadrupoles)

    # l=3: Octupole
    octupoles = None
    if max_moment >= 3:
        spherical_octupoles = multipoles[:, 9:16]
        octupoles = spherical_to_cartesian_octupole(spherical_octupoles)

    return charges, dipoles, quadrupoles, octupoles


def cartesian_multipoles_to_flat(
    charges: NDArray[np.float64],
    dipoles: NDArray[np.float64] | None = None,
    quadrupoles: NDArray[np.float64] | None = None,
    octupoles: NDArray[np.float64] | None = None,
    max_moment: int = 4,
) -> NDArray[np.float64]:
    """Flatten MBIS Cartesian multipoles to a 2D array.

    This keeps the Cartesian representation but flattens tensors.
    All unique components of symmetric tensors are stored.

    Parameters
    ----------
    charges
        MBIS charges of shape (N,) or (N, 1).
    dipoles
        MBIS dipoles of shape (N, 3), or None.
    quadrupoles
        MBIS quadrupoles of shape (N, 3, 3), or None.
    octupoles
        MBIS octupoles of shape (N, 3, 3, 3), or None.
    max_moment
        Maximum multipole moment to include (1-4).

    Returns
    -------
    NDArray[np.float64]
        Flattened Cartesian multipole array of shape (N, n_components).
        Components for each rank are stored in the following order:
        - l=0: q (1 component, index 0)
        - l=1: x, y, z (3 components, indices 1-3)
        - l=2: xx, xy, xz, yy, yz, zz (6 components, indices 4-9)
        - l=3: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
               (10 components, indices 10-19)

    Notes
    -----
    The total number of Cartesian components is:
    - max_moment=1: 1 + 3 = 4
    - max_moment=2: 1 + 3 + 6 = 10
    - max_moment=3: 1 + 3 + 6 + 10 = 20
    - max_moment=4: 1 + 3 + 6 + 10 + 15 = 35
    """
    charges = np.atleast_1d(charges.flatten())
    n_atoms = len(charges)

    # Calculate total Cartesian components: sum of (l+1)(l+2)/2 for l=0 to max
    # l=0: 1, l=1: 3, l=2: 6, l=3: 10, l=4: 15
    n_components = sum((l + 1) * (l + 2) // 2 for l in range(max_moment + 1))
    multipoles = np.zeros((n_atoms, n_components))

    # l=0: Monopole (charge) - 1 component
    multipoles[:, 0] = charges
    idx = 1

    # l=1: Dipole (x, y, z) - 3 components
    if max_moment >= 1 and dipoles is not None:
        multipoles[:, idx : idx + 3] = dipoles
    idx += 3

    # l=2: Quadrupole - 6 unique components (xx, xy, xz, yy, yz, zz)
    if max_moment >= 2 and quadrupoles is not None:
        for i in range(n_atoms):
            q = quadrupoles[i]
            multipoles[i, idx + 0] = q[0, 0]  # xx
            multipoles[i, idx + 1] = q[0, 1]  # xy
            multipoles[i, idx + 2] = q[0, 2]  # xz
            multipoles[i, idx + 3] = q[1, 1]  # yy
            multipoles[i, idx + 4] = q[1, 2]  # yz
            multipoles[i, idx + 5] = q[2, 2]  # zz
    idx += 6

    # l=3: Octupole - 10 unique components
    # Order: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
    if max_moment >= 3 and octupoles is not None:
        for i in range(n_atoms):
            o = octupoles[i]
            multipoles[i, idx + 0] = o[0, 0, 0]  # xxx
            multipoles[i, idx + 1] = o[0, 0, 1]  # xxy
            multipoles[i, idx + 2] = o[0, 0, 2]  # xxz
            multipoles[i, idx + 3] = o[0, 1, 1]  # xyy
            multipoles[i, idx + 4] = o[0, 1, 2]  # xyz
            multipoles[i, idx + 5] = o[0, 2, 2]  # xzz
            multipoles[i, idx + 6] = o[1, 1, 1]  # yyy
            multipoles[i, idx + 7] = o[1, 1, 2]  # yyz
            multipoles[i, idx + 8] = o[1, 2, 2]  # yzz
            multipoles[i, idx + 9] = o[2, 2, 2]  # zzz

    return multipoles
