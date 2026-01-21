"""
Minimal Python port of Fortran constrained MPFIT.

This is a standalone implementation that maps 1:1 to the Fortran code in
mpfit-python/f90/mpfit_source/source_constrain_fit/mpfitroutines.f90

Variable names match Fortran exactly for traceability:
- atomtype: Type label for each atom (e.g., "C1", "H1", "H2")
- quse: Binary mask indicating which atoms affect each site's fit
- allcharge: Per-site charge contributions [n_sites, n_atoms]
- qstore: Total charge on each atom (sum over all site contributions)
- multipoles: Multipole moments from GDMA [n_sites, maxl+1, maxl+1, 2]
- xyzmult: Multipole site coordinates [n_sites, 3]
- xyzcharge: Charge site coordinates [n_atoms, 3]
- lmax: Maximum multipole rank for each site
- rvdw: Van der Waals radius for each site (used to determine quse)

The key constraint mechanism:
- Atoms with the same type label are constrained to have the same total charge (qstore)
- This is enforced by computing the LAST site contribution as:
    allcharge[last_site, i] = qstore[twin] - sum(other contributions)
"""

import numpy as np
from scipy.optimize import minimize

# =============================================================================
# GLOBAL VARIABLES (matching Fortran variables.f90)
# =============================================================================

# Maximum multipole rank (currently cannot be greater than 4)
MAXL = 4

# Fitting shell radii (from JPC 1993, 97, 6628)
R1 = 3.78  # Inner radius in Bohr
R2 = 9.45  # Outer radius in Bohr

# Charge constraint parameters
MOLECULE_CHARGE = 0.0
CONCHG = 0.0  # Weight for total charge constraint (0 means no constraint)

# Negative midpoint charge penalty (for virtual sites)
NEGMIDC = 0.0
NEGMIDE = 0.0

PI = 3.14159265359

# These will be set by setup functions
atomtype = None  # List[str], size = n_atoms
quse = None  # np.ndarray, shape (n_sites, n_atoms)
allcharge = None  # np.ndarray, shape (n_sites, n_atoms)
qstore = None  # np.ndarray, shape (n_atoms,)
multipoles = None  # np.ndarray, shape (n_sites, maxl+1, maxl+1, 2)
xyzmult = None  # np.ndarray, shape (n_sites, 3)
xyzcharge = None  # np.ndarray, shape (n_atoms, 3)
lmax = None  # np.ndarray, shape (n_sites,)
rvdw = None  # np.ndarray, shape (n_sites,)


# =============================================================================
# CORE FUNCTIONS (matching mpfitroutines.f90)
# =============================================================================


def RSH(l: int, m: int, cs: int, xyz: np.ndarray) -> float:
    """
    Evaluate regular solid harmonics at point (x, y, z).

    Parameters
    ----------
    l : int
        Angular momentum quantum number
    m : int
        Magnetic quantum number (0 <= m <= l)
    cs : int
        0 for cosine (real), 1 for sine (imaginary)
    xyz : np.ndarray
        Cartesian coordinates [x, y, z]

    Returns
    -------
    float
        Value of the regular solid harmonic

    Notes
    -----
    Lines 536-576 in Fortran mpfitroutines.f90
    """
    x, y, z = xyz[0], xyz[1], xyz[2]
    rsq = x**2 + y**2 + z**2

    # Pre-compute all needed values
    rsharray = np.zeros((5, 5, 2))

    # l=0 (monopole)
    rsharray[0, 0, 0] = 1.0

    # l=1 (dipole)
    rsharray[1, 0, 0] = z
    rsharray[1, 1, 0] = x
    rsharray[1, 1, 1] = y

    # l=2 (quadrupole)
    rsharray[2, 0, 0] = 0.5 * (3.0 * z**2 - rsq)
    rsharray[2, 1, 0] = np.sqrt(3.0) * x * z
    rsharray[2, 1, 1] = np.sqrt(3.0) * y * z
    rsharray[2, 2, 0] = 0.5 * np.sqrt(3.0) * (x**2 - y**2)
    rsharray[2, 2, 1] = np.sqrt(3.0) * x * y

    # l=3 (octupole)
    rsharray[3, 0, 0] = 0.5 * (5.0 * z**3 - 3.0 * z * rsq)
    rsharray[3, 1, 0] = 0.25 * np.sqrt(6.0) * (4.0 * x * z**2 - x**3 - x * y**2)
    rsharray[3, 1, 1] = 0.25 * np.sqrt(6.0) * (4.0 * y * z**2 - y * x**2 - y**3)
    rsharray[3, 2, 0] = 0.5 * np.sqrt(15.0) * z * (x**2 - y**2)
    rsharray[3, 2, 1] = np.sqrt(15.0) * x * y * z
    rsharray[3, 3, 0] = 0.25 * np.sqrt(10.0) * (x**3 - 3.0 * x * y**2)
    rsharray[3, 3, 1] = 0.25 * np.sqrt(10.0) * (3.0 * x**2 * y - y**3)

    # l=4 (hexadecapole)
    rsharray[4, 0, 0] = 0.125 * (
        8.0 * z**4
        - 24.0 * (x**2 + y**2) * z**2
        + 3.0 * (x**4 + 2.0 * x**2 * y**2 + y**4)
    )
    rsharray[4, 1, 0] = 0.25 * np.sqrt(10.0) * (
        4.0 * x * z**3 - 3.0 * x * z * (x**2 + y**2)
    )
    rsharray[4, 1, 1] = 0.25 * np.sqrt(10.0) * (
        4.0 * y * z**3 - 3.0 * y * z * (x**2 + y**2)
    )
    rsharray[4, 2, 0] = 0.25 * np.sqrt(5.0) * (x**2 - y**2) * (
        6.0 * z**2 - x**2 - y**2
    )
    rsharray[4, 2, 1] = 0.25 * np.sqrt(5.0) * x * y * (6.0 * z**2 - x**2 - y**2)
    rsharray[4, 3, 0] = 0.25 * np.sqrt(70.0) * z * (x**3 - 3.0 * x * y**2)
    rsharray[4, 3, 1] = 0.25 * np.sqrt(70.0) * z * (3.0 * x**2 * y - y**3)
    rsharray[4, 4, 0] = 0.125 * np.sqrt(35.0) * (x**4 - 6.0 * x**2 * y**2 + y**4)
    rsharray[4, 4, 1] = 0.125 * np.sqrt(35.0) * x * y * (x**2 - y**2)

    return rsharray[l, m, cs]


def expandcharge(p0: np.ndarray) -> None:
    """
    Map reduced parameters to full charges with atom-type constraints.

    This is the core constraint mechanism. For atoms with the same type label,
    it enforces qstore[i] == qstore[twin] by computing the last site contribution
    to make up the difference.

    Parameters
    ----------
    p0 : np.ndarray
        Reduced parameter vector

    Notes
    -----
    Lines 585-671 in Fortran mpfitroutines.f90

    The constraint mechanism works as follows:
    1. For the first atom of each type: use all parameters freely
    2. For subsequent atoms of the same type:
       - Use (n_quse - 1) parameters freely
       - Compute the LAST parameter as: qstore[twin] - sum(other contributions)
       - This enforces qstore[i] == qstore[twin]
    """
    global allcharge, qstore

    multsites = xyzmult.shape[0]
    atoms = len(atomtype)
    nmid = 0  # No midpoint charges in this implementation

    allcharge = np.zeros((multsites, atoms + nmid))

    count = 0  # Index into p0

    for i in range(atoms):
        count1 = 0
        charge_sum = 0.0

        if i == 0:
            # First atom: use all parameters freely
            for j in range(multsites):
                if quse[j, i] == 1:
                    allcharge[j, i] = p0[count]
                    charge_sum += p0[count]
                    count += 1
            qstore[i] = charge_sum
        else:
            # Find twin (first atom with same type)
            twin = -1
            for k in range(i):
                if atomtype[i] == atomtype[k]:
                    twin = k
                    break

            if twin >= 0:
                # Constrained atom: same type as twin
                # Count how many sites use this atom
                for j in range(multsites):
                    if quse[j, i] == 1:
                        count1 += 1

                # Fill first (count1 - 1) contributions from p0
                count2 = 1
                for j in range(multsites):
                    if quse[j, i] == 1 and count2 < count1:
                        allcharge[j, i] = p0[count]
                        charge_sum += p0[count]
                        count += 1
                        count2 += 1
                    elif quse[j, i] == 1 and count2 == count1:
                        # LAST site: compute to enforce qstore[i] == qstore[twin]
                        allcharge[j, i] = qstore[twin] - charge_sum
                        qstore[i] = qstore[twin]  # Enforced equality!
            else:
                # First occurrence of this type: use all parameters freely
                for j in range(multsites):
                    if quse[j, i] == 1:
                        allcharge[j, i] = p0[count]
                        charge_sum += p0[count]
                        count += 1
                qstore[i] = charge_sum


def kaisq(p0: np.ndarray) -> float:
    """
    Objective function: sum of squared multipole errors.

    Parameters
    ----------
    p0 : np.ndarray
        Reduced parameter vector

    Returns
    -------
    float
        Value of the objective function

    Notes
    -----
    Lines 212-344 in Fortran mpfitroutines.f90
    """
    global allcharge, qstore

    expandcharge(p0)

    multsites = xyzmult.shape[0]
    natom = multsites
    nmid = xyzcharge.shape[0] - natom

    xyzqatom = xyzcharge[:natom, :]
    xyzqmid = xyzcharge[natom:, :] if nmid > 0 else None

    W = np.zeros(MAXL + 1)

    sumkai = 0.0

    for s in range(multsites):
        q0 = allcharge[s, :]
        rmax = rvdw[s] + R2
        rminn = rvdw[s] + R1

        # Compute W integration factor
        for i in range(int(lmax[s]) + 1):
            W[i] = (1.0 / (1.0 - 2.0 * i)) * (
                rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i)
            )

        site_sum = 0.0
        for l in range(int(lmax[s]) + 1):
            if l == 0:
                sum1 = 0.0
                for j in range(natom):
                    xyz = xyzqatom[j, :] - xyzmult[s, :]
                    sum1 += q0[j] * RSH(0, 0, 0, xyz)
                if nmid > 0:
                    for j in range(nmid):
                        xyz = xyzqmid[j, :] - xyzmult[s, :]
                        sum1 += q0[natom + j] * RSH(0, 0, 0, xyz)
                site_sum = (
                    (4.0 * PI / (2.0 * l + 1.0))
                    * W[0]
                    * (multipoles[s, l, 0, 0] - sum1) ** 2
                )
            else:
                for m in range(l + 1):
                    if m == 0:
                        sum1 = 0.0
                        for j in range(natom):
                            xyz = xyzqatom[j, :] - xyzmult[s, :]
                            sum1 += q0[j] * RSH(l, 0, 0, xyz)
                        if nmid > 0:
                            for j in range(nmid):
                                xyz = xyzqmid[j, :] - xyzmult[s, :]
                                sum1 += q0[natom + j] * RSH(l, 0, 0, xyz)
                        site_sum += (
                            (4.0 * PI / (2.0 * l + 1.0))
                            * W[l]
                            * (multipoles[s, l, 0, 0] - sum1) ** 2
                        )
                    else:
                        # Cosine (real) part
                        sum1 = 0.0
                        for j in range(natom):
                            xyz = xyzqatom[j, :] - xyzmult[s, :]
                            sum1 += q0[j] * RSH(l, m, 0, xyz)
                        if nmid > 0:
                            for j in range(nmid):
                                xyz = xyzqmid[j, :] - xyzmult[s, :]
                                sum1 += q0[natom + j] * RSH(l, m, 0, xyz)
                        site_sum += (
                            (4.0 * PI / (2.0 * l + 1.0))
                            * W[l]
                            * (multipoles[s, l, m, 0] - sum1) ** 2
                        )

                        # Sine (imaginary) part
                        sum1 = 0.0
                        for j in range(natom):
                            xyz = xyzqatom[j, :] - xyzmult[s, :]
                            sum1 += q0[j] * RSH(l, m, 1, xyz)
                        if nmid > 0:
                            for j in range(nmid):
                                xyz = xyzqmid[j, :] - xyzmult[s, :]
                                sum1 += q0[natom + j] * RSH(l, m, 1, xyz)
                        site_sum += (
                            (4.0 * PI / (2.0 * l + 1.0))
                            * W[l]
                            * (multipoles[s, l, m, 1] - sum1) ** 2
                        )

        sumkai += site_sum

    # Add charge constraint penalty
    sumchg = np.sum(qstore)
    sumcon = CONCHG * (sumchg - MOLECULE_CHARGE) ** 2

    # Add penalty for positive midpoint charges (if any)
    if nmid > 0:
        for i in range(nmid):
            if qstore[natom + i] > 0.0:
                sumcon += NEGMIDC * np.exp(NEGMIDE * qstore[natom + i])

    return sumkai + sumcon


def dkaisq(p0: np.ndarray) -> np.ndarray:
    """
    Gradient of kaisq w.r.t. reduced parameters.

    Parameters
    ----------
    p0 : np.ndarray
        Reduced parameter vector

    Returns
    -------
    np.ndarray
        Gradient vector

    Notes
    -----
    Lines 352-527 in Fortran mpfitroutines.f90
    """
    global allcharge, qstore

    expandcharge(p0)

    multsites = xyzmult.shape[0]
    natom = multsites
    npts = xyzcharge.shape[0]
    nmid = npts - natom

    xyzqatom = xyzcharge[:natom, :]
    xyzqmid = xyzcharge[natom:, :] if nmid > 0 else None

    W = np.zeros(MAXL + 1)

    # dparam stores d(kaisq)/d(allcharge[s,j]) before reordering
    dparam = np.zeros(multsites * npts)

    for s in range(multsites):
        q0 = allcharge[s, :]
        rmax = rvdw[s] + R2
        rminn = rvdw[s] + R1

        # Compute W integration factor
        for i in range(int(lmax[s]) + 1):
            W[i] = (1.0 / (1.0 - 2.0 * i)) * (
                rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i)
            )

        for l in range(int(lmax[s]) + 1):
            if l == 0:
                sum1 = 0.0
                for j in range(natom):
                    xyz = xyzqatom[j, :] - xyzmult[s, :]
                    sum1 += q0[j] * RSH(0, 0, 0, xyz)
                if nmid > 0:
                    for j in range(nmid):
                        xyz = xyzqmid[j, :] - xyzmult[s, :]
                        sum1 += q0[natom + j] * RSH(0, 0, 0, xyz)

                coeff = 2.0 * (4.0 * PI / (2.0 * l + 1.0)) * W[0] * (
                    multipoles[s, l, 0, 0] - sum1
                )

                for j in range(natom):
                    xyz = xyzqatom[j, :] - xyzmult[s, :]
                    dparam[s * npts + j] -= coeff * RSH(0, 0, 0, xyz)
                if nmid > 0:
                    for j in range(nmid):
                        xyz = xyzqmid[j, :] - xyzmult[s, :]
                        dparam[s * npts + natom + j] -= coeff * RSH(0, 0, 0, xyz)
            else:
                for m in range(l + 1):
                    if m == 0:
                        sum1 = 0.0
                        for j in range(natom):
                            xyz = xyzqatom[j, :] - xyzmult[s, :]
                            sum1 += q0[j] * RSH(l, 0, 0, xyz)
                        if nmid > 0:
                            for j in range(nmid):
                                xyz = xyzqmid[j, :] - xyzmult[s, :]
                                sum1 += q0[natom + j] * RSH(l, 0, 0, xyz)

                        coeff = 2.0 * (4.0 * PI / (2.0 * l + 1.0)) * W[l] * (
                            multipoles[s, l, 0, 0] - sum1
                        )

                        for j in range(natom):
                            xyz = xyzqatom[j, :] - xyzmult[s, :]
                            dparam[s * npts + j] -= coeff * RSH(l, 0, 0, xyz)
                        if nmid > 0:
                            for j in range(nmid):
                                xyz = xyzqmid[j, :] - xyzmult[s, :]
                                dparam[s * npts + natom + j] -= coeff * RSH(l, 0, 0, xyz)
                    else:
                        # Cosine (real) part
                        sum1 = 0.0
                        for j in range(natom):
                            xyz = xyzqatom[j, :] - xyzmult[s, :]
                            sum1 += q0[j] * RSH(l, m, 0, xyz)
                        if nmid > 0:
                            for j in range(nmid):
                                xyz = xyzqmid[j, :] - xyzmult[s, :]
                                sum1 += q0[natom + j] * RSH(l, m, 0, xyz)

                        coeff = 2.0 * (4.0 * PI / (2.0 * l + 1.0)) * W[l] * (
                            multipoles[s, l, m, 0] - sum1
                        )

                        for j in range(natom):
                            xyz = xyzqatom[j, :] - xyzmult[s, :]
                            dparam[s * npts + j] -= coeff * RSH(l, m, 0, xyz)
                        if nmid > 0:
                            for j in range(nmid):
                                xyz = xyzqmid[j, :] - xyzmult[s, :]
                                dparam[s * npts + natom + j] -= coeff * RSH(l, m, 0, xyz)

                        # Sine (imaginary) part
                        sum1 = 0.0
                        for j in range(natom):
                            xyz = xyzqatom[j, :] - xyzmult[s, :]
                            sum1 += q0[j] * RSH(l, m, 1, xyz)
                        if nmid > 0:
                            for j in range(nmid):
                                xyz = xyzqmid[j, :] - xyzmult[s, :]
                                sum1 += q0[natom + j] * RSH(l, m, 1, xyz)

                        coeff = 2.0 * (4.0 * PI / (2.0 * l + 1.0)) * W[l] * (
                            multipoles[s, l, m, 1] - sum1
                        )

                        for j in range(natom):
                            xyz = xyzqatom[j, :] - xyzmult[s, :]
                            dparam[s * npts + j] -= coeff * RSH(l, m, 1, xyz)
                        if nmid > 0:
                            for j in range(nmid):
                                xyz = xyzqmid[j, :] - xyzmult[s, :]
                                dparam[s * npts + natom + j] -= coeff * RSH(l, m, 1, xyz)

    # Reorder dparam: from [site, atom] indexing to [atom, site] indexing
    # dparam1[(j-1)*multsites + i] = dparam[(i-1)*npts + j]
    dparam1 = np.zeros(multsites * npts)
    for i in range(multsites):
        for j in range(npts):
            dparam1[j * multsites + i] = dparam[i * npts + j]

    # Add charge constraint gradient
    sumchg = np.sum(qstore)
    dparam1 += CONCHG * 2.0 * (sumchg - MOLECULE_CHARGE)

    # Add penalty gradient for positive midpoint charges
    if nmid > 0:
        for i in range(nmid):
            if qstore[natom + i] > 0.0:
                for j in range(multsites):
                    dparam1[(natom + i) * multsites + j] += (
                        NEGMIDE * NEGMIDC * np.exp(NEGMIDE * qstore[natom + i])
                    )

    # Apply chain rule for constraints
    return createdkaisq(dparam1)


def createdkaisq(dparam1: np.ndarray) -> np.ndarray:
    """
    Apply chain rule to convert full gradient to reduced parameter gradient.

    This handles the constraint that atoms with the same type have the same
    total charge. The gradient of the last site contribution for constrained
    atoms must be propagated to all other parameters.

    Parameters
    ----------
    dparam1 : np.ndarray
        Full gradient w.r.t. all allcharge values (atom, site ordering)

    Returns
    -------
    np.ndarray
        Gradient w.r.t. reduced parameters p0

    Notes
    -----
    Lines 674-773 in Fortran mpfitroutines.f90
    """
    atoms = len(atomtype)
    nmid = 0
    multsites = atoms

    # First: combine derivatives for atoms of same type
    # The last site contribution for constrained atoms depends on qstore[twin]
    # which depends on ALL site contributions of the twin
    dparam1_modified = dparam1.copy()

    for i in range(1, atoms):
        twin = -1
        for k in range(i):
            if atomtype[i] == atomtype[k]:
                twin = k
                break

        if twin >= 0:
            # Find the last site index for atom i
            count1 = 0
            for j in range(multsites):
                if quse[j, i] == 1:
                    count1 += 1

            # Find which site is the last one
            count2 = 1
            last_site = -1
            for j in range(multsites):
                if quse[j, i] == 1 and count2 < count1:
                    count2 += 1
                elif quse[j, i] == 1 and count2 == count1:
                    last_site = j
                    break

            if last_site >= 0:
                # Gradient of last site contribution:
                # allcharge[last_site, i] = qstore[twin] - sum(other contributions)
                # d/d(allcharge[k, twin]) = d/d(qstore[twin]) * d(qstore[twin])/d(allcharge[k, twin])
                #                         = dparam1[(i-1)*multsites + last_site] * 1
                # Also, d/d(allcharge[j, i]) for j < last_site gets a -1 factor

                # Add gradient from last site to all twin contributions
                for k in range(multsites):
                    dparam1_modified[twin * multsites + k] += dparam1_modified[
                        i * multsites + last_site
                    ]

                # Subtract gradient from last site from earlier contributions
                for k in range(last_site):
                    dparam1_modified[i * multsites + k] -= dparam1_modified[
                        i * multsites + last_site
                    ]

    # Now order derivatives into dkaisq (reduced parameter gradient)
    n_params = count_parameters()
    dkaisq = np.zeros(n_params)

    count = 0
    for i in range(atoms):
        if i == 0:
            for j in range(multsites):
                if quse[j, i] == 1:
                    dkaisq[count] = dparam1_modified[i * multsites + j]
                    count += 1
        else:
            twin = -1
            for k in range(i):
                if atomtype[i] == atomtype[k]:
                    twin = k
                    break

            if twin >= 0:
                # Constrained atom: only first (count1-1) sites are free
                count1 = 0
                for j in range(multsites):
                    if quse[j, i] == 1:
                        count1 += 1

                count2 = 1
                for j in range(multsites):
                    if quse[j, i] == 1 and count2 < count1:
                        dkaisq[count] = dparam1_modified[i * multsites + j]
                        count2 += 1
                        count += 1
            else:
                # First occurrence of this type
                for j in range(multsites):
                    if quse[j, i] == 1:
                        dkaisq[count] = dparam1_modified[i * multsites + j]
                        count += 1

    return dkaisq


def count_parameters() -> int:
    """
    Count the number of reduced parameters based on constraints.

    Returns
    -------
    int
        Number of free parameters
    """
    atoms = len(atomtype)
    multsites = xyzmult.shape[0]

    n_params = 0
    for i in range(atoms):
        # Count sites using this atom
        n_sites_using = 0
        for j in range(multsites):
            if quse[j, i] == 1:
                n_sites_using += 1

        if i == 0:
            # First atom: all sites are free parameters
            n_params += n_sites_using
        else:
            # Check if this atom has a twin
            twin = -1
            for k in range(i):
                if atomtype[i] == atomtype[k]:
                    twin = k
                    break

            if twin >= 0:
                # Constrained atom: last site is computed, not free
                n_params += n_sites_using - 1
            else:
                # First occurrence of this type
                n_params += n_sites_using

    return n_params


def verify_gradient(p0: np.ndarray, eps: float = 1e-6) -> tuple:
    """
    Verify analytical gradient against numerical finite differences.

    Parameters
    ----------
    p0 : np.ndarray
        Parameter vector
    eps : float
        Finite difference step size

    Returns
    -------
    tuple
        (analytical_gradient, numerical_gradient, max_relative_error)
    """
    analytical = dkaisq(p0)

    numerical = np.zeros_like(p0)
    for i in range(len(p0)):
        p_plus = p0.copy()
        p_minus = p0.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        numerical[i] = (kaisq(p_plus) - kaisq(p_minus)) / (2 * eps)

    # Compute relative error
    max_error = 0.0
    for i in range(len(p0)):
        if abs(numerical[i]) > 1e-10:
            rel_error = abs(analytical[i] - numerical[i]) / abs(numerical[i])
            max_error = max(max_error, rel_error)

    return analytical, numerical, max_error


# =============================================================================
# SETUP AND OPTIMIZATION
# =============================================================================


def setup_water_example():
    """
    Set up a water example with H1 ≡ H2 constraint.

    Water has 3 atoms: O, H1, H2
    The two hydrogens should have the same total charge.
    """
    global atomtype, quse, allcharge, qstore, multipoles, xyzmult, xyzcharge, lmax, rvdw

    # Atom types: O is unique, H1 and H2 are equivalent
    atomtype = ["O", "H1", "H1"]  # H2 has same type as H1

    # Coordinates in Bohr (water geometry)
    # O at origin, H1 and H2 symmetric about x-axis
    xyzcharge = np.array(
        [
            [0.0, 0.0, 0.0],  # O
            [1.43, 1.11, 0.0],  # H1
            [-1.43, 1.11, 0.0],  # H2
        ]
    )

    # Multipole sites are at atom positions
    xyzmult = xyzcharge.copy()

    n_atoms = 3
    n_sites = 3

    # quse matrix: which atoms affect which multipole site
    # For simplicity, use rvdw-based selection
    rvdw = np.array([3.0, 2.27, 2.27])  # O heavier, H lighter

    quse = np.zeros((n_sites, n_atoms), dtype=int)
    for i in range(n_sites):
        for j in range(n_atoms):
            rqm = np.linalg.norm(xyzmult[i] - xyzcharge[j])
            if rqm < rvdw[i]:
                quse[i, j] = 1

    # For this simple example, all atoms affect all sites
    quse = np.ones((n_sites, n_atoms), dtype=int)

    # Initialize allcharge and qstore
    allcharge = np.zeros((n_sites, n_atoms))
    qstore = np.zeros(n_atoms)

    # Maximum multipole rank per site (use rank 2 = quadrupole)
    lmax = np.array([2, 2, 2], dtype=float)

    # Example multipole moments (simplified)
    # In real use, these would come from GDMA
    multipoles = np.zeros((n_sites, MAXL + 1, MAXL + 1, 2))

    # O site: negative charge
    multipoles[0, 0, 0, 0] = -0.8  # Monopole
    multipoles[0, 1, 0, 0] = 0.0  # Dipole z
    multipoles[0, 1, 1, 0] = 0.0  # Dipole x
    multipoles[0, 1, 1, 1] = -0.2  # Dipole y

    # H1 site: positive charge
    multipoles[1, 0, 0, 0] = 0.4  # Monopole

    # H2 site: positive charge (same as H1)
    multipoles[2, 0, 0, 0] = 0.4  # Monopole


def initial_guess() -> np.ndarray:
    """
    Create initial guess for reduced parameters.

    Returns
    -------
    np.ndarray
        Initial parameter vector
    """
    n_params = count_parameters()
    return np.zeros(n_params)


def optimize_constrained(p0_init: np.ndarray, verbose: bool = True) -> dict:
    """
    Optimize charges using scipy's CG method.

    Parameters
    ----------
    p0_init : np.ndarray
        Initial parameter vector
    verbose : bool
        Whether to print progress

    Returns
    -------
    dict
        Optimization result
    """
    if verbose:
        print(f"Starting optimization with {len(p0_init)} parameters")
        print(f"Initial loss: {kaisq(p0_init):.6e}")

    result = minimize(
        kaisq,
        p0_init,
        jac=dkaisq,
        method="CG",
        options={"maxiter": 10000, "gtol": 1e-15, "disp": verbose},
    )

    if verbose:
        print(f"\nOptimization finished: {result.message}")
        print(f"Final loss: {result.fun:.6e}")
        print(f"Iterations: {result.nit}")

    return result


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Constrained MPFIT Minimal Example")
    print("=" * 60)

    # Set up water example
    setup_water_example()

    print("\n--- System Setup ---")
    print(f"Atom types: {atomtype}")
    print(f"Number of atoms: {len(atomtype)}")
    print(f"Number of sites: {xyzmult.shape[0]}")
    print(f"quse matrix:\n{quse}")

    # Count parameters
    n_full = np.sum(quse)
    n_reduced = count_parameters()
    print(f"\nFull parameters: {n_full}")
    print(f"Reduced parameters: {n_reduced}")
    print(f"Reduction: {n_full - n_reduced} parameters constrained")

    # Get initial guess
    p0 = initial_guess()

    # Verify gradient
    print("\n--- Gradient Verification ---")
    _, _, max_error = verify_gradient(p0 + np.random.randn(len(p0)) * 0.1)
    print(f"Max relative error: {max_error:.2e}")
    if max_error < 1e-4:
        print("Gradient check: PASS")
    else:
        print("Gradient check: FAIL")

    # Optimize
    print("\n--- Optimization ---")
    result = optimize_constrained(p0, verbose=True)

    # Expand final parameters and check constraints
    expandcharge(result.x)

    print("\n--- Final Results ---")
    print(f"qstore (total charge per atom):")
    for i, (atype, q) in enumerate(zip(atomtype, qstore)):
        print(f"  Atom {i} ({atype}): {q:.6f}")

    print(f"\nConstraint verification:")
    print(f"  qstore[H1] = {qstore[1]:.6f}")
    print(f"  qstore[H2] = {qstore[2]:.6f}")
    print(f"  Difference: {abs(qstore[1] - qstore[2]):.2e}")
    if np.isclose(qstore[1], qstore[2], rtol=1e-6):
        print("  Constraint satisfied: PASS")
    else:
        print("  Constraint satisfied: FAIL")

    print(f"\nTotal molecular charge: {np.sum(qstore):.6f}")

    print("\n--- Per-site Charge Contributions ---")
    print("allcharge[site, atom]:")
    print(f"{'':>10} ", end="")
    for i, atype in enumerate(atomtype):
        print(f"{atype:>10}", end="")
    print()
    for s in range(xyzmult.shape[0]):
        print(f"Site {s}:   ", end="")
        for a in range(len(atomtype)):
            print(f"{allcharge[s, a]:>10.4f}", end="")
        print()
