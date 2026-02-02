"""Solvers for the MPFIT charge fitting problem."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


class MPFITSolverError(Exception):
    """An exception raised when an MPFIT solver fails."""


class MPFITSolver(abc.ABC):
    """Base class for MPFIT solvers.

    MPFIT uses per-site fitting where each atom has its own design matrix (A)
    and reference vector (b). The solver receives object arrays containing
    these per-site matrices and must solve each site independently.
    """

    @abc.abstractmethod
    def solve(
        self,
        design_matrix: np.ndarray,
        reference_values: np.ndarray,
        ancillary_arrays: dict | None = None,
    ) -> np.ndarray:
        """Solve the MPFIT fitting problem.

        Parameters
        ----------
        design_matrix
            An object array where each element is a site-specific A matrix.
        reference_values
            An object array where each element is a site-specific b vector.
        ancillary_arrays
            Dictionary containing additional arrays needed for the solver.
            For SVD solver, must include 'quse_masks' - list of boolean masks
            indicating which atoms affect each multipole site.

        Returns
        -------
            The set of charge values with shape=(n_atoms, 1)
        """
        raise NotImplementedError


class MPFITSVDSolver(MPFITSolver):
    """Solver that uses SVD to find charges that reproduce multipole moments.

    This solver processes each multipole site independently using SVD
    decomposition, accumulating charge contributions via quse_masks.
    """

    def __init__(self, svd_threshold: float = 1.0e-4) -> None:
        """Initialize the SVD solver.

        Parameters
        ----------
        svd_threshold
            The threshold below which singular values are considered zero.
        """
        self._svd_threshold = svd_threshold

    def solve(
        self,
        design_matrix: np.ndarray,
        reference_values: np.ndarray,
        ancillary_arrays: dict | None = None,
    ) -> np.ndarray:
        """Solve for charges using SVD for each multipole site.

        Parameters
        ----------
        design_matrix
            An object array where each element is a site-specific A matrix.
        reference_values
            An object array where each element is a site-specific b vector.
        ancillary_arrays
            Dictionary containing 'quse_masks' - list of boolean masks
            indicating which atoms affect each multipole site.

        Returns
        -------
            Charge values that reproduce the multipole moments.
        """
        is_object_array = hasattr(
            design_matrix, "dtype"
        ) and design_matrix.dtype == np.dtype("O")
        if is_object_array and len(design_matrix) == 0:
            return np.zeros((0, 1))

        if ancillary_arrays is None or "quse_masks" not in ancillary_arrays:
            raise MPFITSolverError("SVD solver requires quse_masks in ancillary_arrays")

        quse_masks = ancillary_arrays["quse_masks"]

        n_atoms = len(design_matrix)
        charge_values = np.zeros((n_atoms, 1))

        # Solve for each multipole site
        for i in range(len(design_matrix)):
            site_A = np.asarray(design_matrix[i], dtype=np.float64)
            site_b = np.asarray(reference_values[i], dtype=np.float64)
            quse_mask = np.asarray(quse_masks[i], dtype=bool)

            U, S, Vh = np.linalg.svd(site_A)

            S[self._svd_threshold > S] = 0.0

            inv_S = np.zeros_like(S)
            mask_S = S != 0
            inv_S[mask_S] = 1.0 / S[mask_S]

            q = (Vh.T * inv_S) @ (U.T @ site_b)

            charge_values[quse_mask, 0] += q.flatten()

        return charge_values


@dataclass(frozen=True)
class ConstrainedMPFITState:
    """Immutable container for a constrained MPFIT problem."""

    xyzmult: np.ndarray
    xyzcharge: np.ndarray
    multipoles: np.ndarray
    quse: np.ndarray
    atomtype: tuple[str, ...]
    rvdw: np.ndarray
    lmax: np.ndarray
    r1: float
    r2: float
    maxl: int
    atom_counts: tuple[int, ...]
    molecule_charges: tuple[float, ...]


def build_quse_matrix(
    xyzmult: np.ndarray,
    xyzcharge: np.ndarray,
    rvdw: np.ndarray,
) -> np.ndarray:
    """Build binary mask: ``quse[s, i] = 1`` if atom *i* affects site *s*."""
    return (cdist(xyzmult, xyzcharge) < rvdw[:, None]).astype(int)


def _find_twin(atomtype: tuple[str, ...], i: int) -> int | None:
    """Return the index of the first atom with the same type as atom *i*, or None."""
    return next((k for k in range(i) if atomtype[k] == atomtype[i]), None)


def count_parameters(state: ConstrainedMPFITState) -> int:
    """Count free parameters after applying atom-type equivalence constraints."""
    atomtype = state.atomtype
    quse = state.quse
    n_params = 0

    for i in range(len(atomtype)):
        n_sites_using = int(np.sum(quse[:, i]))
        twin = _find_twin(atomtype, i) if i > 0 else None
        if twin is not None:
            n_params += n_sites_using - 1
        else:
            n_params += n_sites_using

    return n_params


def expandcharge(
    p0: np.ndarray,
    state: ConstrainedMPFITState,
) -> tuple[np.ndarray, np.ndarray]:
    """Map reduced parameters to full charges with atom-type constraints.

    Returns
    -------
    allcharge
        Per-site charge contributions, shape ``(n_sites, n_atoms)``.
    qstore
        Total charge per atom, shape ``(n_atoms,)``.
    """
    atomtype = state.atomtype
    quse = state.quse
    n_sites = state.xyzmult.shape[0]
    n_atoms = len(atomtype)

    allcharge = np.zeros((n_sites, n_atoms))
    qstore = np.zeros(n_atoms)
    count = 0

    for i in range(n_atoms):
        twin = _find_twin(atomtype, i) if i > 0 else None
        charge_sum = 0.0

        if twin is not None:
            count1 = int(np.sum(quse[:, i]))
            count2 = 1
            for j in range(n_sites):
                if quse[j, i] == 1 and count2 < count1:
                    allcharge[j, i] = p0[count]
                    charge_sum += p0[count]
                    count += 1
                    count2 += 1
                elif quse[j, i] == 1 and count2 == count1:
                    allcharge[j, i] = qstore[twin] - charge_sum
                    qstore[i] = qstore[twin]
        else:
            for j in range(n_sites):
                if quse[j, i] == 1:
                    allcharge[j, i] = p0[count]
                    charge_sum += p0[count]
                    count += 1
            qstore[i] = charge_sum

    return allcharge, qstore


def createdkaisq(dparam1: np.ndarray, state: ConstrainedMPFITState) -> np.ndarray:
    """Project full-space gradient to reduced parameter space."""
    atomtype = state.atomtype
    quse = state.quse
    n_atoms = len(atomtype)
    n_sites = n_atoms
    dparam1 = dparam1.copy()

    # Pass 1: redistribute constrained site's gradient to twin
    for i in range(1, n_atoms):
        twin = _find_twin(atomtype, i)
        if twin is not None:
            count1 = int(np.sum(quse[:, i]))
            count2 = 1
            for j in range(n_sites):
                if quse[j, i] == 1 and count2 < count1:
                    count2 += 1
                elif quse[j, i] == 1 and count2 == count1:
                    for k in range(n_sites):
                        dparam1[twin * n_sites + k] += dparam1[i * n_sites + j]
                    for k in range(j):
                        dparam1[i * n_sites + k] -= dparam1[i * n_sites + j]

    # Pass 2: extract free-parameter gradient entries
    n_params = count_parameters(state)
    dkaisq_out = np.zeros(n_params)
    count = 0

    for i in range(n_atoms):
        twin = _find_twin(atomtype, i) if i > 0 else None

        if twin is not None:
            count1 = int(np.sum(quse[:, i]))
            count2 = 1
            for j in range(n_sites):
                if quse[j, i] == 1 and count2 < count1:
                    dkaisq_out[count] = dparam1[i * n_sites + j]
                    count2 += 1
                    count += 1
        else:
            for j in range(n_sites):
                if quse[j, i] == 1:
                    dkaisq_out[count] = dparam1[i * n_sites + j]
                    count += 1

    return dkaisq_out


class ConstrainedMPFITSolver(abc.ABC):
    """Base class for constrained MPFIT solvers."""

    def __init__(
        self,
        conchg: float = 1.0,
        method: str = "L-BFGS-B",
        maxiter: int = 1000,
        ftol: float = 1e-8,
        gtol: float = 1e-6,
    ) -> None:
        self._conchg = conchg
        self._method = method
        self._maxiter = maxiter
        self._ftol = ftol
        self._gtol = gtol

    @classmethod
    def loss(
        cls,
        p0: np.ndarray,
        state: ConstrainedMPFITState,
        conchg: float,
    ) -> float:
        """Return the value of the constrained MPFIT objective function.

        Parameters
        ----------
        p0
            The current reduced parameter vector with shape=(n_params,).
            Mapped to full charges via ``expandcharge``.
        state
            The frozen problem state containing coordinates, multipoles,
            equivalence constraints, and per-molecule target charges.
        conchg
            The Lagrange-like penalty weight enforcing per-molecule charge
            conservation.

        Returns
        -------
            The scalar value of the objective function.
        """
        from pympfit.mpfit.core import _regular_solid_harmonic

        allcharge, qstore = expandcharge(p0, state)

        n_sites = state.xyzmult.shape[0]
        maxl = state.maxl
        sumkai = 0.0

        for s in range(n_sites):
            q0 = allcharge[s, :]
            rmax = state.rvdw[s] + state.r2
            rminn = state.rvdw[s] + state.r1

            lmax_s = int(state.lmax[s])
            W = np.zeros(maxl + 1)
            for i in range(lmax_s + 1):
                W[i] = (1.0 / (1.0 - 2.0 * i)) * (
                    rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i)
                )

            dx = state.xyzcharge[:, 0] - state.xyzmult[s, 0]
            dy = state.xyzcharge[:, 1] - state.xyzmult[s, 1]
            dz = state.xyzcharge[:, 2] - state.xyzmult[s, 2]

            site_sum = 0.0
            for l in range(lmax_s + 1):
                weight = (4.0 * np.pi) if l == 0 else (4.0 * np.pi / (2.0 * l + 1.0))
                weight *= W[l]

                for m in range(l + 1):
                    cs_range = [0] if m == 0 else [0, 1]
                    for cs in cs_range:
                        rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz)
                        sum1 = np.dot(q0, rsh_vals)
                        site_sum += weight * (state.multipoles[s, l, m, cs] - sum1) ** 2

            sumkai += site_sum

        # Per-molecule charge conservation penalty
        sumcon = 0.0
        offset = 0
        for n_i, q_target in zip(
            state.atom_counts, state.molecule_charges, strict=False
        ):
            mol_q = np.sum(qstore[offset : offset + n_i])
            sumcon += conchg * (mol_q - q_target) ** 2
            offset += n_i

        return sumkai + sumcon

    @classmethod
    def jacobian(
        cls,
        p0: np.ndarray,
        state: ConstrainedMPFITState,
        conchg: float,
    ) -> np.ndarray:
        """Return the gradient of the objective function w.r.t. reduced parameters.

        Computes the full-space gradient of the multipole fitting error and
        charge conservation penalty, then projects to the reduced parameter
        space via ``createdkaisq``.

        This is a pure function — it does not mutate ``state``.

        Parameters
        ----------
        p0
            The current reduced parameter vector with shape=(n_params,).
            Mapped to full charges via ``expandcharge``.
        state
            The frozen problem state containing coordinates, multipoles,
            equivalence constraints, and per-molecule target charges.
        conchg
            The Lagrange-like penalty weight enforcing per-molecule charge
            conservation.

        Returns
        -------
            The gradient in the reduced parameter space with shape=(n_params,).
        """
        from pympfit.mpfit.core import _regular_solid_harmonic

        allcharge, qstore = expandcharge(p0, state)

        n_sites = state.xyzmult.shape[0]
        n_atoms = n_sites
        maxl = state.maxl
        dparam = np.zeros((n_sites, n_atoms))

        for s in range(n_sites):
            q0 = allcharge[s, :]
            rmax = state.rvdw[s] + state.r2
            rminn = state.rvdw[s] + state.r1

            lmax_s = int(state.lmax[s])
            W = np.zeros(maxl + 1)
            for i in range(lmax_s + 1):
                W[i] = (1.0 / (1.0 - 2.0 * i)) * (
                    rmax ** (1 - 2 * i) - rminn ** (1 - 2 * i)
                )

            dx = state.xyzcharge[:, 0] - state.xyzmult[s, 0]
            dy = state.xyzcharge[:, 1] - state.xyzmult[s, 1]
            dz = state.xyzcharge[:, 2] - state.xyzmult[s, 2]

            for l in range(lmax_s + 1):
                weight = (4.0 * np.pi) if l == 0 else (4.0 * np.pi / (2.0 * l + 1.0))
                weight *= W[l]

                for m in range(l + 1):
                    cs_range = [0] if m == 0 else [0, 1]
                    for cs in cs_range:
                        rsh_vals = _regular_solid_harmonic(l, m, cs, dx, dy, dz)
                        sum1 = np.dot(q0, rsh_vals)
                        coeff = 2.0 * weight * (state.multipoles[s, l, m, cs] - sum1)
                        dparam[s, :] -= coeff * rsh_vals

        # Flatten to atom-major order: dparam1[atom_i * n_sites + site_s]
        dparam1 = dparam.T.flatten()

        # Per-molecule charge conservation gradient
        offset = 0
        for n_i, q_target in zip(
            state.atom_counts, state.molecule_charges, strict=False
        ):
            mol_q = np.sum(qstore[offset : offset + n_i])
            grad_val = conchg * 2.0 * (mol_q - q_target)
            for a in range(offset, offset + n_i):
                dparam1[a * n_sites : a * n_sites + n_sites] += grad_val
            offset += n_i

        return createdkaisq(dparam1, state)

    def initial_guess(self, state: ConstrainedMPFITState) -> np.ndarray:
        """Return a zero initial guess with shape=(n_params,)."""
        return np.zeros(count_parameters(state))

    def solve(self, state: ConstrainedMPFITState) -> np.ndarray:
        """Solve the constrained MPFIT problem and return total charge per atom.

        Parameters
        ----------
        state
            The frozen problem state containing coordinates, multipoles,
            equivalence constraints, and per-molecule target charges.

        Returns
        -------
            Total charge per atom with shape=(n_atoms,).
        """
        p_opt = self._solve(state)
        _, qstore = expandcharge(p_opt, state)
        return qstore

    @abc.abstractmethod
    def _solve(self, state: ConstrainedMPFITState) -> np.ndarray:
        """Implement the optimizer-specific solving logic."""
        ...


class ConstrainedSciPySolver(ConstrainedMPFITSolver):
    """Constrained MPFIT solver using ``scipy.optimize.minimize``."""

    def _solve(self, state: ConstrainedMPFITState) -> np.ndarray:
        from scipy.optimize import minimize

        result = minimize(
            fun=lambda p: self.loss(p, state, self._conchg),
            x0=self.initial_guess(state),
            jac=lambda p: self.jacobian(p, state, self._conchg),
            method=self._method,
            options={
                "maxiter": self._maxiter,
                "ftol": self._ftol,
                "gtol": self._gtol,
            },
        )

        if not result.success:
            raise MPFITSolverError(f"Optimization failed: {result.message}")

        return result.x
