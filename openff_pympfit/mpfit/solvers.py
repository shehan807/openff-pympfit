"""Solvers for the MPFIT charge fitting problem."""

import abc

import numpy as np


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
            This controls numerical stability for ill-conditioned systems.
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

        Raises
        ------
        MPFITSolverError
            If quse_masks is not provided in ancillary_arrays.
        """
        # Handle empty design matrices
        is_object_array = (
            hasattr(design_matrix, "dtype") and design_matrix.dtype == np.dtype("O")
        )
        if is_object_array and len(design_matrix) == 0:
            return np.zeros((0, 1))

        # Check for required quse_masks
        if ancillary_arrays is None or "quse_masks" not in ancillary_arrays:
            raise MPFITSolverError("SVD solver requires quse_masks in ancillary_arrays")

        quse_masks = ancillary_arrays["quse_masks"]

        # Initialize charge values
        n_atoms = len(design_matrix)
        charge_values = np.zeros((n_atoms, 1))

        # Solve for each multipole site
        for i in range(len(design_matrix)):
            site_A = design_matrix[i]
            site_b = reference_values[i]
            quse_mask = np.asarray(quse_masks[i], dtype=bool)

            # Apply SVD to solve the system
            U, S, Vh = np.linalg.svd(site_A, full_matrices=True)

            # Apply threshold to singular values
            S[self._svd_threshold > S] = 0.0

            # Compute pseudo-inverse
            inv_S = np.zeros_like(S)
            mask_S = S != 0
            inv_S[mask_S] = 1.0 / S[mask_S]

            # Solve for charges using SVD
            q = (Vh.T * inv_S) @ (U.T @ site_b)

            # Add the charges to the appropriate atoms using the quse_mask
            charge_values[quse_mask, 0] += q.flatten()

        return charge_values
