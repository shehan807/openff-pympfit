"""Different solvers for solving the MPFIT charge fitting problem."""

import abc
import functools
import warnings
from typing import Literal, cast

import numpy as np


class MPFITSolverError(Exception):
    """An exception raised when an MPFIT solver fails to converge."""


class MPFITSolver(abc.ABC):
    """Base class for finding charges that fit distributed multipole data."""

    @classmethod
    def loss(
        cls,
        charges: np.ndarray,
        design_matrix: np.ndarray,
        reference_values: np.ndarray,
        _constraint_matrix: np.ndarray,
    ) -> np.ndarray:
        """Return the current value of the loss function.

        Parameters
        ----------
        charges
            The current vector of charge values with shape=(n_values,)
        design_matrix
            The design matrix that when right multiplied by ``charges`` yields the
            multipoles due to the charges applied to the molecule of interest with
            shape=(n_grid_points, n_values)
        reference_values
            The reference multipole values with shape=(n_grid_points, 1).
        _constraint_matrix
            A matrix that when right multiplied by the vector of charge values should
            yield a vector that is equal to ``constraint_values`` with
            shape=(n_constraints, n_values).

        Returns
        -------
            The value of the loss function with shape=(1,)
        """
        charges = charges.reshape(-1, 1)

        delta = design_matrix @ charges - reference_values
        return (delta * delta).sum()

    @classmethod
    def jacobian(
        cls,
        charges: np.ndarray,
        design_matrix: np.ndarray,
        reference_values: np.ndarray,
        _constraint_matrix: np.ndarray,
    ) -> np.ndarray:
        """Return the jacobian of the loss function with respect to ``charges``.

        Parameters
        ----------
        charges
            The current vector of charge values with shape=(n_values,)
        design_matrix
            The design matrix that when right multiplied by ``charges`` yields the
            multipoles due to the charges applied to the molecule of interest with
            shape=(n_grid_points, n_values)
        reference_values
            The reference multipole values with shape=(n_grid_points, 1).
        _constraint_matrix
            A matrix that when right multiplied by the vector of charge values should
            yield a vector that is equal to ``constraint_values`` with
            shape=(n_constraints, n_values).

        Returns
        -------
            The value of the jacobian with shape=(n_values,)
        """
        charges = charges.reshape(-1, 1)

        delta = design_matrix @ charges - reference_values
        d_chi_squared = 2.0 * design_matrix.T @ delta

        return d_chi_squared.flatten()

    @classmethod
    def initial_guess(
        cls,
        design_matrix: np.ndarray,
        reference_values: np.ndarray,
        constraint_matrix: np.ndarray,
        constraint_values: np.ndarray,
    ) -> np.ndarray:
        """Compute an initial guess of the charge values.

        Solves the lagrangian constrained ``Ax + b`` equations.

        Parameters
        ----------
        design_matrix
            The design matrix that when right multiplied by ``charges`` yields the
            multipoles due to the charges applied to the molecule of interest with
            shape=(n_grid_points, n_values)
        reference_values
            The reference multipole values with shape=(n_grid_points, 1).
        constraint_matrix
            A matrix that when right multiplied by the vector of charge values should
            yield a vector that is equal to ``constraint_values`` with
            shape=(n_constraints, n_values).
        constraint_values
            The expected values of the constraints with shape=(n_constraints, 1)

        Returns
        -------
            An initial guess of the charge values with shape=(n_values, 1)
        """
        a_matrix = np.block(
            [
                [design_matrix.T @ design_matrix, constraint_matrix.T],
                [constraint_matrix, np.zeros([constraint_matrix.shape[0]] * 2)],
            ]
        )

        b_vector = np.vstack([design_matrix.T @ reference_values, constraint_values])

        initial_values, *_ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
        return initial_values[: design_matrix.shape[1]]

    @abc.abstractmethod
    def _solve(
        self,
        design_matrix: np.ndarray,
        reference_values: np.ndarray,
        constraint_matrix: np.ndarray,
        constraint_values: np.ndarray,
        ancillary_arrays: dict = None,
    ) -> np.ndarray:
        """Implement ``solve`` method.

        Parameters
        ----------
        design_matrix
            The design matrix that when right multiplied by ``charges`` yields the
            multipoles due to the charges applied to the molecule of interest with
            shape=(n_grid_points, n_values)
        reference_values
            The reference multipole values with shape=(n_grid_points, 1).
        constraint_matrix
            A matrix that when right multiplied by the vector of charge values should
            yield a vector that is equal to ``constraint_values`` with
            shape=(n_constraints, n_values).
        constraint_values
            The expected values of the constraints with shape=(n_constraints, 1)
        ancillary_arrays
            Additional arrays needed by specific solvers
            (e.g., quse_masks for SVD solver)

        Raises
        ------
        MPFITSolverError

        Returns
        -------
            The set of charge values that minimize the loss function with
            shape=(n_values, 1)
        """
        raise NotImplementedError

    def solve(
        self,
        design_matrix: np.ndarray,
        reference_values: np.ndarray,
        constraint_matrix: np.ndarray,
        constraint_values: np.ndarray,
        ancillary_arrays: dict = None,
    ) -> np.ndarray:
        """Find a minimum solution to the MPFIT loss function.

        Parameters
        ----------
        design_matrix
            The design matrix that when right multiplied by ``charges`` yields the
            multipoles due to the charges applied to the molecule of interest.
            Can be a regular matrix or an object array of site-specific matrices.
        reference_values
            The reference multipole values. Can be a regular vector or an object array.
        constraint_matrix
            A matrix that when right multiplied by the vector of charge values should
            yield a vector that is equal to ``constraint_values`` with
            shape=(n_constraints, n_values).
        constraint_values
            The expected values of the constraints with shape=(n_constraints, 1)
        ancillary_arrays
            Additional arrays needed by specific solvers
            (e.g., quse_masks for SVD solver)

        Raises
        ------
        MPFITSolverError

        Returns
        -------
            The set of charge values that minimize the loss function with
            shape=(n_values, 1)
        """
        # Handle object arrays for new MPFIT implementation
        is_object_array = hasattr(
            design_matrix, "dtype"
        ) and design_matrix.dtype == np.dtype("O")
        if is_object_array and len(design_matrix) == 0:
            return np.zeros((0, 1))
        if (
            not is_object_array
            and hasattr(design_matrix, "shape")
            and len(design_matrix.shape) > 1
            and design_matrix.shape[1] == 0
        ):
            return np.zeros((0, 1))

        solution = self._solve(
            design_matrix,
            reference_values,
            constraint_matrix,
            constraint_values,
            ancillary_arrays,
        )

        try:
            # Note: This might fail if constraints are incompatible with the
            # SVD solution. This is expected when using non-symmetrized charges.
            predicted_total_charge = constraint_matrix @ solution
            if predicted_total_charge.shape != constraint_values.shape:
                msg = (
                    f"Shape mismatch: {predicted_total_charge.shape} vs "
                    f"{constraint_values.shape}"
                )
                raise ValueError(msg)

            if not np.allclose(predicted_total_charge, constraint_values):
                msg = "The total charge was not conserved by the solver"
                raise MPFITSolverError(msg)
        except (ValueError, MPFITSolverError, TypeError) as e:
            warnings.warn(
                f"Could not verify constraint satisfaction: {e!s}. "
                "Constraint enforcement is not yet fully implemented for MPFIT.",
                UserWarning,
                stacklevel=2,
            )

        return cast(np.ndarray, solution)


class IterativeSolver(MPFITSolver):
    """Find charges that minimize MPFIT loss by repeated least-squares."""

    @classmethod
    def _solve_iteration(
        cls,
        charges: np.ndarray,
        design_matrix: np.ndarray,
        reference_values: np.ndarray,
        constraint_matrix: np.ndarray,
        constraint_values: np.ndarray,
    ) -> np.ndarray:
        charges = charges.reshape(-1, 1)

        a_matrix = np.block(
            [
                [design_matrix.T @ design_matrix, constraint_matrix.T],
                [constraint_matrix, np.zeros([constraint_matrix.shape[0]] * 2)],
            ]
        )

        b_vector = np.vstack([design_matrix.T @ reference_values, constraint_values])

        new_charges, *_ = np.linalg.lstsq(a_matrix, b_vector, rcond=None)
        return new_charges[: design_matrix.shape[1]]

    def _solve(
        self,
        design_matrix: np.ndarray,
        reference_values: np.ndarray,
        constraint_matrix: np.ndarray,
        constraint_values: np.ndarray,
        ancillary_arrays: dict = None,
    ) -> np.ndarray:
        initial_guess = self.initial_guess(
            design_matrix,
            reference_values,
            constraint_matrix,
            constraint_values,
        )

        iteration = 0
        tolerance = 1.0e-5

        current_charges = initial_guess
        while iteration < 300:
            new_charges = self._solve_iteration(
                current_charges,
                design_matrix,
                reference_values,
                constraint_matrix,
                constraint_values,
            )

            charge_difference = new_charges - current_charges

            if np.linalg.norm(charge_difference) / len(new_charges) < tolerance:
                return new_charges

            current_charges = new_charges
            iteration += 1

        raise MPFITSolverError(
            "The iterative solver failed to converge after 300 iterations"
        )


class SciPySolver(MPFITSolver):
    """Find charges that minimize MPFIT loss using scipy.optimize.minimize."""

    def __init__(self, method: Literal["SLSQP", "trust-constr"] = "SLSQP") -> None:
        """Initialize the solver.

        Parameters
        ----------
        method
            The minimizer to use.
        """
        if method not in {"SLSQP", "trust-constr"}:
            msg = f"method must be 'SLSQP' or 'trust-constr', got {method}"
            raise ValueError(msg)
        self._method = method

    def _solve(
        self,
        design_matrix: np.ndarray,
        reference_values: np.ndarray,
        constraint_matrix: np.ndarray,
        constraint_values: np.ndarray,
        ancillary_arrays: dict = None,
    ) -> np.ndarray:
        from scipy.optimize import LinearConstraint, minimize

        loss_function = functools.partial(
            self.loss,
            design_matrix=design_matrix,
            reference_values=reference_values,
            constraint_matrix=constraint_matrix,
        )
        jacobian_function = functools.partial(
            self.jacobian,
            design_matrix=design_matrix,
            reference_values=reference_values,
            constraint_matrix=constraint_matrix,
        )

        initial_guess = self.initial_guess(
            design_matrix,
            reference_values,
            constraint_matrix,
            constraint_values,
        )

        # noinspection PyTypeChecker
        output = minimize(
            fun=loss_function,
            x0=initial_guess.flatten(),
            jac=jacobian_function,
            constraints=(
                LinearConstraint(
                    constraint_matrix,
                    constraint_values.flatten(),
                    constraint_values.flatten(),
                )
                if len(constraint_matrix) > 0
                else ()
            ),
            method=self._method,
            tol=1.0e-5,
        )

        if not output.success:
            raise MPFITSolverError(
                f"SciPy solver with method={self._method} was unsuccessful: "
                f"{output.message}"
            )

        return output.x.reshape(-1, 1)


class MPFITSVDSolver(MPFITSolver):
    """Solver that uses SVD to find charges that reproduce multipole moments."""

    def __init__(self, svd_threshold: float = 1.0e-4) -> None:
        """Initialize the SVD solver.

        Parameters
        ----------
        svd_threshold
            The threshold below which singular values are considered zero.
        """
        self._svd_threshold = svd_threshold

    def _solve(
        self,
        design_matrix: np.ndarray,
        reference_values: np.ndarray,
        constraint_matrix: np.ndarray,
        constraint_values: np.ndarray,
        ancillary_arrays: dict = None,
    ) -> np.ndarray:
        """Solve for charges using SVD for each multipole site.

        Parameters
        ----------
        design_matrix
            An object array where each element is a site-specific A matrix
        reference_values
            An object array where each element is a site-specific b vector
        constraint_matrix
            Charge constraint matrix
        constraint_values
            Constraint values (e.g., total charge)
        ancillary_arrays
            Dictionary containing additional arrays needed for the solver,
            must include 'quse_masks' - list of boolean masks indicating which atoms
            affect each multipole site

        Returns
        -------
            Charge values that reproduce the multipole moments
        """
        # Check if we have quse_masks
        if ancillary_arrays is None or "quse_masks" not in ancillary_arrays:
            raise MPFITSolverError("SVD solver requires quse_masks in ancillary_arrays")

        quse_masks = ancillary_arrays["quse_masks"]

        # Initialize charge values based on the design matrix dimensions
        n_atoms = len(design_matrix)
        charge_values = np.zeros((n_atoms, 1))

        # Issue warning if constraint matrix doesn't match design matrix dimensions
        if constraint_matrix.shape[1] != n_atoms:
            warnings.warn(
                f"Constraint matrix dimensions ({constraint_matrix.shape[1]}) "
                f"don't match the number of atoms/sites ({n_atoms}). "
                "Constraint enforcement with library charges is not yet fully "
                "implemented for MPFIT.",
                UserWarning,
                stacklevel=2,
            )

        # For each multipole site
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
            # quse_mask should already be a boolean array from earlier conversion
            charge_values[quse_mask, 0] += q.flatten()

        return charge_values
