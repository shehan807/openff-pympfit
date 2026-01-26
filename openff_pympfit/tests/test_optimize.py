"""Tests for the MPFIT objective functions.

Ref: fork/_tests/optimize/test_optimize.py (objective term tests)
Pattern: Test objective term creation, combination, design matrix structure

Priority 13: MPFITObjectiveTerm (2 tests)
Priority 14: MPFITObjective.compute_objective_terms (3 tests)
"""

# =============================================================================
# FIRST PASS: test_compute_mpfit_objective_terms
# =============================================================================
# This is the MPFIT analog to test_generate_resp_systems_of_equations from:
#   openff-recharge fork/_tests/charges/resp/test_resp.py
#
# Placement rationale: This test lives in test_optimize.py (not test_mpfit.py)
# because it tests MPFITObjective.compute_objective_terms(), which is defined
# in openff_pympfit/optimize/_optimize.py. The test validates the creation of
# per-site design matrices (A) and reference vectors (b) from GDMA records,
# which is the core optimization infrastructure rather than the high-level
# charge generation API tested in test_mpfit.py.
#
# Key differences from RESP:
# - RESP has a single global design matrix; MPFIT has per-site A matrices
# - RESP includes constraint matrices for charge equivalence; MPFIT uses quse_masks
# - MPFIT returns object arrays where each element is a site-specific matrix/vector
#
# import numpy as np
# import pytest
# from openff.recharge.charges.library import (
#     LibraryChargeCollection,
#     LibraryChargeParameter,
# )
#
# from openff_pympfit.optimize import MPFITObjective, MPFITObjectiveTerm
#
#
# class TestMPFITObjective:
#     """Test MPFITObjective.compute_objective_terms."""
#
#     def test_compute_terms_single_record(self, mock_gdma_record_water):
#         """Test computing objective terms for a single GDMA record."""
#         parameter = LibraryChargeParameter(
#             smiles="[H:1][O:2][H:3]",
#             value=[0.0, 0.0, 0.0],
#             provenance={},
#         )
#         charge_collection = LibraryChargeCollection(parameters=[parameter])
#         charge_parameter_keys = [(parameter.smiles, (0, 1, 2))]
#
#         terms = list(
#             MPFITObjective.compute_objective_terms(
#                 [mock_gdma_record_water],
#                 charge_collection=charge_collection,
#                 charge_parameter_keys=charge_parameter_keys,
#             )
#         )
#
#         assert len(terms) == 1
#         term = terms[0]
#         assert isinstance(term, MPFITObjectiveTerm)
#         assert term.atom_charge_design_matrix is not None
#         assert term.reference_values is not None
#
#     def test_compute_terms_returns_quse_masks(self, mock_gdma_record_water):
#         """Test that return_quse_masks=True includes masks in output."""
#         parameter = LibraryChargeParameter(
#             smiles="[H:1][O:2][H:3]",
#             value=[0.0, 0.0, 0.0],
#             provenance={},
#         )
#         charge_collection = LibraryChargeCollection(parameters=[parameter])
#         charge_parameter_keys = [(parameter.smiles, (0, 1, 2))]
#
#         terms_with_masks = list(
#             MPFITObjective.compute_objective_terms(
#                 [mock_gdma_record_water],
#                 charge_collection=charge_collection,
#                 charge_parameter_keys=charge_parameter_keys,
#                 return_quse_masks=True,
#             )
#         )
#
#         assert len(terms_with_masks) == 1
#         term, mask_dict = terms_with_masks[0]
#         assert isinstance(term, MPFITObjectiveTerm)
#         assert "quse_masks" in mask_dict
#         assert len(mask_dict["quse_masks"]) == 3  # One mask per atom in water
#
#     def test_design_matrix_is_object_array(self, mock_gdma_record_water):
#         """Test that design matrix is an object array of per-site matrices."""
#         parameter = LibraryChargeParameter(
#             smiles="[H:1][O:2][H:3]",
#             value=[0.0, 0.0, 0.0],
#             provenance={},
#         )
#         charge_collection = LibraryChargeCollection(parameters=[parameter])
#         charge_parameter_keys = [(parameter.smiles, (0, 1, 2))]
#
#         terms = list(
#             MPFITObjective.compute_objective_terms(
#                 [mock_gdma_record_water],
#                 charge_collection=charge_collection,
#                 charge_parameter_keys=charge_parameter_keys,
#             )
#         )
#
#         term = terms[0]
#         design_matrix = term.atom_charge_design_matrix
#
#         # MPFIT uses object arrays where each element is a per-site A matrix
#         assert design_matrix.dtype == np.dtype("O")
#         assert len(design_matrix) == 3  # One per atom in water
#
#         # Each element should be a 2D numpy array (the per-site A matrix)
#         for site_matrix in design_matrix:
#             assert isinstance(site_matrix, np.ndarray)
#             assert site_matrix.ndim == 2
#
#     def test_reference_values_is_object_array(self, mock_gdma_record_water):
#         """Test that reference values is an object array of per-site b vectors."""
#         parameter = LibraryChargeParameter(
#             smiles="[H:1][O:2][H:3]",
#             value=[0.0, 0.0, 0.0],
#             provenance={},
#         )
#         charge_collection = LibraryChargeCollection(parameters=[parameter])
#         charge_parameter_keys = [(parameter.smiles, (0, 1, 2))]
#
#         terms = list(
#             MPFITObjective.compute_objective_terms(
#                 [mock_gdma_record_water],
#                 charge_collection=charge_collection,
#                 charge_parameter_keys=charge_parameter_keys,
#             )
#         )
#
#         term = terms[0]
#         reference_values = term.reference_values
#
#         # MPFIT uses object arrays where each element is a per-site b vector
#         assert reference_values.dtype == np.dtype("O")
#         assert len(reference_values) == 3  # One per atom in water
#
#         # Each element should be a 1D numpy array (the per-site b vector)
#         for site_vector in reference_values:
#             assert isinstance(site_vector, np.ndarray)
#             assert site_vector.ndim == 1
#
# =============================================================================


# class TestMPFITObjectiveTerm:
#    """Priority 13: Test MPFITObjectiveTerm class.
#
#    Ref: fork/_tests/optimize/test_optimize.py - ObjectiveTerm tests
#    """
#
#    def test_objective_term_creation(self):
#        """Test that MPFITObjectiveTerm can be instantiated."""
#        pytest.skip("TODO: Implement")
#
#    def test_combine_multiple_terms(self):
#        """Test combining multiple objective terms from different conformers.
#
#        # Ref: fork/_tests/optimize - combine method tests
#        # term1 = MPFITObjectiveTerm(...)
#        # term2 = MPFITObjectiveTerm(...)
#        # combined = MPFITObjectiveTerm.combine(term1, term2)
#        # assert combined.atom_charge_design_matrix is not None
#        """
#        pytest.skip("TODO: Implement")
#
#
# class TestMPFITObjective:
#    """Priority 14: Test MPFITObjective.compute_objective_terms.
#
#    Ref: fork/_tests/optimize/test_optimize.py - Objective class tests
#    """
#
#    def test_compute_terms_single_record(self, mock_gdma_record_water):
#        """Test computing objective terms for a single GDMA record."""
#        pytest.skip("TODO: Implement")
#
#    def test_compute_terms_returns_quse_masks(self, mock_gdma_record_water):
#        """Test that return_quse_masks=True includes masks in output."""
#        pytest.skip("TODO: Implement")
#
#    def test_design_matrix_is_object_array(self, mock_gdma_record_water):
#        """Test that design matrix is an object array of per-site matrices."""
#        pytest.skip("TODO: Implement")
