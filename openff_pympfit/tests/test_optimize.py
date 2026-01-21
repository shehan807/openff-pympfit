"""Tests for the MPFIT objective functions.

Ref: fork/_tests/optimize/test_optimize.py (objective term tests)
Pattern: Test objective term creation, combination, design matrix structure

Priority 13: MPFITObjectiveTerm (2 tests)
Priority 14: MPFITObjective.compute_objective_terms (3 tests)
"""

# import pytest


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
#class TestMPFITObjective:
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
