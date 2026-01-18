"""
Unit and regression test for the openff_pympfit package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import openff_pympfit


def test_openff_pympfit_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "openff_pympfit" in sys.modules
