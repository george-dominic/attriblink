"""Tests for the Carino linking method."""

import numpy as np
import pandas as pd
import pytest

from attriblink import link
from attriblink.exceptions import ZeroExcessReturnError
from attriblink.methods.carino import carino_link, get_k_factor


class TestCarinoBasic:
    """Basic Carino method tests."""

    def test_simple_two_period(self):
        """Test simple two-period case with known results."""
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.015, 0.02])

        # Effects: allocation and selection
        effects = pd.DataFrame(
            {"allocation": [0.005, 0.008], "selection": [0.002, 0.005]},
        )

        result = link(effects, portfolio, benchmark, method="carino")

        # Verify additivity
        excess_return = (portfolio - benchmark).sum()
        assert np.isclose(result.sum(), excess_return, rtol=1e-10)

    def test_three_period(self):
        """Test three-period case."""
        portfolio = pd.Series([0.02, 0.03, 0.015])
        benchmark = pd.Series([0.015, 0.02, 0.01])

        effects = pd.DataFrame(
            {
                "allocation": [0.005, 0.008, 0.003],
                "selection": [0.002, 0.005, 0.004],
                "interaction": [0.001, 0.002, 0.001],
            },
        )

        result = link(effects, portfolio, benchmark, method="carino")

        excess_return = (portfolio - benchmark).sum()
        assert np.isclose(result.sum(), excess_return, rtol=1e-10)

    def test_single_period(self):
        """Test single period (should just return sums)."""
        portfolio = pd.Series([0.02])
        benchmark = pd.Series([0.015])

        # Effects must sum to excess return for proper attribution
        # excess = 0.02 - 0.015 = 0.005
        effects = pd.DataFrame(
            {"allocation": [0.003], "selection": [0.002]},  # sums to 0.005
        )

        result = link(effects, portfolio, benchmark, method="carino")

        # Single period: linked = sum (k = 1)
        expected_sum = effects.sum()
        assert np.allclose(result.values, expected_sum.values)

        # Also verify additivity (effects sum to excess)
        excess_return = (portfolio - benchmark).sum()
        assert np.isclose(result.sum(), excess_return, rtol=1e-10)


class TestCarinoEdgeCases:
    """Edge case tests for Carino method."""

    def test_zero_excess_return(self):
        """Test when cumulative excess return is exactly zero."""
        portfolio = pd.Series([0.02, -0.02])
        benchmark = pd.Series([0.02, -0.02])

        effects = pd.DataFrame(
            {"effect": [0.0, 0.0]},
        )

        result = link(effects, portfolio, benchmark, method="carino")

        # With zero excess, should use k=1
        assert np.isclose(result.sum(), 0.0)

    def test_near_zero_excess_return(self):
        """Test when cumulative excess return is near zero."""
        portfolio = pd.Series([0.010001, -0.01])
        benchmark = pd.Series([0.01, -0.01])

        effects = pd.DataFrame(
            {"effect": [0.000001, 0.0]},
        )

        result = link(effects, portfolio, benchmark, method="carino")

        # Should complete without error
        excess_return = (portfolio - benchmark).sum()
        assert np.isclose(result.sum(), excess_return, rtol=1e-6)

    def test_negative_excess_return(self):
        """Test with negative cumulative excess return."""
        portfolio = pd.Series([0.01, -0.02])
        benchmark = pd.Series([0.015, -0.01])

        effects = pd.DataFrame(
            {"allocation": [-0.005, -0.01], "selection": [0.0, 0.0]},
        )

        result = link(effects, portfolio, benchmark, method="carino")

        excess_return = (portfolio - benchmark).sum()
        assert np.isclose(result.sum(), excess_return, rtol=1e-10)


class TestKFactor:
    """Tests for k-factor computation."""

    def test_k_factor_basic(self):
        """Test basic k-factor calculation."""
        portfolio = np.array([0.02, 0.03])
        benchmark = np.array([0.015, 0.02])

        k = get_k_factor(portfolio, benchmark)

        # k should be positive and typically less than or equal to 1
        # for positive excess returns
        assert k > 0

    def test_k_factor_zero_excess(self):
        """Test k-factor with zero excess return."""
        portfolio = np.array([0.02, -0.02])
        benchmark = np.array([0.02, -0.02])

        k = get_k_factor(portfolio, benchmark)

        # With zero excess, k should be 1
        assert np.isclose(k, 1.0)


class TestCarinoNumericalStability:
    """Tests for numerical stability."""

    def test_small_returns(self):
        """Test with very small returns."""
        portfolio = pd.Series([0.0001, 0.0002])
        benchmark = pd.Series([0.00005, 0.0001])

        effects = pd.DataFrame(
            {"effect": [0.00005, 0.0001]},
        )

        result = link(effects, portfolio, benchmark, method="carino")

        excess_return = (portfolio - benchmark).sum()
        assert np.isclose(result.sum(), excess_return, rtol=1e-6)

    def test_large_returns(self):
        """Test with large returns."""
        portfolio = pd.Series([0.50, 0.30])
        benchmark = pd.Series([0.20, 0.10])

        effects = pd.DataFrame(
            {"allocation": [0.25, 0.15], "selection": [0.05, 0.05]},
        )

        result = link(effects, portfolio, benchmark, method="carino")

        excess_return = (portfolio - benchmark).sum()
        assert np.isclose(result.sum(), excess_return, rtol=1e-10)
