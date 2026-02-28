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

        # Disable validation - Carino handles effects that don't sum to excess
        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        # Verify additivity to geometric cumulative excess return (CER)
        cr_port = (1 + portfolio).prod() - 1
        cr_bench = (1 + benchmark).prod() - 1
        cumulative_excess = cr_port - cr_bench
        linked_sum = result['allocation'] + result['selection']
        assert np.isclose(linked_sum, cumulative_excess, rtol=1e-10)

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

        # Disable validation - Carino handles effects that don't sum to excess
        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        cr_port = (1 + portfolio).prod() - 1
        cr_bench = (1 + benchmark).prod() - 1
        cumulative_excess = cr_port - cr_bench
        linked_sum = result['allocation'] + result['selection'] + result['interaction']
        assert np.isclose(linked_sum, cumulative_excess, rtol=1e-10)

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
        expected_sum = effects.sum().sum()
        actual_sum = result['allocation'] + result['selection']
        assert np.isclose(actual_sum, expected_sum, rtol=1e-10)

        # Also verify additivity (effects sum to excess).
        # For a single period, arithmetic and geometric excess coincide.
        cr_port = (1 + portfolio).prod() - 1
        cr_bench = (1 + benchmark).prod() - 1
        cumulative_excess = cr_port - cr_bench
        assert np.isclose(actual_sum, cumulative_excess, rtol=1e-10)


class TestCarinoEdgeCases:
    """Edge case tests for Carino method."""

    def test_zero_excess_return(self):
        """Test when cumulative excess return is exactly zero."""
        portfolio = pd.Series([0.02, -0.02])
        benchmark = pd.Series([0.02, -0.02])

        effects = pd.DataFrame(
            {"effect": [0.0, 0.0]},
        )

        # Disable validation - this is an edge case test
        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        # With zero excess, should use k=1
        linked_sum = result['effect']
        assert np.isclose(linked_sum, 0.0)

    def test_near_zero_excess_return(self):
        """Test when cumulative excess return is near zero."""
        portfolio = pd.Series([0.010001, -0.01])
        benchmark = pd.Series([0.01, -0.01])

        effects = pd.DataFrame(
            {"effect": [0.000001, 0.0]},
        )

        # Disable validation - this is an edge case test
        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        # Should complete without error and match geometric cumulative excess
        cr_port = (1 + portfolio).prod() - 1
        cr_bench = (1 + benchmark).prod() - 1
        cumulative_excess = cr_port - cr_bench
        linked_sum = result['effect']
        assert np.isclose(linked_sum, cumulative_excess, rtol=1e-6)

    def test_negative_excess_return(self):
        """Test with negative cumulative excess return."""
        portfolio = pd.Series([0.01, -0.02])
        benchmark = pd.Series([0.015, -0.01])

        effects = pd.DataFrame(
            {"allocation": [-0.005, -0.01], "selection": [0.0, 0.0]},
        )

        # Disable validation - this is an edge case test
        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        cr_port = (1 + portfolio).prod() - 1
        cr_bench = (1 + benchmark).prod() - 1
        cumulative_excess = cr_port - cr_bench
        linked_sum = result['allocation'] + result['selection']
        assert np.isclose(linked_sum, cumulative_excess, rtol=1e-10)


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

        # Disable validation - this is a numerical stability test
        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        cr_port = (1 + portfolio).prod() - 1
        cr_bench = (1 + benchmark).prod() - 1
        cumulative_excess = cr_port - cr_bench
        linked_sum = result['effect']
        assert np.isclose(linked_sum, cumulative_excess, rtol=1e-6)

    def test_large_returns(self):
        """Test with large returns."""
        portfolio = pd.Series([0.50, 0.30])
        benchmark = pd.Series([0.20, 0.10])

        effects = pd.DataFrame(
            {"allocation": [0.25, 0.15], "selection": [0.05, 0.05]},
        )

        # Disable validation - this is a numerical stability test
        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        cr_port = (1 + portfolio).prod() - 1
        cr_bench = (1 + benchmark).prod() - 1
        cumulative_excess = cr_port - cr_bench
        linked_sum = result['allocation'] + result['selection']
        assert np.isclose(linked_sum, cumulative_excess, rtol=1e-10)


class TestAttributionResult:
    """Tests for AttributionResult class."""

    def test_result_has_k_factor(self):
        """Test that result has k_factor attribute."""
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.015, 0.02])
        effects = pd.DataFrame(
            {"allocation": [0.005, 0.008], "selection": [0.002, 0.005]},
        )

        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        assert hasattr(result, 'k_factor')
        assert isinstance(result.k_factor, float)

    def test_result_has_dataframe(self):
        """Test that result has data DataFrame."""
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.015, 0.02])
        effects = pd.DataFrame(
            {"allocation": [0.005, 0.008], "selection": [0.002, 0.005]},
        )

        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        assert hasattr(result, 'data')
        assert isinstance(result.data, pd.DataFrame)

    def test_result_summary(self):
        """Test that result.summary() works."""
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.015, 0.02])
        effects = pd.DataFrame(
            {"allocation": [0.005, 0.008], "selection": [0.002, 0.005]},
        )

        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "Attribution Summary" in summary

    def test_result_str(self):
        """Test that str(result) returns summary."""
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.015, 0.02])
        effects = pd.DataFrame(
            {"allocation": [0.005, 0.008], "selection": [0.002, 0.005]},
        )

        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        summary_str = str(result)
        assert "Attribution Summary" in summary_str
