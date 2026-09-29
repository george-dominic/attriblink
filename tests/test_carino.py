"""Tests for the Carino linking method."""

import numpy as np
import pandas as pd
import pytest

from attriblink import link
from attriblink.methods.carino import get_k_factor


class TestCarinoBasic:
    """Basic Carino method tests."""

    def test_simple_two_period(self):
        """Test simple two-period case with known results."""
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.015, 0.02])

        # Effects: allocation and selection
        effects = pd.DataFrame(
            {"allocation": [0.003, 0.008], "selection": [0.002, 0.002]},
        )

        # Period effects reconcile to active return before linking
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
                "allocation": [0.002, 0.006, 0.002],
                "selection": [0.002, 0.002, 0.002],
                "interaction": [0.001, 0.002, 0.001],
            },
        )

        # Period effects reconcile to active return before linking
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

        # Single period: period and cumulative coefficients cancel
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

        # Equal returns use the finite coefficient limit
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

        # Equal cumulative returns use 1 / (1 + cumulative return)
        assert np.isclose(k, 1 / ((1 + portfolio).prod()))


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
            {"allocation": [0.003, 0.008], "selection": [0.002, 0.002]},
        )

        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        assert hasattr(result, 'k_factor')
        assert isinstance(result.k_factor, float)

    def test_result_has_dataframe(self):
        """Test that result has data DataFrame."""
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.015, 0.02])
        effects = pd.DataFrame(
            {"allocation": [0.003, 0.008], "selection": [0.002, 0.002]},
        )

        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        assert hasattr(result, 'data')
        assert isinstance(result.data, pd.DataFrame)

    def test_result_summary(self):
        """Test that result.summary() works."""
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.015, 0.02])
        effects = pd.DataFrame(
            {"allocation": [0.003, 0.008], "selection": [0.002, 0.002]},
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
            {"allocation": [0.003, 0.008], "selection": [0.002, 0.002]},
        )

        result = link(effects, portfolio, benchmark, method="carino", check_effects_sum=False)

        summary_str = str(result)
        assert "Attribution Summary" in summary_str


class TestCarinoRegression:
    """Independent expected effects catch errors hidden by total reconciliation."""

    def test_period_specific_weights(self):
        portfolio = pd.Series([0.2, -0.1])
        benchmark = pd.Series([0.1, -0.05])
        effects = pd.DataFrame({'allocation': [0.1, 0], 'selection': [0, -0.05]})
        result = link(effects, portfolio, benchmark, strict=True)
        np.testing.assert_allclose(result.linked_effects, [0.092441227530, -0.057441227530], atol=1e-12)
        assert np.isclose(result.k_factor, get_k_factor(portfolio.values, benchmark.values))

    def test_arithmetic_cancellation(self):
        result = link(pd.DataFrame({'allocation': [0.1, 0], 'selection': [0, -0.1]}),
                      pd.Series([0.1, 0.2]), pd.Series([0, 0.3]), strict=True)
        np.testing.assert_allclose(result.linked_effects, [0.124853910311, -0.104853910311], atol=1e-12)
        assert np.isclose(result.linked_effects.sum(), 0.02)

    def test_cumulative_cancellation(self):
        result = link(pd.DataFrame({'allocation': [0.1, 0], 'selection': [0, -0.1]}),
                      pd.Series([0.1, 0]), pd.Series([0, 0.1]), strict=True)
        np.testing.assert_allclose(result.linked_effects, [0.104841197785, -0.104841197785], atol=1e-12)
        assert np.isclose(result.k_factor, 1 / 1.1)

    def test_equal_period_returns_with_offsetting_effects(self):
        result = link(pd.DataFrame({'allocation': [0.01, 0], 'selection': [-0.01, 0.1]}),
                      pd.Series([0.1, 0.2]), pd.Series([0.1, 0.1]), strict=True)
        np.testing.assert_allclose(result.linked_effects, [0.011492749966699, 0.098507250033301], atol=1e-12)

    def test_small_active_returns(self):
        result = link(pd.DataFrame({'effect': [1e-12, 2e-12]}),
                      pd.Series([0.1 + 1e-12, 0.2 + 2e-12]), pd.Series([0.1, 0.2]), strict=True)
        np.testing.assert_allclose(result.linked_effects, [3.4e-12], rtol=1e-10, atol=1e-22)

    def test_mismatched_effects_are_not_rescaled(self):
        portfolio = pd.Series([0.1, 0.2])
        benchmark = pd.Series([0, 0.1])
        effects = pd.DataFrame({'effect': [0.05, 0.05]})
        result = link(effects, portfolio, benchmark, check_effects_sum=False)
        assert np.isclose(result['effect'], 0.11)
        assert not np.isclose(result['effect'], 0.22)

    def test_large_negative_active_return_is_valid(self):
        result = link(pd.DataFrame({'effect': [-1.3, 0.1]}),
                      pd.Series([-0.8, 0.2]), pd.Series([0.5, 0.1]), strict=True)
        assert np.isclose(result['effect'], -1.41)

    def test_invalid_portfolio_log_domain(self):
        with pytest.raises(ValueError, match='<= -1'):
            link(pd.DataFrame({'effect': [-1.0]}), pd.Series([-1.0]), pd.Series([0.0]))

    @pytest.mark.parametrize("factor, unit", [(1, "decimal"), (100, "percent"), (10000, "bps")])
    def test_output_units(self, factor, unit):
        effects = pd.DataFrame({'allocation': [0.1, 0], 'selection': [0, -0.05]})
        result = link(effects * factor, pd.Series([0.2, -0.1]) * factor,
                      pd.Series([0.1, -0.05]) * factor, unit=unit, strict=True)
        np.testing.assert_allclose(result.linked_effects / factor,
                                   [0.092441227530, -0.057441227530], atol=1e-12)
