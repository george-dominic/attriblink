"""Tests for key invariants (additivity, etc.)."""

import numpy as np
import pandas as pd
import pytest

from attriblink import link
from attriblink.result import AttributionResult


class TestAdditivityInvariant:
    """Tests for the additivity invariant: sum(linked_effects) == cumulative_excess_return."""

    def test_additivity_basic(self):
        """Test basic additivity."""
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.015, 0.02])

        effects = pd.DataFrame(
            {"allocation": [0.005, 0.008], "selection": [0.002, 0.005]},
        )

        result = link(effects, portfolio, benchmark, method="carino")
        cumulative_excess = (portfolio - benchmark).sum()

        # Access linked effects via result.linked_effects or result['effect_name']
        assert np.isclose(result.linked_effects.sum(), cumulative_excess, rtol=1e-10)

    def test_additivity_many_periods(self):
        """Test additivity with many periods."""
        n_periods = 12
        portfolio = pd.Series(np.random.uniform(0.01, 0.05, n_periods))
        benchmark = portfolio * 0.8  # Benchmark is 80% of portfolio

        effects = pd.DataFrame(
            {
                f"effect_{i}": np.random.uniform(0.001, 0.01, n_periods)
                for i in range(5)
            },
        )

        result = link(effects, portfolio, benchmark, method="carino")
        cumulative_excess = (portfolio - benchmark).sum()

        assert np.isclose(result.linked_effects.sum(), cumulative_excess, rtol=1e-10)

    def test_additivity_single_effect(self):
        """Test additivity with single effect column."""
        portfolio = pd.Series([0.02, 0.03, 0.015])
        benchmark = pd.Series([0.015, 0.02, 0.01])

        # One effect that equals excess return
        excess = portfolio - benchmark
        effects = pd.DataFrame({"excess": excess.values})

        result = link(effects, portfolio, benchmark, method="carino")
        cumulative_excess = excess.sum()

        assert np.isclose(result.linked_effects.sum(), cumulative_excess, rtol=1e-10)

    def test_additivity_many_effects(self):
        """Test additivity with many effect columns."""
        n_periods = 5
        n_effects = 20

        portfolio = pd.Series([0.02, 0.03, -0.01, 0.025, 0.015])
        benchmark = pd.Series([0.015, 0.02, -0.015, 0.02, 0.01])

        # Effects that sum to excess return per period
        excess = portfolio - benchmark
        effects_data = {}
        for i in range(n_effects):
            # Distribute excess across effects
            weights = np.random.dirichlet(np.ones(n_effects))
            effects_data[f"effect_{i}"] = (excess * weights[i]).values

        effects = pd.DataFrame(effects_data)

        result = link(effects, portfolio, benchmark, method="carino")
        cumulative_excess = excess.sum()

        assert np.isclose(result.linked_effects.sum(), cumulative_excess, rtol=1e-10)

    def test_additivity_negative_excess(self):
        """Test additivity with negative excess return."""
        portfolio = pd.Series([0.01, -0.02, 0.015])
        benchmark = pd.Series([0.015, -0.01, 0.02])

        effects = pd.DataFrame(
            {"allocation": [-0.005, -0.01, -0.005], "selection": [0.0, 0.0, 0.0]},
        )

        result = link(effects, portfolio, benchmark, method="carino")
        cumulative_excess = (portfolio - benchmark).sum()

        assert np.isclose(result.linked_effects.sum(), cumulative_excess, rtol=1e-10)

    def test_additivity_zero_excess(self):
        """Test additivity when excess is exactly zero."""
        portfolio = pd.Series([0.02, -0.02])
        benchmark = pd.Series([0.02, -0.02])

        effects = pd.DataFrame(
            {"effect": [0.0, 0.0]},
        )

        result = link(effects, portfolio, benchmark, method="carino")
        cumulative_excess = (portfolio - benchmark).sum()

        assert np.isclose(result.linked_effects.sum(), cumulative_excess, rtol=1e-10)


class TestFloatingPointDrift:
    """Tests to ensure no floating-point drift accumulates."""

    def test_no_drift_many_periods(self):
        """Test no drift with many periods."""
        n_periods = 100
        portfolio = pd.Series(np.random.uniform(-0.1, 0.2, n_periods))
        benchmark = portfolio * np.random.uniform(0.7, 1.1, n_periods)

        excess = portfolio - benchmark
        # Make effects sum to excess per period
        effects = pd.DataFrame(
            {
                "effect1": excess * 0.6,
                "effect2": excess * 0.3,
                "effect3": excess * 0.1,
            },
        )

        result = link(effects, portfolio, benchmark, method="carino")
        cumulative_excess = excess.sum()

        # Should be exact within floating-point tolerance
        diff = abs(result.linked_effects.sum() - cumulative_excess)
        assert diff < 1e-10, f"Floating-point drift detected: {diff}"

    def test_no_drift_extreme_values(self):
        """Test no drift with extreme return values."""
        portfolio = pd.Series([10.0, -5.0, 3.0])
        benchmark = pd.Series([5.0, -3.0, 1.0])

        excess = portfolio - benchmark
        effects = pd.DataFrame(
            {
                "allocation": excess * 0.5,
                "selection": excess * 0.3,
                "interaction": excess * 0.2,
            },
        )

        result = link(effects, portfolio, benchmark, method="carino")
        cumulative_excess = excess.sum()

        assert np.isclose(result.linked_effects.sum(), cumulative_excess, rtol=1e-10)


class TestResultProperties:
    """Tests for properties of the result."""

    def test_result_is_attribution_result(self):
        """Test that result is AttributionResult."""
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.015, 0.02])
        effects = pd.DataFrame({"col": [0.005, 0.01]})

        result = link(effects, portfolio, benchmark, method="carino")

        assert isinstance(result, AttributionResult)

    def test_result_linked_effects_index_matches_effects_columns(self):
        """Test that linked_effects index matches effect column names."""
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.015, 0.02])
        effects = pd.DataFrame(
            {"allocation": [0.005, 0.008], "selection": [0.002, 0.005]},
        )

        result = link(effects, portfolio, benchmark, method="carino")

        assert list(result.linked_effects.index) == list(effects.columns)

    def test_result_has_k_factor(self):
        """Test that result has k_factor attribute."""
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.015, 0.02])
        effects = pd.DataFrame({"col": [0.005, 0.01]})

        result = link(effects, portfolio, benchmark, method="carino")

        assert hasattr(result, 'k_factor')
        assert isinstance(result.k_factor, float)

    def test_result_dataframe(self):
        """Test that result.data returns a DataFrame."""
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.015, 0.02])
        effects = pd.DataFrame(
            {"allocation": [0.005, 0.008], "selection": [0.002, 0.005]},
        )

        result = link(effects, portfolio, benchmark, method="carino")

        assert hasattr(result, 'data')
        assert isinstance(result.data, pd.DataFrame)
