"""Tests for input validation."""

import numpy as np
import pandas as pd
import pytest

from attriblink import link
from attriblink.exceptions import (
    AlignmentError,
    InvalidEffectsError,
    InvalidMethodError,
    InvalidReturnsError,
)
from attriblink.validators import (
    validate_alignment,
    validate_effects,
    validate_not_missing,
    validate_returns,
)


class TestEffectsValidation:
    """Tests for effects validation."""

    def test_invalid_type_list(self):
        """Test that list is rejected."""
        with pytest.raises(InvalidEffectsError):
            validate_effects([1, 2, 3])

    def test_invalid_type_dict(self):
        """Test that dict is rejected."""
        with pytest.raises(InvalidEffectsError):
            validate_effects({"a": [1, 2, 3]})

    def test_empty_dataframe(self):
        """Test that empty DataFrame is rejected."""
        with pytest.raises(InvalidEffectsError):
            validate_effects(pd.DataFrame())

    def test_no_columns(self):
        """Test that DataFrame with no columns is rejected."""
        effects = pd.DataFrame(index=[0, 1, 2])
        with pytest.raises(InvalidEffectsError):
            validate_effects(effects)

    def test_non_numeric_column(self):
        """Test that non-numeric columns are rejected."""
        effects = pd.DataFrame(
            {"text_col": ["a", "b", "c"]},
        )
        with pytest.raises(InvalidEffectsError):
            validate_effects(effects)

    def test_infinite_values(self):
        """Test that infinite values are rejected."""
        effects = pd.DataFrame(
            {"col": [1.0, np.inf, 2.0]},
        )
        with pytest.raises(InvalidEffectsError):
            validate_effects(effects)

    def test_all_nan_column(self):
        """Test that all-NaN columns are rejected."""
        effects = pd.DataFrame(
            {"col1": [1.0, 2.0, 3.0], "col2": [np.nan, np.nan, np.nan]},
        )
        with pytest.raises(InvalidEffectsError):
            validate_effects(effects)


class TestReturnsValidation:
    """Tests for returns validation."""

    def test_invalid_type_list(self):
        """Test that list is rejected for returns."""
        with pytest.raises(InvalidReturnsError):
            validate_returns([1, 2, 3], "returns")

    def test_invalid_type_dataframe(self):
        """Test that DataFrame is rejected for returns."""
        with pytest.raises(InvalidReturnsError):
            validate_returns(pd.DataFrame({"a": [1, 2]}), "returns")

    def test_empty_series(self):
        """Test that empty Series is rejected."""
        with pytest.raises(InvalidReturnsError):
            validate_returns(pd.Series(dtype=float), "returns")

    def test_non_numeric_series(self):
        """Test that non-numeric Series is rejected."""
        with pytest.raises(InvalidReturnsError):
            validate_returns(pd.Series(["a", "b", "c"]), "returns")

    def test_infinite_values(self):
        """Test that infinite values are rejected."""
        with pytest.raises(InvalidReturnsError):
            validate_returns(pd.Series([1.0, np.inf, 2.0]), "returns")


class TestAlignmentValidation:
    """Tests for alignment validation."""

    def test_mismatched_effects_portfolio_index(self):
        """Test that mismatched effects/portfolio index is rejected."""
        effects = pd.DataFrame(
            {"col": [1.0, 2.0]},
        )
        portfolio = pd.Series([0.01, 0.02], index=[2, 3])
        benchmark = pd.Series([0.005, 0.01])

        with pytest.raises(AlignmentError):
            validate_alignment(effects, portfolio, benchmark)

    def test_mismatched_effects_benchmark_index(self):
        """Test that mismatched effects/benchmark index is rejected."""
        effects = pd.DataFrame(
            {"col": [1.0, 2.0]},
        )
        portfolio = pd.Series([0.01, 0.02])
        benchmark = pd.Series([0.005, 0.01], index=[2, 3])

        with pytest.raises(AlignmentError):
            validate_alignment(effects, portfolio, benchmark)

    def test_duplicate_indices(self):
        """Test that duplicate indices are rejected."""
        idx = [0, 0]
        effects = pd.DataFrame({"col": [1.0, 2.0]}, index=idx)
        portfolio = pd.Series([0.01, 0.02], index=idx)
        benchmark = pd.Series([0.005, 0.01], index=idx)

        with pytest.raises(AlignmentError):
            validate_alignment(effects, portfolio, benchmark)


class TestMissingValidation:
    """Tests for missing value validation."""

    def test_nan_in_effects(self):
        """Test that NaN in effects is rejected."""
        effects = pd.DataFrame(
            {"col": [1.0, np.nan, 3.0]},
        )
        portfolio = pd.Series([0.01, 0.02, 0.03])
        benchmark = pd.Series([0.005, 0.01, 0.015])

        with pytest.raises(InvalidEffectsError):
            validate_not_missing(effects, portfolio, benchmark)

    def test_nan_in_portfolio(self):
        """Test that NaN in portfolio returns is rejected."""
        effects = pd.DataFrame(
            {"col": [1.0, 2.0, 3.0]},
        )
        portfolio = pd.Series([0.01, np.nan, 0.03])
        benchmark = pd.Series([0.005, 0.01, 0.015])

        with pytest.raises(InvalidReturnsError):
            validate_not_missing(effects, portfolio, benchmark)

    def test_nan_in_benchmark(self):
        """Test that NaN in benchmark returns is rejected."""
        effects = pd.DataFrame(
            {"col": [1.0, 2.0, 3.0]},
        )
        portfolio = pd.Series([0.01, 0.02, 0.03])
        benchmark = pd.Series([0.005, 0.01, np.nan])

        with pytest.raises(InvalidReturnsError):
            validate_not_missing(effects, portfolio, benchmark)


class TestMethodValidation:
    """Tests for method validation."""

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        portfolio = pd.Series([0.01, 0.02])
        benchmark = pd.Series([0.005, 0.01])
        effects = pd.DataFrame({"col": [0.005, 0.01]})

        with pytest.raises(InvalidMethodError):
            link(effects, portfolio, benchmark, method="invalid_method")

    def test_unsupported_method(self):
        """Test that unsupported method raises error."""
        portfolio = pd.Series([0.01, 0.02])
        benchmark = pd.Series([0.005, 0.01])
        effects = pd.DataFrame({"col": [0.005, 0.01]})

        with pytest.raises(InvalidMethodError):
            link(effects, portfolio, benchmark, method="geometric")
