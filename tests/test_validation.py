"""Tests for input validation."""

import warnings

import numpy as np
import pandas as pd
import pytest

from attriblink import link
from attriblink.exceptions import (
    AlignmentError,
    EffectsSumMismatchError,
    InvalidEffectsError,
    InvalidMethodError,
    InvalidReturnsError,
)
from attriblink.validators import (
    validate_alignment,
    validate_effects,
    validate_effects_sum,
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


class TestEffectsSumValidation:
    """Tests for effects sum validation."""

    def test_effects_sum_matches_excess_no_warning(self):
        """Test that matching effects don't trigger warning."""
        # Effects sum to excess: 0.005 + 0.002 = 0.007 (equals 0.02 - 0.013 = 0.007)
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.013, 0.025])
        effects = pd.DataFrame(
            {"allocation": [0.005, 0.003], "selection": [0.002, 0.002]},
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = link(effects, portfolio, benchmark, check_effects_sum=True, strict=False)

        # Should succeed without error
        assert result is not None

    def test_effects_sum_mismatch_strict_true_raises_error(self):
        """Test that mismatch with strict=True raises error."""
        import warnings
        from attriblink.exceptions import EffectsSumMismatchError

        # Effects sum to 0.01 but excess is 0.02 - mismatch!
        portfolio = pd.Series([0.03, 0.04])  # excess = 0.02 + 0.03 = 0.05
        benchmark = pd.Series([0.01, 0.01])  # excess per period = 0.02, 0.03
        effects = pd.DataFrame(
            {"allocation": [0.005, 0.005]},  # sum = 0.01, but should be 0.05
        )

        with pytest.raises(EffectsSumMismatchError):
            link(effects, portfolio, benchmark, check_effects_sum=True, strict=True)

    def test_effects_sum_mismatch_strict_false_warns(self):
        """Test that mismatch with strict=False issues warning."""
        import warnings
        from attriblink.exceptions import EffectsSumMismatchError

        # Effects sum to 0.01 but excess is 0.05 - mismatch!
        portfolio = pd.Series([0.03, 0.04])  # excess = 0.02 + 0.03 = 0.05
        benchmark = pd.Series([0.01, 0.01])
        effects = pd.DataFrame(
            {"allocation": [0.005, 0.005]},  # sum = 0.01, but should be 0.05
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = link(effects, portfolio, benchmark, check_effects_sum=True, strict=False)

            # Should have issued a warning
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Effects do not sum to excess return" in str(w[0].message)

        # But should still return a result (continues with warning)
        assert result is not None

    def test_check_effects_sum_disabled_no_warning(self):
        """Test that check_effects_sum=False skips validation."""
        import warnings
        from attriblink.exceptions import EffectsSumMismatchError

        # Mismatch case
        portfolio = pd.Series([0.03, 0.04])
        benchmark = pd.Series([0.01, 0.01])
        effects = pd.DataFrame(
            {"allocation": [0.005, 0.005]},  # sum = 0.01, but should be 0.05
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = link(effects, portfolio, benchmark, check_effects_sum=False, strict=False)

            # Should NOT have issued any warnings
            assert len(w) == 0

        assert result is not None

    def test_default_parameters(self):
        """Test that default parameters are check_effects_sum=True, strict=False."""
        import warnings

        # Matching case - should work with defaults
        portfolio = pd.Series([0.02, 0.03])
        benchmark = pd.Series([0.013, 0.025])
        effects = pd.DataFrame(
            {"allocation": [0.005, 0.003], "selection": [0.002, 0.002]},
        )

        # Should work without explicit parameters (defaults)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = link(effects, portfolio, benchmark)

        assert result is not None
