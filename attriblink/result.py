"""AttributionResult class for comprehensive attribution output."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class AttributionResult:
    """Result object containing linked attribution effects and summary info.
    
    This class provides a comprehensive return type for the link() function,
    including:
    - Total portfolio and benchmark returns (compounded)
    - Active return (portfolio - benchmark)
    - Linked attribution effects for each source
    - Summary display with period breakdown
    """

    def __init__(
        self,
        linked_effects: pd.Series,
        k_factor: float,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        effects: pd.DataFrame,
    ):
        """Initialize AttributionResult.
        
        Args:
            linked_effects: Series of linked attribution effects.
            k_factor: Carino k-factor used for linking.
            portfolio_returns: Original portfolio return series.
            benchmark_returns: Original benchmark return series.
            effects: Original effects DataFrame.
        """
        self._linked_effects = linked_effects
        self._k_factor = k_factor
        self._portfolio_returns = portfolio_returns
        self._benchmark_returns = benchmark_returns
        self._effects = effects

    @property
    def data(self) -> pd.DataFrame:
        """Get the full DataFrame with all attribution data.
        
        Returns a DataFrame where:
        - Each period is a row
        - Columns ordered: Portfolio Return, Benchmark Return, Active Return, [effects...]
        - A 'Total' row at the bottom contains geometric linked returns
        """
        # Get period indices
        periods = [str(i) for i in self._effects.index]
        
        # Calculate geometric linked returns for totals
        total_port = (1 + self._portfolio_returns).prod() - 1
        total_bench = (1 + self._benchmark_returns).prod() - 1
        total_active = total_port - total_bench
        
        # Build column order: Portfolio, Benchmark, Active, effects
        effect_cols = list(self._effects.columns)
        column_order = ['Portfolio Return', 'Benchmark Return', 'Active Return'] + effect_cols
        
        # Build data row by row (one row per period)
        rows_data = []
        
        for i, period in enumerate(periods):
            row = {
                'Portfolio Return': self._portfolio_returns.iloc[i],
                'Benchmark Return': self._benchmark_returns.iloc[i],
                'Active Return': self._portfolio_returns.iloc[i] - self._benchmark_returns.iloc[i],
            }
            # Add each effect for this period
            for effect in effect_cols:
                row[effect] = self._effects[effect].iloc[i]
            rows_data.append(row)
        
        # Add Total row at bottom with geometric linked values
        total_row = {
            'Portfolio Return': total_port,
            'Benchmark Return': total_bench,
            'Active Return': total_active,
        }
        # Total for each effect = Carino linked effect
        for effect in effect_cols:
            total_row[effect] = self._linked_effects[effect]
        
        rows_data.append(total_row)
        
        # Create DataFrame with periods as index (including 'Total' for last row)
        row_labels = periods + ['Total']
        df = pd.DataFrame(rows_data, index=row_labels)
        
        # Reorder columns to match desired order
        df = df[column_order]
        
        return df

    @property
    def k_factor(self) -> float:
        """Get the Carino k-factor."""
        return self._k_factor

    @property
    def effects(self) -> pd.DataFrame:
        """Get the original effects DataFrame."""
        return self._effects

    @property
    def portfolio_returns(self) -> pd.Series:
        """Get the original portfolio returns."""
        return self._portfolio_returns

    @property
    def benchmark_returns(self) -> pd.Series:
        """Get the original benchmark returns."""
        return self._benchmark_returns

    @property
    def linked_effects(self) -> pd.Series:
        """Get the linked effects Series."""
        return self._linked_effects

    @property
    def date_range(self) -> tuple[str, str]:
        """Get the start and end dates from the index."""
        index = self._portfolio_returns.index
        start = index[0]
        end = index[-1]
        return (str(start), str(end))

    @property
    def num_periods(self) -> int:
        """Get the number of periods."""
        return len(self._portfolio_returns)

    @property
    def effect_columns(self) -> list[str]:
        """Get the list of effect column names."""
        return list(self._effects.columns)

    def _format_percent(self, value: float) -> str:
        """Format a value as a percentage string."""
        if value is None:
            return "-"
        return f"{value * 100:>7.2f}%"

    def _format_row_label(self, idx: Any) -> str:
        """Format index value as row label (date or period)."""
        try:
            dt = pd.to_datetime(idx)
            return dt.strftime('%b %Y')
        except Exception:
            return str(idx)

    def summary(self) -> str:
        """Generate a summary table of the attribution results.
        
        Returns a transposed table matching the data property layout:
        - Rows: each period (by date) and Total
        - Columns: Portfolio Return, Benchmark Return, Active Return, [effects...]
        
        Returns:
            Formatted string with the attribution summary table.
        """
        # Compute totals (geometric linked returns)
        total_port = (1 + self._portfolio_returns).prod() - 1
        total_bench = (1 + self._benchmark_returns).prod() - 1
        total_active = total_port - total_bench
        linked_effects_sum = self._linked_effects.sum()

        # Column order: Portfolio, Benchmark, Active, effects (same as data property)
        effect_cols = self.effect_columns
        column_order = ['Portfolio Return', 'Benchmark Return', 'Active Return'] + effect_cols

        # Compute column widths - fit percentage values (e.g. "  -1.23%")
        col_width = 10
        label_width = max(12, max(len(self._format_row_label(i)) for i in self._effects.index) + 2, 6)

        # Short display names for long column headers
        col_display = {
            'Portfolio Return': 'Portfolio',
            'Benchmark Return': 'Benchmark',
            'Active Return': 'Active',
        }

        # Build the header
        lines = []
        lines.append("Attribution Summary (Carino Method)")
        lines.append("=" * (label_width + len(column_order) * col_width))
        lines.append("")

        # Header row - column names
        header = "".ljust(label_width)
        for col in column_order:
            hdr = col_display.get(col, col)
            header += f"{hdr[:col_width]:>{col_width}}"
        lines.append(header)
        lines.append("-" * (label_width + len(column_order) * col_width))

        # Data rows - one row per period
        for i in range(len(self._portfolio_returns)):
            row_label = self._format_row_label(self._portfolio_returns.index[i])
            row = f"{row_label:<{label_width}}"
            row += f"{self._format_percent(self._portfolio_returns.iloc[i]):>{col_width}}"
            row += f"{self._format_percent(self._benchmark_returns.iloc[i]):>{col_width}}"
            row += f"{self._format_percent(self._portfolio_returns.iloc[i] - self._benchmark_returns.iloc[i]):>{col_width}}"
            for effect in effect_cols:
                row += f"{self._format_percent(self._effects[effect].iloc[i]):>{col_width}}"
            lines.append(row)

        # Total row
        row = f"{'Total':<{label_width}}"
        row += f"{self._format_percent(total_port):>{col_width}}"
        row += f"{self._format_percent(total_bench):>{col_width}}"
        row += f"{self._format_percent(total_active):>{col_width}}"
        for effect in effect_cols:
            row += f"{self._format_percent(self._linked_effects[effect]):>{col_width}}"
        lines.append(row)

        # Footer info
        lines.append("")
        lines.append(f"Smoothing Factor (k): {self._k_factor:.4f}")

        # Sum check
        sum_check = np.isclose(linked_effects_sum, total_active, rtol=1e-6)
        check_symbol = "✓" if sum_check else "✗"
        lines.append(f"Sum Check: {check_symbol}")

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation."""
        return self.summary()

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"AttributionResult(periods={self.num_periods}, effects={len(self.effect_columns)}, k_factor={self._k_factor:.4f})"

    # Allow dictionary-like access to linked effects
    def __getitem__(self, key: str) -> float:
        """Access linked effect by name."""
        return self._linked_effects[key]

    # Allow iteration over effect columns
    def __iter__(self):
        """Iterate over effect column names."""
        return iter(self._effects.columns)
    
    # Add numpy array compatibility
    def __array__(self) -> np.ndarray:
        """Return linked effects as numpy array."""
        return self._linked_effects.values
