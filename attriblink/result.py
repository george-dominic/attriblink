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
        - Columns ordered: Portfolio Return, Benchmark Return, Active Return, [effects...], Total
        - A 'Total' row at the bottom contains geometric linked returns
        """
        # Get period indices
        periods = [str(i) for i in self._effects.index]
        
        # Calculate geometric linked returns for totals
        total_port = (1 + self._portfolio_returns).prod() - 1
        total_bench = (1 + self._benchmark_returns).prod() - 1
        total_active = total_port - total_bench
        
        # Build column order: Portfolio, Benchmark, Active, effects, Total
        effect_cols = list(self._effects.columns)
        column_order = ['Portfolio Return', 'Benchmark Return', 'Active Return'] + effect_cols + ['Total']
        
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
            # Total for this period = sum of effects
            row['Total'] = self._effects.iloc[i].sum()
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
        # Total of totals = sum of linked effects
        total_row['Total'] = self._linked_effects.sum()
        
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

    def summary(self) -> str:
        """Generate a summary table of the attribution results.
        
        Returns:
            Formatted string with the attribution summary table.
        """
        # Get date range
        start_date, end_date = self.date_range
        num_periods = self.num_periods
        
        # Format dates nicely
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            start_str = start_dt.strftime('%b %Y')
            end_str = end_dt.strftime('%b %Y')
            if start_dt.year == end_dt.year and start_dt.month == end_dt.month:
                date_str = start_str
            else:
                date_str = f"{start_str} - {end_str}"
        except Exception:
            date_str = f"{start_date} to {end_date}"
        
        period_label = date_str if num_periods == 1 else f"{date_str} ({num_periods} periods)"
        
        # Build the header
        lines = []
        lines.append("Attribution Summary (Carino Method)")
        lines.append("=" * 60)
        lines.append(f"Period: {period_label}")
        lines.append("")
        
        # Get period columns (numeric indices)
        period_cols = [str(i) for i in self._effects.index]
        
        # Header row
        header = " " * 24
        for col in period_cols:
            header += f"{col:>8}"
        header += f"{'Total':>8}{'Linked':>8}"
        lines.append(header)
        lines.append("-" * 60)
        
        # Portfolio Return row
        row = "Portfolio Return:      "
        for val in self._portfolio_returns.values:
            row += f"{self._format_percent(val):>8}"
        total_port = (1 + self._portfolio_returns).prod() - 1
        row += f"{self._format_percent(total_port):>8}{'-':>8}"
        lines.append(row)
        
        # Benchmark Return row
        row = "Benchmark Return:      "
        for val in self._benchmark_returns.values:
            row += f"{self._format_percent(val):>8}"
        total_bench = (1 + self._benchmark_returns).prod() - 1
        row += f"{self._format_percent(total_bench):>8}{'-':>8}"
        lines.append(row)
        
        # Active Return row
        row = "Active Return:         "
        excess_vals = self._portfolio_returns - self._benchmark_returns
        for val in excess_vals.values:
            row += f"{self._format_percent(val):>8}"
        total_active = total_port - total_bench
        row += f"{self._format_percent(total_active):>8}{'-':>8}"
        lines.append(row)
        
        lines.append("")  # Empty line before effects
        
        # Effect rows
        effect_names = self.effect_columns
        
        for effect in effect_names:
            # Get period values from original effects
            period_vals = self._effects[effect].values
            
            row = f"{effect.capitalize():<24}"
            for val in period_vals:
                row += f"{self._format_percent(val):>8}"
            
            # Total (sum of periods)
            total_val = sum(period_vals)
            row += f"{self._format_percent(total_val):>8}"
            
            # Linked value
            linked_val = self._linked_effects[effect]
            row += f"{self._format_percent(linked_val):>8}"
            
            lines.append(row)
        
        # Total Effects row
        lines.append("-" * 60)
        
        # Sum of period effects for each period
        period_totals = self._effects.sum(axis=1).values
        
        row = f"{'Total Effects':<24}"
        for val in period_totals:
            row += f"{self._format_percent(val):>8}"
        
        # Total sum
        total_effects = sum(period_totals)
        row += f"{self._format_percent(total_effects):>8}"
        
        # Linked sum
        linked_effects_sum = self._linked_effects.sum()
        row += f"{self._format_percent(linked_effects_sum):>8}"
        
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
