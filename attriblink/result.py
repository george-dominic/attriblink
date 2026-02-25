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
        """Get the full DataFrame with all attribution data."""
        # Build a DataFrame with period columns + Total + Linked
        periods = [str(i) for i in self._effects.index]
        cols = periods + ['Total', 'Linked']
        
        # Create rows for returns
        portfolio_vals = list(self._portfolio_returns.values)
        benchmark_vals = list(self._benchmark_returns.values)
        active_vals = list((self._portfolio_returns - self._benchmark_returns).values)
        
        # Calculate totals (compounded)
        total_port = (1 + self._portfolio_returns).prod() - 1
        total_bench = (1 + self._benchmark_returns).prod() - 1
        total_active = total_port - total_bench
        
        data = {
            'Portfolio Return': portfolio_vals + [total_port, None],
            'Benchmark Return': benchmark_vals + [total_bench, None],
            'Active Return': active_vals + [total_active, None],
        }
        
        # Add effect rows
        for col in self._effects.columns:
            period_vals = list(self._effects[col].values)
            total = sum(period_vals)
            linked = self._linked_effects[col]
            data[col] = period_vals + [total, linked]
        
        # Add Total Effects row
        period_totals = list(self._effects.sum(axis=1).values)
        total_sum = sum(period_totals)
        linked_sum = self._linked_effects.sum()
        data['Total Effects'] = period_totals + [total_sum, linked_sum]
        
        df = pd.DataFrame(data, index=cols).T
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
