"""Batch linking functionality for multi-fund attribution."""

from __future__ import annotations

import pandas as pd

from .core import link


__all__ = ["link_batch"]


def link_batch(
    data: pd.DataFrame,
    group_by: str,
    date_col: str,
    effects_cols: list[str],
    portfolio_col: str,
    benchmark_col: str,
    unit: str = "decimal",
    method: str = "carino",
    check_effects_sum: bool = True,
) -> pd.DataFrame:
    """Apply attribution linking to grouped data across multiple funds.

    This function groups data by a specified column, applies the link()
    function to each group, and combines the results into a single DataFrame.

    Args:
        data: DataFrame containing all the attribution data.
        group_by: Column name to group by (e.g., 'FUND_ID').
        date_col: Column name containing dates.
        effects_cols: List of effect column names to include in attribution.
        portfolio_col: Column name for portfolio returns.
        benchmark_col: Column name for benchmark returns.
        unit: Unit of the input effects and returns. Can be:
            - "decimal": values as decimals (e.g., 0.02 for 2%)
            - "bps": values in basis points (e.g., 200 for 2%)
            - "percent": values in percent (e.g., 2 for 2%)
            Default is "decimal".
        method: Linking method to use. Currently only "carino" is supported.
        check_effects_sum: If True, validates that period-by-period effects
            sum to period-by-period excess returns. Default is True.

    Returns:
        DataFrame with columns:
            - DATE: Dates from date_col
            - FUND_ID: Group identifier (renamed from group_by column)
            - portfolio_return: Portfolio return for each period
            - benchmark_return: Benchmark return for each period
            - active_return: Portfolio return - benchmark return
            - Linked effect columns (from effects_cols)

    Example:
        >>> import pandas as pd
        >>> from attriblink import link_batch
        >>> data = pd.DataFrame({
        ...     'FUND_ID': ['FUND_A', 'FUND_A', 'FUND_B', 'FUND_B'],
        ...     'DATE': ['2024-01-31', '2024-02-29', '2024-01-31', '2024-02-29'],
        ...     'allocation': [0.005, 0.008, 0.004, 0.006],
        ...     'selection': [0.002, 0.005, 0.003, 0.004],
        ...     'portfolio': [0.02, 0.03, 0.018, 0.025],
        ...     'benchmark': [0.015, 0.02, 0.012, 0.018],
        ... })
        >>> result = link_batch(
        ...     data,
        ...     group_by='FUND_ID',
        ...     date_col='DATE',
        ...     effects_cols=['allocation', 'selection'],
        ...     portfolio_col='portfolio',
        ...     benchmark_col='benchmark',
        ... )
    """
    results: list[pd.DataFrame] = []

    # Group by the specified column
    grouped = data.groupby(group_by)

    for group_name, group_data in grouped:
        # Sort by date to ensure proper linking
        group_data = group_data.sort_values(date_col)

        # Extract required series for link()
        effects = group_data[effects_cols]
        portfolio_returns = group_data[portfolio_col]
        benchmark_returns = group_data[benchmark_col]

        # Call the existing link function
        result = link(
            effects=effects,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            method=method,
            unit=unit,
            check_effects_sum=check_effects_sum,
        )

        # Build the output row for this group
        # Get the total row (linked effects) from result.data
        result_data = result.data

        # Extract period data (all rows except 'Total')
        period_data = result_data.drop(index='Total', errors='ignore')

        # Build output DataFrame for this group
        output_df = pd.DataFrame({
            'DATE': group_data[date_col].values,
            'FUND_ID': group_name,
            'portfolio_return': period_data['Portfolio Return'].values,
            'benchmark_return': period_data['Benchmark Return'].values,
            'active_return': period_data['Active Return'].values,
        })

        # Add linked effect columns
        for effect_col in effects_cols:
            # Get the linked effect value from the Total row
            linked_value = result_data.loc['Total', effect_col]
            # For each period, we could either:
            # 1. Repeat the linked value (shows same total for each period)
            # 2. Just show period-level effects
            # Based on the requirements, let's include the linked effect 
            # Repeat the linked total for each row (meaningful for comparison)
            output_df[effect_col] = linked_value

        results.append(output_df)

    # Combine all group results
    combined = pd.concat(results, ignore_index=True)

    # Reorder columns: DATE, FUND_ID, returns, effects
    column_order = ['DATE', 'FUND_ID', 'portfolio_return', 'benchmark_return', 'active_return'] + effects_cols
    combined = combined[column_order]

    return combined
