from __future__ import annotations

"""Public analysis helper API used by the main notebook and tests.

The implementation still delegates to the historical module so older codepaths
keep working, but callers should import from this module going forward.
"""

from helpers.legacy_emissions_stats import (
    LEGACY_DEFAULT_CONFIG as DEFAULT_CONFIG,
    build_legacy_plot_data as build_aligned_plot_data,
    compare_reconstructed_to_published as compare_results_to_reference,
    published_reference_table as reference_stats_table,
    reconstruct_legacy_stats_table as reconstruct_stats_table,
)
