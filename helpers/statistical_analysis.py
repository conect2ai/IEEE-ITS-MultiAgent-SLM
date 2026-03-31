from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, kruskal, norm, rankdata, shapiro, wilcoxon


def aggregate_metric_by_block(
    df: pd.DataFrame,
    metric: str,
    block_col: str = "vehicle",
    model_col: str = "model",
    aggfunc: str = "median",
) -> pd.DataFrame:
    required = {block_col, model_col, metric}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    aggregated = (
        df[[block_col, model_col, metric]]
        .dropna(subset=[metric])
        .groupby([block_col, model_col], as_index=False)[metric]
        .agg(aggfunc)
        .pivot(index=block_col, columns=model_col, values=metric)
        .dropna(axis=0, how="any")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return aggregated


def shapiro_tests_by_model(pivot_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for model in pivot_df.columns:
        values = pivot_df[model].dropna().to_numpy()
        if len(values) < 3:
            rows.append(
                {
                    "model": model,
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "is_normal": False,
                }
            )
            continue

        statistic, p_value = shapiro(values)
        rows.append(
            {
                "model": model,
                "statistic": statistic,
                "p_value": p_value,
                "is_normal": bool(p_value > alpha),
            }
        )

    return pd.DataFrame(rows)


def friedman_test(pivot_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    clean = pivot_df.dropna(axis=0, how="any")
    n_blocks = len(clean)
    n_models = len(clean.columns)

    if n_models < 3 or n_blocks < 1:
        raise ValueError(
            "Friedman test requires at least 3 model columns and 1 complete block "
            f"after dropping missing values; got {n_models} model columns and {n_blocks} complete blocks."
        )

    statistic, p_value = friedmanchisquare(*[clean[col].to_numpy() for col in clean.columns])

    kendalls_w = statistic / (n_blocks * (n_models - 1))

    return {
        "statistic": statistic,
        "p_value": p_value,
        "is_significant": bool(p_value < alpha),
        "kendalls_w": kendalls_w,
        "n_blocks": n_blocks,
        "n_models": n_models,
    }


def pairwise_wilcoxon_tests(pivot_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for model_a, model_b in combinations(pivot_df.columns, 2):
        paired = pivot_df[[model_a, model_b]].dropna()
        x = paired[model_a].to_numpy()
        y = paired[model_b].to_numpy()

        statistic, p_value = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "statistic": statistic,
                "p_value": p_value,
                "rank_biserial": _rank_biserial_from_pairs(x, y),
                "a12": _vargha_delaney_a12(x, y),
            }
        )

    results = pd.DataFrame(rows)
    results["p_value_holm"] = _holm_adjust(results["p_value"].to_list())
    results["is_significant"] = results["p_value_holm"] < alpha
    return results.sort_values(["p_value_holm", "p_value"]).reset_index(drop=True)


def kruskal_wallis_test(
    df: pd.DataFrame,
    metric: str,
    model_col: str = "model",
    alpha: float = 0.05,
) -> dict:
    clean = df[[model_col, metric]].dropna()
    groups = [
        group[metric].to_numpy()
        for _, group in clean.groupby(model_col, sort=True)
    ]

    statistic, p_value = kruskal(*groups)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "is_significant": bool(p_value < alpha),
        "n_models": len(groups),
        "n_observations": len(clean),
    }


def pairwise_dunn_tests(
    df: pd.DataFrame,
    metric: str,
    model_col: str = "model",
    alpha: float = 0.05,
) -> pd.DataFrame:
    clean = df[[model_col, metric]].dropna().copy()
    clean["_rank"] = rankdata(clean[metric].to_numpy())
    n_total = len(clean)
    tie_correction = _tie_correction_factor(clean[metric].to_numpy())

    rows = []
    for model_a, model_b in combinations(sorted(clean[model_col].unique()), 2):
        group_a = clean[clean[model_col] == model_a]
        group_b = clean[clean[model_col] == model_b]

        mean_rank_a = group_a["_rank"].mean()
        mean_rank_b = group_b["_rank"].mean()
        n_a = len(group_a)
        n_b = len(group_b)

        variance = (n_total * (n_total + 1) / 12.0) * tie_correction * ((1.0 / n_a) + (1.0 / n_b))
        z_value = (mean_rank_a - mean_rank_b) / np.sqrt(variance)
        p_value = 2 * norm.sf(abs(z_value))

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "z_value": z_value,
                "p_value": p_value,
                "cliffs_delta": cliffs_delta(group_a[metric].to_numpy(), group_b[metric].to_numpy()),
            }
        )

    results = pd.DataFrame(rows)
    m = len(results)
    results["p_value_bonferroni"] = np.minimum(1.0, results["p_value"] * m)
    results["is_significant"] = results["p_value_bonferroni"] < alpha
    return results.sort_values(["p_value_bonferroni", "p_value"]).reset_index(drop=True)


def cliffs_delta(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    greater = 0
    lower = 0
    for x_val in x_arr:
        greater += np.sum(x_val > y_arr)
        lower += np.sum(x_val < y_arr)
    return float((greater - lower) / (len(x_arr) * len(y_arr)))


def _holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (original_index, p_value) in enumerate(indexed):
        candidate = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max

    return adjusted


def _tie_correction_factor(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    numerator = np.sum(counts**3 - counts)
    denominator = len(values) ** 3 - len(values)
    if denominator == 0:
        return 1.0
    return float(1.0 - (numerator / denominator))


def _rank_biserial_from_pairs(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    non_zero = diff[diff != 0]
    if len(non_zero) == 0:
        return 0.0

    ranks = rankdata(np.abs(non_zero))
    positive_rank_sum = ranks[non_zero > 0].sum()
    negative_rank_sum = ranks[non_zero < 0].sum()
    total_rank_sum = len(non_zero) * (len(non_zero) + 1) / 2
    return float((positive_rank_sum - negative_rank_sum) / total_rank_sum)


def _vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    greater = 0
    equal = 0
    for x_val in x:
        greater += np.sum(x_val > y)
        equal += np.sum(x_val == y)
    return float((greater + 0.5 * equal) / (len(x) * len(y)))
