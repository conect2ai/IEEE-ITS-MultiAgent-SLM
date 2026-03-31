from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from helpers.statistical_analysis import cliffs_delta, kruskal_wallis_test


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LegacySourceConfig:
    llm_snapshot_dir: Path
    codecarbon_dir: Path


def _default_llm_snapshot_dir() -> Path:
    preferred = Path("data/processed/emissions")
    fallback = Path("data/processed/emissinos_pendrive")

    if (REPO_ROOT / preferred).exists():
        return preferred
    return fallback


LEGACY_DEFAULT_CONFIG = LegacySourceConfig(
    llm_snapshot_dir=_default_llm_snapshot_dir(),
    codecarbon_dir=Path("data/processed/codecarbon"),
)


def _resolve_repo_path(path_like: Path | str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def load_legacy_llm_snapshot(base_dir: Path | str) -> pd.DataFrame:
    base_path = _resolve_repo_path(base_dir)
    all_dfs: list[pd.DataFrame] = []

    for model in ("gemma", "qwen", "smollm2"):
        for csv_path in sorted((base_path / model).glob("*.csv")):
            vehicle = csv_path.stem.replace("_emissions", "")
            df = pd.read_csv(csv_path, low_memory=False)
            df["model"] = model
            df["vehicle"] = vehicle
            all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No legacy LLM snapshot CSVs found under {base_path}")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_llm = df_all[df_all["llm_total_tokens"].notna()].copy()

    for column in ("llm_total_ms_client", "llm_output_tokens", "llm_total_tokens"):
        df_llm[column] = pd.to_numeric(df_llm[column], errors="coerce")

    df_llm = df_llm[df_llm["llm_total_ms_client"] > 0].copy()
    df_llm["llm_tps"] = df_llm["llm_output_tokens"] / (df_llm["llm_total_ms_client"] / 1000.0)
    return df_llm


def load_codecarbon_snapshot(base_dir: Path | str) -> pd.DataFrame:
    base_path = _resolve_repo_path(base_dir)
    all_dfs: list[pd.DataFrame] = []

    for csv_path in sorted(base_path.glob("*_emissions.csv")):
        stem = csv_path.stem.replace("_emissions", "")
        model, vehicle = stem.split("_", 1)
        df = pd.read_csv(csv_path, low_memory=False)
        df["model"] = model
        df["vehicle"] = vehicle
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No CodeCarbon CSVs found under {base_path}")

    df_all = pd.concat(all_dfs, ignore_index=True)
    for column in ("duration", "energy_consumed", "emissions"):
        df_all[column] = pd.to_numeric(df_all[column], errors="coerce")
    return df_all


def build_legacy_plot_data(config: LegacySourceConfig = LEGACY_DEFAULT_CONFIG) -> pd.DataFrame:
    llm_df = load_legacy_llm_snapshot(config.llm_snapshot_dir).copy()
    codecarbon_df = load_codecarbon_snapshot(config.codecarbon_dir).copy()

    llm_df["inference_idx"] = llm_df.groupby(["model", "vehicle"]).cumcount()
    codecarbon_df["inference_idx"] = codecarbon_df.groupby(["model", "vehicle"]).cumcount()

    merged = llm_df.merge(
        codecarbon_df[
            ["model", "vehicle", "inference_idx", "duration", "energy_consumed", "emissions"]
        ],
        on=["model", "vehicle", "inference_idx"],
        how="inner",
        validate="one_to_one",
    )
    return merged


def independent_stats_row(df: pd.DataFrame, metric: str, label: str) -> dict[str, float | str]:
    metric_df = df[["model", metric]].dropna().copy()
    if metric_df["model"].nunique() != 3:
        raise ValueError(f"Metric {metric} does not contain all 3 models")

    result = kruskal_wallis_test(metric_df, metric)
    gemma = metric_df.loc[metric_df["model"] == "gemma", metric].to_numpy()
    qwen = metric_df.loc[metric_df["model"] == "qwen", metric].to_numpy()
    smollm2 = metric_df.loc[metric_df["model"] == "smollm2", metric].to_numpy()

    return {
        "Metric": label,
        "H": round(float(result["statistic"]), 1),
        "Q vs. S": round(cliffs_delta(qwen, smollm2), 3),
        "G vs. Q": round(cliffs_delta(gemma, qwen), 3),
        "G vs. S": round(cliffs_delta(gemma, smollm2), 3),
        "n": int(len(metric_df)),
    }


def reconstruct_legacy_stats_table(config: LegacySourceConfig = LEGACY_DEFAULT_CONFIG) -> pd.DataFrame:
    llm_df = load_legacy_llm_snapshot(config.llm_snapshot_dir)
    codecarbon_df = load_codecarbon_snapshot(config.codecarbon_dir)

    rows = [
        independent_stats_row(llm_df, "llm_total_ms_client", "Latency"),
        independent_stats_row(llm_df, "llm_output_tokens", "Output tokens"),
        independent_stats_row(llm_df, "llm_tps", "Throughput (TPS)"),
        independent_stats_row(codecarbon_df, "energy_consumed", "Energy"),
        independent_stats_row(codecarbon_df, "emissions", "CO2 emissions"),
    ]
    return pd.DataFrame(rows)


def published_reference_table() -> pd.DataFrame:
    rows = [
        {"Metric": "Latency", "H": 1596.0, "Q vs. S": 0.906, "G vs. Q": -0.677, "G vs. S": 0.593},
        {"Metric": "Output tokens", "H": 513.4, "Q vs. S": 0.595, "G vs. Q": -0.367, "G vs. S": 0.107},
        {"Metric": "Throughput (TPS)", "H": 1214.3, "Q vs. S": -0.732, "G vs. Q": 0.701, "G vs. S": -0.499},
        {"Metric": "Energy", "H": 1369.7, "Q vs. S": 0.856, "G vs. Q": -0.695, "G vs. S": 0.394},
        {"Metric": "CO2 emissions", "H": 1357.5, "Q vs. S": 0.852, "G vs. Q": -0.691, "G vs. S": 0.398},
    ]
    return pd.DataFrame(rows)


def compare_reconstructed_to_published(config: LegacySourceConfig = LEGACY_DEFAULT_CONFIG) -> pd.DataFrame:
    published = published_reference_table()
    reconstructed = reconstruct_legacy_stats_table(config)

    merged = published.merge(reconstructed, on="Metric", suffixes=("_published", "_reconstructed"))
    for column in ("H", "Q vs. S", "G vs. Q", "G vs. S"):
        merged[f"{column} delta"] = (
            merged[f"{column}_reconstructed"] - merged[f"{column}_published"]
        ).round(3)
    return merged
