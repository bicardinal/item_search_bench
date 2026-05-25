import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ENGINE_ORDER = [
	"brinicle",
	"opensearch",
	"weaviate",
	"typesense",
	"meilisearch",
]

ENGINE_LABELS = {
	"brinicle": "Brinicle",
	"opensearch": "OpenSearch",
	"weaviate": "Weaviate",
	"typesense": "Typesense",
	"meilisearch": "Meilisearch",
}

DATASET_ORDER = ["WANDS", "ESCI"]

KS_DEFAULT = [1, 5, 10, 20, 50, 100]

RELEVANCE_FAMILIES = ["nDCG", "Recall", "Hit", "MRR"]

PAPER_DPI = 240

ENGINE_HATCHES = {
    "brinicle": "///",
    "opensearch": "\\\\\\",
    "weaviate": "xxx",
    "typesense": "...",
    "meilisearch": "++",
}

ENGINE_MARKERS = {
    "brinicle": "o",
    "opensearch": "s",
    "weaviate": "^",
    "typesense": "D",
    "meilisearch": "P",
}

LINESTYLES = {
    "brinicle": "-",
    "opensearch": "--",
    "weaviate": "-.",
    "typesense": ":",
    "meilisearch": (0, (5, 2, 1, 2)),
}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "0.7",
        "axes.edgecolor": "0.75",
        "axes.linewidth": 1.0,
        "grid.color": "0.85",
        "grid.linewidth": 0.8,
    })

def read_json(path: Path) -> Dict[str, Any]:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def normalize_engine_name(value: Optional[str], path: Path) -> str:
	if value:
		value = value.strip().lower()
		for engine in ENGINE_ORDER:
			if value == engine:
				return engine

	name = path.name.lower()
	for engine in ENGINE_ORDER:
		if name.startswith(engine + "_") or engine in name:
			return engine

	raise ValueError(f"Could not infer engine from: {path}")


def normalize_dataset_name(value: Optional[str], path: Path) -> str:
	if value:
		value = value.strip().upper()
		if value in {"ESCI", "WANDS"}:
			return value

	parts = [p.lower() for p in path.parts]
	joined = "/".join(parts)

	if "esci_results" in joined or "esci" in path.name.lower():
		return "ESCI"

	if "wands_results" in joined or "wands" in path.name.lower():
		return "WANDS"

	raise ValueError(f"Could not infer dataset from: {path}")


def get_results_block(data: Dict[str, Any]) -> Dict[str, Any]:

	if isinstance(data.get("results"), dict):
		return data["results"]
	return data


def get_ks(data: Dict[str, Any], results: Dict[str, Any]) -> List[int]:
	if isinstance(data.get("ks"), list) and data["ks"]:
		return [int(k) for k in data["ks"]]

	found = set()
	pattern = re.compile(r"^(nDCG|Recall|Hit|MRR)@(\d+)$")
	for key in results.keys():
		m = pattern.match(key)
		if m:
			found.add(int(m.group(2)))

	if found:
		return sorted(found)

	return KS_DEFAULT


def seconds_to_ms(value: Optional[float]) -> Optional[float]:
	if value is None:
		return None
	return float(value) * 1000.0


def safe_float(value: Any) -> Optional[float]:
	if value is None:
		return None
	try:
		return float(value)
	except Exception:
		return None


def collect_results(root: Path) -> pd.DataFrame:
	json_paths = sorted(root.glob("*_results/*.json"))

	if not json_paths:
		raise FileNotFoundError(
			f"No JSON result files found under {root}/<dataset>_results/*.json"
		)

	rows: List[Dict[str, Any]] = []

	for path in json_paths:
		data = read_json(path)
		results = get_results_block(data)

		engine = normalize_engine_name(data.get("engine"), path)
		dataset = normalize_dataset_name(data.get("dataset"), path)
		ks = get_ks(data, results)

		build_profile = results.get("build_memory_profile") or {}

		row: Dict[str, Any] = {
			"dataset": dataset,
			"engine": engine,
			"engine_label": ENGINE_LABELS.get(engine, engine),
			"file": str(path),
			"docs": data.get("docs"),
			"queries": data.get("queries"),
			"eval_queries": data.get("eval_queries"),
			"trials": data.get("trials"),
			"retrieval_mode": data.get("retrieval_mode"),
			"search_mode": data.get("search_mode"),
			"search_batch_size": data.get("search_batch_size"),
			"search_batch_jobs": data.get("search_batch_jobs"),
			"tune_metric": (data.get("tuning") or {}).get("metric"),
			"best_alpha": (data.get("tuning") or {}).get("best_alpha"),
			"search_avg_latency_ms": seconds_to_ms(results.get("search_avg_latency")),
			"search_p50_latency_ms": seconds_to_ms(results.get("search_p50_latency")),
			"search_p95_latency_ms": seconds_to_ms(results.get("search_p95_latency")),
			"search_p99_latency_ms": seconds_to_ms(results.get("search_p99_latency")),
			"qps": safe_float(results.get("qps")),
			"search_total_query_time_s": safe_float(results.get("search_total_query_time")),
			"search_raw_peak_mb": safe_float(results.get("raw_peak_mb")),
			"search_working_set_peak_mb": safe_float(results.get("working_set_peak_mb")),
			"search_anon_peak_mb": safe_float(results.get("anon_peak_mb")),
			"search_file_peak_mb": safe_float(results.get("file_peak_mb")),
			"build_latency_s": safe_float(results.get("build_latency")),
			"build_raw_peak_mb": safe_float(build_profile.get("raw_peak_mb")),
			"build_working_set_peak_mb": safe_float(build_profile.get("working_set_peak_mb")),
			"build_anon_peak_mb": safe_float(build_profile.get("anon_peak_mb")),
			"build_file_peak_mb": safe_float(build_profile.get("file_peak_mb")),
		}

		for family in RELEVANCE_FAMILIES:
			for k in ks:
				row[f"{family}@{k}"] = safe_float(results.get(f"{family}@{k}"))

		rows.append(row)

	df = pd.DataFrame(rows)

	df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
	df["engine"] = pd.Categorical(df["engine"], ENGINE_ORDER, ordered=True)
	df = df.sort_values(["dataset", "engine"]).reset_index(drop=True)

	return df


def make_relevance_long(df: pd.DataFrame) -> pd.DataFrame:
	rows = []

	for _, row in df.iterrows():
		for family in RELEVANCE_FAMILIES:
			for k in KS_DEFAULT:
				col = f"{family}@{k}"
				if col in df.columns and pd.notna(row.get(col)):
					rows.append(
						{
							"dataset": row["dataset"],
							"engine": row["engine"],
							"engine_label": row["engine_label"],
							"metric": family,
							"k": k,
							"value": float(row[col]),
						}
					)

	return pd.DataFrame(rows)


def savefig(out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)

    path_png = out / f"{name}.png"
    path_svg = out / f"{name}.svg"
    path_pdf = out / f"{name}.pdf"

    plt.tight_layout()
    plt.savefig(path_png, dpi=PAPER_DPI, bbox_inches="tight")
    plt.savefig(path_svg, bbox_inches="tight")
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.close()

def format_engine_axis(ax) -> None:
	ax.grid(True, axis="x", alpha=0.25)
	ax.set_axisbelow(True)


def plot_latency_bars(df: pd.DataFrame, out: Path) -> None:
    for dataset in DATASET_ORDER:
        sub = df[df["dataset"] == dataset].copy()
        sub = sub.sort_values("search_p99_latency_ms", ascending=True)

        fig, ax = plt.subplots(figsize=(11, 6))

        bars = ax.bar(
            sub["engine_label"],
            sub["search_p99_latency_ms"],
            color="white",
            edgecolor="black",
            linewidth=1.4,
        )

        for bar, engine in zip(bars, sub["engine"]):
            bar.set_hatch(ENGINE_HATCHES.get(str(engine), "///"))

        ax.set_yscale("log")
        ax.set_ylabel("p99 Query Latency (ms, log scale)")
        ax.set_xlabel("Search Engine")
        ax.set_title(f"p99 Query Latency ({dataset})")

        ax.grid(True, axis="y", alpha=0.55)
        ax.set_axisbelow(True)

        for bar, value in zip(bars, sub["search_p99_latency_ms"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.xticks(rotation=35, ha="right")
        savefig(out, f"{dataset.lower()}_p99_latency_bw")

def plot_memory_bars(df: pd.DataFrame, out: Path) -> None:
    for dataset in DATASET_ORDER:
        sub = df[df["dataset"] == dataset].copy()
        sub = sub.sort_values("search_raw_peak_mb", ascending=True)

        fig, ax = plt.subplots(figsize=(11, 6))

        bars = ax.bar(
            sub["engine_label"],
            sub["search_raw_peak_mb"],
            color="white",
            edgecolor="black",
            linewidth=1.4,
        )

        for bar, engine in zip(bars, sub["engine"]):
            bar.set_hatch(ENGINE_HATCHES.get(str(engine), "///"))

        ax.set_yscale("log")
        ax.set_ylabel("Peak Search Memory (MB, log scale)")
        ax.set_xlabel("Search Engine")
        ax.set_title(f"Peak Search Memory ({dataset})")

        ax.grid(True, axis="y", alpha=0.55)
        ax.set_axisbelow(True)

        for bar, value in zip(bars, sub["search_raw_peak_mb"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.xticks(rotation=35, ha="right")
        savefig(out, f"{dataset.lower()}_search_memory_bw")

def plot_build_vs_search_memory(df: pd.DataFrame, out: Path) -> None:
    for dataset in DATASET_ORDER:
        sub = df[df["dataset"] == dataset].copy()
        sub = sub.sort_values("search_raw_peak_mb", ascending=True)

        x = np.arange(len(sub))
        width = 0.34

        fig, ax = plt.subplots(figsize=(12, 6.5))

        build = ax.bar(
            x - width / 2,
            sub["build_raw_peak_mb"],
            width,
            label="Build Memory",
            color="white",
            edgecolor="black",
            linewidth=1.4,
            hatch="//",
        )

        search = ax.bar(
            x + width / 2,
            sub["search_raw_peak_mb"],
            width,
            label="Search Memory",
            color="0.86",
            edgecolor="black",
            linewidth=1.4,
            hatch="\\\\",
        )

        ax.set_ylabel("Memory Usage (MB)")
        ax.set_xlabel("Search Engine")
        ax.set_title(f"Memory Usage: Build vs Search ({dataset})")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["engine_label"], rotation=35, ha="right")

        ax.grid(True, axis="y", alpha=0.45)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left")

        ymax = max(
            sub["build_raw_peak_mb"].max(),
            sub["search_raw_peak_mb"].max(),
        )
        ax.set_ylim(0, ymax * 1.18)

        for bars in [build, search]:
            for bar in bars:
                value = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        savefig(out, f"{dataset.lower()}_build_vs_search_memory_bw")


def plot_build_cost(df: pd.DataFrame, out: Path) -> None:
    for dataset in DATASET_ORDER:
        sub = df[df["dataset"] == dataset].copy()

        fig, ax = plt.subplots(figsize=(9, 6.5))

        for _, row in sub.iterrows():
            x = row["build_latency_s"]
            y = row["build_raw_peak_mb"]

            if pd.isna(x) or pd.isna(y):
                continue

            engine = str(row["engine"])

            ax.scatter(
                x,
                y,
                s=160,
                marker=ENGINE_MARKERS.get(engine, "o"),
                facecolors="white",
                edgecolors="black",
                linewidths=1.5,
            )

            ax.text(
                x,
                y,
                f" {row['engine_label']}",
                va="center",
                fontsize=10,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Build Time (seconds, log scale)")
        ax.set_ylabel("Peak Build Memory (MB, log scale)")
        ax.set_title(f"Build Cost: Time vs Memory ({dataset})")

        ax.grid(True, alpha=0.45)
        ax.set_axisbelow(True)

        savefig(out, f"{dataset.lower()}_build_cost_time_vs_memory_bw")


def plot_pareto(df: pd.DataFrame, out: Path, relevance_metric: str) -> None:
    for dataset in DATASET_ORDER:
        sub = df[df["dataset"] == dataset].copy()

        if relevance_metric not in sub.columns:
            continue

        fig, ax = plt.subplots(figsize=(9, 6.5))

        memory = sub["search_raw_peak_mb"].astype(float)
        finite_memory = memory.replace([np.inf, -np.inf], np.nan).dropna()

        if finite_memory.empty:
            sizes = pd.Series(np.repeat(180, len(sub)), index=sub.index)
        else:
            min_m = finite_memory.min()
            max_m = finite_memory.max()

            if math.isclose(min_m, max_m):
                sizes = pd.Series(np.repeat(220, len(sub)), index=sub.index)
            else:
                sizes = 120 + 850 * ((memory - min_m) / (max_m - min_m))

        for idx, row in sub.iterrows():
            x = row["search_p99_latency_ms"]
            y = row[relevance_metric]

            if pd.isna(x) or pd.isna(y):
                continue

            engine = str(row["engine"])

            ax.scatter(
                x,
                y,
                s=float(sizes.loc[idx]),
                marker=ENGINE_MARKERS.get(engine, "o"),
                facecolors="white",
                edgecolors="black",
                linewidths=1.5,
                label=row["engine_label"],
            )

            ax.text(
                x,
                y,
                f" {row['engine_label']}",
                va="center",
                fontsize=10,
            )

        ax.set_xscale("log")
        ax.set_xlabel("p99 Query Latency (ms, log scale)")
        ax.set_ylabel(relevance_metric)
        ax.set_title(f"Hybrid Search Pareto: {relevance_metric} vs p99 Latency ({dataset})")

        ax.grid(True, alpha=0.45)
        ax.set_axisbelow(True)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best", title="Bubble size = memory")

        savefig(out, f"{dataset.lower()}_pareto_{relevance_metric.replace('@', '_at_')}_bw")

def plot_relevance_curves(long_df: pd.DataFrame, out: Path) -> None:
    for dataset in DATASET_ORDER:
        for metric in RELEVANCE_FAMILIES:
            sub = long_df[
                (long_df["dataset"] == dataset)
                & (long_df["metric"] == metric)
            ].copy()

            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            for engine in ENGINE_ORDER:
                e = sub[sub["engine"] == engine].sort_values("k")
                if e.empty:
                    continue

                label = ENGINE_LABELS.get(engine, engine)

                ax.plot(
                    e["k"],
                    e["value"],
                    label=label,
                    color="black",
                    linewidth=2.0,
                    marker=ENGINE_MARKERS.get(engine, "o"),
                    markersize=6,
                    linestyle=LINESTYLES.get(engine, "-"),
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=1.2,
                )

            ax.set_xscale("log")
            ax.set_xticks(KS_DEFAULT)
            ax.set_xticklabels([str(k) for k in KS_DEFAULT])
            ax.set_xlabel("k")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric}@k ({dataset})")

            ax.grid(True, alpha=0.45)
            ax.set_axisbelow(True)
            ax.legend(loc="best")

            savefig(out, f"{dataset.lower()}_{metric.lower()}_curve_bw")

def plot_compact_leaderboard(df: pd.DataFrame, out: Path) -> None:
	cols = [
		"dataset",
		"engine_label",
		"nDCG@10",
		"nDCG@100",
		"Recall@100",
		"Hit@100",
		"MRR@100",
		"search_p99_latency_ms",
		"search_raw_peak_mb",
		"build_latency_s",
		"build_raw_peak_mb",
	]

	existing = [c for c in cols if c in df.columns]
	compact = df[existing].copy()

	compact = compact.rename(
		columns={
			"engine_label": "engine",
			"search_p99_latency_ms": "p99_latency_ms",
			"search_raw_peak_mb": "search_peak_mb",
			"build_latency_s": "build_time_s",
			"build_raw_peak_mb": "build_peak_mb",
		}
	)

	compact.to_csv(out / "compact_leaderboard.csv", index=False)


def print_summary(df: pd.DataFrame) -> None:
	print("\nLoaded benchmark files:")
	for _, row in df.iterrows():
		print(
			f"  {row['dataset']:5s} | {row['engine_label']:11s} | "
			f"p99={row['search_p99_latency_ms']:.3f} ms | "
			f"mem={row['search_raw_peak_mb']:.1f} MB | "
			f"nDCG@100={row.get('nDCG@100', float('nan')):.4f}"
		)

	print("\nBest by dataset/metric:")

	metrics = [
		("nDCG@10", False),
		("nDCG@100", False),
		("Recall@100", False),
		("MRR@100", False),
		("search_p99_latency_ms", True),
		("search_raw_peak_mb", True),
		("build_raw_peak_mb", True),
	]

	for dataset in DATASET_ORDER:
		sub = df[df["dataset"] == dataset]
		if sub.empty:
			continue

		print(f"\n{dataset}")

		for metric, lower_is_better in metrics:
			if metric not in sub.columns:
				continue

			valid = sub[pd.notna(sub[metric])]
			if valid.empty:
				continue

			if lower_is_better:
				best = valid.loc[valid[metric].idxmin()]
				direction = "lowest"
			else:
				best = valid.loc[valid[metric].idxmax()]
				direction = "highest"

			print(
				f"  {metric:22s} {direction:7s}: "
				f"{best['engine_label']} ({best[metric]:.4f})"
			)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--root",
		type=Path,
		default=Path("benchmark"),
		help="Benchmark root directory containing esci_results/ and wands_results/",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=Path("benchmark_figures"),
		help="Output directory for figures and CSV files",
	)
	parser.add_argument(
		"--pareto-metrics",
		nargs="+",
		default=["nDCG@10", "nDCG@100"],
		help="Relevance metrics to use for Pareto charts",
	)

	args = parser.parse_args()

	args.out.mkdir(parents=True, exist_ok=True)

	df = collect_results(args.root)
	long_df = make_relevance_long(df)

	df.to_csv(args.out / "benchmark_summary_wide.csv", index=False)
	long_df.to_csv(args.out / "benchmark_relevance_long.csv", index=False)
	apply_paper_style()
	plot_compact_leaderboard(df, args.out)

	for metric in args.pareto_metrics:
		plot_pareto(df, args.out, metric)

	plot_relevance_curves(long_df, args.out)
	plot_latency_bars(df, args.out)
	plot_memory_bars(df, args.out)
	plot_build_cost(df, args.out)
	plot_build_vs_search_memory(df, args.out)
	print_summary(df)

	print(f"\nDone. Figures and CSV files written to: {args.out.resolve()}")


if __name__ == "__main__":
	main()
