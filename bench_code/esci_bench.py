import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

from aux.shared_aux import (
	load_queries,
	build_qrels_from_labels,
	evaluate_results,
	summarize_latencies,
	run_search,
	TypesenseAdapter,
	BrinicleItemAdapter,
	MeilisearchAdapter,
	BrinicleLocalItemAdapter,
	WeaviateAdapter,
	OpenSearchAdapter,
)

try:
	from aux.memory_inspect import CgroupMemoryMonitor
except ImportError:
	CgroupMemoryMonitor = None


DEFAULT_ESCI_PATH = Path("datasets/ESCI/esci_us_test_title_only.jsonl")
DEFAULT_OUTPUT_DIR = Path("benchmark/esci_results")

DEFAULT_KS = [1, 5, 10, 20, 50, 100]

SHARED_ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

CONTAINER_NAMES = {
	"brinicle": "brinicle_container",
	"typesense": "typesense",
	"meilisearch": "meilisearch",
	"weaviate": "weaviate",
	"opensearch": "opensearch",
}

ESCI_LABEL_MAPS = {
	"graded": {
		"E": 1.0,
		"S": 0.1,
		"C": 0.01,
		"I": 0.0,
	},
	"exact": {
		"E": 1.0,
		"S": 0.0,
		"C": 0.0,
		"I": 0.0,
	},
}

def _resolve_sidecar_path(
	jsonl_path: Path,
	meta: Dict[str, Any],
	key: str,
	default_suffix: str,
) -> Path:

	embedding_meta = meta.get("embedding", {}) or {}
	raw_path = embedding_meta.get(key)

	candidates = []

	if raw_path:
		p = Path(raw_path)
		candidates.append(p)

		if not p.is_absolute():
			candidates.append(jsonl_path.parent / p)
			candidates.append(jsonl_path.parent / p.name)

	stem = jsonl_path.with_suffix("")
	candidates.append(stem.with_suffix(default_suffix))

	for candidate in candidates:
		if candidate.exists():
			return candidate

	raise FileNotFoundError(
		f"Could not resolve sidecar path for {key}. Tried: "
		+ ", ".join(str(c) for c in candidates)
	)


def load_esci_preprocessed_jsonl(
	path: Path,
	source: Optional[str] = None,
	retrieval_mode: str = "lexical",
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:

	if retrieval_mode not in {"lexical", "hybrid"}:
		raise ValueError("retrieval_mode must be one of: lexical, hybrid")

	load_vectors = retrieval_mode == "hybrid"

	items: List[Dict[str, Any]] = []
	queries: List[Dict[str, Any]] = []
	labels: List[Dict[str, Any]] = []
	meta: Dict[str, Any] = {}

	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue

			record = json.loads(line)
			record_type = record.get("type")

			if record_type == "meta":
				meta = record

			elif record_type == "document":
				items.append({
					"id": str(record["id"]),
					"title": str(record.get("title", "") or ""),
					"category": "",
					"subcategory": "",
					"attributes": {},
					"vector_index": (
						int(record["vector_index"])
						if "vector_index" in record
						else None
					),
				})

			elif record_type == "query":
				if source is not None and record.get("source") != source:
					continue

				queries.append({
					"query_id": int(record["query_id"]),
					"query": str(record.get("query", "") or ""),
					"split": record.get("split"),
					"source": record.get("source"),
					"small_version": int(record.get("small_version", 0)),
					"large_version": int(record.get("large_version", 1)),
					"vector_index": (
						int(record["vector_index"])
						if "vector_index" in record
						else None
					),
				})

			elif record_type == "qrel":
				if source is not None and record.get("source") != source:
					continue

				labels.append({
					"query_id": int(record["query_id"]),
					"doc_id": str(record["doc_id"]),
					"esci_label": str(record["esci_label"]),
					"gain_exact": float(record.get("gain_exact", 0.0)),
					"gain_graded": float(record.get("gain_graded", 0.0)),
					"split": record.get("split"),
					"source": record.get("source"),
					"small_version": int(record.get("small_version", 0)),
					"large_version": int(record.get("large_version", 1)),
				})

	if not items:
		raise ValueError(f"No documents loaded from {path}")

	if not queries:
		raise ValueError(f"No queries loaded from {path}")

	if not labels:
		raise ValueError(f"No qrels loaded from {path}")

	if load_vectors:
		document_vectors_path = _resolve_sidecar_path(
			jsonl_path=path,
			meta=meta,
			key="document_vectors_path",
			default_suffix=".documents.npy",
		)

		query_vectors_path = _resolve_sidecar_path(
			jsonl_path=path,
			meta=meta,
			key="query_vectors_path",
			default_suffix=".queries.npy",
		)

		print(f"[load] document_vectors={document_vectors_path}")
		print(f"[load] query_vectors={query_vectors_path}")

		document_vectors = np.load(document_vectors_path, mmap_mode="r")
		query_vectors = np.load(query_vectors_path, mmap_mode="r")

		if document_vectors.ndim != 2:
			raise ValueError(
				f"Document vectors must be 2-D, got shape={document_vectors.shape}"
			)

		if query_vectors.ndim != 2:
			raise ValueError(
				f"Query vectors must be 2-D, got shape={query_vectors.shape}"
			)

		for item in items:
			vector_index = item.get("vector_index")
			if vector_index is None:
				raise ValueError(
					f"Document {item['id']} has no vector_index in hybrid mode"
				)

			if vector_index < 0 or vector_index >= len(document_vectors):
				raise IndexError(
					f"Bad document vector_index={vector_index} for {item['id']}"
				)

			item["vector"] = document_vectors[vector_index]

		for query in queries:
			vector_index = query.get("vector_index")
			if vector_index is None:
				raise ValueError(
					f"Query {query['query_id']} has no vector_index in hybrid mode"
				)

			if vector_index < 0 or vector_index >= len(query_vectors):
				raise IndexError(
					f"Bad query vector_index={vector_index} for query_id={query['query_id']}"
				)

			query["vector"] = query_vectors[vector_index]

		meta.setdefault("embedding", {})
		meta["embedding"]["document_vectors_loaded_from"] = str(document_vectors_path)
		meta["embedding"]["query_vectors_loaded_from"] = str(query_vectors_path)
		meta["embedding"]["resolved_dimension"] = int(document_vectors.shape[1])

		if document_vectors.shape[1] != query_vectors.shape[1]:
			raise ValueError(
				f"Document/query vector dims mismatch: "
				f"{document_vectors.shape[1]} vs {query_vectors.shape[1]}"
			)

	queries_df = pd.DataFrame(queries)
	labels_df = pd.DataFrame(labels)

	return items, queries_df, labels_df, meta

def set_adapter_alpha(args, adapter, alpha: float):
	alpha = float(alpha)

	if args.engine in {"brinicle", "brinicle_inprocess"}:
		args.brinicle_alpha = alpha
		adapter.set_alpha(alpha)
		return

	if args.engine == "typesense":
		args.typesense_alpha = alpha
		adapter.alpha = alpha
		return

	if args.engine == "meilisearch":
		args.meilisearch_alpha = alpha
		adapter.semantic_ratio = alpha
		return

	if args.engine == "weaviate":
		args.weaviate_alpha = alpha
		adapter.alpha = alpha
		return

	if args.engine == "opensearch":
		args.opensearch_alpha = alpha
		adapter.set_hybrid_alpha(alpha)
		return

	raise ValueError(f"Alpha tuning is not supported for engine={args.engine}")


def make_tune_eval_query_sets(
	queries_df: pd.DataFrame,
	tune_alpha: bool,
	tune_query_count: int,
	seed: int,
):

	if "large_version" not in queries_df.columns:
		queries_df = queries_df.copy()
		queries_df["large_version"] = 1

	if "small_version" not in queries_df.columns:
		queries_df = queries_df.copy()
		queries_df["small_version"] = 0

	large_df = queries_df[queries_df["large_version"] == 1].copy()

	if not tune_alpha:
		return None, large_df, set()

	small_df = large_df[large_df["small_version"] == 1].copy()

	if small_df.empty:
		raise ValueError("Cannot tune alpha: no small_version == 1 queries found")

	tune_count = min(int(tune_query_count), len(small_df))

	tune_df = (
		small_df
		.sample(n=tune_count, random_state=seed)
		.sort_values("query_id")
		.reset_index(drop=True)
	)

	tune_qids = {int(qid) for qid in tune_df["query_id"].tolist()}

	eval_df = (
		large_df[~large_df["query_id"].isin(tune_qids)]
		.sort_values("query_id")
		.reset_index(drop=True)
	)

	return tune_df, eval_df, tune_qids

def build_adapter(args):
	if args.engine == "typesense":
		return TypesenseAdapter(
			host=args.typesense_host,
			api_key=args.typesense_api_key,
			collection=args.typesense_collection,
			timeout_s=args.timeout_s,
			num_typos=0,
			query_by="title",
			query_by_weights="8",
			retrieval_mode=args.retrieval_mode,
			vector_dim=args.vector_dim,
			alpha=args.typesense_alpha,
			m=args.m,
			efc=args.efc,
			efs=max(args.top_k, args.efs),
			vector_k=args.typesense_vector_k,
			batch_size=args.typesense_batch_size,
		)
	if args.engine == "brinicle":
		return BrinicleItemAdapter(
			host=args.brinicle_host,
			index_name=args.brinicle_index,
			dim=args.brinicle_lexical_dim,
			vector_dim=args.vector_dim,
			m=args.m,
			efc=args.efc,
			efs=args.efs,
			alpha=args.brinicle_alpha,
			seed=args.seed,
			retrieval_mode=args.retrieval_mode,
			build_n_threads=args.build_n_jobs,
			search_batch_jobs=args.search_batch_jobs,
			batch_size=args.brinicle_batch_size,
		)
	if args.engine == "meilisearch":
		return MeilisearchAdapter(
			host=args.meilisearch_host,
			api_key=args.meilisearch_api_key,
			index_name=args.meilisearch_index,
			timeout_s=args.timeout_s,
			batch_size=args.meilisearch_batch_size,
			typo_tolerance=False,
			retrieval_mode=args.retrieval_mode,
			vector_dim=args.vector_dim,
			embedder_name="default",
			semantic_ratio=args.meilisearch_alpha,
		)
	if args.engine == "brinicle_inprocess":
		return BrinicleLocalItemAdapter(
			index_name=args.brinicle_index,
			dim=args.brinicle_lexical_dim,
			vector_dim=args.vector_dim,
			m=args.m,
			efc=args.efc,
			efs=args.efs,
			alpha=args.brinicle_alpha,
			retrieval_mode=args.retrieval_mode,
			search_batch_jobs=args.search_batch_jobs,
			build_n_threads=args.build_n_jobs,
			seed=args.seed,
		)

	if args.engine == "weaviate":
		return WeaviateAdapter(
			host=args.weaviate_host,
			grpc_port=args.weaviate_grpc_port,
			collection=args.weaviate_collection,
			retrieval_mode=args.retrieval_mode,
			alpha=args.weaviate_alpha,
			batch_size=args.weaviate_batch_size,
			m=args.m,
			efc=args.efc,
			efs=max(args.top_k, args.efs),
		)

	if args.engine == "opensearch":
		return OpenSearchAdapter(
			host=args.opensearch_host,
			index_name=args.opensearch_index,
			retrieval_mode=args.retrieval_mode,
			vector_dim=args.vector_dim,
			alpha=args.opensearch_alpha,
			m=args.m,
			efc=args.efc,
			efs=max(args.top_k, args.efs),
			batch_size=args.opensearch_batch_size,
			verify_certs=args.opensearch_verify_certs,
		)

	raise ValueError(f"Unknown engine: {args.engine}")


def build_with_optional_memory_monitor(args, adapter, items):
	local_engines = {"brinicle_inprocess", }

	if args.engine in local_engines or CgroupMemoryMonitor is None:
		build_latency = adapter.build(items)
		build_peak_mb = None
		return build_latency, build_peak_mb

	container_name = args.container_name or CONTAINER_NAMES.get(args.engine, args.engine)

	mon = CgroupMemoryMonitor(
		container_name=container_name,
		interval_s=0.01,
	).start()

	build_latency = adapter.build(items)

	mon.stop()
	# build_peak_mb = mon.peak_bytes / (1024 * 1024)

	return build_latency, mon.peak_report_mb()


def run_trial_with_optional_memory_monitor(args, adapter, eval_queries):
	local_engines = {"brinicle_inprocess",}

	if args.engine in local_engines or CgroupMemoryMonitor is None:
		results_by_qid, latencies = run_search(
			adapter=adapter,
			queries=eval_queries,
			top_k=args.top_k,
			warmup=min(args.warmup, len(eval_queries)),
			search_batch_size=args.search_batch_size,
			search_mode=args.search_mode,
			retrieval_mode=args.retrieval_mode,
		)
		return results_by_qid, latencies, None

	container_name = args.container_name or CONTAINER_NAMES.get(args.engine, args.engine)

	mon = CgroupMemoryMonitor(
		container_name=container_name,
		interval_s=0.01,
	).start()

	results_by_qid, latencies = run_search(
		adapter=adapter,
		queries=eval_queries,
		top_k=args.top_k,   
		warmup=min(args.warmup, len(eval_queries)),
		search_batch_size=args.search_batch_size,
		search_mode=args.search_mode,
		retrieval_mode=args.retrieval_mode,
	)

	mon.stop()

	return results_by_qid, latencies, mon.peak_report_mb()


def make_output_suffix(args) -> str:
	if args.engine == "typesense":
		return f"_"
	if args.engine == "brinicle":
		return f"_{args.m}m_{args.efc}efc_{args.efs}efs_{args.brinicle_alpha}alpha"
	if args.engine == "meilisearch":
		return f"_"
	if args.engine == "brinicle_inprocess":
		return f"_{args.m}m_{args.efc}efc_{args.efs}efs"

	return ""


def main():
	p = argparse.ArgumentParser(description="Benchmark on Amazon ESCI title-only retrieval")

	p.add_argument(
		"--engine",
		choices=["brinicle", "typesense", "meilisearch", "brinicle_inprocess", "weaviate", "opensearch"],
		required=True,
	)

	p.add_argument("--data", type=Path, default=DEFAULT_ESCI_PATH)
	p.add_argument("--qrels-mode", choices=["graded", "exact"], default="graded")

	p.add_argument("--top-k", type=int, default=100)
	p.add_argument("--max-queries", type=int, default=None)
	p.add_argument("--sample", action="store_true")
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--warmup", type=int, default=10)
	p.add_argument("--trials", type=int, default=1)

	p.add_argument("--source", type=str, default=None)

	p.add_argument("--search-batch-jobs", type=int, default=32, help="Number of worker jobs used by adapter batch search.")
	p.add_argument("--search-batch-size", type=int, default=32, help="Number of queries for batch search mode.")
	p.add_argument("--timeout-s", type=float, default=40)
	p.add_argument("--build-n-jobs", type=int, default=1, help="build num threads")
	p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	p.add_argument("--container-name", type=str, default=None)
	p.add_argument("--search-mode", choices=["sequential", "batch"], default="sequential", help="Search execution mode: sequential one-query requests or batched multi-search.")
	p.add_argument("--retrieval-mode", choices=["lexical", "hybrid"], default="lexical", help="Retrieval mode: lexical text-only search or hybrid text+vector search.",
	)

	p.add_argument("--m", type=int, default=8)
	p.add_argument("--efc", type=int, default=512)
	p.add_argument("--efs", type=int, default=1024)

	# brinicle params
	p.add_argument("--brinicle-host", type=str, default="http://localhost:1984")
	p.add_argument("--brinicle-index", type=str, default="esci_item_title_only_bench")
	p.add_argument("--brinicle-lexical-dim", type=int, default=70, help="brinicle lexical encoder dimension.")
	p.add_argument("--brinicle-alpha", type=float, default=0.5, help="brinicle alpha.")
	p.add_argument("--brinicle-batch-size", type=int, default=2048)

	# Typesense params
	p.add_argument("--typesense-alpha", type=float, default=0.5)
	p.add_argument("--typesense-vector-k", type=int, default=None)
	p.add_argument("--typesense-host", type=str, default="http://localhost:8108")
	p.add_argument("--typesense-api-key", type=str, default=os.getenv("TYPESENSE_API_KEY", "xyz"))
	p.add_argument("--typesense-batch-size", type=int, default=2048)
	p.add_argument("--typesense-collection", type=str, default="esci_products_title_only")

	p.add_argument("--meilisearch-host", type=str, default="http://localhost:7700")
	p.add_argument("--meilisearch-api-key", type=str, default=os.getenv("MEILI_MASTER_KEY", "benchmark_master_key_123"))
	p.add_argument("--meilisearch-index", type=str, default="esci_products_title_only")
	p.add_argument("--meilisearch-batch-size", type=int, default=2048)
	p.add_argument("--meilisearch-alpha", type=float, default=0.5)

	p.add_argument("--weaviate-host", type=str, default="http://localhost:8080")
	p.add_argument("--weaviate-grpc-port", type=int, default=50051)
	p.add_argument("--weaviate-collection", type=str, default="EsciProductsTitleOnly")
	p.add_argument("--weaviate-alpha", type=float, default=0.5)
	p.add_argument("--weaviate-batch-size", type=int, default=2048)

	p.add_argument("--opensearch-host", type=str, default="http://localhost:9200")
	p.add_argument("--opensearch-index", type=str, default="esci_products_title_only")
	p.add_argument("--opensearch-alpha", type=float, default=0.5)
	p.add_argument("--opensearch-batch-size", type=int, default=2048)
	p.add_argument("--opensearch-verify-certs", action="store_true")

	p.add_argument("--tune-metric", type=str, default="nDCG@10", help="Metric used to select best alpha.")
	p.add_argument("--tune-alpha", action="store_true", help="Run shared alpha grid search before final evaluation.")
	p.add_argument("--tune-query-count", type=int, default=2000, help="Number of small_version queries used for alpha tuning.")

	args = p.parse_args()
	print(args.search_mode)
	print("[load] ESCI preprocessed:", args.data)

	items, queries_df, labels_df, meta = load_esci_preprocessed_jsonl(
		path=args.data,
		source=args.source,
		retrieval_mode=args.retrieval_mode,
	)

	if args.retrieval_mode == "hybrid":
		inferred_vector_dim = None
		embedding_meta = meta.get("embedding", {}) or {}
		if "resolved_dimension" in embedding_meta:
			inferred_vector_dim = int(embedding_meta["resolved_dimension"])
		elif "dimension" in embedding_meta:
			inferred_vector_dim = int(embedding_meta["dimension"])
		args.vector_dim = inferred_vector_dim
		print(f"[hybrid] vector_dim={args.vector_dim}")
	else:
		args.vector_dim = 0

	tune_queries_df, eval_queries_df, tune_qids = make_tune_eval_query_sets(
		queries_df=queries_df,
		tune_alpha=args.tune_alpha,
		tune_query_count=args.tune_query_count,
		seed=args.seed,
	)

	if args.tune_alpha:
		tune_queries = load_queries(
			tune_queries_df,
			max_queries=None,
			sample=False,
			seed=args.seed,
		)
	else:
		tune_queries = []

	eval_queries = load_queries(
		eval_queries_df,
		max_queries=args.max_queries,
		sample=args.sample,
		seed=args.seed,
	)

	if args.retrieval_mode == "hybrid":
		query_vector_by_qid = {
			int(row.query_id): row.vector
			for row in queries_df.itertuples(index=False)
		}

		for query_list in [tune_queries, eval_queries]:
			for q in query_list:
				qid = int(q["query_id"])

				if qid not in query_vector_by_qid:
					raise KeyError(f"Missing query vector for query_id={qid}")

				q["vector"] = query_vector_by_qid[qid]

	label_map = ESCI_LABEL_MAPS[args.qrels_mode]

	qrels = build_qrels_from_labels(
		labels=labels_df,
		query_col="query_id",
		doc_col="doc_id",
		label_col="esci_label",
		label_map=label_map,
	)

	# Restrict qrels to evaluated queries only.
	eval_qids = {int(q["query_id"]) for q in eval_queries}
	tune_qids_loaded = {int(q["query_id"]) for q in tune_queries}

	eval_qrels = {
		qid: rels
		for qid, rels in qrels.items()
		if int(qid) in eval_qids
	}

	tune_qrels = {
		qid: rels
		for qid, rels in qrels.items()
		if int(qid) in tune_qids_loaded
	}

	qrels = eval_qrels

	ks = [k for k in DEFAULT_KS if k <= args.top_k]

	print(
		f"[load] docs={len(items)}, "
		f"tune_queries={len(tune_queries)}, "
		f"eval_queries={len(eval_queries)}, "
		f"qrels_queries={len(qrels)}, "
		f"qrels_mode={args.qrels_mode}, ks={ks}"
	)

	if args.source:
		print(f"[load] source_filter={args.source}")

	print("[engine]", args.engine)

	adapter = build_adapter(args)

	build_latency, build_memory = build_with_optional_memory_monitor(
		args=args,
		adapter=adapter,
		items=items,
	)

	trial_reports = []
	search_memory_profiles = []
	tuning_report = {"enabled": False}

	if args.tune_alpha:
		if args.retrieval_mode != "hybrid":
			raise ValueError("--tune-alpha requires --retrieval-mode hybrid")

		print("[tune] shared alpha grid search enabled")
		print(
			f"[tune] queries={len(tune_queries)}, "
			f"qrels_queries={len(tune_qrels)}, "
			f"metric={args.tune_metric}"
		)

		tune_candidates = []

		for alpha in SHARED_ALPHA_GRID:
			print(f"[tune] alpha={alpha}")

			set_adapter_alpha(
				args=args,
				adapter=adapter,
				alpha=alpha,
			)

			results_by_qid, latencies = run_search(
				adapter=adapter,
				queries=tune_queries,
				top_k=args.top_k,
				warmup=min(args.warmup, len(tune_queries)),
				search_batch_size=args.search_batch_size,
				search_mode=args.search_mode,
				retrieval_mode=args.retrieval_mode,
			)

			metrics = evaluate_results(
				results_by_qid=results_by_qid,
				qrels_by_qid=tune_qrels,
				ks=ks,
			)

			latency_report = summarize_latencies(latencies)

			if args.tune_metric not in metrics:
				raise KeyError(
					f"Unknown tune metric: {args.tune_metric}. "
					f"Available metrics: {sorted(metrics.keys())}"
				)

			candidate = {
				"alpha": float(alpha),
				"score": float(metrics[args.tune_metric]),
				"metric": args.tune_metric,
				"results": {
					**metrics,
					**latency_report,
				},
			}

			tune_candidates.append(candidate)

			print(
				f"[tune] alpha={alpha} "
				f"{args.tune_metric}={candidate['score']:.6f}"
			)

		tune_candidates.sort(key=lambda x: x["score"], reverse=True)

		best_tune = tune_candidates[0]
		best_alpha = float(best_tune["alpha"])

		tuning_report = {
			"enabled": True,
			"grid": SHARED_ALPHA_GRID,
			"metric": args.tune_metric,
			"best_alpha": best_alpha,
			"best_score": best_tune["score"],
			"candidates": tune_candidates,
			"tune_queries": len(tune_queries),
			"eval_excludes_tune_queries": True,
		}

		print(
			f"[tune] selected alpha={best_alpha} "
			f"{args.tune_metric}={best_tune['score']:.6f}"
		)

		set_adapter_alpha(
			args=args,
			adapter=adapter,
			alpha=best_alpha,
		)

	for trial in range(args.trials):
		print(f"[trial] {trial + 1}/{args.trials}")

		results_by_qid, latencies, search_memory_profile = run_trial_with_optional_memory_monitor(
			args=args,
			adapter=adapter,
			eval_queries=eval_queries,
		)

		metrics = evaluate_results(
			results_by_qid=results_by_qid,
			qrels_by_qid=qrels,
			ks=ks,
		)

		latency_report = summarize_latencies(latencies)

		report = {}
		report.update(metrics)
		report.update(latency_report)
		if search_memory_profile is not None:
			report.update(search_memory_profile)
		trial_reports.append(report)

	averaged = {}
	keys = trial_reports[0].keys()

	for key in keys:
		vals = [r[key] for r in trial_reports if r[key] is not None]
		averaged[key] = float(np.mean(vals)) if vals else None

	averaged["build_latency"] = float(build_latency)
	averaged["build_memory_profile"] = (
		build_memory if build_memory is not None else None
	)

	output = {
		"engine": args.engine,
		"dataset": "ESCI",
		"task": "title_only_full_corpus_retrieval",
		"retrieval_mode": args.retrieval_mode,
		"data_path": str(args.data),
		"docs": len(items),
		"queries": len(eval_queries),
		"tune_queries": len(tune_queries),
		"eval_queries": len(eval_queries),
		"excluded_tune_queries_from_eval": bool(args.tune_alpha),
		"qrels_queries": len(qrels),
		"qrels_mode": args.qrels_mode,
		"source_filter": args.source,
		"fields": ["title"],
		"top_k": args.top_k,
		"ks": ks,
		"trials": args.trials,
		"dataset_meta": meta,
		"search_mode": args.search_mode,
		"search_batch_size": args.search_batch_size,
		"search_batch_jobs": args.search_batch_jobs,
		"tuning": tuning_report,
		"params": {
			"retrieval_mode": args.retrieval_mode,
			"vector_dim": args.vector_dim,
			"M": args.m if args.engine in {"brinicle", "brinicle_inprocess"} else None,
			"ef_construction": args.efc if args.engine in {"brinicle", "brinicle_inprocess"} else None,
			"ef_search": args.efs if args.engine in {"brinicle", "brinicle_inprocess"} else None,
			"meilisearch_batch_size": (
				args.meilisearch_batch_size
				if args.engine == "meilisearch"
				else None
			),
			"meilisearch_alpha": (
				args.meilisearch_alpha
				if args.engine == "meilisearch"
				else None
			),
			"weaviate_alpha": (
				args.weaviate_alpha
				if args.engine == "weaviate"
				else None
			),
			"opensearch_alpha": (
				args.opensearch_alpha
				if args.engine == "opensearch"
				else None
			),
			"opensearch_batch_size": (
				args.opensearch_batch_size
				if args.engine == "opensearch"
				else None
			),
			"typesense_alpha": (
				args.typesense_alpha
				if args.engine == "typesense"
				else None
			),
			"typesense_batch_size": (
				args.typesense_batch_size
				if args.engine == "typesense"
				else None
			),
			"tune_alpha": args.tune_alpha,
			"tune_query_count": args.tune_query_count if args.tune_alpha else None,
			"tune_metric": args.tune_metric if args.tune_alpha else None,
			"brinicle_alpha": (
				args.brinicle_alpha
				if args.engine in {"brinicle", "brinicle_inprocess"}
				else None
			),
			"brinicle_build_batch_size": args.brinicle_batch_size,
			"brinicle_lexical_dim": args.brinicle_lexical_dim,
		},
		"results": averaged,
		"trial_results": trial_reports,
	}

	args.output_dir.mkdir(parents=True, exist_ok=True)

	suffix = make_output_suffix(args)

	source_suffix = f"_{args.source}" if args.source else ""

	out_path = (
		args.output_dir
		/ f"{args.engine}_esci_title_only_{args.retrieval_mode}_{args.search_mode}"
		  f"_B{args.search_batch_size}_{args.qrels_mode}{source_suffix}{suffix}.json"
	)

	with out_path.open("w", encoding="utf-8") as f:
		json.dump(output, f, indent=2, ensure_ascii=False)

	print(f"[save] {out_path}")
	print(json.dumps(output["results"], indent=4))


if __name__ == "__main__":
	main()
