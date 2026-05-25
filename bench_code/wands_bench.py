import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

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


DEFAULT_PREFIX = Path("datasets/WANDS/preprocessed/wands_title")
DEFAULT_OUTPUT_DIR = Path("benchmark/wands_results")

DEFAULT_KS = [1, 5, 10, 20, 50, 100]

ALPHA_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

CONTAINER_NAMES = {
    "brinicle": "brinicle_container",
    "typesense": "typesense",
    "meilisearch": "meilisearch",
    "weaviate": "weaviate",
    "opensearch": "opensearch",
}

WANDS_LABEL_MAPS = {
    "exact": {
        "Exact": 1.0,
        "Partial": 0.0,
        "Irrelevant": 0.0,
    },
    "graded": {
        "Exact": 1.0,
        "Partial": 0.1,
        "Irrelevant": 0.0,
    },
}


def p(prefix: Path, suffix: str) -> Path:
    return Path(str(prefix) + suffix)


def load_wands_preprocessed(
    prefix: Path,
    retrieval_mode: str,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame, int]:
    docs_path = p(prefix, ".docs.jsonl")
    queries_path = p(prefix, ".queries.jsonl")
    qrels_path = p(prefix, ".qrels.jsonl")

    print("[load] docs:", docs_path)
    print("[load] queries:", queries_path)
    print("[load] qrels:", qrels_path)

    docs = pd.read_json(docs_path, lines=True)
    queries = pd.read_json(queries_path, lines=True)
    qrels = pd.read_json(qrels_path, lines=True)

    if docs.empty:
        raise ValueError(f"No docs loaded from {docs_path}")
    if queries.empty:
        raise ValueError(f"No queries loaded from {queries_path}")
    if qrels.empty:
        raise ValueError(f"No qrels loaded from {qrels_path}")

    docs["id"] = docs["id"].astype(str)
    docs["title"] = docs["title"].fillna("").astype(str)
    docs["category"] = docs.get("category", "").fillna("").astype(str)
    docs["subcategory"] = docs.get("subcategory", "").fillna("").astype(str)

    queries["query_id"] = queries["query_id"].astype(int)
    queries["query"] = queries["query"].fillna("").astype(str)

    qrels["query_id"] = qrels["query_id"].astype(int)
    qrels["doc_id"] = qrels["doc_id"].astype(str)
    qrels["wands_label"] = qrels["wands_label"].fillna("").astype(str)

    items = []

    for row in docs.itertuples(index=False):
        item = {
            "id": str(row.id),
            "title": str(row.title),
            "category": str(row.category),
            "subcategory": str(row.subcategory),
            "attributes": {},
            "vector": None,
        }

        if hasattr(row, "vector_index"):
            item["vector_index"] = int(row.vector_index)

        items.append(item)

    vector_dim = 0

    if retrieval_mode == "hybrid":
        document_vectors_path = p(prefix, ".documents.npy")
        query_vectors_path = p(prefix, ".queries.npy")

        print("[load] document_vectors:", document_vectors_path)
        print("[load] query_vectors:", query_vectors_path)

        document_vectors = np.load(document_vectors_path, mmap_mode="r")
        query_vectors = np.load(query_vectors_path, mmap_mode="r")

        if document_vectors.ndim != 2:
            raise ValueError(f"Bad document vector shape: {document_vectors.shape}")
        if query_vectors.ndim != 2:
            raise ValueError(f"Bad query vector shape: {query_vectors.shape}")
        if document_vectors.shape[1] != query_vectors.shape[1]:
            raise ValueError(
                f"Vector dim mismatch: docs={document_vectors.shape[1]}, "
                f"queries={query_vectors.shape[1]}"
            )

        vector_dim = int(document_vectors.shape[1])

        if "vector_index" not in docs.columns:
            raise ValueError("docs jsonl has no vector_index column")
        if "vector_index" not in queries.columns:
            raise ValueError("queries jsonl has no vector_index column")

        for item in items:
            idx = int(item["vector_index"])
            item["vector"] = document_vectors[idx]

        queries = queries.copy()
        queries["vector"] = [
            query_vectors[int(idx)]
            for idx in queries["vector_index"].tolist()
        ]

    return items, queries, qrels, vector_dim


def make_tune_eval_queries(
    queries_df: pd.DataFrame,
    tune_alpha: bool,
    tune_query_count: int,
    seed: int,
):
    queries_df = queries_df.sort_values("query_id").reset_index(drop=True)

    if not tune_alpha:
        return None, queries_df, set()

    tune_count = min(int(tune_query_count), len(queries_df))

    tune_df = (
        queries_df
        .sample(n=tune_count, random_state=seed)
        .sort_values("query_id")
        .reset_index(drop=True)
    )

    tune_qids = {int(qid) for qid in tune_df["query_id"].tolist()}

    eval_df = (
        queries_df[~queries_df["query_id"].isin(tune_qids)]
        .sort_values("query_id")
        .reset_index(drop=True)
    )

    return tune_df, eval_df, tune_qids


def attach_vectors(loaded_queries, source_df: pd.DataFrame):
    if "vector" not in source_df.columns:
        return

    vector_by_qid = {
        int(row.query_id): row.vector
        for row in source_df.itertuples(index=False)
    }

    for q in loaded_queries:
        qid = int(q["query_id"])
        q["vector"] = vector_by_qid[qid]


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

    raise ValueError(f"Alpha tuning not supported for engine={args.engine}")


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
    local_engines = {"brinicle_inprocess"}

    if args.engine in local_engines or CgroupMemoryMonitor is None:
        return adapter.build(items), None

    container_name = args.container_name or CONTAINER_NAMES.get(args.engine, args.engine)

    mon = CgroupMemoryMonitor(
        container_name=container_name,
        interval_s=0.01,
    ).start()

    build_latency = adapter.build(items)

    mon.stop()
    return build_latency, mon.peak_report_mb()


def run_trial_with_optional_memory_monitor(args, adapter, eval_queries):
    local_engines = {"brinicle_inprocess"}

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


def main():
    p = argparse.ArgumentParser(description="Benchmark WANDS product retrieval")

    p.add_argument(
        "--engine",
        choices=[
            "brinicle",
            "typesense",
            "meilisearch",
            "brinicle_inprocess",
            "weaviate",
            "opensearch",
        ],
        required=True,
    )

    p.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    p.add_argument("--retrieval-mode", choices=["lexical", "hybrid"], default="lexical")
    p.add_argument("--qrels-mode", choices=["graded", "exact"], default="graded")

    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--trials", type=int, default=1)

    p.add_argument("--search-mode", choices=["sequential", "batch"], default="sequential")
    p.add_argument("--search-batch-size", type=int, default=32)
    p.add_argument("--search-batch-jobs", type=int, default=16)
    p.add_argument("--timeout-s", type=float, default=40)
    p.add_argument("--build-n-jobs", type=int, default=1)
    p.add_argument("--container-name", type=str, default=None)

    p.add_argument("--m", type=int, default=8)
    p.add_argument("--efc", type=int, default=512)
    p.add_argument("--efs", type=int, default=1024)

    # Brinicle.
    p.add_argument("--brinicle-host", type=str, default="http://localhost:1984")
    p.add_argument("--brinicle-index", type=str, default="wands_item_bench")
    p.add_argument("--brinicle-lexical-dim", type=int, default=70)
    p.add_argument("--brinicle-alpha", type=float, default=0.5)
    p.add_argument("--brinicle-batch-size", type=int, default=2048)

    # Typesense.
    p.add_argument("--typesense-host", type=str, default="http://localhost:8108")
    p.add_argument("--typesense-api-key", type=str, default=os.getenv("TYPESENSE_API_KEY", "xyz"))
    p.add_argument("--typesense-collection", type=str, default="wands_products")
    p.add_argument("--typesense-batch-size", type=int, default=2048)
    p.add_argument("--typesense-alpha", type=float, default=0.5)
    p.add_argument("--typesense-vector-k", type=int, default=None)

    # Meilisearch.
    p.add_argument("--meilisearch-host", type=str, default="http://localhost:7700")
    p.add_argument("--meilisearch-api-key", type=str, default=os.getenv("MEILI_MASTER_KEY", "benchmark_master_key_123"))
    p.add_argument("--meilisearch-index", type=str, default="wands_products")
    p.add_argument("--meilisearch-batch-size", type=int, default=2048)
    p.add_argument("--meilisearch-alpha", type=float, default=0.5)

    # Weaviate.
    p.add_argument("--weaviate-host", type=str, default="http://localhost:8080")
    p.add_argument("--weaviate-grpc-port", type=int, default=50051)
    p.add_argument("--weaviate-collection", type=str, default="WandsProducts")
    p.add_argument("--weaviate-batch-size", type=int, default=2048)
    p.add_argument("--weaviate-alpha", type=float, default=0.5)

    # OpenSearch.
    p.add_argument("--opensearch-host", type=str, default="http://localhost:9200")
    p.add_argument("--opensearch-index", type=str, default="wands_products")
    p.add_argument("--opensearch-batch-size", type=int, default=2048)
    p.add_argument("--opensearch-alpha", type=float, default=0.5)
    p.add_argument("--opensearch-verify-certs", action="store_true")

    # Hybrid alpha tuning.
    p.add_argument("--tune-alpha", action="store_true")
    p.add_argument("--tune-query-count", type=int, default=30)
    p.add_argument("--tune-metric", type=str, default="nDCG@5")

    args = p.parse_args()

    print("[engine]", args.engine)
    print("[mode]", args.retrieval_mode)
    print("[prefix]", args.prefix)

    items, queries_df, qrels_df, vector_dim = load_wands_preprocessed(
        prefix=args.prefix,
        retrieval_mode=args.retrieval_mode,
    )

    args.vector_dim = vector_dim

    if args.retrieval_mode == "hybrid":
        print(f"[hybrid] vector_dim={args.vector_dim}")
    else:
        args.vector_dim = 0

    if args.tune_alpha and args.retrieval_mode != "hybrid":
        raise ValueError("--tune-alpha requires --retrieval-mode hybrid")

    tune_queries_df, eval_queries_df, tune_qids = make_tune_eval_queries(
        queries_df=queries_df,
        tune_alpha=args.tune_alpha or (args.engine in {"brinicle", "brinicle_inprocess"} and args.tune_alpha is False),
        tune_query_count=args.tune_query_count,
        seed=args.seed,
    )

    tune_queries = []
    if args.tune_alpha:
        tune_queries = load_queries(
            tune_queries_df,
            max_queries=None,
            sample=False,
            seed=args.seed,
        )
        attach_vectors(tune_queries, tune_queries_df)

    eval_queries = load_queries(
        eval_queries_df,
        max_queries=args.max_queries,
        sample=args.sample,
        seed=args.seed,
    )
    attach_vectors(eval_queries, eval_queries_df)

    label_map = WANDS_LABEL_MAPS[args.qrels_mode]

    all_qrels = build_qrels_from_labels(
        labels=qrels_df,
        query_col="query_id",
        doc_col="doc_id",
        label_col="wands_label",
        label_map=label_map,
    )

    eval_qids = {int(q["query_id"]) for q in eval_queries}
    tune_qids_loaded = {int(q["query_id"]) for q in tune_queries}

    eval_qrels = {
        int(qid): rels
        for qid, rels in all_qrels.items()
        if int(qid) in eval_qids
    }

    tune_qrels = {
        int(qid): rels
        for qid, rels in all_qrels.items()
        if int(qid) in tune_qids_loaded
    }

    ks = [k for k in DEFAULT_KS if k <= args.top_k]

    print(
        f"[load] docs={len(items)}, "
        f"tune_queries={len(tune_queries)}, "
        f"eval_queries={len(eval_queries)}, "
        f"eval_qrels_queries={len(eval_qrels)}, "
        f"qrels_mode={args.qrels_mode}, ks={ks}"
    )

    adapter = build_adapter(args)

    build_latency, build_memory = build_with_optional_memory_monitor(
        args=args,
        adapter=adapter,
        items=items,
    )

    tuning_report = {"enabled": False}

    if args.tune_alpha:
        print("[tune] alpha grid enabled")
        print(
            f"[tune] queries={len(tune_queries)}, "
            f"qrels_queries={len(tune_qrels)}, "
            f"metric={args.tune_metric}"
        )

        candidates = []

        for alpha in ALPHA_GRID:
            print(f"[tune] alpha={alpha}")

            set_adapter_alpha(args, adapter, alpha)

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
                    f"Available: {sorted(metrics.keys())}"
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

            candidates.append(candidate)

            print(
                f"[tune] alpha={alpha} "
                f"{args.tune_metric}={candidate['score']:.6f}"
            )
        candidates.sort(key=lambda x: x["score"], reverse=True)

        best = candidates[0]
        best_alpha = float(best["alpha"])

        print(
            f"[tune] selected alpha={best_alpha} "
            f"{args.tune_metric}={best['score']:.6f}"
        )
        if args.engine in {"brinicle", "brinicle_inprocess"}:
            print(
                f"\nRerun the brinicle docker, and run this script with {best_alpha} as alpha. Set it with --brinicle-alpha."
                f"\nIn addition, do not use --tune-alpha since we already have the tuned alpha. This happens only if the engine is brinicle."
                f"\nFor a fair comparison, brinicle requires tuned alpha both for build and search, not just search."
            )
            exit()
        set_adapter_alpha(args, adapter, best_alpha)

        tuning_report = {
            "enabled": True,
            "grid": ALPHA_GRID,
            "metric": args.tune_metric,
            "best_alpha": best_alpha,
            "best_score": best["score"],
            "candidates": candidates,
            "tune_queries": len(tune_queries),
            "eval_excludes_tune_queries": True,
        }

    trial_reports = []

    for trial in range(args.trials):
        print(f"[trial] {trial + 1}/{args.trials}")

        results_by_qid, latencies, search_memory = run_trial_with_optional_memory_monitor(
            args=args,
            adapter=adapter,
            eval_queries=eval_queries,
        )

        metrics = evaluate_results(
            results_by_qid=results_by_qid,
            qrels_by_qid=eval_qrels,
            ks=ks,
        )

        latency_report = summarize_latencies(latencies)

        report = {}
        report.update(metrics)
        report.update(latency_report)

        if search_memory is not None:
            report.update(search_memory)

        trial_reports.append(report)

    averaged = {}

    for key in trial_reports[0].keys():
        vals = [r[key] for r in trial_reports if r.get(key) is not None]
        averaged[key] = float(np.mean(vals)) if vals else None

    averaged["build_latency"] = float(build_latency)
    averaged["build_memory_profile"] = build_memory if build_memory is not None else None

    output = {
        "engine": args.engine,
        "dataset": "WANDS",
        "task": "product_retrieval",
        "prefix": str(args.prefix),
        "retrieval_mode": args.retrieval_mode,
        "docs": len(items),
        "queries": len(eval_queries),
        "tune_queries": len(tune_queries),
        "eval_queries": len(eval_queries),
        "excluded_tune_queries_from_eval": bool(args.tune_alpha),
        "qrels_queries": len(eval_qrels),
        "qrels_mode": args.qrels_mode,
        "fields": ["title"],
        "top_k": args.top_k,
        "ks": ks,
        "trials": args.trials,
        "search_mode": args.search_mode,
        "search_batch_size": args.search_batch_size,
        "search_batch_jobs": args.search_batch_jobs,
        "tuning": tuning_report,
        "params": {
            "retrieval_mode": args.retrieval_mode,
            "vector_dim": args.vector_dim,
            "M": args.m,
            "ef_construction": args.efc,
            "ef_search": args.efs,
            "brinicle_alpha": args.brinicle_alpha if args.engine in {"brinicle", "brinicle_inprocess"} else None,
            "brinicle_lexical_dim": args.brinicle_lexical_dim,
            "brinicle_batch_size": args.brinicle_batch_size,
            "typesense_alpha": args.typesense_alpha if args.engine == "typesense" else None,
            "typesense_batch_size": args.typesense_batch_size if args.engine == "typesense" else None,
            "typesense_vector_k": args.typesense_vector_k if args.engine == "typesense" else None,
            "meilisearch_alpha": args.meilisearch_alpha if args.engine == "meilisearch" else None,
            "meilisearch_batch_size": args.meilisearch_batch_size if args.engine == "meilisearch" else None,
            "weaviate_alpha": args.weaviate_alpha if args.engine == "weaviate" else None,
            "weaviate_batch_size": args.weaviate_batch_size if args.engine == "weaviate" else None,
            "opensearch_alpha": args.opensearch_alpha if args.engine == "opensearch" else None,
            "opensearch_batch_size": args.opensearch_batch_size if args.engine == "opensearch" else None,
            "tune_alpha": args.tune_alpha,
            "tune_query_count": args.tune_query_count if args.tune_alpha else None,
            "tune_metric": args.tune_metric if args.tune_alpha else None,
        },
        "results": averaged,
        "trial_results": trial_reports,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_path = (
        args.output_dir
        / f"{args.engine}_wands_{args.prefix.name}_{args.retrieval_mode}"
          f"_{args.search_mode}_B{args.search_batch_size}_{args.qrels_mode}.json"
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[save] {out_path}")
    print(json.dumps(output["results"], indent=4))


if __name__ == "__main__":
    main()
