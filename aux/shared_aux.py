import re
import os
import json
import time
import math
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import requests
import shutil
from urllib.parse import urlparse

from collections import Counter, defaultdict

from brinicle.lexical_encoder import LexicalEncoder
import brinicle

import weaviate
# import weaviate.classes as wvc
import weaviate.classes.config as wvc

from weaviate.util import generate_uuid5
from weaviate.classes.query import HybridFusion


from opensearchpy import OpenSearch, helpers



def load_queries(
	queries: pd.DataFrame,
	max_queries: Optional[int],
	sample: bool,
	seed: int,
):
	rows = queries[["query_id", "query"]].copy()

	if max_queries is not None:
		max_queries = min(max_queries, len(rows))

		if sample:
			rows = rows.sample(n=max_queries, random_state=seed)
		else:
			rows = rows.iloc[:max_queries]

	out = []
	for _, row in rows.iterrows():
		out.append({
			"query_id": int(row["query_id"]),
			"query": str(row["query"]),
		})

	return out

def build_qrels_from_labels(
	labels: pd.DataFrame,
	query_col: str,
	doc_col: str,
	label_col: str,
	label_map: Dict[str, float],
) -> Dict[int, Dict[str, float]]:
	labels = labels.copy()
	labels["rel"] = labels[label_col].map(label_map).fillna(0.0)

	qrels: Dict[int, Dict[str, float]] = {}

	for qid, group in labels.groupby(query_col):
		qrels[int(qid)] = {
			str(row[doc_col]): float(row["rel"])
			for _, row in group.iterrows()
		}

	return qrels



def load_jsonl(path: Path) -> List[Dict[str, Any]]:
	items = []
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if line:
				items.append(json.loads(line))
	return items


def dcg(rels: List[float]) -> float:
	return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(result_ids: List[str], qrels: Dict[str, float], k: int) -> Optional[float]:
	if not qrels:
		return None

	rels = [qrels.get(str(pid), 0.0) for pid in result_ids[:k]]
	ideal = sorted(qrels.values(), reverse=True)[:k]

	ideal_dcg = dcg(ideal)
	if ideal_dcg <= 0:
		return None

	return dcg(rels) / ideal_dcg


def recall_at_k(result_ids: List[str], qrels: Dict[str, float], k: int) -> Optional[float]:
	relevant = {pid for pid, rel in qrels.items() if rel > 0}
	if not relevant:
		return None

	retrieved = set(str(pid) for pid in result_ids[:k])
	return len(retrieved & relevant) / len(relevant)


def hit_at_k(result_ids: List[str], qrels: Dict[str, float], k: int) -> Optional[float]:
	if not qrels:
		return None

	for pid in result_ids[:k]:
		if qrels.get(str(pid), 0.0) > 0:
			return 1.0
	return 0.0


def mrr_at_k(result_ids: List[str], qrels: Dict[str, float], k: int) -> Optional[float]:
	if not qrels:
		return None

	for rank, pid in enumerate(result_ids[:k], start=1):
		if qrels.get(str(pid), 0.0) > 0:
			return 1.0 / rank
	return 0.0



def mean_ignore_none(values: List[Optional[float]]) -> Optional[float]:
	values = [v for v in values if v is not None]
	if not values:
		return None
	return float(np.mean(values))



def evaluate_results(
	results_by_qid: Dict[int, List[str]],
	qrels_by_qid: Dict[int, Dict[str, float]],
	ks: List[int],
) -> Dict[str, float]:
	report = {}

	for k in ks:
		report[f"nDCG@{k}"] = mean_ignore_none([
			ndcg_at_k(results_by_qid.get(qid, []), qrels, k)
			for qid, qrels in qrels_by_qid.items()
		])

		report[f"Recall@{k}"] = mean_ignore_none([
			recall_at_k(results_by_qid.get(qid, []), qrels, k)
			for qid, qrels in qrels_by_qid.items()
		])

		report[f"Hit@{k}"] = mean_ignore_none([
			hit_at_k(results_by_qid.get(qid, []), qrels, k)
			for qid, qrels in qrels_by_qid.items()
		])

		report[f"MRR@{k}"] = mean_ignore_none([
			mrr_at_k(results_by_qid.get(qid, []), qrels, k)
			for qid, qrels in qrels_by_qid.items()
		])

	return report


def summarize_latencies(latencies: List[float]) -> Dict[str, float]:
	arr = np.array(latencies, dtype=np.float64)
	total = float(np.sum(arr))

	return {
		"search_avg_latency": float(np.mean(arr)),
		"search_p50_latency": float(np.percentile(arr, 50)),
		"search_p95_latency": float(np.percentile(arr, 95)),
		"search_p99_latency": float(np.percentile(arr, 99)),
		"qps": float(len(arr) / total) if total > 0 else float("inf"),
		"search_total_query_time": total,
	}


def _search_one(adapter, query_text: str, top_k: int, vector=None):
	if vector is None:
		return adapter.search_one(query_text, top_k)

	return adapter.search_one(query_text, top_k, vector=vector)

def _search_batch(adapter, query_texts: List[str], top_k: int, vectors=None):
	if vectors is None:
		return adapter.search_batch(query_texts, top_k)

	return adapter.search_batch(query_texts, top_k, vectors=vectors)

def run_search(
	adapter,
	queries: List[Dict[str, Any]],
	top_k: int,
	warmup: int = 10,
	search_batch_size: int = 32,
	search_mode: str = "sequential",
	retrieval_mode: str = "lexical",
) -> Tuple[Dict[int, List[str]], List[float]]:
	"""
	Run benchmark search.

	search_mode:
		sequential:
			one query at a time

		batch:
			query batches

	retrieval_mode:
		lexical:
			pass only text

		hybrid:
			pass text + vector
	"""

	if search_mode not in {"sequential", "batch"}:
		raise ValueError("search_mode must be one of: sequential, batch")

	if retrieval_mode not in {"lexical", "hybrid"}:
		raise ValueError("retrieval_mode must be one of: lexical, hybrid")

	use_vectors = retrieval_mode == "hybrid"

	# Warmup: do not record latency/results.
	warmup_queries = queries[:warmup]
	for q in warmup_queries:
		query_text = q["query"]
		vector = q["vector"] if use_vectors else None
		_search_one(adapter, query_text, top_k, vector=vector)

	results_by_qid: Dict[int, List[str]] = {}
	latencies: List[float] = []

	if search_mode == "sequential":
		for i, q in enumerate(queries):
			qid = q["query_id"]
			query_text = q["query"]
			vector = q["vector"] if use_vectors else None

			if use_vectors and vector is None:
				raise ValueError(f"Missing vector for query_id={qid}")

			t0 = time.perf_counter()
			result_ids = _search_one(
				adapter=adapter,
				query_text=query_text,
				top_k=top_k,
				vector=vector,
			)
			latency = time.perf_counter() - t0

			results_by_qid[qid] = result_ids
			latencies.append(latency)
			if i%100 == 0:
				print(f"[search] {i}", end="\r")

	else:
		for start in range(0, len(queries), search_batch_size):
			batch = queries[start:start + search_batch_size]

			query_ids = [q["query_id"] for q in batch]
			query_texts = [q["query"] for q in batch]

			vectors = None
			if use_vectors:
				vectors = [q["vector"] for q in batch]

				for qid, vector in zip(query_ids, vectors):
					if vector is None:
						raise ValueError(f"Missing vector for query_id={qid}")

			t0 = time.perf_counter()
			batch_results = _search_batch(
				adapter=adapter,
				query_texts=query_texts,
				top_k=top_k,
				vectors=vectors,
			)
			batch_latency = time.perf_counter() - t0

			if len(batch_results) != len(batch):
				raise RuntimeError(
					f"Batch result length mismatch: "
					f"got {len(batch_results)}, expected {len(batch)}"
				)

			per_query_latency = batch_latency / max(1, len(batch))

			for qid, result_ids in zip(query_ids, batch_results):
				results_by_qid[qid] = result_ids
				latencies.append(per_query_latency)

			print(f"[search] {start}", end="\r")
	return results_by_qid, latencies



# -----------------------------
# Typesense adapter
# -----------------------------
"""
sudo docker run -p 8108:8108 --memory="16gb" --memory-swap="16gb" --cpus="16" -v/tmp:/data mirror2.chabokan.net/typesense/typesense:30.2 --data-dir /data --api-key=xyz
"""

class TypesenseAdapter:
	def __init__(
		self,
		host: str,
		api_key: str,
		collection: str,
		timeout_s: float,
		num_typos: int,
		query_by: str = "title",
		query_by_weights: str = "8",
		retrieval_mode: str = "lexical",
		vector_dim: int = 0,
		vector_field: str = "embedding",
		alpha: float = 0.5,
		m: int = 32,
		efc: int = 2048,
		efs: int = 2048,
		vector_k: Optional[int] = None,
		drop_tokens_threshold: int = 0,
		batch_size: int = 2048,
	):
		if retrieval_mode not in {"lexical", "hybrid"}:
			raise ValueError("retrieval_mode must be one of: lexical, hybrid")

		self.host = host.rstrip("/")
		self.api_key = api_key
		self.collection = collection
		self.timeout_s = timeout_s
		self.num_typos = int(num_typos)

		self.query_by = query_by
		self.query_by_weights = query_by_weights

		self.retrieval_mode = retrieval_mode
		self.vector_dim = int(vector_dim or 0)
		self.vector_field = vector_field
		self.alpha = float(alpha)

		self.m = int(m)
		self.efc = int(efc)
		self.efs = int(efs)
		self.vector_k = int(vector_k) if vector_k is not None else None
		self.drop_tokens_threshold = int(drop_tokens_threshold)

		if self.retrieval_mode == "hybrid" and self.vector_dim <= 0:
			raise ValueError("hybrid mode requires vector_dim > 0")

		if not 0.0 <= self.alpha <= 1.0:
			raise ValueError("alpha must be between 0.0 and 1.0")

		self.session = requests.Session()
		self.session.headers.update({
			"X-TYPESENSE-API-KEY": self.api_key,
		})
		self.batch_size = int(batch_size)

	def _url(self, path: str) -> str:
		return f"{self.host}{path}"

	def _vector_to_list(self, vector: Any) -> List[float]:
		if vector is None:
			raise ValueError("hybrid mode requires a query/document vector")

		arr = np.asarray(vector, dtype=np.float32)

		if arr.ndim != 1:
			raise ValueError(f"vector must be 1-D, got shape={arr.shape}")

		if arr.shape[0] != self.vector_dim:
			raise ValueError(
				f"vector dim mismatch: got {arr.shape[0]}, expected {self.vector_dim}"
			)

		return arr.tolist()

	def _vector_query(self, vector: Any, top_k: int) -> str:
		vec = self._vector_to_list(vector)

		k = self.vector_k or top_k
		k = max(int(k), int(top_k))
		ef = max(k, self.efs)

		vector_json = json.dumps(vec, separators=(",", ":"))

		return (
			f"{self.vector_field}:("
			f"{vector_json}, "
			f"k:{k}, "
			f"alpha:{self.alpha}, "
			f"ef:{ef}"
			f")"
		)

	def _search_payload(
		self,
		query: str,
		top_k: int,
		vector: Optional[Any] = None,
	) -> Dict[str, Any]:
		payload = {
			"collection": self.collection,
			"q": str(query or ""),
			"query_by": self.query_by,
			"query_by_weights": self.query_by_weights,
			"text_match_type": "sum_score",
			"prefix": "false",
			"num_typos": str(self.num_typos),
			"prioritize_exact_match": "true",
			"prioritize_num_matching_fields": "true",
			"per_page": int(top_k),
			"exclude_fields": self.vector_field,
		}

		if self.retrieval_mode == "hybrid":
			payload["vector_query"] = self._vector_query(vector, top_k=top_k)
			payload["sort_by"] = "_text_match:desc"

			# Typesense recommends this for multi-word hybrid queries to reduce
			# redundant internal keyword searches and CPU usage.
			payload["drop_tokens_threshold"] = self.drop_tokens_threshold

		return payload

	def _import_batch(self, docs: List[Dict[str, Any]]) -> int:
		if not docs:
			return 0

		jsonl = "\n".join(
			json.dumps(doc, ensure_ascii=False)
			for doc in docs
		)

		r = self.session.post(
			self._url(f"/collections/{self.collection}/documents/import"),
			params={
				"action": "create",
			},
			data=jsonl.encode("utf-8"),
			headers={
				"X-TYPESENSE-API-KEY": self.api_key,
				"Content-Type": "text/plain",
			},
			timeout=max(self.timeout_s, 120),
		)

		r.raise_for_status()

		failures = 0
		first_failure = None

		for line in r.text.splitlines():
			if not line.strip():
				continue

			obj = json.loads(line)

			if not obj.get("success", False):
				failures += 1

				if first_failure is None:
					first_failure = obj

		if failures:
			raise RuntimeError(
				f"Typesense import failures in batch: {failures}. "
				f"First failure: {first_failure}"
			)

		return len(docs)

	def build(self, items: List[Dict[str, Any]]) -> float:
		print(
			f"[typesense/build] collection={self.collection}, docs={len(items)}, "
			f"mode={self.retrieval_mode}, vector_dim={self.vector_dim}, "
			f"M={self.m}, efc={self.efc}, efs={self.efs}, alpha={self.alpha}"
		)

		self.session.delete(
			self._url(f"/collections/{self.collection}"),
			timeout=self.timeout_s,
		)

		fields = [
			{"name": "title", "type": "string"},
			{"name": "category", "type": "string", "facet": True},
			{"name": "subcategory", "type": "string", "facet": True},
		]

		if self.retrieval_mode == "hybrid":
			fields.append({
				"name": self.vector_field,
				"type": "float[]",
				"num_dim": self.vector_dim,
				"hnsw_params": {
					"M": self.m,
					"ef_construction": self.efc,
				},
			})

		schema = {
			"name": self.collection,
			"fields": fields,
		}

		r = self.session.post(
			self._url("/collections"),
			json=schema,
			timeout=self.timeout_s,
		)
		r.raise_for_status()

		t0 = time.perf_counter()

		docs = []
		imported = 0

		for i, item in enumerate(items, start=1):
			doc = {
				"id": str(item["id"]),
				"title": str(item.get("title", "") or ""),
				"category": str(item.get("category", "") or ""),
				"subcategory": str(item.get("subcategory", "") or ""),
			}

			if self.retrieval_mode == "hybrid":
				vector = item.get("vector")

				if vector is None:
					raise ValueError(
						f"Missing vector for item {item['id']} in hybrid mode"
					)

				doc[self.vector_field] = self._vector_to_list(vector)

			docs.append(doc)

			if len(docs) >= self.batch_size:
				imported += self._import_batch(docs)
				docs.clear()

				print(
					f"[typesense/build] imported {imported}/{len(items)}",
					end="\r",
				)

		if docs:
			imported += self._import_batch(docs)
			docs.clear()

			print(
				f"[typesense/build] imported {imported}/{len(items)}",
				end="\r",
			)

		build_time = time.perf_counter() - t0

		print(f"\n[typesense/build] done in {build_time:.3f}s")
		return build_time

	def search_one(
		self,
		query: str,
		top_k: int,
		vector: Optional[Any] = None,
	) -> List[str]:
		results = self.search_batch(
			queries=[query],
			top_k=top_k,
			vectors=[vector] if self.retrieval_mode == "hybrid" else None,
		)

		return results[0]

	def search_batch(
		self,
		queries: List[str],
		top_k: int,
		vectors: Optional[List[Any]] = None,
	) -> List[List[str]]:
		if not queries:
			return []

		if self.retrieval_mode == "hybrid":
			if vectors is None:
				raise ValueError("hybrid batch search requires query vectors")

			if len(queries) != len(vectors):
				raise ValueError(
					f"queries/vectors length mismatch: {len(queries)} vs {len(vectors)}"
				)

		searches = []

		for i, query in enumerate(queries):
			vector = vectors[i] if self.retrieval_mode == "hybrid" else None

			searches.append(
				self._search_payload(
					query=query,
					top_k=top_k,
					vector=vector,
				)
			)

		payload = {
			"searches": searches,
		}

		r = self.session.post(
			self._url("/multi_search"),
			json=payload,
			timeout=max(self.timeout_s, 120),
		)
		r.raise_for_status()

		data = r.json()
		results = []

		for result in data.get("results", []):
			if "error" in result:
				raise RuntimeError(f"Typesense multi-search query failed: {result}")

			hits = result.get("hits", [])
			ids = [str(hit["document"]["id"]) for hit in hits]
			results.append(ids)

		if len(results) != len(queries):
			raise RuntimeError(
				f"Typesense multi-search returned {len(results)} results "
				f"for {len(queries)} queries"
			)

		return results

	def close(self):
		self.session.close()

# -----------------------------
# brinicle adapter
# -----------------------------

def _jsonable(value):
	if hasattr(value, "tolist"):
		return value.tolist()

	if isinstance(value, dict):
		return {k: _jsonable(v) for k, v in value.items()}

	if isinstance(value, (list, tuple)):
		return [_jsonable(v) for v in value]

	return value

class BrinicleItemAdapter:
	def __init__(
		self,
		host: str,
		index_name: str,
		dim: int,
		vector_dim: int,
		m: int,
		efc: int,
		efs: int,
		alpha: float,
		seed: int,
		retrieval_mode: str = "lexical",
		search_batch_jobs: int = 32,
		build_n_threads: int = 4,
		ingest_print_every: int = 10_000,
		batch_size: int = 2048,
	):
		if retrieval_mode not in {"lexical", "hybrid"}:
			raise ValueError("retrieval_mode must be one of: lexical, hybrid")

		self.host = host.rstrip("/")
		self.index_name = index_name

		self.dim = int(dim)
		self.vector_dim = int(vector_dim or 0)

		self.m = int(m)
		self.efc = int(efc)
		self.efs = int(efs)
		self.alpha = float(alpha)
		self.seed = int(seed)

		self.retrieval_mode = retrieval_mode

		if self.retrieval_mode == "hybrid" and self.vector_dim <= 0:
			raise ValueError("hybrid mode requires vector_dim > 0")

		self.search_batch_jobs = int(search_batch_jobs)
		self.build_n_threads = int(build_n_threads)

		self.ingest_print_every = int(ingest_print_every)
		self.batch_size = int(batch_size)

		self.session = requests.Session()


	def _url(self, path: str) -> str:
		return f"{self.host}{path}"



	def _post(self, path: str, payload: Dict[str, Any]):
		payload = _jsonable(payload)

		body = json.dumps(
			payload,
			ensure_ascii=False,
			separators=(",", ":"),
		).encode("utf-8")

		r = self.session.post(
			self._url(path),
			data=body,
			headers={"content-type": "application/json"},
			timeout=None,
		)

		if not r.ok:
			print("\n[brinicle/_post] request failed")
			print("url:", self._url(path))
			print("status:", r.status_code)
			print("payload:", json.dumps(payload, ensure_ascii=False, indent=2)[:4000])
			print("response:", r.text[:4000])
			r.raise_for_status()

		return r.json() if r.text.strip() else {}

	def _delete_if_loaded(self, destroy: bool = False):
		url = self._url(f"/indexes/{self.index_name}")
		r = self.session.delete(
			url,
			params={"destroy": str(bool(destroy)).lower()},
			timeout=None,
		)

		if r.status_code == 404:
			return

		if not r.ok:
			print("\n[brinicle/delete] request failed")
			print("url:", url)
			print("status:", r.status_code)
			print("response:", r.text[:4000])
			r.raise_for_status()

	def set_alpha(self, alpha):
		self.alpha = float(alpha)
		self.load()

	def _index_payload(self) -> Dict[str, Any]:
		kwargs = {
			"index_name": self.index_name,
			"dim": self.dim,
			"params": {
				"M": self.m,
				"ef_construction": self.efc,
				"ef_search": self.efs,
				"alpha": self.alpha,
				"build_n_threads": self.build_n_threads,
				"rng_seed": self.seed,
			},
		}
		if self.vector_dim > 0:
			kwargs["vector_dim"] = self.vector_dim
			kwargs["vector_normalized"] = True

		return kwargs

	def _create_index(self):
		self._post("/indexes", self._index_payload())

	def build(self, items: List[Dict[str, Any]]) -> float:
		print(
			f"[brinicle_api/build] index={self.index_name}, "
			f"docs={len(items)}, lexical_dim={self.dim}, "
			f"vector_dim={self.vector_dim}, mode={self.retrieval_mode}, "
			f"efc={self.efc}, efs={self.efs}, M={self.m}, alpha={self.alpha}"
		)

		self._delete_if_loaded(destroy=True)
		self._create_index()

		t0 = time.perf_counter()

		self._post("/init", {
			"index_name": self.index_name,
			"mode": "build",
		})

		ingest_url = self._url("/ingest/batch")

		r = self.session.post(
			ingest_url,
			params={"index_name": self.index_name},
			data=self._iter_ndjson_items(items),
			headers={"content-type": "application/x-ndjson"},
			timeout=None,
		)

		if not r.ok:
			print("\n[brinicle_api/build] batch ingest failed")
			print("url:", ingest_url)
			print("status:", r.status_code)
			print("response:", r.text[:4000])
			r.raise_for_status()

		ingest_response = r.json() if r.text.strip() else {}
		ingested_count = int(ingest_response.get("count", -1))

		if ingested_count != len(items):
			raise RuntimeError(
				f"Server ingested {ingested_count} items, expected {len(items)}"
			)

		self._post("/finalize", {
			"index_name": self.index_name,
		})

		build_time = time.perf_counter() - t0
		print(f"\n[brinicle_api/build] done in {build_time:.3f}s")

		return build_time

	def _iter_ndjson_items(self, items: List[Dict[str, Any]]):
		buf = bytearray()
		batch_count = 0

		for i, item in enumerate(items, start=1):
			payload = {
				"id": str(item["id"]),
				"title": item.get("title") or "",
			}

			vector = item.get("vector")
			if vector is not None:
				payload["vector"] = vector

			line = (
				json.dumps(
					_jsonable(payload),
					ensure_ascii=False,
					separators=(",", ":"),
				).encode("utf-8")
				+ b"\n"
			)

			buf.extend(line)
			batch_count += 1

			if batch_count >= self.batch_size:
				yield bytes(buf)
				buf.clear()
				batch_count = 0

			if i % self.ingest_print_every == 0 or i == len(items):
				print(
					f"[brinicle_api/build] streamed {i}/{len(items)}",
					end="\r",
				)

		if buf:
			yield bytes(buf)

	def load(self) -> float:
		print(
			f"[brinicle_api/load] index={self.index_name}, "
			f"lexical_dim={self.dim}, vector_dim={self.vector_dim}, "
			f"mode={self.retrieval_mode}, alpha={self.alpha}"
		)

		t0 = time.perf_counter()

		self._post("/indexes/load", self._index_payload())

		load_time = time.perf_counter() - t0
		print(f"[brinicle_api/load] loaded in {load_time:.3f}s")

		return load_time

	def search_one(self, query: str, top_k: int, vector=None) -> List[str]:
		payload = {
			"index_name": self.index_name,
			"query": query,
			"k": int(top_k),
			"ef_search": self.efs,
		}

		if self.retrieval_mode == "hybrid":
			if vector is None:
				raise ValueError("hybrid search_one requires vector")
			payload["vector"] = vector

		data = self._post("/search", payload)

		return [str(x) for x in data]

	def search_batch(
		self,
		queries: List[str],
		top_k: int,
		vectors=None,
	) -> List[List[str]]:
		if not queries:
			return []

		payload = {
			"index_name": self.index_name,
			"queries": queries,
			"k": int(top_k),
			"ef_search": self.efs,
			"n_jobs": self.search_batch_jobs,
		}

		if self.retrieval_mode == "hybrid":
			if vectors is None:
				raise ValueError("hybrid search_batch requires vectors")

			if len(vectors) != len(queries):
				raise ValueError(
					f"vectors length {len(vectors)} does not match "
					f"queries length {len(queries)}"
				)

			payload["vectors"] = vectors

		data = self._post("/search/batch", payload)

		if len(data) != len(queries):
			raise RuntimeError(
				f"Brinicle batch search returned {len(data)} result lists "
				f"for {len(queries)} queries"
			)

		return [[str(x) for x in ids] for ids in data]

	def close(self):
		try:
			self._delete_if_loaded(destroy=False)
		finally:
			self.session.close()

# -----------------------------
# Meilisearch adapter
# -----------------------------

class MeilisearchAdapter:
	def __init__(
		self,
		host: str,
		api_key: str,
		index_name: str,
		timeout_s: float = 30.0,
		batch_size: int = 2048,
		typo_tolerance: bool = False,
		retrieval_mode: str = "lexical",
		vector_dim: int = 0,
		embedder_name: str = "default",
		semantic_ratio: float = 0.5,
	):
		if retrieval_mode not in {"lexical", "hybrid"}:
			raise ValueError("retrieval_mode must be one of: lexical, hybrid")

		self.host = host.rstrip("/")
		self.api_key = api_key
		self.index_name = index_name
		self.timeout_s = timeout_s
		self.batch_size = batch_size
		self.typo_tolerance = typo_tolerance

		self.retrieval_mode = retrieval_mode
		self.vector_dim = int(vector_dim or 0)
		self.embedder_name = str(embedder_name)
		self.semantic_ratio = float(semantic_ratio)

		if self.retrieval_mode == "hybrid" and self.vector_dim <= 0:
			raise ValueError("hybrid mode requires vector_dim > 0")

		if not 0.0 <= self.semantic_ratio <= 1.0:
			raise ValueError("semantic_ratio must be between 0.0 and 1.0")

		self.session = requests.Session()
		self.session.headers.update({
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
		})

		self.safe_to_original_id: Dict[str, str] = {}

	def _to_meili_id(self, doc_id: str) -> str:
		return str(doc_id).replace("::", "__").replace(":", "_")

	def _from_meili_id(self, meili_id: str) -> str:
		return self.safe_to_original_id.get(str(meili_id), str(meili_id))

	def _url(self, path: str) -> str:
		return f"{self.host}{path}"

	def _request(self, method: str, path: str, **kwargs):
		r = self.session.request(
			method,
			self._url(path),
			timeout=self.timeout_s,
			**kwargs,
		)

		# Deleting a missing index is fine for benchmark rebuilds.
		if method.upper() == "DELETE" and r.status_code == 404:
			return {}

		r.raise_for_status()

		if not r.text.strip():
			return {}

		return r.json()

	def _wait_task(self, task_uid: int):
		while True:
			data = self._request("GET", f"/tasks/{task_uid}")
			status = data.get("status")

			if status == "succeeded":
				return data

			if status in {"failed", "canceled"}:
				raise RuntimeError(f"Meilisearch task failed: {data}")

			time.sleep(0.05)

	def _maybe_wait(self, response: Dict[str, Any]):
		task_uid = response.get("taskUid")
		if task_uid is not None:
			self._wait_task(task_uid)

	def _index_exists(self) -> bool:
		r = self.session.get(
			self._url(f"/indexes/{self.index_name}"),
			timeout=self.timeout_s,
		)

		if r.status_code == 200:
			return True

		if r.status_code == 404:
			return False

		r.raise_for_status()
		return False

	def _vector_to_list(self, vector: Any) -> List[float]:
		if vector is None:
			raise ValueError("vector is required in hybrid mode")

		arr = np.asarray(vector, dtype=np.float32)

		if arr.ndim != 1:
			raise ValueError(f"vector must be 1-D, got shape={arr.shape}")

		if arr.shape[0] != self.vector_dim:
			raise ValueError(
				f"vector dim mismatch: got {arr.shape[0]}, expected {self.vector_dim}"
			)

		return arr.tolist()

	def build(self, items: List[Dict[str, Any]]) -> float:
		print(
			f"[meilisearch/build] index={self.index_name}, "
			f"docs={len(items)}, mode={self.retrieval_mode}, "
			f"vector_dim={self.vector_dim}, embedder={self.embedder_name}, "
			f"semantic_ratio={self.semantic_ratio}, "
			f"typo_tolerance={self.typo_tolerance}"
		)

		if self._index_exists():
			print(f"[meilisearch/build] deleting existing index={self.index_name}")
			delete_resp = self._request("DELETE", f"/indexes/{self.index_name}")
			self._maybe_wait(delete_resp)
		else:
			print("[meilisearch/build] index does not exist, skipping delete")

		create_resp = self._request(
			"POST",
			"/indexes",
			json={
				"uid": self.index_name,
				"primaryKey": "id",
			},
		)
		self._maybe_wait(create_resp)

		displayed_attributes = [
			"id",
			"original_id",
		]

		settings = {
			"searchableAttributes": [
				"title",
			],
			"displayedAttributes": displayed_attributes,
			"typoTolerance": {
				"enabled": bool(self.typo_tolerance),
			},
		}

		if self.retrieval_mode == "hybrid":
			settings["embedders"] = {
				self.embedder_name: {
					"source": "userProvided",
					"dimensions": self.vector_dim,
				}
			}

		settings_resp = self._request(
			"PATCH",
			f"/indexes/{self.index_name}/settings",
			json=settings,
		)
		self._maybe_wait(settings_resp)

		t0 = time.perf_counter()

		for start in range(0, len(items), self.batch_size):
			batch = items[start:start + self.batch_size]

			docs = []

			for item in batch:
				original_id = str(item["id"])
				meili_id = self._to_meili_id(original_id)

				self.safe_to_original_id[meili_id] = original_id

				doc = {
					"id": meili_id,
					"original_id": original_id,
					"title": str(item.get("title", "") or ""),
				}

				if self.retrieval_mode == "hybrid":
					vector = item.get("vector")
					if vector is None:
						raise ValueError(
							f"Missing vector for item {original_id} in hybrid mode"
						)

					doc["_vectors"] = {
						self.embedder_name: self._vector_to_list(vector)
					}

				docs.append(doc)

			resp = self._request(
				"POST",
				f"/indexes/{self.index_name}/documents",
				json=docs,
			)
			self._maybe_wait(resp)

			print(
				f"[meilisearch/build] ingested {start + len(batch)}/{len(items)}",
				end="\r",
			)

		build_time = time.perf_counter() - t0

		print(f"\n[meilisearch/build] done in {build_time:.3f}s")
		return build_time

	def _make_search_payload(
		self,
		query: str,
		top_k: int,
		vector: Optional[Any] = None,
	) -> Dict[str, Any]:
		payload = {
			"q": str(query or ""),
			"limit": int(top_k),
			"attributesToRetrieve": ["id", "original_id"],
		}

		if self.retrieval_mode == "hybrid":
			payload["vector"] = self._vector_to_list(vector)
			payload["hybrid"] = {
				"embedder": self.embedder_name,
				"semanticRatio": self.semantic_ratio,
			}

		return payload

	def search_one(
		self,
		query: str,
		top_k: int,
		vector: Optional[Any] = None,
	) -> List[str]:
		data = self._request(
			"POST",
			f"/indexes/{self.index_name}/search",
			json=self._make_search_payload(
				query=query,
				top_k=top_k,
				vector=vector,
			),
		)

		hits = data.get("hits", [])

		out = []

		for hit in hits:
			if "original_id" in hit:
				out.append(str(hit["original_id"]))
			else:
				out.append(self._from_meili_id(str(hit["id"])))

		return out

	def search_batch(
		self,
		queries: List[str],
		top_k: int,
		vectors: Optional[List[Any]] = None,
	) -> List[List[str]]:
		if not queries:
			return []

		if self.retrieval_mode == "hybrid":
			if vectors is None:
				raise ValueError("hybrid batch search requires query vectors")

			if len(queries) != len(vectors):
				raise ValueError(
					f"queries/vectors length mismatch: "
					f"{len(queries)} vs {len(vectors)}"
				)

		multi_queries = []

		for i, query in enumerate(queries):
			vector = vectors[i] if self.retrieval_mode == "hybrid" else None

			payload = self._make_search_payload(
				query=query,
				top_k=top_k,
				vector=vector,
			)

			payload["indexUid"] = self.index_name
			multi_queries.append(payload)

		data = self._request(
			"POST",
			"/multi-search",
			json={
				"queries": multi_queries,
			},
		)

		results = []

		for result in data.get("results", []):
			if "error" in result:
				raise RuntimeError(f"Meilisearch multi-search query failed: {result}")

			ids = []

			for hit in result.get("hits", []):
				if "original_id" in hit:
					ids.append(str(hit["original_id"]))
				else:
					ids.append(self._from_meili_id(str(hit["id"])))

			results.append(ids)

		if len(results) != len(queries):
			raise RuntimeError(
				f"Meilisearch multi-search returned {len(results)} results "
				f"for {len(queries)} queries"
			)

		return results

	def close(self):
		self.session.close()

# -----------------------------
# Brinicle in-process adapter
# -----------------------------

class BrinicleLocalItemAdapter:

	def __init__(
		self,
		index_name: str,
		dim: int,
		vector_dim: int,
		m: int,
		efc: int,
		efs: int,
		alpha: float,
		seed: int,
		retrieval_mode: str = "lexical",
		search_batch_jobs: int = 32,
		build_n_threads: int = 4,
	):
		if retrieval_mode not in {"lexical", "hybrid"}:
			raise ValueError("retrieval_mode must be one of: lexical, hybrid")

		self.index_name = index_name
		self.dim = int(dim)
		self.vector_dim = int(vector_dim or 0)
		self.retrieval_mode = retrieval_mode

		if self.retrieval_mode == "hybrid" and self.vector_dim <= 0:
			raise ValueError("hybrid mode requires vector_dim > 0")

		self.engine = None
		self.m = m
		self.efc = efc
		self.efs = efs
		self.seed = seed
		self.build_n_threads = build_n_threads
		self.search_batch_jobs = search_batch_jobs
		self.alpha = alpha

	def _make_engine(self):
		kwargs = {
			"dim": self.dim,
			"M": self.m,
			"ef_construction": self.efc,
			"ef_search": self.efs,
			"build_n_threads": self.build_n_threads,
			"alpha": self.alpha,
			"seed": self.seed,
		}

		if self.vector_dim > 0:
			kwargs["vector_dim"] = self.vector_dim
			kwargs["vector_normalized"] = True

		return brinicle.ItemSearchEngine(
			self.index_name,
			**kwargs,
		)

	def set_alpha(self, alpha):
		self.alpha = alpha
		self.engine = self._make_engine()

	def build(self, items: List[Dict[str, Any]]) -> float:
		print(
			f"[brinicle_local/build] index={self.index_name}, "
			f"docs={len(items)}, lexical_dim={self.dim}, "
			f"vector_dim={self.vector_dim}, mode={self.retrieval_mode}, "
			f"efc={self.efc}, efs={self.efs}, M={self.m}, alpha={self.alpha}"
		)

		self.engine = self._make_engine()

		t0 = time.perf_counter()

		self.engine.init(mode="build")

		for i, item in enumerate(items, start=1):
			self.engine.ingest(
				external_id=str(item["id"]),
				title=item["title"],
				vector=item["vector"],
			)
			if i % 10000 == 0 or i == len(items):
				print(
					f"[brinicle_local/build] ingested {i}/{len(items)}",
					end="\r",
				)

		self.engine.finalize()

		build_time = time.perf_counter() - t0
		print(f"\n[brinicle_local/build] done in {build_time:.3f}s")

		return build_time

	def load(self) -> float:
		print(
			f"[brinicle_local/load] index={self.index_name}, "
			f"lexical_dim={self.dim}, vector_dim={self.vector_dim}, "
			f"mode={self.retrieval_mode}"
		)

		t0 = time.perf_counter()
		self.engine = self._make_engine()
		load_time = time.perf_counter() - t0
		print(f"[brinicle_local/load] loaded in {load_time:.3f}s")

		return load_time

	def search_one(self, query: str, top_k: int, vector=None) -> List[str]:
		return self.engine.search(
			query,
			k=top_k,
			vector=vector,
			efs=self.efs,
		)

	def search_batch(
		self,
		queries: List[str],
		top_k: int,
		vectors=None,
	) -> List[List[str]]:

		return self.engine.search_batch(
			queries,
			k=top_k,
			efs=self.efs,
			vectors=vectors,
			n_jobs=self.search_batch_jobs,
		)


	def close(self):
		self.engine = None

# -----------------------------
# Weaviate adapter
# -----------------------------

class WeaviateAdapter:
	def __init__(
		self,
		host: str = "http://localhost:8080",
		grpc_port: int = 50051,
		collection: str = "EsciProductsTitleOnly",
		retrieval_mode: str = "lexical",
		alpha: float = 0.5,
		batch_size: int = 2048,
		m: int = 32,
		efc: int = 2048,
		efs: int = 2048,
	):
		if retrieval_mode not in {"lexical", "hybrid"}:
			raise ValueError("retrieval_mode must be one of: lexical, hybrid")

		self.host = host
		self.grpc_port = int(grpc_port)
		self.collection_name = collection
		self.retrieval_mode = retrieval_mode
		self.alpha = float(alpha)
		self.batch_size = int(batch_size)

		self.m = int(m)
		self.efc = int(efc)
		self.efs = int(efs)

		self.client = None
		self.collection = None

	def _connect(self):
		parsed = urlparse(self.host)

		if self.host in {"http://localhost:8080", "localhost", "local"}:
			self.client = weaviate.connect_to_local()
			return

		scheme = parsed.scheme or "http"
		hostname = parsed.hostname or "localhost"
		http_port = parsed.port or (443 if scheme == "https" else 8080)
		secure = scheme == "https"

		self.client = weaviate.connect_to_custom(
			http_host=hostname,
			http_port=http_port,
			http_secure=secure,
			grpc_host=hostname,
			grpc_port=self.grpc_port,
			grpc_secure=secure,
		)

	def _create_collection(self):
		if self.client.collections.exists(self.collection_name):
			self.client.collections.delete(self.collection_name)

		if self.retrieval_mode == "hybrid":
			self.collection = self.client.collections.create(
				name=self.collection_name,
				vector_index_config=wvc.Configure.VectorIndex.hnsw(
					distance_metric=wvc.VectorDistances.COSINE,
					max_connections=self.m,
					ef_construction=self.efc,
					ef=self.efs,
				),
				properties=[
					wvc.Property(name="external_id", data_type=wvc.DataType.TEXT),
					wvc.Property(name="title", data_type=wvc.DataType.TEXT),
					wvc.Property(name="category", data_type=wvc.DataType.TEXT),
					wvc.Property(name="subcategory", data_type=wvc.DataType.TEXT),
				],
			)
		else:
			self.collection = self.client.collections.create(
				name=self.collection_name,
				properties=[
					wvc.Property(name="external_id", data_type=wvc.DataType.TEXT),
					wvc.Property(name="title", data_type=wvc.DataType.TEXT),
					wvc.Property(name="category", data_type=wvc.DataType.TEXT),
					wvc.Property(name="subcategory", data_type=wvc.DataType.TEXT),
				],
			)

	def build(self, items: List[Dict[str, Any]]) -> float:
		print(
			f"[weaviate/build] collection={self.collection_name}, "
			f"docs={len(items)}, mode={self.retrieval_mode}, "
			f"M={self.m}, efc={self.efc}, efs={self.efs}, "
			f"alpha={self.alpha}, batch_size={self.batch_size}"
		)

		if self.client is None:
			self._connect()

		t0 = time.perf_counter()

		self._create_collection()

		with self.collection.batch.fixed_size(batch_size=self.batch_size) as batch:
			for i, item in enumerate(items, start=1):
				external_id = str(item["id"])

				properties = {
					"external_id": external_id,
					"title": str(item.get("title", "") or ""),
					"category": str(item.get("category", "") or ""),
					"subcategory": str(item.get("subcategory", "") or ""),
				}

				if self.retrieval_mode == "hybrid":
					vector = item.get("vector")
					if vector is None:
						raise ValueError(
							f"Missing vector for item {external_id} in hybrid mode"
						)

					batch.add_object(
						properties=properties,
						vector=np.asarray(vector, dtype=np.float32).tolist(),
						uuid=generate_uuid5(external_id),
					)
				else:
					batch.add_object(
						properties=properties,
						uuid=generate_uuid5(external_id),
					)

				if i % 10000 == 0 or i == len(items):
					print(f"[weaviate/build] ingested {i}/{len(items)}", end="\r")

		failed = self.collection.batch.failed_objects
		if failed:
			raise RuntimeError(
				f"Weaviate batch import failed for {len(failed)} objects. "
				f"First failure: {failed[0]}"
			)

		build_time = time.perf_counter() - t0
		print(f"\n[weaviate/build] done in {build_time:.3f}s")

		return build_time

	def load(self) -> float:
		t0 = time.perf_counter()

		if self.client is None:
			self._connect()

		self.collection = self.client.collections.get(self.collection_name)

		load_time = time.perf_counter() - t0
		print(f"[weaviate/load] loaded in {load_time:.3f}s")

		return load_time

	def search_one(
		self,
		query: str,
		top_k: int,
		vector: Optional[Any] = None,
	) -> List[str]:
		if self.collection is None:
			self.load()

		if self.retrieval_mode == "lexical":
			response = self.collection.query.bm25(
				query=query,
				query_properties=["title"],
				limit=top_k,
				return_properties=["external_id"],
			)
		else:
			if vector is None:
				raise ValueError("hybrid mode requires a query vector")

			response = self.collection.query.hybrid(
				query=query,
				vector=np.asarray(vector, dtype=np.float32).tolist(),
				query_properties=["title"],
				alpha=self.alpha,
				fusion_type=HybridFusion.RELATIVE_SCORE,
				limit=top_k,
				return_properties=["external_id"],
			)

		return [
			str(obj.properties["external_id"])
			for obj in response.objects
		]

	def search_batch(
		self,
		queries: List[str],
		top_k: int,
		vectors: Optional[List[Any]] = None,
	) -> List[List[str]]:
		if self.retrieval_mode == "hybrid":
			if vectors is None:
				raise ValueError("hybrid batch search requires query vectors")

			return [
				self.search_one(query=q, top_k=top_k, vector=v)
				for q, v in zip(queries, vectors)
			]

		return [
			self.search_one(query=q, top_k=top_k)
			for q in queries
		]

	def close(self):
		if self.client is not None:
			self.client.close()

		self.client = None
		self.collection = None


# -----------------------------
# OpenSearch adapter
# -----------------------------

"""
docker pull mirror-docker.runflare.com/opensearchproject/opensearch:latest && docker run -it -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" mirror-docker.runflare.com/opensearchproject/opensearch:latest
"""

class OpenSearchAdapter:
	def __init__(
		self,
		host: str = "http://localhost:9200",
		index_name: str = "esci_products_title_only",
		retrieval_mode: str = "lexical",
		vector_dim: int = 0,
		alpha: float = 0.5,
		m: int = 32,
		efc: int = 2048,
		efs: int = 2048,
		batch_size: int = 2048,
		username: Optional[str] = None,
		password: Optional[str] = None,
		verify_certs: bool = False,
		pipeline_name: Optional[str] = None,
	):
		if retrieval_mode not in {"lexical", "hybrid"}:
			raise ValueError("retrieval_mode must be one of: lexical, hybrid")

		self.host = host
		self.index_name = index_name
		self.retrieval_mode = retrieval_mode
		self.vector_dim = int(vector_dim or 0)
		self.alpha = float(alpha)

		if self.retrieval_mode == "hybrid" and self.vector_dim <= 0:
			raise ValueError("hybrid mode requires vector_dim > 0")

		self.m = int(m)
		self.efc = int(efc)
		self.efs = int(efs)
		self.batch_size = int(batch_size)

		self.username = username
		self.password = password
		self.verify_certs = verify_certs
		self.pipeline_name = pipeline_name or f"{self.index_name}_hybrid_pipeline"

		self.client = None

	def set_hybrid_alpha(self, alpha: float):
		self.alpha = float(alpha)

		if self.retrieval_mode != "hybrid":
			return

		if self.client is None:
			self.load()

		self._create_hybrid_pipeline()

	def _connect(self):
		parsed = urlparse(self.host)
		scheme = parsed.scheme or "http"
		host = parsed.hostname or "localhost"
		port = parsed.port or (443 if scheme == "https" else 9200)

		auth = None
		if self.username is not None and self.password is not None:
			auth = (self.username, self.password)

		self.client = OpenSearch(
			hosts=[{"host": host, "port": port}],
			http_auth=auth,
			use_ssl=(scheme == "https"),
			verify_certs=self.verify_certs,
			ssl_show_warn=False,
			http_compress=True,
			timeout=120,
			max_retries=3,
			retry_on_timeout=True,
		)

	def _create_hybrid_pipeline(self):
		# weights order matches hybrid query order:
		# 1) lexical match
		# 2) vector knn
		body = {
			"description": "Hybrid BM25 + vector score normalization pipeline",
			"phase_results_processors": [
				{
					"normalization-processor": {
						"normalization": {
							"technique": "min_max",
						},
						"combination": {
							"technique": "arithmetic_mean",
							"parameters": {
								"weights": [
									float(1.0 - self.alpha),
									float(self.alpha),
								]
							},
						},
					}
				}
			],
		}

		self.client.transport.perform_request(
			method="PUT",
			url=f"/_search/pipeline/{self.pipeline_name}",
			body=body,
		)

	def _create_index(self):
		if self.client.indices.exists(index=self.index_name):
			self.client.indices.delete(index=self.index_name)

		properties = {
			"external_id": {
				"type": "keyword",
			},
			"title": {
				"type": "text",
			},
			"category": {
				"type": "keyword",
			},
			"subcategory": {
				"type": "keyword",
			},
		}

		settings = {
			"index": {
				"number_of_shards": 1,
				"number_of_replicas": 0,
				"refresh_interval": "-1",
			}
		}

		if self.retrieval_mode == "hybrid":
			settings["index"]["knn"] = True
			settings["index"]["knn.algo_param.ef_search"] = self.efs

			properties["embedding"] = {
				"type": "knn_vector",
				"dimension": self.vector_dim,
				"method": {
					"name": "hnsw",
					"space_type": "cosinesimil",
					"engine": "faiss",
					"parameters": {
						"ef_construction": self.efc,
						"m": self.m,
					},
				},
			}

		body = {
			"settings": settings,
			"mappings": {
				"properties": properties,
			},
		}

		self.client.indices.create(index=self.index_name, body=body)

		if self.retrieval_mode == "hybrid":
			self._create_hybrid_pipeline()

	def build(self, items: List[Dict[str, Any]]) -> float:
		print(
			f"[opensearch/build] index={self.index_name}, "
			f"docs={len(items)}, mode={self.retrieval_mode}, "
			f"vector_dim={self.vector_dim}, M={self.m}, efc={self.efc}, "
			f"efs={self.efs}, alpha={self.alpha}, batch_size={self.batch_size}"
		)

		if self.client is None:
			self._connect()

		t0 = time.perf_counter()

		self._create_index()

		actions = []

		for i, item in enumerate(items, start=1):
			external_id = str(item["id"])

			source = {
				"external_id": external_id,
				"title": str(item.get("title", "") or ""),
				"category": str(item.get("category", "") or ""),
				"subcategory": str(item.get("subcategory", "") or ""),
			}

			if self.retrieval_mode == "hybrid":
				vector = item.get("vector")
				if vector is None:
					raise ValueError(
						f"Missing vector for item {external_id} in hybrid mode"
					)

				source["embedding"] = np.asarray(
					vector,
					dtype=np.float32,
				).tolist()

			actions.append(
				{
					"_op_type": "index",
					"_index": self.index_name,
					"_id": external_id,
					"_source": source,
				}
			)

			if len(actions) >= self.batch_size:
				helpers.bulk(
					self.client,
					actions,
					chunk_size=self.batch_size,
					request_timeout=120,
					raise_on_error=True,
				)
				actions.clear()

			if i % 10000 == 0 or i == len(items):
				print(f"[opensearch/build] ingested {i}/{len(items)}", end="\r")

		if actions:
			helpers.bulk(
				self.client,
				actions,
				chunk_size=self.batch_size,
				request_timeout=120,
				raise_on_error=True,
			)

		self.client.indices.refresh(index=self.index_name)

		# Restore normal refresh behavior after benchmark build.
		self.client.indices.put_settings(
			index=self.index_name,
			body={
				"index": {
					"refresh_interval": "1s",
				}
			},
		)

		build_time = time.perf_counter() - t0
		print(f"\n[opensearch/build] done in {build_time:.3f}s")

		return build_time

	def load(self) -> float:
		t0 = time.perf_counter()

		if self.client is None:
			self._connect()

		if not self.client.indices.exists(index=self.index_name):
			raise RuntimeError(f"OpenSearch index does not exist: {self.index_name}")

		load_time = time.perf_counter() - t0
		print(f"[opensearch/load] loaded in {load_time:.3f}s")

		return load_time

	def search_one(
		self,
		query: str,
		top_k: int,
		vector: Optional[Any] = None,
	) -> List[str]:
		if self.client is None:
			self.load()

		ef_search = max(top_k, self.efs)
		if self.retrieval_mode == "lexical":
			body = {
				"size": top_k,
				"_source": ["external_id"],
				"query": {
					"match": {
						"title": {
							"query": query,
						}
					}
				},
			}

			response = self.client.search(
				index=self.index_name,
				body=body,
				request_timeout=120,
			)

		else:
			if vector is None:
				raise ValueError("hybrid mode requires a query vector")

			query_vector = np.asarray(vector, dtype=np.float32).tolist()

			body = {
				"size": top_k,
				"track_total_hits": False,
				"_source": False,
				"query": {
					"hybrid": {
						"queries": [
							{
								"match": {
									"title": {
										"query": query,
									}
								}
							},
							{
								"knn": {
									"embedding": {
										"vector": query_vector,
										"k": top_k,
										"method_parameters": {
											"ef_search": ef_search,
										},
									}
								}
							},
						]
					}
				},
			}

			response = self.client.search(
				index=self.index_name,
				body=body,
				params={
					"search_pipeline": self.pipeline_name,
				},
				request_timeout=120,
			)

		hits = response.get("hits", {}).get("hits", [])

		return [str(hit["_id"]) for hit in response["hits"]["hits"]]

	def search_batch(
		self,
		queries: List[str],
		top_k: int,
		vectors: Optional[List[Any]] = None,
	) -> List[List[str]]:
		if self.client is None:
			self.load()

		ef_search = max(top_k, self.efs)

		if self.retrieval_mode == "hybrid":
			if vectors is None:
				raise ValueError("hybrid batch search requires query vectors")

			if len(queries) != len(vectors):
				raise ValueError(
					f"queries/vectors length mismatch: {len(queries)} vs {len(vectors)}"
				)

		lines = []

		for i, query in enumerate(queries):
			# Metadata line.
			lines.append(
				json.dumps(
					{
						"index": self.index_name,
					}
				)
			)

			if self.retrieval_mode == "lexical":
				body = {
					"size": top_k,
					"track_total_hits": False,
					"_source": False,
					"query": {
						"match": {
							"title": {
								"query": query,
							}
						}
					},
				}

			else:
				query_vector = np.asarray(vectors[i], dtype=np.float32).tolist()

				body = {
					"size": top_k,
					"track_total_hits": False,
					"_source": False,

					# OpenSearch supports search_pipeline inside each msearch body.
					"search_pipeline": self.pipeline_name,

					"query": {
						"hybrid": {
							"queries": [
								{
									"match": {
										"title": {
											"query": query,
										}
									}
								},
								{
									"knn": {
										"embedding": {
											"vector": query_vector,
											"k": top_k,
											"method_parameters": {
												"ef_search": ef_search,
											},
										}
									}
								},
							]
						}
					},
				}
			lines.append(json.dumps(body))
		# OpenSearch _msearch requires newline-delimited JSON ending with newline.
		ndjson_body = "\n".join(lines) + "\n"

		response = self.client.transport.perform_request(
			method="POST",
			url="/_msearch",
			body=ndjson_body,
		)

		responses = response.get("responses", [])

		if len(responses) != len(queries):
			raise RuntimeError(
				f"OpenSearch msearch returned {len(responses)} responses "
				f"for {len(queries)} queries"
			)

		results: List[List[str]] = []

		for idx, item in enumerate(responses):
			if "error" in item:
				raise RuntimeError(
					f"OpenSearch msearch failed for query index {idx}: {item['error']}"
				)

			hits = item.get("hits", {}).get("hits", [])

			# Because we indexed _id = external_id, no _source fetch is needed.
			results.append([str(hit["_id"]) for hit in hits])

		return results

	def close(self):
		if self.client is not None:
			self.client.close()

		self.client = None