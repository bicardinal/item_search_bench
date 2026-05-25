from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# MODEL_PATH = (
#     "/home/swan/.cache/huggingface/hub/"
#     "models--nomic-ai--nomic-embed-text-v1.5/"
#     "snapshots/e9b6763023c676ca8431644204f50c2b100d9aab"
# )

MODEL_PATH = "nomic-ai/nomic-embed-text-v1.5"

model = SentenceTransformer(
    MODEL_PATH,
    device="cuda",
    trust_remote_code=True,
)


def preprocess_esci_title_only(
    dataset_dir: str,
    output_path: str,
    locale: str = "us",
    split: str | None = "test",
    index_corpus: str = "all_locale_products",
    embedding_dim: int = 256,
    batch_size: int = 256,
):

    global model

    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_stem = output_path.with_suffix("")

    document_vectors_path = output_stem.with_suffix(".documents.npy")
    query_vectors_path = output_stem.with_suffix(".queries.npy")
    document_ids_path = output_stem.with_suffix(".document_ids.json")
    query_ids_path = output_stem.with_suffix(".query_ids.json")

    examples_path = dataset_dir / "shopping_queries_dataset_examples.parquet"
    products_path = dataset_dir / "shopping_queries_dataset_products.parquet"
    sources_path = dataset_dir / "shopping_queries_dataset_sources.csv"

    def clean_text(value):
        if value is None or pd.isna(value):
            return ""
        value = str(value)
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    def make_doc_id(product_locale, product_id):
        return f"{product_locale}::{product_id}"

    def encode_with_prefix(texts, prefix: str):
        prefixed = [f"{prefix}: {text}" for text in texts]

        vectors = model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            truncate_dim=embedding_dim,
        )

        return np.asarray(vectors, dtype=np.float32)

    label_gain_exact = {
        "E": 1.0,
        "S": 0.0,
        "C": 0.0,
        "I": 0.0,
    }

    label_gain_graded = {
        "E": 1.0,
        "S": 0.1,
        "C": 0.01,
        "I": 0.0,
    }

    examples = pd.read_parquet(
        examples_path,
        columns=[
            "example_id",
            "query",
            "query_id",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "large_version",
            "split",
        ],
    )

    examples = examples[examples["product_locale"] == locale].copy()
    examples = examples[examples["large_version"] == 1].copy()

    if split is not None:
        examples = examples[examples["split"] == split].copy()

    sources = pd.read_csv(
        sources_path,
        usecols=["query_id", "source"],
    )

    examples = examples.merge(sources, on="query_id", how="left")

    examples["query"] = examples["query"].map(clean_text)
    examples["source"] = examples["source"].map(clean_text)

    examples["small_version"] = examples["small_version"].astype(int)
    examples["large_version"] = examples["large_version"].astype(int)

    examples["doc_id"] = [
        make_doc_id(loc, pid)
        for loc, pid in zip(examples["product_locale"], examples["product_id"])
    ]

    products = pd.read_parquet(
        products_path,
        columns=[
            "product_id",
            "product_title",
            "product_locale",
        ],
    )

    products = products[products["product_locale"] == locale].copy()

    products["doc_id"] = [
        make_doc_id(loc, pid)
        for loc, pid in zip(products["product_locale"], products["product_id"])
    ]

    products["title"] = products["product_title"].map(clean_text)

    products = products[
        ["doc_id", "product_id", "product_locale", "title"]
    ].drop_duplicates("doc_id")

    products = products[products["title"] != ""].copy()

    if index_corpus == "all_locale_products":
        pass
    elif index_corpus == "judged_products_only":
        judged_doc_ids = set(examples["doc_id"])
        products = products[products["doc_id"].isin(judged_doc_ids)].copy()
    else:
        raise ValueError(
            "index_corpus must be 'all_locale_products' or 'judged_products_only'"
        )

    products = products.sort_values("doc_id").reset_index(drop=True)
    products["vector_index"] = np.arange(len(products), dtype=np.int64)

    indexed_doc_ids = set(products["doc_id"])

    examples = examples[examples["doc_id"].isin(indexed_doc_ids)].copy()

    query_flags = (
        examples
        .groupby("query_id", as_index=False)
        .agg(
            query=("query", "first"),
            split=("split", "first"),
            source=("source", "first"),
            small_version=("small_version", "max"),
            large_version=("large_version", "max"),
        )
    )

    queries = (
        query_flags
        .sort_values("query_id")
        .reset_index(drop=True)
    )

    queries["vector_index"] = np.arange(len(queries), dtype=np.int64)

    qrels = examples[
        [
            "query_id",
            "doc_id",
            "product_id",
            "product_locale",
            "esci_label",
            "split",
            "source",
            "small_version",
            "large_version",
        ]
    ].copy()

    qrels["gain_exact"] = qrels["esci_label"].map(label_gain_exact)
    qrels["gain_graded"] = qrels["esci_label"].map(label_gain_graded)


    print(f"Encoding {len(products):,} documents at dim={embedding_dim}...")
    document_vectors = encode_with_prefix(
        products["title"].tolist(),
        prefix="search_document",
    )

    if document_vectors.shape != (len(products), embedding_dim):
        raise RuntimeError(
            f"Bad document vector shape: {document_vectors.shape}, "
            f"expected {(len(products), embedding_dim)}"
        )

    np.save(document_vectors_path, document_vectors)
    del document_vectors

    print(f"Encoding {len(queries):,} queries at dim={embedding_dim}...")
    query_vectors = encode_with_prefix(
        queries["query"].tolist(),
        prefix="search_query",
    )

    if query_vectors.shape != (len(queries), embedding_dim):
        raise RuntimeError(
            f"Bad query vector shape: {query_vectors.shape}, "
            f"expected {(len(queries), embedding_dim)}"
        )

    np.save(query_vectors_path, query_vectors)
    del query_vectors

    with document_ids_path.open("w", encoding="utf-8") as f:
        json.dump(products["doc_id"].tolist(), f, ensure_ascii=False)

    with query_ids_path.open("w", encoding="utf-8") as f:
        json.dump(
            [int(qid) for qid in queries["query_id"].tolist()],
            f,
            ensure_ascii=False,
        )

    with output_path.open("w", encoding="utf-8") as f:
        meta = {
            "type": "meta",
            "dataset": "amazon_esci",
            "task": "title_only_retrieval",
            "locale": locale,
            "split": split,
            "index_corpus": index_corpus,
            "query_universe": "large_version",
            "tune_query_filter": "small_version == 1",
            "eval_query_filter": "large_version == 1 and small_version == 0",
            "embedding": {
                "model": str(MODEL_PATH),
                "dimension": embedding_dim,
                "dtype": "float32",
                "normalized": True,
                "document_prefix": "search_document",
                "query_prefix": "search_query",
                "document_vectors_path": str(document_vectors_path),
                "query_vectors_path": str(query_vectors_path),
                "document_ids_path": str(document_ids_path),
                "query_ids_path": str(query_ids_path),
                "mapping": "record.vector_index is row index in the corresponding .npy file",
            },
            "document_fields": [
                "id",
                "product_id",
                "locale",
                "title",
                "vector_index",
            ],
            "query_fields": [
                "query_id",
                "query",
                "split",
                "source",
                "small_version",
                "large_version",
                "vector_index",
            ],
            "qrel_fields": [
                "query_id",
                "doc_id",
                "esci_label",
                "gain_exact",
                "gain_graded",
                "small_version",
                "large_version",
            ],
            "label_meaning": {
                "E": "exact",
                "S": "substitute",
                "C": "complement",
                "I": "irrelevant",
            },
        }

        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        for row in products.itertuples(index=False):
            record = {
                "type": "document",
                "id": row.doc_id,
                "product_id": row.product_id,
                "locale": row.product_locale,
                "title": row.title,
                "vector_index": int(row.vector_index),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        for row in queries.itertuples(index=False):
            record = {
                "type": "query",
                "query_id": int(row.query_id),
                "query": row.query,
                "split": row.split,
                "source": row.source,
                "small_version": int(row.small_version),
                "large_version": int(row.large_version),
                "vector_index": int(row.vector_index),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        for row in qrels.itertuples(index=False):
            record = {
                "type": "qrel",
                "query_id": int(row.query_id),
                "doc_id": row.doc_id,
                "product_id": row.product_id,
                "locale": row.product_locale,
                "esci_label": row.esci_label,
                "gain_exact": float(row.gain_exact),
                "gain_graded": float(row.gain_graded),
                "split": row.split,
                "source": row.source,
                "small_version": int(row.small_version),
                "large_version": int(row.large_version),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "output_path": str(output_path),
        "document_vectors_path": str(document_vectors_path),
        "query_vectors_path": str(query_vectors_path),
        "document_ids_path": str(document_ids_path),
        "query_ids_path": str(query_ids_path),
        "documents": int(len(products)),
        "queries": int(len(queries)),
        "small_queries": int(queries["small_version"].sum()),
        "large_queries": int(queries["large_version"].sum()),
        "large_only_queries": int(
            ((queries["large_version"] == 1) & (queries["small_version"] == 0)).sum()
        ),
        "qrels": int(len(qrels)),
        "small_qrels": int(qrels["small_version"].sum()),
        "large_qrels": int(qrels["large_version"].sum()),
        "large_only_qrels": int(
            ((qrels["large_version"] == 1) & (qrels["small_version"] == 0)).sum()
        ),
        "embedding_dim": int(embedding_dim),
        "locale": locale,
        "split": split,
        "index_corpus": index_corpus,
    }

    return stats


stats = preprocess_esci_title_only(
    dataset_dir="datasets/ESCI",
    output_path="datasets/ESCI/esci_us_test_title_only.jsonl",
    locale="us",
    split="test",
    index_corpus="all_locale_products",
    embedding_dim=128,
    batch_size=256,
)

print(json.dumps(stats, indent=2))

# stats = preprocess_esci_title_only(
#     dataset_dir="datasets/ESCI",
#     output_path="datasets/ESCI/esci_us_large_test_title_only.jsonl",
#     locale="us",
#     version="large",
#     split="test",
#     index_corpus="all_locale_products",
# )

# print(stats)


# stats = preprocess_esci_title_only(
#     dataset_dir="datasets/ESCI",
#     output_path="datasets/ESCI/esci_jp_small_test_title_only_all_products.jsonl",
#     locale="jp",
#     version="small",
#     split="test",
#     index_corpus="all_locale_products",
# )

# print(stats)
