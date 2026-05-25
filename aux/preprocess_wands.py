from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA = Path("datasets/WANDS")
OUT = DATA / "preprocessed"
OUT.mkdir(parents=True, exist_ok=True)

# MODEL_PATH = (
#     "/home/swan/.cache/huggingface/hub/"
#     "models--nomic-ai--nomic-embed-text-v1.5/"
#     "snapshots/e9b6763023c676ca8431644204f50c2b100d9aab"
# )

MODEL_PATH = "nomic-ai/nomic-embed-text-v1.5"


EMBEDDING_DIM = 128
BATCH_SIZE = 256

COMPOSITION = "title"


def clean(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def leaf_category(x):
    if pd.isna(x):
        return ""
    parts = [p.strip() for p in str(x).split("/") if p.strip()]
    return parts[-1] if parts else ""


def join_text(*parts):
    return " ".join(p for p in (clean(x) for x in parts) if p)


products = pd.read_csv(DATA / "product.csv", sep="\t")
queries = pd.read_csv(DATA / "query.csv", sep="\t")
labels = pd.read_csv(DATA / "label.csv", sep="\t")

# -----------------------------
# Products
# -----------------------------

products["product_id"] = products["product_id"].astype(str)
products["product_name"] = products["product_name"].map(clean)
products["product_class"] = products["product_class"].map(clean)
products["category hierarchy"] = products["category hierarchy"].map(clean)
products["subcategory"] = products["category hierarchy"].map(leaf_category)

if "product_description" in products.columns:
    products["product_description"] = products["product_description"].map(clean)
else:
    products["product_description"] = ""

if "product_features" in products.columns:
    products["product_features"] = products["product_features"].map(clean)
else:
    products["product_features"] = ""

products["text"] = products["product_name"]

docs = products[
    [
        "product_id",
        "text",
        "product_name",
        "product_class",
        "subcategory",
    ]
].copy()

docs = docs.rename(columns={
    "product_id": "id",
    "text": "title",
    "product_class": "category",
})

docs = docs[docs["title"] != ""].drop_duplicates("id").reset_index(drop=True)
docs["vector_index"] = np.arange(len(docs), dtype=np.int64)

queries["query_id"] = queries["query_id"].astype(int)
queries["query"] = queries["query"].map(clean)

query_df = queries[["query_id", "query"]].copy()
query_df = query_df[query_df["query"] != ""].drop_duplicates("query_id").reset_index(drop=True)
query_df["vector_index"] = np.arange(len(query_df), dtype=np.int64)


labels["query_id"] = labels["query_id"].astype(int)
labels["product_id"] = labels["product_id"].astype(str)
labels["label"] = labels["label"].map(lambda x: clean(x).title())

label_gain_exact = {
    "Exact": 1.0,
    "Partial": 0.0,
    "Irrelevant": 0.0,
}

label_gain_graded = {
    "Exact": 1.0,
    "Partial": 0.1,
    "Irrelevant": 0.0,
}

qrels = labels[["query_id", "product_id", "label"]].copy()
qrels = qrels.rename(columns={
    "product_id": "doc_id",
    "label": "wands_label",
})

qrels["gain_exact"] = qrels["wands_label"].map(label_gain_exact).fillna(0.0)
qrels["gain_graded"] = qrels["wands_label"].map(label_gain_graded).fillna(0.0)

valid_doc_ids = set(docs["id"])
valid_query_ids = set(query_df["query_id"])

qrels = qrels[
    qrels["doc_id"].isin(valid_doc_ids)
    & qrels["query_id"].isin(valid_query_ids)
].reset_index(drop=True)

query_df = query_df[query_df["query_id"].isin(set(qrels["query_id"]))].reset_index(drop=True)
query_df["vector_index"] = np.arange(len(query_df), dtype=np.int64)


prefix = OUT / f"wands_{COMPOSITION}"

docs.to_json(
    prefix.with_suffix(".docs.jsonl"),
    orient="records",
    lines=True,
    force_ascii=False,
)

query_df.to_json(
    prefix.with_suffix(".queries.jsonl"),
    orient="records",
    lines=True,
    force_ascii=False,
)

qrels.to_json(
    prefix.with_suffix(".qrels.jsonl"),
    orient="records",
    lines=True,
    force_ascii=False,
)

with prefix.with_suffix(".doc_ids.json").open("w", encoding="utf-8") as f:
    json.dump(docs["id"].tolist(), f, ensure_ascii=False)

with prefix.with_suffix(".query_ids.json").open("w", encoding="utf-8") as f:
    json.dump(query_df["query_id"].astype(int).tolist(), f, ensure_ascii=False)


model = SentenceTransformer(
    MODEL_PATH,
    device="cuda",
    trust_remote_code=True,
)

doc_vectors = model.encode(
    [f"search_document: {x}" for x in docs["title"].tolist()],
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
    truncate_dim=EMBEDDING_DIM,
).astype(np.float32)

query_vectors = model.encode(
    [f"search_query: {x}" for x in query_df["query"].tolist()],
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
    truncate_dim=EMBEDDING_DIM,
).astype(np.float32)

np.save(prefix.with_suffix(".documents.npy"), doc_vectors)
np.save(prefix.with_suffix(".queries.npy"), query_vectors)

print("done")
print("composition:", COMPOSITION)
print("docs:", len(docs), doc_vectors.shape)
print("queries:", len(query_df), query_vectors.shape)
print("qrels:", len(qrels))
print("saved prefix:", prefix)