
# Brinicle Benchmark Suite

This repository contains the benchmark code used to compare hybrid product search across:

- Brinicle
- Typesense
- Meilisearch
- Weaviate
- OpenSearch

The benchmark is run on two product-search datasets:

- **WANDS** — about 4K products
- **Amazon ESCI** — about 1.2M products

The task is simple:

> Given a product query and a large list of product documents, retrieve the exact matching product.

This benchmark focuses on **hybrid search**, not pure lexical search and not pure vector search.

Each engine was benchmarked with the same container resource limit:

- **16 GB RAM**
- **16 CPU cores**

Only one engine container should be running during each benchmark run.

---

## 1. Prepare datasets

Download the datasets manually:

- WANDS from its GitHub repository
- Amazon ESCI from its GitHub repository

Place them like this:

```bash
datasets/
  WANDS/
    ...
  ESCI/
    ...
````

Then run preprocessing:

```bash
python aux/preprocess_wands.py
python aux/preprocess_esci.py
```

The benchmark scripts expect the preprocessed files to be available under:

```bash
datasets/WANDS/preprocessed/
datasets/ESCI/
```

---

## 2. Run one engine container

Run only the engine you want to benchmark.

Do **not** run all containers at once.

### Typesense

```bash
bash typesense_run.sh
```

### Meilisearch

```bash
bash meilisearch_run.sh
```

### Weaviate

```bash
bash weaviate_run.sh
```

### OpenSearch

```bash
bash opensearch.sh
```

### Brinicle

Brinicle is built from its own repository:

```bash
git clone https://github.com/bicardinal/brinicle.git
cd brinicle

bash build.sh
make docker-build
make docker-run
```

Stop Brinicle with:

```bash
make docker-stop
make docker-clean
```

For a clean Brinicle benchmark run:

```bash
rm -rf brinicle_data
make docker-stop
make docker-clean
make docker-build
make docker-run
```

To change Brinicle Docker RAM limits, edit the Docker command in the `Makefile`.

---

## 3. Run WANDS benchmark

From this benchmark repository:

```bash
python bench_code/wands_bench.py \
  --engine typesense \
  --retrieval-mode hybrid \
  --qrels-mode exact \
  --search-mode batch \
  --search-batch-size 32 \
  --top-k 100 \
  --tune-alpha
```

Replace `typesense` with one of:

```bash
brinicle
typesense
meilisearch
weaviate
opensearch
```

Example:

```bash
python bench_code/wands_bench.py \
  --engine meilisearch \
  --retrieval-mode hybrid \
  --qrels-mode exact \
  --search-mode batch \
  --search-batch-size 32 \
  --top-k 100 \
  --tune-alpha
```

Results are written to:

```bash
benchmark/wands_results/
```

---

## 4. Run ESCI benchmark

```bash
python bench_code/esci_bench.py \
  --engine typesense \
  --retrieval-mode hybrid \
  --qrels-mode exact \
  --search-mode batch \
  --search-batch-size 32 \
  --top-k 100 \
  --tune-alpha
```

Replace `typesense` with one of:

```bash
brinicle
typesense
meilisearch
weaviate
opensearch
```

Example:

```bash
python bench_code/esci_bench.py \
  --engine opensearch \
  --retrieval-mode hybrid \
  --qrels-mode exact \
  --search-mode batch \
  --search-batch-size 32 \
  --top-k 100 \
  --tune-alpha
```

Results are written to:

```bash
benchmark/esci_results/
```

---

## 5. Brinicle alpha note

For most engines, alpha is only a search-time parameter.

For Brinicle, alpha affects the graph itself, so it should be used at build time and search time.

For WANDS, when tuning Brinicle alpha, first run:

```bash
python bench_code/wands_bench.py \
  --engine brinicle \
  --retrieval-mode hybrid \
  --qrels-mode exact \
  --search-mode batch \
  --search-batch-size 32 \
  --top-k 100 \
  --tune-alpha
```

The script prints the best alpha and exits.

Then restart Brinicle cleanly:

```bash
cd path/to/brinicle

rm -rf brinicle_data
make docker-stop
make docker-clean
make docker-build
make docker-run
```

Then rerun the benchmark without `--tune-alpha` and pass the selected alpha:

```bash
cd path/to/benchmark-repo

python bench_code/wands_bench.py \
  --engine brinicle \
  --retrieval-mode hybrid \
  --qrels-mode exact \
  --search-mode batch \
  --search-batch-size 32 \
  --top-k 100 \
  --brinicle-alpha <BEST_ALPHA>
```

Example:

```bash
python bench_code/wands_bench.py \
  --engine brinicle \
  --retrieval-mode hybrid \
  --qrels-mode exact \
  --search-mode batch \
  --search-batch-size 32 \
  --top-k 100 \
  --brinicle-alpha 0.9
```

---

## 6. Useful Brinicle parameters

Brinicle exposes HNSW and build/search parameters:

```bash
--m 8
--efc 512
--efs 1024
--build-n-jobs 1
--search-batch-jobs 32
--brinicle-alpha 0.9
```

Example:

```bash
python bench_code/esci_bench.py \
  --engine brinicle \
  --retrieval-mode hybrid \
  --qrels-mode exact \
  --search-mode batch \
  --search-batch-size 32 \
  --top-k 100 \
  --m 8 \
  --efc 512 \
  --efs 1024 \
  --build-n-jobs 1 \
  --search-batch-jobs 32 \
  --brinicle-alpha 0.9
```

Increasing `--build-n-jobs` can make Brinicle builds faster, usually with a small accuracy tradeoff.

---

## 7. Output files

Benchmark results are saved as JSON files.

WANDS:

```bash
benchmark/wands_results/
```

ESCI:

```bash
benchmark/esci_results/
```

Each result file contains:

* engine name
* dataset name
* retrieval mode
* query count
* document count
* qrels mode
* benchmark parameters
* latency statistics
* build latency
* memory profile, when available
* ranking metrics such as nDCG, Recall, Hit, and MRR

---

## 8. Notes on fairness

We tried to keep the benchmark as fair as possible:

* same datasets
* same task
* same container resource limits
* same top-k evaluation
* hybrid search enabled for all engines
* alpha tuning supported where applicable
* one engine container running at a time

---

## 9. Minimal command checklist

```bash
# 1. Prepare datasets
python aux/preprocess_wands.py
python aux/preprocess_esci.py

# 2. Start exactly one engine container
bash typesense_run.sh
# or
bash meilisearch_run.sh
# or
bash weaviate_run.sh
# or
bash opensearch.sh

# 3. Run WANDS
python bench_code/wands_bench.py \
  --engine typesense \
  --retrieval-mode hybrid \
  --qrels-mode exact \
  --search-mode batch \
  --search-batch-size 32 \
  --top-k 100 \
  --tune-alpha

# 4. Run ESCI
python bench_code/esci_bench.py \
  --engine typesense \
  --retrieval-mode hybrid \
  --qrels-mode exact \
  --search-mode batch \
  --search-batch-size 32 \
  --top-k 100 \
  --tune-alpha
```


There is also one script for creating figures one can utilize:
```bash
python -m aux.figures --root benchmark --out benchmark_figures
```

Thank you!