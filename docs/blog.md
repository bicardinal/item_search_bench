
# Abstract

Hybrid search is commonly implemented by combining lexical retrieval over an inverted index with semantic retrieval over a vector index, followed by score fusion or reranking. This paper studies an alternative formulation: representing lexical and semantic product-search signals inside a single HNSW graph. Brinicle encodes product-title tokens and dense title embeddings into one searchable representation. A custom distance function combines symbolic title matching and vector similarity during graph traversal, allowing lexical, semantic, and hybrid retrieval behavior to be expressed through the same graph structure. We evaluate this approach on WANDS and US-filtered Amazon ESCI using title-based hybrid product retrieval. Brinicle is compared with Weaviate, Meilisearch, Typesense, and OpenSearch under shared resource limits and the same precomputed embedding model. Across both datasets, Brinicle achieves competitive retrieval quality while reducing search memory usage and P99 latency relative to the compared systems. These results indicate that, for title-based product retrieval, hybrid search can be modeled as a single-graph retrieval problem rather than as post-hoc fusion over separate lexical and vector retrieval structures.

---

# 1. Introduction

Hybrid retrieval is commonly implemented as a coordination problem between two search systems. A lexical index retrieves documents through exact or near-exact term matching, while a vector index retrieves documents through dense semantic similarity. The final ranking is then produced by score fusion, reranking, or another combination strategy.

This architecture has become a practical default for modern search applications. Lexical retrieval preserves exact terms, identifiers, numbers, and product-specific fragments. Vector retrieval improves tolerance to vocabulary mismatch and natural-language variation. Used together, they often provide better retrieval behavior than either method alone.

The architectural cost is that hybrid retrieval usually requires multiple retrieval structures. A system may need to store and operate an inverted index, a vector index, and a fusion layer with its own scoring assumptions. This increases memory usage, tuning surface area, and operational complexity.

This paper studies a different formulation of hybrid retrieval: representing lexical and semantic signals inside a single HNSW graph.

## 1.1 Hybrid retrieval as a graph-distance problem

Brinicle treats hybrid product retrieval as a distance-function problem over one encoded representation.

Each item is encoded with symbolic title evidence and, in hybrid mode, a dense title embedding. Queries are encoded in the same representation family. Search is then performed through one HNSW graph using a custom distance function that combines title-token agreement and vector similarity during graph traversal.

The retrieval mode is controlled by the distance configuration. A lexical configuration emphasizes title-token matching. A vector configuration emphasizes embedding similarity. A hybrid configuration combines both signals inside the same graph search process.

This differs from the common architecture in which lexical and vector retrieval produce separate candidate sets that are merged afterward. In Brinicle, candidate exploration itself is hybrid-aware because the graph traversal uses the combined distance.

## 1.2 Product search as a motivating task

Product retrieval is a useful setting for evaluating this idea because it requires both exactness and tolerance.

A product query may contain short fragments that carry precise meaning:

```text
iphone 15 256gb
rtx 4060
sony wh-1000xm5
m2 macbook air
```

In these cases, numbers, model identifiers, and capacities are not incidental text. They are part of the user's intent. A semantically related result with the wrong model or capacity may be commercially incorrect.

At the same time, product titles are often long and noisy. They may contain brands, colors, editions, years, bundle descriptions, packaging terms, seller formatting, and marketing phrases. Users rarely type the full title. A retrieval system therefore needs to tolerate partial queries and vocabulary mismatch without losing exact symbolic evidence.

This makes product search a natural hybrid retrieval task. Lexical matching helps preserve exact constraints. Dense embeddings help recover semantically related titles when surface forms differ.

## 1.3 Brinicle's approach

Brinicle encodes product-title tokens and dense embeddings into a single HNSW-searchable representation. The graph is built over this representation, and a custom scorer defines how the symbolic and semantic components contribute to distance.

At a high level, the method consists of three parts:

```text
encoded item representation
+ single HNSW graph
+ hybrid-aware distance function
```

The encoded representation stores title-token evidence and, when enabled, a dense vector. The HNSW graph provides approximate nearest-neighbor traversal. The distance function determines whether the search behaves lexically, semantically, or as a hybrid of both.

This paper focuses on title-based hybrid retrieval. The evaluated configuration uses product titles and precomputed title embeddings for both documents and queries. Brinicle's broader item-search representation can include structured fields such as category, subcategory, and attributes, but the benchmark isolates the title + vector retrieval setting.

## 1.4 Evaluation overview

We evaluate Brinicle on WANDS and US-filtered Amazon ESCI, comparing it with Weaviate, Meilisearch, Typesense, and OpenSearch. All systems are tested under shared CPU and memory limits using the same precomputed embedding model.

The evaluation reports ranking quality, search latency, throughput, memory usage, and build cost. The main result is a system-level trade-off: Brinicle achieves competitive retrieval quality while reducing search memory usage and P99 latency in the tested setup.

The results support the architectural claim that hybrid product retrieval can be expressed through one graph and one distance function, rather than requiring post-hoc fusion over separate lexical and vector retrieval structures.

## 1.5 Contributions

This paper makes four contributions.

First, it presents a single-graph formulation for hybrid product retrieval, where lexical and semantic evidence are represented inside one HNSW-searchable object.

Second, it describes Brinicle's encoded item representation and hybrid-aware distance function, including symbolic title matching, dense-vector similarity, and the alpha mechanism used to control semantic bias.

Third, it evaluates the approach on two product-search benchmarks against four established hybrid search systems under shared resource limits.

Fourth, it reports the resulting trade-off between retrieval quality, memory usage, and search latency, showing that a single-graph design can provide competitive hybrid retrieval behavior with a smaller search-time resource footprint.

---

# 2. One-Graph Hybrid Retrieval

Hybrid retrieval is often described as a fusion problem: lexical retrieval produces one ranked list, semantic retrieval produces another, and a combination layer merges the two into a final ranking. Brinicle uses a different formulation. It treats hybrid retrieval as graph traversal over a representation that contains both symbolic and semantic evidence.

In this formulation, each item is encoded into one HNSW-searchable object. The object contains lexical title evidence and, in hybrid mode, a dense embedding. The HNSW graph is built over these encoded objects, and retrieval is controlled by a distance function that can read and combine the different regions of the representation.

At a high level, the retrieval pipeline is 1. document title + optional dense embedding 2. encoded item representation 3. single HNSW graph 4. hybrid-aware distance function 5. ranked results.
The key design choice is that lexical and semantic evidence participate in the same graph traversal. Candidate exploration is therefore influenced by the combined distance, rather than by a post-processing step over independently retrieved lexical and vector candidates.

## 2.1 Retrieval as distance over structured representations

Brinicle represents each item as a structured numeric object rather than as an ordinary dense vector alone. The representation contains enough information for the distance function to interpret different components separately.

For title-based hybrid retrieval, the relevant components are:

```text
title-token evidence
+ dense title embedding
```

The query is encoded in the same representation family:

```text
query-token evidence
+ dense query embedding
```

The distance function then compares the query and document through both symbolic and semantic components. Title-token overlap contributes lexical evidence. Vector similarity contributes semantic evidence. The final distance is a weighted combination of these signals.

This makes the HNSW graph a retrieval structure over hybrid-search objects. The graph organizes items according to the distance function used during construction, and the same family of distance functions is used during search.

## 2.2 Unified candidate exploration

In a two-index hybrid system, candidate generation is usually split across retrieval structures. A lexical index explores term-based candidates, while a vector index explores embedding-based candidates. Fusion happens after those candidate sets have already been produced. Brinicle moves the hybrid decision earlier. Since graph traversal uses a distance function that includes both title-token matching and vector similarity, lexical and semantic evidence affect candidate exploration directly. This changes the role of the hybrid scorer. It is not only a final ranking function. It also helps define local neighborhoods in the graph and influences which candidates are reached during approximate search. The result is a single candidate-exploration structure: 1. encoded query 2. HNSW traversal using hybrid distance 3. candidate set 4. ranked results.

This is the central architectural distinction. Hybrid behavior is part of the graph-search process itself.

## 2.3 Retrieval modes as distance configurations

Brinicle's retrieval modes are expressed through distance configuration.

The same encoded representation can support lexical, vector, or hybrid retrieval by changing the active components and their weights:

```text
lexical retrieval:
  title distance active
  vector distance inactive

vector retrieval:
  vector distance active
  title distance inactive

hybrid retrieval:
  title distance active
  vector distance active
```

This gives the system a single conceptual model:

```text
same representation family
same graph structure
different distance configurations
```

In lexical mode, retrieval is driven by symbolic title evidence. In vector mode, retrieval is driven by embedding similarity. In hybrid mode, both signals contribute to the distance used during graph traversal.

The benchmark in this paper focuses on the hybrid configuration, where product titles and title embeddings are both active.

## 2.4 Product-title hybrid retrieval

Product titles provide a useful test case for one-graph hybrid retrieval because they combine short exact identifiers with longer noisy descriptions. A title may contain model numbers, capacities, color names, brand names, technical variants, and marketing text. Some tokens are highly specific and must be matched carefully. Other parts of the title provide broader semantic context. Brinicle's representation preserves title-token evidence explicitly while also attaching dense semantic vectors. This allows the distance function to reward exact symbolic matches and semantic proximity within the same graph search. For example, a query such as "iphone 15 256gb" benefits from exact matching on `iphone`, `15`, and `256gb`, while vector similarity can still help when relevant product titles use different surrounding language.

The same principle applies to product queries involving model identifiers, abbreviated names, or partial descriptions. The graph does not need to choose between symbolic and semantic retrieval as separate execution paths. Both signals are available to the distance function.

## 2.5 Summary

One-graph hybrid retrieval can be summarized as follows:

```text
encoded item =
  lexical title evidence
  + optional dense vector

retrieval structure =
  one HNSW graph over encoded items

retrieval behavior =
  distance configuration over lexical and vector components
```

This formulation makes hybrid product search a graph-distance problem. The next section describes how Brinicle encodes items and queries so that the distance function can compare symbolic and semantic evidence inside one representation.

---

# 3. Encoding Items and Queries

One-graph hybrid retrieval requires documents and queries to be represented in a form that can be compared by a single distance function. Brinicle uses a structured numeric representation for this purpose. The representation is compact enough to be indexed by HNSW, while preserving separate regions for lexical, structured, and semantic evidence.

In the benchmarked configuration, each document is represented by its product title and a dense embedding of that title. Each query is represented by query text and a dense query embedding. Both are encoded into the same representation family, allowing the distance function to compare symbolic and semantic evidence during graph traversal.

## 3.1 Encoded object layout

Each encoded object begins with a fixed-size header followed by a variable-length payload.

The header stores metadata needed by the distance function:

```text
[
  version,
  title_count,
  attr_pair_count,
  category_id,
  subcategory_id,
  vector_dim,
  payload...
]
```

The payload stores the searchable content:

```text
title token ids
+ optional attribute key/value ids
+ optional dense vector
```

The header allows the scorer to parse the representation without external metadata. It can determine how many title tokens are present, whether structured fields exist, whether a dense vector is attached, and where each region begins.

For title-based hybrid retrieval, the active regions are:

```text
title token ids
+ dense title embedding
```

This layout keeps the representation numeric while preserving internal structure. The scorer can interpret title evidence and vector evidence separately instead of treating the object as an opaque dense vector.

## 3.2 Title-token encoding

Product titles are converted into sorted token identifiers. The title encoding pipeline is: 1. title text 2. normalization 3. isolated tokenization 4. token-id extraction 5. special-token filtering 6. term-frequency packing 7. sorted title-token representation.
The tokenizer preserves short product-specific fragments such as numbers, model names, and compact identifiers. These fragments are important in product retrieval because small textual differences can change the target item.

Examples include:

```text
4060
256gb
13 inch
a54
m2
wh-1000xm5
```

Dense embeddings can place related products near each other, but exact fragments still need to remain available to the scorer. Brinicle therefore stores symbolic title evidence explicitly as part of the indexed representation.

## 3.3 Term-frequency packing

Title tokens include a small saturated term-frequency signal. Conceptually, each stored title token combines a token id with a compact frequency component:

```text
packed_title_token = token_id + small_tf_component
```

The frequency component allows repeated title terms to contribute additional evidence without making repetition dominate the score. This is useful for product titles, where repeated words may reflect emphasis, formatting, or seller-side noise rather than true relevance.

The term-frequency signal is intentionally bounded. A repeated token can matter slightly more than a single occurrence, but excessive repetition is saturated by the scorer.

## 3.4 Dense vector attachment

In hybrid mode, Brinicle appends a dense embedding to the lexical representation. The benchmark uses title embeddings for documents and query embeddings for queries.

A document is encoded as:

```text
header
+ title-token representation
+ dense title embedding
```

A query is encoded as:

```text
header
+ query-token representation
+ dense query embedding
```

The vector region is parsed using the `vector_dim` value stored in the header. This allows the same distance function to combine token-based title matching with vector similarity.

The resulting object is still a single HNSW-searchable representation, but the scorer can evaluate its regions separately.

## 3.5 Optional structured fields

Brinicle's general item representation can also encode structured fields:

```text
category
subcategory
attributes
```

Category and subcategory are stored as stable identifiers. Attributes are stored as sorted key/value id pairs:

```text
[
  key_id_1, value_id_1,
  key_id_2, value_id_2,
  ...
]
```

This allows structured evidence to participate in the same distance function as title tokens and dense vectors. For example, a product item may include title evidence, category identity, and attribute matches inside one encoded object.

The experiments in this paper use the title + vector configuration, but the same representation layout supports richer item-search configurations.

## 3.6 Shared representation family

Documents and queries are encoded into the same representation family. This is what allows HNSW traversal to operate over hybrid-search objects directly.

A document may contain:

```text
product-title tokens
+ product-title embedding
```

A query may contain:

```text
query tokens
+ query embedding
```

The distance function compares the two encoded objects by reading their corresponding regions:

```text
title-token agreement
+ optional structured-field agreement
+ vector similarity
```

This shared representation is central to the one-graph design. The graph stores encoded items, and the query enters the graph as a comparable encoded object.

## 3.7 Encoding summary

Brinicle's item/query representation can be summarized as:

```text id="772ssj"
encoded object =
  header
  + lexical title evidence
  + optional structured evidence
  + optional dense vector
```

The representation is numeric, but not unstructured. Its internal layout allows the distance function to combine symbolic and semantic evidence during graph traversal.

The next section defines the distance function used to compare these encoded objects.

---

Here is the rewritten **Section 4: Distance Function**. I kept the technical core, removed defensive explanation, and made the alpha/build-time behavior neutral and method-oriented.

# 4. Distance Function

Brinicle's encoded representation becomes searchable through a custom distance function. The distance function reads the structured regions of the encoded query and document, computes component-wise distances, and combines them into a single value used by HNSW during graph construction and search.

For the general item-search representation, the distance has the form:

```text
D(q, d) =
w_title       · D_title(q, d)
+ w_attr      · D_attr(q, d)
+ w_category  · D_category(q, d)
+ w_subcat    · D_subcat(q, d)
+ w_vector    · D_vector(q, d)
```

where `q` is the encoded query, `d` is the encoded document, and each component measures one region of the representation.

The benchmarked hybrid configuration uses the title and vector components:

```text
D(q, d) =
w_title  · D_title(q, d)
+ w_vector · D_vector(q, d)
```

Structured-field components are part of the broader scorer, but the main experiments isolate title-based hybrid retrieval.

## 4.1 Title distance

The title component measures symbolic agreement between query tokens and document-title tokens. Product queries are usually shorter than product titles, so the title scorer uses an asymmetric overlap measure.

Brinicle uses a Tversky-style similarity:

```text
S_title(q, d) =
matched / (matched + α_title · only_query + β_title · extra_document)
```

The corresponding distance is:

```text
D_title(q, d) = 1 - S_title(q, d)
```

Here:

```text
matched        = weighted title-token matches
only_query     = query tokens missing from the document title
extra_document = document-title tokens not present in the query
```

The parameters `α_title` and `β_title` control the relative cost of missing query tokens and extra document tokens.

This is useful for product retrieval because a relevant product title may contain all query terms plus additional descriptive text. For example, query "iphone 15 256gb", and document title "Apple iPhone 15 256GB Blue Unlocked Smartphone 2023".
The extra document terms provide context, but missing query terms usually represent a stronger mismatch. The asymmetric title distance reflects this behavior.

## 4.2 Term-frequency saturation

Title-token matches use the packed term-frequency signal described in Section 3. Repeated terms are passed through a saturation function before contributing to the title score:

```text
tf_sat(tf) =
(tf · (k1 + 1)) / (tf + k1)
```

The saturation limits the effect of repeated title terms. A repeated token can increase the contribution of a match, but repeated words do not scale linearly without bound.

This gives the title component a controlled lexical signal. Token match is a positive evidence, Repeated token is slightly stronger evidence, and excess repetition is saturated contribution.

## 4.3 Build-time and search-time title configuration

Brinicle can use different title-distance settings during graph construction and query-time search.

During graph construction, the default title configuration is symmetric:

| Phase | `α_title` | `β_title` | Behavior          |
| ----- | --------: | --------: | ----------------- |
| Build |       1.0 |       1.0 | Symmetric overlap |

During search, the default title configuration is query-oriented:

| Phase  | `α_title` | `β_title` | Behavior                                  |
| ------ | --------: | --------: | ----------------------------------------- |
| Search |       1.0 |      0.06 | Stronger penalty for missing query tokens |

The build-time configuration shapes graph neighborhoods using balanced title overlap. The search-time configuration gives more weight to query coverage, which is appropriate for short product queries matched against longer product titles.

## 4.4 Vector distance

The vector component measures semantic similarity between the query embedding and the document embedding. Brinicle uses scaled cosine distance:

```text
D_vector(q, d) = 0.5 · (1 - cos(q, d))
```

The scaling maps cosine distance into a range compatible with the lexical distance components. Since cosine similarity lies in `[-1, 1]`, the unscaled expression `1 - cos(q, d)` lies in `[0, 2]`; multiplying by `0.5` maps it to `[0, 1]`.

When vectors are normalized, cosine similarity can be computed through a dot product. The distance function can also use the general cosine path when normalization is not assumed.

## 4.5 Structured-field distances

The general Brinicle scorer can also compare structured fields. Category and subcategory are treated as identifier matches. Attribute fields are treated as sorted key/value pairs.

For category-like identifiers, the distance is direct:

```text
D_id(a, b) =
0                if a or b is unknown
0                if a = b
field_penalty    otherwise
```

For attributes, the scorer compares matching keys and evaluates whether their values agree:

```text
same key + same value       → no penalty
same key + different value  → mismatch penalty
missing field information   → neutral or soft contribution
```

These structured components allow category, subcategory, and attribute evidence to participate in the same distance function as title tokens and dense vectors. In the experiments reported in this paper, the active retrieval configuration uses title and vector evidence.

## 4.6 Hybrid weighting

The general distance function is controlled through component weights. In title-based hybrid retrieval, the active weights are `w_title` and `w_vector`.

A lexical configuration sets the vector contribution to zero:

```text
w_title > 0
w_vector = 0
```

A vector configuration sets the title contribution to zero:

```text
w_title = 0
w_vector > 0
```

A hybrid configuration activates both:

```text
w_title > 0
w_vector > 0
```

This makes retrieval behavior a property of the distance configuration. The same encoded representation can be searched with different component weights depending on the desired retrieval mode.

## 4.7 Brinicle alpha

Brinicle uses an alpha parameter to control the balance between semantic distance and lexical correction.

For `0 < p < 1`, alpha `p` is converted into:

```text
w_vector = 1
w_lexical = (1 - p) / p
```

The vector component keeps full weight, while the lexical components are scaled by `w_lexical`.

At the boundaries:

```text
p = 1 → vector retrieval
p = 0 → lexical retrieval
```

For example, when `p = 0.90`:

```text
w_lexical = (1 - 0.90) / 0.90 = 0.1111
```

If the base title weight is `0.45`, the effective title weight becomes:

```text
0.45 · 0.1111 = 0.0500
```

while the vector weight remains:

```text
w_vector = 1.0
```

This parameterization treats dense-vector distance as the primary semantic geometry and uses lexical evidence as a correction term. Lower alpha values increase the strength of lexical correction. Higher alpha values make retrieval more vector-oriented.

## 4.8 Alpha and graph construction

In Brinicle, the distance function is used during graph construction as well as query-time search. Therefore, the selected hybrid configuration affects both neighborhood formation and query traversal.

The build process uses the configured distance function to decide how items connect inside the HNSW graph. A more lexical configuration creates neighborhoods influenced more strongly by title-token overlap. A more semantic configuration creates neighborhoods influenced more strongly by vector similarity.

The same principle applies during search: the query traverses the graph using the configured distance function, and candidates are ranked according to the resulting distances.

This makes alpha part of the index configuration. In the benchmark, Brinicle indexes are built with the selected alpha value for each dataset.

## 4.9 Distance-function summary

Brinicle's distance function combines interpretable regions of the encoded representation:

```text
title tokens        → Tversky-style symbolic distance
dense vector        → scaled cosine distance
structured fields   → identifier and key/value penalties
```

For title-based hybrid retrieval, the main distance is:

```text
D(q, d) =
w_title  · D_title(q, d)
+ w_vector · D_vector(q, d)
```

This distance is used by HNSW for graph construction and search, making hybrid behavior part of candidate exploration rather than a separate fusion stage.


```text
encoded item:
  title tokens
  + optional structured fields
  + optional dense vector

distance function:
  title Tversky distance
  + optional structured penalties
  + scaled cosine vector distance

retrieval behavior:
  lexical, vector, or hybrid depending on weights
```

The next section describes how this method is evaluated experimentally against existing hybrid-search systems.

---

# 5. Experimental Setup

The experiments evaluate title-based hybrid product retrieval. Each engine receives a product query and returns a ranked list of product identifiers from a fixed corpus. Documents are indexed using product titles and precomputed dense title embeddings. Queries are represented using query text and precomputed dense query embeddings.

The benchmark compares Brinicle with four existing search systems under the same host environment, container resource limits, embedding model, indexed field, and top-k retrieval setting.

## 5.1 Retrieval task

For each query, the engine receives:

```text
query text
+ query embedding
```

The corpus contains documents represented as:

```text
product title
+ product title embedding
```

Each engine returns the top `K` product identifiers. The returned identifiers are compared against the relevance judgments provided by the dataset.

All experiments use:

```text
top_k = 100
```

Metrics are reported at:

```text
K = 1, 5, 10, 20, 50, 100
```
# 5.2 Datasets

The benchmark uses two product-search datasets: WANDS and Amazon ESCI.

| Dataset                | Documents | Queries | Tuning queries | Evaluation queries | Indexed field |
| ---------------------- | --------: | ------: | -------------: | -----------------: | ------------- |
| WANDS                  |    42,994 |     450 |             30 |                420 | Title         |
| Amazon ESCI, US locale | 1,215,854 |  20,458 |          2,000 |             18,458 | Title         |

The tuning queries are used to select each engine's hybrid parameter. They are excluded from final evaluation.

Both datasets are evaluated using exact-match relevance only. For Amazon ESCI, only products labeled `E` are treated as relevant. `S`, `C`, and `I` labels are treated as non-relevant. The same binary relevance protocol is used for WANDS, where only exact relevance judgments are counted as relevant.

This produces a stricter retrieval setting: a result must match the exact target product intent to be considered relevant.

## 5.3 Compared systems

The benchmark compares five systems:

| System      | Retrieval configuration         |
| ----------- | ------------------------------- |
| Brinicle    | Single-graph hybrid retrieval   |
| Weaviate    | Hybrid BM25/vector retrieval    |
| Meilisearch | Hybrid keyword/vector retrieval |
| Typesense   | Hybrid keyword/vector retrieval |
| OpenSearch  | Hybrid BM25/vector retrieval    |

All systems index the same product title field and use the same precomputed dense embeddings.

Brinicle is evaluated through its server adapter. This makes the measurement closer to a service-style deployment.

## 5.4 Embedding model

Dense embeddings are generated using:

```text
nomic-ai/nomic-embed-text-v1.5
```

Document embeddings use the prefix:

```text
search_document: {title}
```

Query embeddings use the prefix:

```text
search_query: {query}
```

Embeddings are computed before the benchmark runs. Search latency measurements therefore cover retrieval-engine behavior and do not include embedding generation.

## 5.5 Indexed fields

All engines index the product title as the lexical search field:

```text
title
```

For hybrid retrieval, each document also contains a dense vector field holding the precomputed title embedding.

The Brinicle configuration used in the benchmark activates title-token evidence and dense-vector evidence. Structured fields such as category, subcategory, and attributes are part of Brinicle's general item representation, but they are not active in this benchmark configuration.

## 5.6 Runtime environment

All benchmark runs are performed on the same host machine:

| Component             | Value                 |
| --------------------- | --------------------- |
| Host OS               | Ubuntu 25.10          |
| CPU                   | Intel Core i7-13650HX |
| Host RAM              | 32 GiB                |
| Storage               | NVMe SSD              |
| Docker version        | 29.2.1                |
| Docker storage driver | overlay2              |

Each engine runs inside Docker with the same resource limits:

| Resource  |  Limit |
| --------- | -----: |
| CPU cores |     16 |
| RAM       | 16 GiB |

Only one engine container is active during each benchmark run.

## 5.7 Retrieval parameters

Where supported, HNSW-related parameters are configured as follows:

| Parameter         | Value |
| ----------------- | ----: |
| `M`               |     8 |
| `ef_construction` |   512 |
| `ef_search`       |  1024 |
| `top_k`           |   100 |

For Brinicle, the lexical representation uses:

| Parameter         | Value |
| ----------------- | ----: |
| Lexical dimension |    70 |

Lexical dimension specifies how many slots do we have for storage. The more space, the less title truncation, the more memory usage.

These parameters define the benchmark configuration used for the reported experiments.

## 5.8 Hybrid parameter tuning

Each system exposes its own parameter for controlling the lexical-semantic balance. The parameters are tuned separately for each engine and dataset using the held-out tuning queries.

The selected values are:

| Dataset | Brinicle | Meilisearch | OpenSearch | Typesense | Weaviate |
| ------- | -------: | ----------: | ---------: | --------: | -------: |
| WANDS   |     0.95 |        0.55 |       0.60 |      0.80 |     0.70 |
| ESCI    |     0.90 |        0.40 |       0.40 |      0.20 |     0.50 |

For Brinicle, the selected alpha is part of the index configuration because the distance function is used during graph construction. Brinicle indexes are therefore built with the tuned alpha value for each dataset.

## 5.9 Benchmark procedure

Each benchmark run has two phases. First, the engine builds or ingests the index. During this phase, the benchmark records build time and build memory. Second, the benchmark runs the evaluation queries. During this phase, the benchmark records returned product identifiers, per-query latency, throughput, and search memory.
The measured search outputs include:

```text
ranked product ids
per-query latency
total query time
container memory profile
```

## 5.10 Memory measurement

Memory is measured separately for build and search. The benchmark records multiple memory counters, including:

```
raw peak memory
working-set peak memory
anonymous memory
file-backed memory
kernel memory
slab memory
```

The main results report peak search memory. Additional memory counters are included in the appendix.

## 5.11 Evaluation metrics

The benchmark reports ranking metrics at `K = 1, 5, 10, 20, 50, 100`.

The relevance metrics are:

| Metric     | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| `Hit@K`    | Whether at least one relevant product appears in the top `K` |
| `Recall@K` | Fraction of relevant products retrieved in the top `K`       |
| `nDCG@K`   | Graded ranking quality in the top `K`                        |
| `MRR@K`    | Reciprocal rank of the first relevant product                |

The system metrics are:

| Metric         | Description                                |
| -------------- | ------------------------------------------ |
| Build time     | Time required to build or ingest the index |
| Search latency | Per-query retrieval latency                |
| QPS            | Queries processed per second               |
| Build memory   | Peak memory during index construction      |
| Search memory  | Peak memory during query execution         |

The main results focus on ranking quality, P99 latency, and peak search memory. Full metric tables are reported in the appendix.


---


# 6. Results

This section reports the main retrieval and system results on WANDS and US-filtered Amazon ESCI. The main text focuses on exact-relevance retrieval quality, P99 search latency, and peak search memory. Full metric tables, throughput measurements, build-time measurements, and additional memory counters are reported in the appendix.

## 6.1 WANDS results

Table 1 reports the main WANDS results under the title-based hybrid retrieval configuration.

| Engine      |      Hit@1 |    nDCG@10 |    Hit@100 |  P99 latency | Peak search memory |
| ----------- | ---------: | ---------: | ---------: | -----------: | -----------------: |
| Brinicle    |     0.4844 |     0.5851 |     0.7444 | **0.516 ms** |         **129 MB** |
| Meilisearch |     0.4844 |     0.5724 |     0.7311 |     7.433 ms |             239 MB |
| OpenSearch  | **0.4956** | **0.5855** | **0.7467** |     1.480 ms |           9,552 MB |
| Typesense   |     0.4844 |     0.5779 |     0.7311 |     7.574 ms |           1,016 MB |
| Weaviate    |     0.4622 |     0.5631 |     0.7333 |    10.758 ms |             597 MB |

**Table 1. WANDS main results.** Relevance is evaluated using exact-match labels. Latency is reported as per-query P99 latency. Memory is reported as peak search memory.

On WANDS, OpenSearch has the highest `Hit@1`, `nDCG@10`, and `Hit@100`. Brinicle is close on all three relevance metrics, with the lowest P99 latency and the lowest peak search memory among the compared systems.

The WANDS results show a narrow relevance spread among the strongest systems. OpenSearch reaches `0.4956` Hit@1, while Brinicle, Meilisearch, and Typesense each reach `0.4844`. At `Hit@100`, OpenSearch reaches `0.7467`, while Brinicle reaches `0.7444`.

The system measurements show a larger separation. Brinicle records `0.516 ms` P99 latency and `129 MB` peak search memory. The closest non-Brinicle P99 latency is OpenSearch at `1.480 ms`, while the closest non-Brinicle search memory is Meilisearch at `239 MB`.

## 6.2 ESCI results

Table 2 reports the main results on US-filtered Amazon ESCI.

| Engine      |      Hit@1 |    nDCG@10 |    Hit@100 |  P99 latency | Peak search memory |
| ----------- | ---------: | ---------: | ---------: | -----------: | -----------------: |
| Brinicle    | **0.4280** | **0.3661** |     0.8932 | **0.773 ms** |       **1,731 MB** |
| Meilisearch |     0.4175 |     0.3566 |     0.8862 |    19.768 ms |           5,671 MB |
| OpenSearch  |     0.4226 |     0.3601 |     0.9009 |     3.407 ms |          11,716 MB |
| Typesense   |     0.4191 |     0.3525 |     0.8793 |    12.160 ms |           8,041 MB |
| Weaviate    |     0.4203 |     0.3588 | **0.9054** |     9.483 ms |           4,794 MB |

**Table 2. ESCI main results.** Relevance is evaluated using exact labels only. Latency is reported as per-query P99 latency. Memory is reported as peak search memory.

On ESCI, Brinicle has the highest `Hit@1` and `nDCG@10`. Weaviate has the highest `Hit@100`, followed by OpenSearch. This indicates a difference between early exact-match ranking and deeper top-k retrieval.

Brinicle records `0.4280` Hit@1 and `0.3661` nDCG@10. The strongest non-Brinicle Hit@1 is OpenSearch at `0.4226`, and the strongest non-Brinicle nDCG@10 is also OpenSearch at `0.3601`. At `Hit@100`, Weaviate reaches `0.9054`, OpenSearch reaches `0.9009`, and Brinicle reaches `0.8932`.

The system measurements again show the largest differences in latency and memory. Brinicle records `0.773 ms` P99 latency and `1,731 MB` peak search memory. The closest non-Brinicle P99 latency is OpenSearch at `3.407 ms`. The closest non-Brinicle peak search memory is Weaviate at `4,794 MB`.

## 6.3 P99 search latency

  
  ![Figure 1](https://github.com/bicardinal/item_search_bench/blob/main/benchmark_figures/wands_p99_latency_bw.png?raw=true)  
  *Figure 1. P99 search latency on WANDS.*
  
![Figure 2](https://github.com/bicardinal/item_search_bench/blob/main/benchmark_figures/esci_p99_latency_bw.png?raw=true)  
*Figure 2. P99 search latency on ESCI.*
  

Figures 1 and 2 compare P99 search latency across engines on both datasets.

| Dataset |     Brinicle | Meilisearch | OpenSearch | Typesense |  Weaviate |
| ------- | -----------: | ----------: | ---------: | --------: | --------: |
| WANDS   | **0.516 ms** |    7.433 ms |   1.480 ms |  7.574 ms | 10.758 ms |
| ESCI    | **0.773 ms** |   19.768 ms |   3.407 ms | 12.160 ms |  9.483 ms |

**Table 3. P99 search latency.**

Brinicle has the lowest measured P99 latency on both datasets. On WANDS, its P99 latency is `0.516 ms`, compared with `1.480 ms` for OpenSearch, the closest non-Brinicle system. On ESCI, its P99 latency is `0.773 ms`, compared with `3.407 ms` for OpenSearch.

## 6.4 Search memory

![Figure 3](https://github.com/bicardinal/item_search_bench/blob/main/benchmark_figures/wands_search_memory_bw.png?raw=true)  
*Figure 3. Peak search memory on WANDS.*

![Figure 4](https://github.com/bicardinal/item_search_bench/blob/main/benchmark_figures/esci_search_memory_bw.png?raw=true)  
*Figure 4. Peak search memory on ESCI.*
  
Table 4 reports peak search memory for both datasets.

| Dataset |     Brinicle | Meilisearch | OpenSearch | Typesense | Weaviate |
| ------- | -----------: | ----------: | ---------: | --------: | -------: |
| WANDS   |   **129 MB** |      239 MB |   9,552 MB |  1,016 MB |   597 MB |
| ESCI    | **1,731 MB** |    5,671 MB |  11,716 MB |  8,041 MB | 4,794 MB |

**Table 4. Peak search memory.**

Brinicle has the lowest measured search memory on both datasets. On WANDS, Brinicle uses `129 MB`, followed by Meilisearch at `239 MB`. On ESCI, Brinicle uses `1,731 MB`, followed by Weaviate at `4,794 MB`.

The memory difference is larger on ESCI, where the corpus is substantially larger. In that setting, Brinicle's peak search memory is less than half of the closest non-Brinicle measurement.

## 6.5 Hit@K curves

![Figure 5](https://github.com/bicardinal/item_search_bench/blob/main/benchmark_figures/wands_hit_curve_bw.png?raw=true)  
*Figure 5. Hit@K curve on WANDS using exact relevance.*

![Figure 6](https://github.com/bicardinal/item_search_bench/blob/main/benchmark_figures/esci_hit_curve_bw.png?raw=true)  
*Figure 6. Hit@K curve on ESCI using exact relevance.*  
  

  
Figures 5 and 6 report Hit@K curves across `K = 1, 5, 10, 20, 50, 100`.

On WANDS, OpenSearch is slightly ahead across the main reported relevance points, while Brinicle remains close. On ESCI, Brinicle leads at early ranking points reported in Table 2, while Weaviate and OpenSearch reach higher `Hit@100`.

The full Hit@K, Recall@K, nDCG@K, and MRR@K tables are provided in the appendix.

## 6.6 Result summary

Across both datasets, the results show three main patterns.

First, relevance is competitive across systems. On WANDS, OpenSearch has the strongest exact-relevance metrics among the reported values. On ESCI, Brinicle has the strongest `Hit@1` and `nDCG@10`, while Weaviate has the strongest `Hit@100`.

Second, Brinicle has the lowest measured P99 search latency on both datasets.

Third, Brinicle has the lowest measured peak search memory on both datasets.

These results support the single-graph formulation as a practical retrieval design for title-based hybrid product search: lexical and semantic evidence can be combined during graph traversal while maintaining competitive exact-relevance quality and a smaller search-time resource footprint.

---

# 7. Discussion

The results show that hybrid product retrieval can be implemented through a single HNSW graph while preserving competitive exact-relevance quality. Brinicle's main distinction is not a single isolated relevance score, but the combination of retrieval quality, low search memory, and low search latency under the tested configuration.

This section discusses the implications of the benchmark results for hybrid retrieval design, product-title search, and deployment trade-offs.

## 7.1 Interpreting the retrieval trade-off

The relevance results differ across datasets and ranking depths.

On WANDS, OpenSearch has the strongest reported exact-relevance metrics. Brinicle remains close across the main relevance points, with a small difference in `Hit@1`, `nDCG@10`, and `Hit@100`.

On ESCI, Brinicle has the strongest `Hit@1` and `nDCG@10`, while Weaviate has the strongest `Hit@100`. This indicates that Brinicle performs strongly in early ranking, while other systems retrieve more exact matches at deeper top-k positions.

This pattern is useful because it separates two retrieval behaviors:

```text
early ranking quality
deep candidate coverage
```

For product search, both behaviors can matter. Early ranking is important when results are shown directly to users. Deeper candidate coverage is important when the retrieval stage feeds reranking, recommendation, or downstream selection.

The benchmark results therefore describe an operating profile rather than a single leaderboard. Brinicle's profile is strongest in search-time efficiency and early exact-match ranking on the larger ESCI benchmark, while other systems show advantages in specific relevance metrics and deeper retrieval settings.

## 7.2 Hybrid retrieval inside graph traversal

The central architectural result is that lexical and semantic evidence can participate in the same graph traversal.

In a conventional hybrid system, lexical and vector retrieval are usually performed through separate structures, and hybrid behavior is introduced through score fusion or reranking. Brinicle moves this combination into the distance function used by HNSW.

This has two consequences. First, hybrid scoring affects candidate exploration, not only final ranking. The graph traversal is guided by a distance function that includes both symbolic title evidence and dense-vector similarity. Second, the retrieval system has a smaller structural surface. The benchmarked Brinicle configuration uses one encoded representation, one HNSW graph, and one hybrid-aware distance function for title-based hybrid retrieval.

The results suggest that this design is sufficient to produce competitive retrieval behavior on the evaluated product-search tasks.

## 7.3 Early ranking and deeper top-k behavior

The ESCI results show a clear distinction between early ranking and deeper top-k retrieval. Brinicle leads the reported early-ranking metrics:

```text
Hit@1
nDCG@10
```

Weaviate leads the reported deeper metric:

```text
Hit@100
```

This distinction is important for interpreting hybrid retrieval systems. A method can be strong at placing an exact result near the top while another method can be stronger at retrieving more exact results somewhere inside a larger candidate set. The appropriate retrieval profile depends on the application. A direct product-search interface benefits from strong early ranking. A multi-stage ranking system may prefer broader top-k coverage before reranking. In this benchmark, Brinicle's strongest relevance behavior appears in early exact-match ranking on ESCI, while its strongest system behavior appears consistently in latency and memory across both datasets.

## 7.4 Search-time resource profile

The memory and latency measurements show the clearest separation between Brinicle and the compared systems.

Brinicle has the lowest measured peak search memory on both WANDS and ESCI. The difference is especially visible on ESCI, where Brinicle uses less than half the search memory of the closest non-Brinicle system. Brinicle also has the lowest measured P99 latency on both datasets. This result is consistent across the smaller WANDS corpus and the larger ESCI corpus. Together, these measurements show that the single-graph design changes the search-time resource profile of hybrid retrieval. The system does not maintain separate lexical and vector retrieval structures for the benchmarked hybrid task, and the measured search memory reflects that architectural choice.

## 7.5 Alpha as index configuration

Brinicle's alpha affects graph construction as well as query-time search. This makes the hybrid parameter part of the index configuration rather than only a runtime fusion parameter. When the graph is built, the configured distance function influences neighborhood formation. A more semantic configuration creates graph neighborhoods shaped more strongly by vector similarity. A stronger lexical correction changes how symbolic title evidence contributes to those neighborhoods. This is different from hybrid systems where the lexical and vector indexes are built independently and the hybrid parameter only affects query-time score combination. In the benchmark, each Brinicle index is built using the tuned alpha selected for that dataset. This means the reported Brinicle results reflect both the encoded representation and the graph topology produced by the selected hybrid distance.

## 7.6 Deployment implications

The measured trade-off is relevant for deployments where search memory and latency are important constraints.

A lower search-time memory footprint can reduce infrastructure cost, allow more indexes to run on the same machine, or leave more memory available for application logic. Lower latency can improve interactive search behavior and increase the headroom available for additional downstream processing. Brinicle's design is therefore most directly relevant to search systems where hybrid retrieval is needed but maintaining multiple retrieval structures is expensive. The benchmarked setting is title-based product retrieval, but the architectural pattern is broader: encode multiple retrieval signals into one comparable object, then use a distance function that combines those signals during graph traversal.

## 7.7 Discussion summary

The results support three main observations. First, title-based hybrid product retrieval can be expressed through one HNSW graph and one hybrid-aware distance function. Second, Brinicle achieves competitive exact-relevance quality on both evaluated datasets, with stronger early-ranking results on ESCI and close relevance results on WANDS. Third, Brinicle shows a consistent search-time resource advantage in the reported measurements, with the lowest P99 latency and lowest peak search memory on both datasets. These observations support the paper's main claim: hybrid product retrieval can be modeled as a single-graph retrieval problem, with lexical and semantic evidence combined during graph traversal rather than through post-hoc fusion over separate retrieval structures.

---
# 8. Limitations

This benchmark evaluates title-based hybrid product retrieval using precomputed embeddings and exact-match relevance labels. It does not measure multi-field ranking, faceted filtering, personalized retrieval, distributed deployment, or reranking pipelines. The results should therefore be interpreted as evidence for the tested title + vector retrieval setting, not as a complete evaluation of every product-search workload. The compared systems were tuned through held-out queries under a shared benchmark configuration, but each engine has additional parameters and deployment modes that may change its behavior. Brinicle's alpha also affects graph construction, so changing the hybrid balance requires rebuilding the index. Future experiments should evaluate richer metadata, structured filters, additional datasets, and multi-stage retrieval pipelines.

---

# 9. Conclusion

This paper studied a single-graph formulation for hybrid product retrieval. Instead of combining separate lexical and vector retrieval results through post-hoc fusion, Brinicle encodes title-token evidence and dense embeddings into one HNSW-searchable representation. A custom distance function then combines symbolic and semantic evidence during graph construction and search.

The experiments on WANDS and US-filtered Amazon ESCI show that this approach achieves competitive exact-relevance quality under the tested title + vector configuration. Brinicle has the lowest measured P99 latency and peak search memory on both datasets, while relevance leadership varies by dataset and metric.

The main result is architectural: hybrid product retrieval can be modeled as graph traversal over a structured representation, rather than as coordination between separate retrieval structures. For workloads where exact product identifiers and semantic tolerance both matter, this opens a practical design space for lower-memory and lower-latency hybrid search.

Future work should evaluate the same approach with richer product metadata, structured filters, additional datasets, different embedding models, and multi-stage reranking pipelines.

---
Here is a paper-ready appendix draft. I kept it factual and compact, with repository links, dataset citations, configuration, and full result tables. Numeric tables are derived from your raw result dump. 

# Appendix A. Code, Data, and Citations

The implementation and benchmark materials are available at the following repositories:

| Resource            | Repository                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------ |
| Brinicle            | [github.com/bicardinal/brinicle](https://github.com/bicardinal/brinicle)                   |
| Benchmark harness   | [github.com/bicardinal/item_search_bench](https://github.com/bicardinal/item_search_bench) |
| Amazon ESCI dataset | [github.com/amazon-science/esci-data](https://github.com/amazon-science/esci-data/)        |
| WANDS dataset       | [github.com/wayfair/WANDS](https://github.com/wayfair/WANDS)                               |

### WANDS

```bibtex
@InProceedings{wands,
  title = {WANDS: Dataset for Product Search Relevance Assessment},
  author = {Chen, Yan and Liu, Shujian and Liu, Zheng and Sun, Weiyi and Baltrunas, Linas and Schroeder, Benjamin},
  booktitle = {Proceedings of the 44th European Conference on Information Retrieval},
  year = {2022},
  numpages = {12}
}
```

### Amazon ESCI

```bibtex
@article{reddy2022shopping,
  title = {Shopping Queries Dataset: A Large-Scale {ESCI} Benchmark for Improving Product Search},
  author = {Chandan K. Reddy and Lluís Màrquez and Fran Valero and Nikhil Rao and Hugo Zaragoza and Sambaran Bandyopadhyay and Arnab Biswas and Anlu Xing and Karthik Subbian},
  year = {2022},
  eprint = {2206.06588},
  archivePrefix = {arXiv}
}
```

# Appendix B. Benchmark Configuration

## B.1 Dataset configuration

| Dataset                | Documents | Queries | Tuning queries | Evaluation queries | Indexed field |
| ---------------------- | --------: | ------: | -------------: | -----------------: | ------------- |
| WANDS                  |    42,994 |     450 |             30 |                420 | Title         |
| Amazon ESCI, US locale | 1,215,854 |  20,458 |          2,000 |             18,458 | Title         |

Both datasets are evaluated using exact relevance only. For ESCI, only `E` labels are treated as relevant; `S`, `C`, and `I` labels are treated as non-relevant. The same binary exact-relevance protocol is used for WANDS.

## B.2 Shared retrieval configuration

| Parameter             |                 Value |
| --------------------- | --------------------: |
| Indexed lexical field |                 Title |
| Retrieval mode        | Hybrid title + vector |
| `top_k`               |                   100 |
| Reported K values     | 1, 5, 10, 20, 50, 100 |

## B.3 Embedding configuration

| Parameter        | Value                            |
| ---------------- | -------------------------------- |
| Embedding model  | `nomic-ai/nomic-embed-text-v1.5` |
| Document prefix  | `search_document: {title}`       |
| Query prefix     | `search_query: {query}`          |
| Embedding timing | Precomputed before benchmark     |

Search latency does not include embedding generation.

## B.4 Runtime environment

| Component             | Value                 |
| --------------------- | --------------------- |
| Host OS               | Ubuntu 25.10          |
| CPU                   | Intel Core i7-13650HX |
| Host RAM              | 32 GiB                |
| Storage               | NVMe SSD              |
| Docker version        | 29.2.1                |
| Docker storage driver | overlay2              |

Each engine is run inside Docker with the same resource limits:

| Resource  |  Limit |
| --------- | -----: |
| CPU cores |     16 |
| RAM       | 16 GiB |

Only one engine container is active during each benchmark run.

## B.5 HNSW and Brinicle configuration

Where supported, HNSW-related parameters use the following configuration:

| Parameter         | Value |
| ----------------- | ----: |
| `M`               |     8 |
| `ef_construction` |   512 |
| `ef_search`       |  1024 |
| `top_k`           |   100 |

For Brinicle:

| Parameter         | Value |
| ----------------- | ----: |
| Lexical dimension |    70 |

# Appendix C. Hybrid Parameter Tuning

Each engine's hybrid parameter is selected using held-out tuning queries and then applied to the evaluation split.

| Dataset | Brinicle | Meilisearch | OpenSearch | Typesense | Weaviate |
| ------- | -------: | ----------: | ---------: | --------: | -------: |
| WANDS   |     0.95 |        0.55 |       0.60 |      0.80 |     0.70 |
| ESCI    |     0.90 |        0.40 |       0.40 |      0.20 |     0.50 |

For Brinicle, the selected alpha is part of the index configuration because the distance function is used during graph construction.

# Appendix D. Full Relevance Metrics

All relevance metrics are computed using exact relevance only.

## D.1 WANDS Hit@K

| Engine      |     @1 |     @5 |    @10 |    @20 |    @50 |   @100 |
| ----------- | -----: | -----: | -----: | -----: | -----: | -----: |
| Brinicle    | 0.4844 | 0.5911 | 0.6356 | 0.6911 | 0.7222 | 0.7444 |
| Meilisearch | 0.4844 | 0.5844 | 0.6267 | 0.6778 | 0.7133 | 0.7311 |
| OpenSearch  | 0.4956 | 0.6022 | 0.6467 | 0.6867 | 0.7333 | 0.7467 |
| Typesense   | 0.4844 | 0.5956 | 0.6333 | 0.6822 | 0.7156 | 0.7311 |
| Weaviate    | 0.4622 | 0.5778 | 0.6356 | 0.6733 | 0.7089 | 0.7333 |

## D.2 WANDS Recall@K

| Engine      |     @1 |     @5 |    @10 |    @20 |    @50 |   @100 |
| ----------- | -----: | -----: | -----: | -----: | -----: | -----: |
| Brinicle    | 0.1238 | 0.2074 | 0.2637 | 0.3545 | 0.4949 | 0.6122 |
| Meilisearch | 0.1250 | 0.2054 | 0.2576 | 0.3395 | 0.4630 | 0.5730 |
| OpenSearch  | 0.1271 | 0.2112 | 0.2701 | 0.3549 | 0.4870 | 0.6008 |
| Typesense   | 0.1223 | 0.2107 | 0.2653 | 0.3488 | 0.4720 | 0.5803 |
| Weaviate    | 0.1198 | 0.1990 | 0.2537 | 0.3355 | 0.4557 | 0.5518 |

## D.3 WANDS nDCG@K

| Engine      |     @1 |     @5 |    @10 |    @20 |    @50 |   @100 |
| ----------- | -----: | -----: | -----: | -----: | -----: | -----: |
| Brinicle    | 0.6124 | 0.5910 | 0.5851 | 0.5827 | 0.5804 | 0.5937 |
| Meilisearch | 0.6124 | 0.5812 | 0.5724 | 0.5659 | 0.5560 | 0.5642 |
| OpenSearch  | 0.6264 | 0.5925 | 0.5855 | 0.5799 | 0.5731 | 0.5838 |
| Typesense   | 0.6124 | 0.5869 | 0.5779 | 0.5724 | 0.5620 | 0.5707 |
| Weaviate    | 0.5843 | 0.5685 | 0.5631 | 0.5569 | 0.5468 | 0.5472 |

## D.4 WANDS MRR@K

| Engine      |     @1 |     @5 |    @10 |    @20 |    @50 |   @100 |
| ----------- | -----: | -----: | -----: | -----: | -----: | -----: |
| Brinicle    | 0.4844 | 0.5249 | 0.5308 | 0.5345 | 0.5354 | 0.5357 |
| Meilisearch | 0.4844 | 0.5244 | 0.5304 | 0.5340 | 0.5352 | 0.5355 |
| OpenSearch  | 0.4956 | 0.5349 | 0.5411 | 0.5440 | 0.5456 | 0.5457 |
| Typesense   | 0.4844 | 0.5262 | 0.5313 | 0.5348 | 0.5359 | 0.5362 |
| Weaviate    | 0.4622 | 0.5094 | 0.5172 | 0.5199 | 0.5211 | 0.5215 |

## D.5 ESCI Hit@K

| Engine      |     @1 |     @5 |    @10 |    @20 |    @50 |   @100 |
| ----------- | -----: | -----: | -----: | -----: | -----: | -----: |
| Brinicle    | 0.4280 | 0.6631 | 0.7444 | 0.8068 | 0.8634 | 0.8932 |
| Meilisearch | 0.4175 | 0.6438 | 0.7243 | 0.7876 | 0.8506 | 0.8862 |
| OpenSearch  | 0.4226 | 0.6600 | 0.7416 | 0.8046 | 0.8653 | 0.9009 |
| Typesense   | 0.4191 | 0.6493 | 0.7244 | 0.7875 | 0.8475 | 0.8793 |
| Weaviate    | 0.4203 | 0.6652 | 0.7475 | 0.8090 | 0.8727 | 0.9054 |

## D.6 ESCI Recall@K

| Engine      |     @1 |     @5 |    @10 |    @20 |    @50 |   @100 |
| ----------- | -----: | -----: | -----: | -----: | -----: | -----: |
| Brinicle    | 0.0631 | 0.1952 | 0.2859 | 0.3826 | 0.4991 | 0.5789 |
| Meilisearch | 0.0625 | 0.1898 | 0.2774 | 0.3689 | 0.4769 | 0.5518 |
| OpenSearch  | 0.0630 | 0.1930 | 0.2815 | 0.3775 | 0.4917 | 0.5701 |
| Typesense   | 0.0610 | 0.1869 | 0.2718 | 0.3618 | 0.4674 | 0.5398 |
| Weaviate    | 0.0628 | 0.1919 | 0.2809 | 0.3775 | 0.4944 | 0.5760 |

## D.7 ESCI nDCG@K

| Engine      |     @1 |     @5 |    @10 |    @20 |    @50 |   @100 |
| ----------- | -----: | -----: | -----: | -----: | -----: | -----: |
| Brinicle    | 0.4280 | 0.3847 | 0.3661 | 0.3773 | 0.4268 | 0.4585 |
| Meilisearch | 0.4175 | 0.3747 | 0.3566 | 0.3656 | 0.4110 | 0.4406 |
| OpenSearch  | 0.4226 | 0.3784 | 0.3601 | 0.3710 | 0.4190 | 0.4498 |
| Typesense   | 0.4191 | 0.3733 | 0.3525 | 0.3604 | 0.4046 | 0.4332 |
| Weaviate    | 0.4203 | 0.3772 | 0.3588 | 0.3702 | 0.4196 | 0.4516 |

## D.8 ESCI MRR@K

| Engine      |     @1 |     @5 |    @10 |    @20 |    @50 |   @100 |
| ----------- | -----: | -----: | -----: | -----: | -----: | -----: |
| Brinicle    | 0.4280 | 0.5176 | 0.5285 | 0.5329 | 0.5348 | 0.5352 |
| Meilisearch | 0.4175 | 0.5037 | 0.5146 | 0.5190 | 0.5211 | 0.5216 |
| OpenSearch  | 0.4226 | 0.5124 | 0.5234 | 0.5279 | 0.5299 | 0.5304 |
| Typesense   | 0.4191 | 0.5067 | 0.5169 | 0.5213 | 0.5233 | 0.5238 |
| Weaviate    | 0.4203 | 0.5136 | 0.5247 | 0.5291 | 0.5312 | 0.5316 |

# Appendix E. Latency and Throughput

Latency values are reported in milliseconds. Total query time is reported in seconds.

## E.1 WANDS latency and throughput

| Engine      | Avg ms | P50 ms | P95 ms | P99 ms |    QPS | Total query time |
| ----------- | -----: | -----: | -----: | -----: | -----: | ---------------: |
| Brinicle    |  0.427 |  0.428 |  0.516 |  0.516 | 2357.6 |            0.192 |
| Meilisearch |  7.090 |  7.093 |  7.433 |  7.433 |  141.1 |            3.191 |
| OpenSearch  |  1.083 |  1.029 |  1.480 |  1.480 |  926.8 |            0.487 |
| Typesense   |  6.774 |  6.696 |  7.574 |  7.574 |  147.7 |            3.048 |
| Weaviate    |  9.427 |  9.565 | 10.758 | 10.758 |  106.2 |            4.242 |

## E.2 ESCI latency and throughput

| Engine      | Avg ms | P50 ms | P95 ms | P99 ms |    QPS | Total query time |
| ----------- | -----: | -----: | -----: | -----: | -----: | ---------------: |
| Brinicle    |  0.556 |  0.549 |  0.692 |  0.773 | 1800.1 |           11.366 |
| Meilisearch | 15.057 | 15.122 | 17.408 | 19.768 |   66.4 |          308.043 |
| OpenSearch  |  2.704 |  2.671 |  3.053 |  3.407 |  371.0 |           55.321 |
| Typesense   |  9.334 |  9.202 | 10.379 | 12.160 |  107.1 |          190.950 |
| Weaviate    |  8.597 |  8.749 |  9.283 |  9.483 |  116.3 |          175.882 |

# Appendix F. Search Memory

Memory values are reported in MB.

## F.1 WANDS search memory

| Engine      | Raw peak | Working set | Anonymous | File-backed | Kernel | Slab |
| ----------- | -------: | ----------: | --------: | ----------: | -----: | ---: |
| Brinicle    |    129.3 |       128.4 |      79.8 |        41.3 |    4.8 |  3.5 |
| Meilisearch |    238.8 |       238.8 |     124.9 |       107.4 |    4.2 |  1.6 |
| OpenSearch  |   9551.5 |      9544.2 |    9390.0 |       128.9 |   29.6 |  7.3 |
| Typesense   |   1016.4 |      1016.4 |     637.6 |       349.7 |   26.3 |  9.2 |
| Weaviate    |    596.8 |       596.8 |     471.4 |       119.1 |    2.6 |  0.9 |

## F.2 ESCI search memory

| Engine      | Raw peak | Working set | Anonymous | File-backed | Kernel | Slab |
| ----------- | -------: | ----------: | --------: | ----------: | -----: | ---: |
| Brinicle    |   1731.1 |      1714.7 |     480.8 |      1203.7 |   43.7 | 39.0 |
| Meilisearch |   5671.0 |      5671.0 |     955.0 |      4687.0 |   26.3 | 12.1 |
| OpenSearch  |  11716.2 |     11704.5 |   10171.0 |      1505.4 |   35.7 | 10.9 |
| Typesense   |   8040.5 |      8040.3 |    1601.3 |      6391.7 |   44.4 | 24.8 |
| Weaviate    |   4794.0 |      4794.0 |    2326.7 |      2446.1 |   17.3 |  6.6 |

# Appendix G. Build Time and Build Memory

## G.1 Build time

| Dataset | Brinicle | Meilisearch | OpenSearch | Typesense | Weaviate |
| ------- | -------: | ----------: | ---------: | --------: | -------: |
| WANDS   |   11.4 s |      15.4 s |     27.7 s |     8.9 s |    4.8 s |
| ESCI    |  405.8 s |    3136.3 s |    697.9 s |   339.2 s |  227.8 s |

## G.2 WANDS build memory

| Engine      | Raw peak | Working set | Anonymous | File-backed | Kernel | Slab |
| ----------- | -------: | ----------: | --------: | ----------: | -----: | ---: |
| Brinicle    |    160.2 |       126.9 |      82.1 |        76.7 |    5.5 |  4.4 |
| Meilisearch |    849.4 |       849.4 |     733.5 |       111.5 |    5.5 |  1.7 |
| OpenSearch  |   9522.8 |      9515.4 |    9317.3 |       178.5 |   28.2 |  7.0 |
| Typesense   |    897.0 |       897.0 |     312.0 |       574.1 |   25.6 |  9.9 |
| Weaviate    |    535.2 |       535.2 |     410.4 |       117.6 |    2.5 |  0.9 |

## G.3 ESCI build memory

| Engine      | Raw peak | Working set | Anonymous | File-backed | Kernel | Slab |
| ----------- | -------: | ----------: | --------: | ----------: | -----: | ---: |
| Brinicle    |   2600.7 |      1665.9 |     320.4 |      2204.2 |   74.0 | 68.3 |
| Meilisearch |   7059.7 |      6989.1 |    2276.1 |      4795.6 |   31.2 | 15.3 |
| OpenSearch  |  12963.4 |     12953.9 |    9645.5 |      3616.9 |   36.8 | 15.0 |
| Typesense   |   8432.8 |      8432.5 |    1708.3 |      7106.8 |   44.0 | 26.4 |
| Weaviate    |   6228.4 |      6228.4 |    3770.3 |      2720.5 |   16.5 |  7.2 |

# Appendix H. Figure Data

The main figures can be generated from the appendix tables as follows:

| Figure                        | Source table                  |
| ----------------------------- | ----------------------------- |
| P99 latency comparison        | Appendix E                    |
| Peak search memory comparison | Appendix F                    |
| Hit@K curves                  | Appendix D.1 and Appendix D.5 |

# Appendix I. Raw Result Fields

The benchmark output uses the following fields:

| Field                     | Meaning                                                            |
| ------------------------- | ------------------------------------------------------------------ |
| `nDCG@K`                  | Normalized discounted cumulative gain at `K`                       |
| `Recall@K`                | Fraction of exact-relevant products retrieved in the top `K`       |
| `Hit@K`                   | Whether at least one exact-relevant product appears in the top `K` |
| `MRR@K`                   | Reciprocal rank of the first exact-relevant product in the top `K` |
| `search_avg_latency`      | Mean per-query search latency, in seconds                          |
| `search_p50_latency`      | 50th percentile per-query search latency, in seconds               |
| `search_p95_latency`      | 95th percentile per-query search latency, in seconds               |
| `search_p99_latency`      | 99th percentile per-query search latency, in seconds               |
| `qps`                     | Queries processed per second                                       |
| `search_total_query_time` | Total measured search time, in seconds                             |
| `raw_peak_mb`             | Peak raw memory usage, in MB                                       |
| `working_set_peak_mb`     | Peak working-set memory usage, in MB                               |
| `anon_peak_mb`            | Peak anonymous memory usage, in MB                                 |
| `file_peak_mb`            | Peak file-backed memory usage, in MB                               |
| `kernel_peak_mb`          | Peak kernel memory usage, in MB                                    |
| `slab_peak_mb`            | Peak slab memory usage, in MB                                      |
| `build_latency`           | Index build or ingestion time, in seconds                          |
| `build_memory_profile`    | Memory profile recorded during index build or ingestion            |
