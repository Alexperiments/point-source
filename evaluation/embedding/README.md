# Design Doc: Selection of Open-Source Embedding Model for Math-Heavy Scientific Text

**Candidate Models Evaluated**

* **Qwen/Qwen3-Embedding-0.6B** (Qwen2TokenizerFast)
* **google/embeddinggemma-300m** (GemmaTokenizerFast)
* **BAAI/bge-m3** (XLMRobertaTokenizerFast)
* **intfloat/e5-large-v2** (BertTokenizerFast)
* **Alibaba-NLP/gte-large-en-v1.5** (BertTokenizerFast)

---

## Tokenization

### Context and Requirements

The target corpus consists of scientific and technical documents with:

* Inline and display LaTeX (`\omega`, `\epsilon`, `\mathrm{}`, subscripts/superscripts),
* Unicode math symbols (`ϵ, ∑, ≤, √, ∞, μ, ω, π`),
* Mixed prose and equations,
* A requirement for robust semantic retrieval across equivalent representations (Unicode vs TeX vs ASCII).

For embeddings, **tokenization quality directly affects semantic consistency**. In particular:

* Tokenizers that emit `[UNK]` for math symbols are unacceptable.
* Byte-level or fragmented tokenization is acceptable only if it is internally coherent and does not cause large semantic drift in embeddings.
* Ideally, equivalent mathematical representations should yield similar embeddings.

---

### Evaluation

#### Hard Exclusion Criterion

**Any tokenizer that produces `[UNK]` for common math symbols is disqualified.**

* **e5-large-v2**: Produces `[UNK]` for key symbols such as `ϵ`.
  → **Rejected**
* **gte-large-en-v1.5**: Same failure mode as e5.
  → **Rejected**

These models are not suitable for math-heavy corpora.

---

### Remaining Models: Tokenization Findings

#### 1. `embeddinggemma-300m` (GemmaTokenizerFast)

**Strengths**

* Clean, native tokenization of Unicode math symbols (`ϵ, ε, μ, ω, ≤, ∑, √, ∞`).
* No `[UNK]` tokens observed in any test.
* TeX macros (`\omega`, `\epsilon`, `\mathrm`, `\frac`, `\sqrt`) are consistently and compactly tokenized.
* Lowest or near-lowest token counts on math-dense text.
* Embedding invariance tests show:

  * Strong equivalence for relational operators (`≤` vs `<=`).
  * Reasonable (but not perfect) equivalence for Unicode vs TeX math symbols.

**Weaknesses**

* Does not fully collapse all equivalent math representations (e.g., `ϵ` vs `\epsilon` cosine ≈ 0.87).
* Some domain concepts (e.g., `micron` vs `μm`) still require normalization for best results.

**Assessment**

* Best overall tokenizer behavior for math-heavy text.
* Serves as the baseline for “clean” math tokenization.

---

#### 2. `qwen3-embedding-0.6B` (Qwen2TokenizerFast)

**Strengths**

* No `[UNK]` tokens for math symbols.
* Very strong handling of LaTeX structure:

  * Stable tokens for `_{`, `}=` , `\mathrm`, `\frac`, `\sqrt`, subscripts/superscripts.
* Token counts comparable to Gemma on dense math passages.

**Observed Issue: Mojibake Appearance**

* Unicode symbols (e.g., `ϵ`) appear as mojibake-looking byte tokens in debug output.
* Round-trip tests confirm this is a **display artifact**, not corruption.

**Semantic Impact (Important)**

* Embedding invariance tests show **moderate sensitivity** to representation:

  * `ϵ` vs `ε`: ~0.87 cosine
  * `\omega` vs `ω`: ~0.87 cosine
  * `≈` vs `~=`: ~0.79 cosine
* Behavior is comparable to, and sometimes slightly better than, Gemma for certain pairs (`μm` vs `\mu m`), but worse for others.

**Assessment**

* Tokenization is internally coherent and safe (no `[UNK]`).
* However, Unicode math symbols are not treated as “atomic semantic units” in the same way as Gemma.
* Acceptable if math symbols are normalized prior to embedding.

---

#### 3. `bge-m3` (XLMRobertaTokenizerFast)

**Strengths**

* No `[UNK]` tokens for math symbols.
* Unicode math symbols are tokenized as real tokens.
* Broad multilingual coverage.

**Weaknesses**

* Heavy fragmentation of core TeX macros:

  * `\omega` → `om` + `ega`
  * `\sqrt` → `s` + `q` + `rt`
* Higher token counts on math-dense passages.
* More fragmentation of scientific prose (`frequency`, `opacities`, etc.).

**Assessment**

* Tokenization is symbol-safe but inefficient for LaTeX-heavy scientific text.
* Higher token counts and macro fragmentation may reduce embedding stability.

---

## Summary of Tokenization Quality

| Model                | `[UNK]` for Math | Unicode Symbols      | TeX Macro Compactness | Token Efficiency | Verdict                       |
| -------------------- | ---------------- | -------------------- | --------------------- | ---------------- | ----------------------------- |
| embeddinggemma-300m  | No               | Excellent            | Excellent             | High             | Preferred                     |
| qwen3-embedding-0.6B | No               | Byte-level, coherent | Very good             | High             | Preferred |
| bge-m3               | No               | Good                 | Weak (fragmented)     | Low              | Secondary                     |
| e5-large-v2          | Yes              | Fails                | —                     | —                | Reject                        |
| gte-large-en-v1.5    | Yes              | Fails                | —                     | —                | Reject                        |

---

## Design Decision

1. **Reject any embedding model whose tokenizer emits `[UNK]` for common math symbols.**
2. **Primary choice:** `embeddinggemma-300m`

   * Best tokenizer behavior for math-heavy scientific text.
   * Clean Unicode handling and compact TeX tokenization.
3. **Secondary option:** `qwen3-embedding-0.6B`

   * Acceptable if input text is normalized (Unicode ↔ TeX) prior to embedding.
   * Mojibake appearance is not a correctness issue, but representation sensitivity remains.
4. **Optional fallback:** `bge-m3`

   * Use only if multilingual coverage is required and token efficiency is less critical.

---

## Recommendation for Production Use

Regardless of model choice, apply **math normalization before embedding**:

* Canonicalize Unicode ↔ TeX (e.g., `ϵ, ε → \epsilon`, `ω → \omega`, `μ → \mu`).
* Normalize relational operators (`≤ → <=` or `\leq`, consistently).
* Normalize units (`μm → \mu m` or `micron`).

This reduces representation variance and improves retrieval robustness across all evaluated models.

# Embedding benchmark

Goal: build a small, reproducible benchmark to compare embedding models and their
quantized variants on a 2-stage pipeline (semantic retrieval -> reranking).

## Dataset

- 1000 documents, possibly from very similar physics sub-domains.
- 30 questions.
- every question has 5-10 relevant documents that will count as "positive" in decreasing order of importance

## Metrics

Retrieval metrics:
- Recall@50
- Recall@100
- Recall@200

Reranking metrics:
- nDCG@10
- MRR

Supplementary:
- Latency distribution (p95/p99) for query embedding + retrieval

## Models to test

- Qwen3-Embedding-0.6B
- embeddinggemma-300m
- e5-large-v2
- gte-large-en-v1.5
- bai-bge-m3
- text-embedding-3-small

## Storage layout (new tables)

Tables live under the `evaluation` schema for cleanliness:
- `evaluation.documents`: canonical documents.
- `evaluation.queries`: canonical queries.
- `evaluation.pair_metrics`: wide per-pair stats (query/document/model/quantization,
  retrieval and rerank ranks/scores, relevance grade, and latency fields).

Embeddings are stored as pgvector columns in per-model tables:
- `evaluation.document_embeddings_<model_key>`
- `evaluation.query_embeddings_<model_key>`

Embedding tables include storage columns for:
- `embedding` (vector / fp32)
- `embedding_halfvec` (halfvec / fp16)
- `embedding_bit` (bit / binary)

Quantization tags are stored in the `quantization` column. For IVF+SQ8, the vector
is stored in `embedding` and the index will be configured later.

Each metrics row is keyed by:
- `dataset_name` to group corpora
- `run_name` to separate repeatable evaluation runs
- `model_name` + `quantization` to compare fp32 vs quantized variants

Runner (WIP): `evaluation/embedding/benchmark.py`
- Documents JSONL line: `{"source_id": "...", "text": "...", "metadata": {...}}`
- Queries JSONL line: `{"text": "...", "metadata": {...}, "relevance": {"source_id": 1}}`
- `--reset-run` clears dataset rows/metrics before re-running.

Dataset creation (example):
- `evaluation/embedding/datasets/baseline_1_query_1_doc/create_dataset.py`
  - Outputs to `evaluation/embedding/datasets/baseline_1_query_1_doc/documents.jsonl`
  - Outputs to `evaluation/embedding/datasets/baseline_1_query_1_doc/queries.jsonl`

## Results

Below is a **compact markdown summary table** followed by a **short interpretive recap** tailored to your setup.

---

## Retrieval Evaluation Summary (Astrophysics Corpus)

| Model                    | Recall@50 | Recall@100 | Recall@200 | nDCG@10   | MRR       | p95 Latency (ms) |
| ------------------------ | --------- | ---------- | ---------- | --------- | --------- | ---------------- |
| **qwen3_embedding_0_6b** | **0.532** | **0.625**  | **0.762**  | **0.220** | 0.571     | 173              |
| embeddinggemma_300m      | 0.440     | 0.574      | 0.681      | 0.206     | 0.572     | 80               |
| e5_large_v2              | 0.398     | 0.524      | 0.669      | 0.169     | 0.507     | 61               |
| gte_large_en_v1_5        | 0.443     | 0.558      | 0.694      | 0.207     | 0.551     | **60**           |
| bge_m3                   | 0.458     | 0.562      | 0.699      | 0.145     | 0.509     | **60**           |
| text-embedding-3-small   | 0.493     | 0.600      | 0.718      | 0.182     | **0.611** | 713              |

---

## Key Findings

* **Best overall retrieval quality:** `qwen3_embedding_0_6b`

  * Highest Recall@50 / @100 / @200
  * Best nDCG@10

* **Best early recovery:** `text-embedding-3-small`
  * Highest MRR

* **Fastest models:** `e5_large_v2`, `bge_m3`, `gte_large_en_v1_5`

  * p95 ≈ 60 ms
  * Clear recall/ranking gap vs Qwen


# Latency and optimizations
After testing the latency on real-chunks embeddings it's clear that running the embedding locally for the current corpus is unfeasible (for ca. 850k chunks it would take few days of compute and results sometimes in crashes).
It's necessary to find light-weighted versions of the chosen embedding model. Best bet is to work with an optimized version for Apple silicon (MLX).

- **Models compared**
  - Full-precision **Qwen3-Embedding-0.6B** (SentenceTransformer, PyTorch)
  - **Qwen3-Embedding-0.6B-4bit-DWQ** (MLX, Apple Silicon)

- **Correctness**
  - Output dimensionality: **1024**
  - Ranking preservation (Spearman ρ): **0.98**
  - Conclusion: quantized embeddings preserve relative similarities extremely well

- **Latency (end-to-end, 32 sentences × 256 tokens)**
  - Full precision: **~5.0 s**
  - 4-bit DWQ: **~2.5 s**
  - Speedup: **~2×**

- **Memory footprint**
  - 4-bit DWQ (MLX): **~0.4 GB peak memory**
  - Full precision: **significantly higher (>1 GB typical)**
  - Quantization substantially reduces memory usage, improving deployability on resource-constrained systems

- **Overall**
  - Quantization yields a **significant latency reduction** with **negligible impact on embedding quality**
  - Suitable for production retrieval and similarity search workloads
