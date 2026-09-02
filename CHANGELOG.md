# Changelog

All notable changes to **Verascan** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.3.0] - 2026-09-02

### Added
- **Leak-Free Dataset Splitting (`verascan.split`)**:
  - `verascan.split(data, eval_size=0.2, methods=["exact", "fuzzy"], seed=42)` partitions a dataset into training and evaluation sets while guaranteeing zero exact, fuzzy, or semantic leakage.
  - Iterative leak purification: detects candidate leakage between splits and moves contaminated candidate eval examples out of eval (into train by default, preserving all samples).
  - Supports `move_to="train"` (default) or `move_to="drop"`.
  - Type-safe format preservation: returns `list[str]` for string list inputs, and `pandas.DataFrame` (preserving all original columns) for DataFrames, CSV, JSONL, and HuggingFace Datasets.
  - Direct file output options: `output_train` and `output_eval` supporting `.csv`, `.jsonl`, and `.json`.
- **CLI Command (`verascan split`)**:
  - `verascan split --input data.jsonl --eval-size 0.2 --output-train train.jsonl --output-eval eval.jsonl`.
  - Configurable `--methods`, `--threshold`, `--column`, `--seed`, `--move-to`, and `--no-progress`.
- **`ngram` detection method**: GPT-3-style word *n*-gram overlap (Brown et al., 2020).
  - Lowercases text, strips punctuation per token, builds word 13-grams by default.
  - Drops training grams appearing in ≥ `ngram_max_count` documents (default 10).
  - Flags eval examples that share any retained gram; score is the per-pair overlap ratio.
  - API: `methods=["ngram"]` with optional `ngram_n` / `ngram_max_count`.
  - CLI: `--methods ngram` (combinable with `exact`, `fuzzy`, `semantic`).
- **Total Matches Property**:
  - Added `report.total_matches` convenience property on `ContaminationReport`.

## [0.2.0] - 2026-08-31

### Added
- **Cleaned eval export**: After `check()`, emit a decontaminated evaluation set with flagged rows removed.
  - `ContaminationReport.cleaned_eval()` returns a `list[str]` or `pandas.DataFrame` matching the original eval input (extra columns preserved).
  - `ContaminationReport.to_cleaned(path)` writes CSV / JSONL / JSON.
  - `contaminated_eval()` / `to_contaminated(path)` export the flagged rows.
  - CLI: `verascan check --output-cleaned PATH`.
  - Empty-contamination case returns the original eval unchanged.

## [0.1.0] - 2026-08-22

### Added
- **Core Detection Engines**:
  - `exact`: Ultra-fast SHA-256 hash-based matching with case and whitespace normalisation.
  - `fuzzy`: Near-duplicate detection using MinHash and Locality-Sensitive Hashing (LSH) via `datasketch`.
  - `semantic`: Embedding-based semantic similarity search with `sentence-transformers` and FAISS vector indexing.
- **Unified Data Loading**:
  - Direct support for Python `list[str]`, `pandas.DataFrame`, `JSONL`, `CSV`, and HuggingFace `datasets.Dataset`.
  - Custom column selection for tabular and nested datasets.
- **Reporting & Visualization**:
  - `ContaminationReport` class with query helpers (`flagged()`, `contamination_rate`, method breakdown metrics).
  - Modern, self-contained, responsive HTML reports with interactive filters, search, word-level diffs, and health indicators.
  - Machine-readable JSON export (`to_json()`) and dictionary conversion (`to_dict()`).
- **Command-Line Interface (CLI)**:
  - `verascan check` command with multi-method selection, threshold configuration, and output path options.
  - `--fail-above` CI gate flag to exit with non-zero status when contamination exceeds acceptable limits.
- **Quiet & Clean Runtime**:
  - Automatic environment configuration to silence noisy backend C++/oneDNN and framework deprecation warnings.
  - Zero startup latency for core exact/fuzzy operations.
