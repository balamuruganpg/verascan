# Changelog

All notable changes to **Verascan** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
