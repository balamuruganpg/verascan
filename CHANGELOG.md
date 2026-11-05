# Changelog

All notable changes to **Verascan** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.6.0.post1] - 2026-11-05

### Changed
- Reworked the README around the 0.6.0 GPU and reporting workflow, with new
  dashboard captures, the supplied dark/light screenshots, a banner and a workflow diagram.
- Package builds now derive the PyPI Markdown description from the README and
  convert screenshot references to absolute GitHub URLs. Removed obsolete HTML report previews.
- Replaced the hardcoded PyPI version badge with a live badge and scoped the CI
  badge to pushes on `main`.

## [0.6.0] - 2026-11-03

### Added
- **Dedicated NVIDIA GPU auditing** through `verascan.gpu_audit()` and `verascan gpu`.
  - CUDA availability checks before dataset loading, selected `cuda:N` devices,
    and CLI device name, CUDA runtime, free/total memory diagnostics.
  - SentenceTransformer embeddings explicitly on CUDA; GPU-preferred FAISS
    inner-product search over normalized float32 embeddings.
  - Visible FAISS-CPU index fallback after GPU index creation, insertion, or
    search failure, while embeddings remain on CUDA. CUDA/embedding failures
    report actionable errors rather than silently using CPU embeddings.
  - Default GPU batch size 256, batched encoding/indexing/search with tqdm
    progress, optional `max_train` / `--max-train` and per-benchmark
    `max_eval` / `--max-eval`, model and dataset split/config controls.
  - Up to ten nearest training neighbors per evaluation row, filtered by score.
- **PDF export** via `report.to_pdf("report.pdf")` on both contamination and audit
  reports, plus `.pdf` CLI output for `check`, `audit`, and `gpu`.
  - Summary cover with contamination rate, train/eval sizes, methods, benchmarks;
    paginated score tables and full highlighted word-level differences.
  - ReportLab rendering independent of the HTML template, with bundled licensed
    DejaVu fonts, safe dataset markup escaping, page footers, and long-text support.
- **Local Streamlit dashboard** via `verascan dashboard report.json`.
  - Custom monochrome workspace with pure black/white themes and a header
    toggle that preserves filters and selection. Grayscale charts, controls,
    semantic pair table and word differences; color reserved for status.
  - Clean/Low/High status card, train/eval/match/rate metrics, and a dedicated
    clean state without empty charts.
  - Method and benchmark charts for reports with matches, score/method/benchmark
    filters, literal search, paginated pairs, side-by-side source text and diffs.
  - `--demo` previews five planted pairs without modifying the real report.
  - Compatible with existing JSON exports; optional `[dashboard]` extra
    (Python 3.11+), launched on localhost using the active Python environment.
- **GPU extra**: `pip install "verascan[gpu]"` requests sentence-transformers,
  faiss-gpu, torch, and datasets; documented CUDA PyTorch and FAISS platform constraints.
- GPU mocking/fallback tests, PDF content/pagination tests, dashboard widget tests,
  and CLI coverage for the new paths.

### Changed
- FAISS capability checks distinguish CPU installations before indexing and
  emit one clear CUDA-preserving fallback line. GPU installations prefer the
  standard resource/transfer API, with direct GPU flat-index construction when
  the transfer helper is unavailable. CPU-only detection avoids duplicate encoding.
- PDF cover now uses a navy summary panel, status/rate, size tiles and a
  dedicated audit-configuration section.
- GPU installation selects FAISS-CPU on Windows and non-x86-64 Linux platforms
  rather than requesting unavailable FAISS-GPU wheels. CUDA embeddings remain
  mandatory; Linux x86-64 continues to prefer FAISS-GPU.
- CPU-only PyTorch builds now produce an explicit CUDA installation diagnostic,
  even when an NVIDIA GPU and driver are present.
- PDF and the JSON dashboard are the primary reporting workflows; JSON and
  self-contained legacy HTML exports remain supported.
- GPU HF inputs default to the full training split; existing CPU dataset bounds,
  matching behavior, and `check` / `audit` / `split` defaults are unchanged.
- Added ReportLab to core dependencies. Version remains sourced by Hatch from
  `src/verascan/_version.py`, now `0.6.0`.

## [0.5.0] - 2026-10-01

### Added
- **Hugging Face Dataset Ingestion by Name**:
  - Automatically load Hugging Face datasets when passing a dataset name string (e.g. `train="imdb"`, `train="gsm8k"`, or `train="openai/gsm8k"`).
  - Safe default partition bounds: defaults to `train[:2000]` for training inputs and `test` for evaluation inputs to prevent accidental multi-gigabyte downloads.
  - Granular split & configuration controls: `train_split`, `eval_split`, `split`, and `config` parameters in `verascan.check()`, `verascan.audit()`, `verascan.split()`, and the CLI.
  - Helpful error messages: raises `ImportError: Install with pip install "verascan[hf]"` if the `datasets` dependency is missing, or a descriptive error when a dataset is unresolvable.
  - Seamless interoperability: works identically across `verascan.check()`, `verascan.audit()`, and `verascan.split()` without breaking existing inputs (`list[str]`, `list[dict]`, `pandas.DataFrame`, `datasets.Dataset`, or local files).

### Changed
- **Silent Report Summary (`report.summary()`)**:
  - `ContaminationReport.summary()`, `AuditReport.summary()`, and `SplitResult.summary()` now return the formatted summary string without writing to `stdout`.
  - CLI commands (`verascan check`, `verascan audit`, `verascan split`) print the returned summary using `typer.echo`.

## [0.4.1] - 2026-09-24

### Fixed
- **Contamination Rate Display**:
  - Never display `"0.0%"` when matches exist; formats very small non-zero rates with high precision (e.g. `0.006%` or `0.013%`) so the status banner and metrics are non-contradictory.
  - Aligned singular/plural wording for match counts in the status banner.
- **Loader Support for `list[dict]`**:
  - Added support for passing `list[dict]` to `check()`, `split()`, `load_texts()`, and `load_eval_payload()`, extracting the specified `column` (e.g. `column="question"`).
  - Preserves tabular records and column structure for downstream cleaned dataset export.
- **Clearer Dataset Name / Missing File Error**:
  - Improved error when passing a non-existent path or dataset name string (e.g. `'imdb'`): raises `FileNotFoundError: Not a data file: 'imdb'. Pass a file path, list[str], DataFrame, or Hugging Face Dataset object.`

## [0.4.0] - 2026-09-07

### Added
- **Public Benchmark Contamination Audit (`verascan.audit`)**:
  - `verascan.audit(train="train.jsonl", benchmarks=["mmlu", "gsm8k", "humaneval"])` audits a training dataset against popular public evaluation benchmarks.
  - Built on top of existing `check()` detection engines (`exact`, `ngram`, `fuzzy`, `semantic`) without duplicating logic.
  - Returns an `AuditReport` subclass of `ContaminationReport` with `benchmark_counts` and `by_benchmark()` helper methods.
- **Benchmark Presets**:
  - `mmlu`: `cais/mmlu` (config `all`, `test` split), extracting `question`.
  - `gsm8k`: `gsm8k` / `openai/gsm8k` (`test` split), extracting `question`.
  - `humaneval`: `openai_humaneval` (`test` split), extracting `prompt`.
- **Synthetic Stand-Ins (`--synthetic` / `synthetic=True`)**:
  - High-fidelity synthetic representations of `mmlu`, `gsm8k`, and `humaneval` for offline and deterministic CI/local testing without network downloads.
  - `allow_synthetic=True` fallback when Hugging Face download is unreachable.
- **CLI Audit Command (`verascan audit`)**:
  - `verascan audit --train train.jsonl --benchmarks mmlu,gsm8k,humaneval --output audit.html`.
  - Supports `--methods`, `--threshold`, `--column`, `--output`, `--output-json`, `--fail-above`, and `--synthetic`.
- **HTML Report Benchmark Visualizations**:
  - Added benchmark provenance badges (`[gsm8k]`, `[humaneval]`, `[mmlu]`) on match cards.
  - Interactive benchmark filter tabs (`All Benchmarks`, `mmlu`, `gsm8k`, `humaneval`) with dynamic count chips.
  - Summary metrics card displaying audited benchmark count and contamination breakdown.
- **GitHub Actions CI**:
  - Added `.github/workflows/ci.yml` testing pushes and pull requests on Python 3.11 with `pytest`.

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
