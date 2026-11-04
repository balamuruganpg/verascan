# GPU and Reporting Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Work in the current checkout as requested; leave changes uncommitted for review.

**Goal:** Deliver the user's GPU and reporting release as Verascan 0.6.0.
**Architecture:** Separate GPU engine/orchestrator, inherited PDF exporter,
optional Streamlit dashboard. Reuse existing benchmark loaders and report model.
**Tech Stack:** Python, torch, sentence-transformers, FAISS, ReportLab, Streamlit.
**Spec:** `docs/superpowers/specs/2026-11-03-gpu-reporting-design.md`

## Global Constraints

- Existing CPU check/audit detection remains unchanged.
- GPU embeddings require NVIDIA CUDA; only FAISS indexing/search may fall back.
- Python >=3.9 core compatibility; optional extras resolve compatible versions.
- Keep JSON and secondary HTML exports; do not publish.

## Review Focus

- GPU search failure after partial progress must neither lose nor duplicate rows.
- Selected CUDA device must be used consistently by model, FAISS and diagnostics.
- Large inputs must not require a full embedding matrix or one giant PDF table row.
- Empty or ordinary 0.5.0 JSON reports must render without benchmark metadata.
- Input text with markup or regex characters must remain inert and searchable.

## Task 1: GPU path

Files: create `src/verascan/gpu.py`, `tests/test_gpu.py`; modify `src/verascan/__init__.py`.
Produces: `gpu_audit(train, benchmarks, *, column, device, batch_size, threshold,
model_name, max_train, max_eval, train_split, config, benchmark_split,
show_progress) -> AuditReport`; `cuda_info(device) -> dict[str, Any]`.

- [x] Write hardware-independent tests with deterministic embeddings and real FAISS CPU search; cover success, creation/add/search fallback, CUDA/missing dependency errors, invalid arguments, row bounds and custom columns.
- [x] Run `pytest tests/test_gpu.py -v --tb=short`; confirm failures for missing API.
- [x] Implement bounded encoding/indexing/search and public API with lazy optional imports.
- [x] Re-run targeted tests; expect all passing. Run Ruff on new files.

## Task 2: PDF reports

Files: create `src/verascan/pdf_report.py`, `tests/test_pdf.py`; modify `report.py`, `pyproject.toml`, `cli.py`.
Consumes: `ContaminationReport.to_dict()` (audit includes per-match benchmark).
Produces: `ContaminationReport.to_pdf(path: str) -> None`.

- [x] Write PDF tests for both report types, empty reports, multi-page tables, long text, safe markup and Unicode.
- [x] Run tests to demonstrate missing export method.
- [x] Implement professional ReportLab summary/table/diffs with pagination, add PDF dependency and existing CLI PDF branches.
- [x] Run targeted PDF/CLI tests; render and inspect representative PDF pages.

## Task 3: Dashboard and CLI

Files: create `src/verascan/dashboard.py`, `tests/test_dashboard.py`; modify `cli.py`, `pyproject.toml`, `tests/test_gpu.py`.
Consumes: existing JSON schema; GPU API from task 1 and PDF from task 2.
Produces: `verascan gpu ...`, `verascan dashboard REPORT_JSON`.

- [x] Write failing CLI tests for CUDA diagnostics, PDF + JSON output, error handling and dashboard launch; AppTest filtering/search/empty report tests.
- [x] Implement lazy Streamlit launcher and dashboard metrics/charts/filters/diffs with bounded pages.
- [x] Verify targeted tests plus CPU CLI tests and lint/type checks.

## Task 4: Release verification

Files: `_version.py`, `pyproject.toml`, `README.md`, `CHANGELOG.md`.

- [x] Set version 0.6.0; add GPU/dashboard extras and README installation/platform notes and examples. Describe limits, top-10 matching and fallback policy.
- [x] Run full `pytest tests/ -v --tb=short`, `ruff check .`, `ruff format --check .`, `mypy src/`.
- [x] Build with `python -m build`; inspect metadata and dashboard resources.
- [x] Obtain independent review, resolve material findings, repeat affected verification; report results and physical-GPU validation limitations.


## Execution record

- Current checkout reused; no commits/pushes/publishing requested or performed.
- Pre-flight: GPU returns AuditReport; PDF consumes to_dict; dashboard consumes
  unchanged JSON schema. Interfaces agree, CPU detection modules untouched.
- Design check: independent reviewer found no important unanswered questions.
- Task 1: 19 tests observed failing for missing API, then passing; combined with
  existing loader/check coverage: 57 passed.
- Task 2: 6 tests observed failing for missing PDF method/output branches. Font
  coverage test exposed Vera's missing Greek glyphs; bundled licensed DejaVu.
  PDF/CLI/existing report verification: 37 passed. Three PDF pages rendered and
  visually inspected, including highlighted diffs.
- Task 3: 11 new dashboard/CLI tests observed failing; corrected test doubles to
  include module metadata and used installed Streamlit chart element name.
  Dashboard + GPU verification: 30 passed. Strict mypy: 20 files clean.
- Environment ruling: Streamlit AppTest needs a local asyncio socketpair;
  Windows network sandbox blocks it before app execution. Relevant tests run
  with approved local socket access; no runtime code workaround introduced.
- Dependency ruling: Streamlit >=1.66 for APIs verified against installed docs;
  dashboard requires Python 3.11+. Existing [all] CPU extras remain unchanged.
- Review completed: confirmed three material findings with five failing
  regressions. Fixed traceback/resource release before fallback encoding, bounded
  plain explicit HF training splits, and rejected colliding GPU output paths.
  All 26 GPU tests passed, including weak-reference destruction assertions.
- Final verification: 184 passed, 2 skipped, 1 pre-existing Google dependency
  FutureWarning (85.73 seconds). Ruff lint and formatting pass; strict mypy
  reports no issues in 20 files. Existing CPU semantic functional tests passed.
- Built wheel and sdist as 0.6.0; inspected all required modules, fonts/license,
  HTML and typing resources plus GPU/dashboard extras in metadata. Installed the
  wheel in an isolated target and verified PDF generation/font loading, JSON
  roundtrip and both new CLI help entry points from the installed copy.
- Initial environment used CPU-only PyTorch, so the initial verification used
  CUDA mocks plus real FAISS CPU search; no publishing performed.
- Windows follow-up: reproduced unavailable faiss-gpu wheels, then selected
  faiss-cpu through platform markers while retaining mandatory CUDA embeddings.
  Added platform-selection regressions and an explicit CPU-only PyTorch error.
  The exact editable GPU/dashboard installation now succeeds on Windows.
- Follow-up verification before replacing PyTorch: 188 passed, 2 skipped,
  1 existing dependency warning. Ruff lint/format and mypy passed; wheel/sdist
  rebuilt and platform-specific dependency metadata inspected.
- Installed PyTorch 2.8.0+cu126 into the local virtual environment after verifying
  the wheel against the publisher SHA-256. CUDA 12.6 detects the RTX 4060 Laptop
  GPU (8 GiB), and a real CUDA tensor calculation succeeds.
- Physical GPU CLI verification completed using 64 IMDb training rows and 8
  evaluation rows each from MMLU and GSM8K. CUDA embeddings plus the expected
  Windows FAISS-CPU fallback produced report.pdf and report.json successfully.
  This bounded smoke run found zero matches; it is not a full dataset audit.
- Full suite repeated with the CUDA build: 188 passed, 2 skipped, 1 existing
  dependency warning (69.40 seconds). Ruff lint/format and strict mypy passed.
- Generated PDF is 44,743 bytes across two pages; JSON reloads successfully.
  Streamlit AppTest opens the real report without exceptions, shows correct
  training/evaluation metrics and renders all three charts.
