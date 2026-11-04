# Verascan 0.6.0: GPU and reporting

The user's release specification is the authority. Add a dedicated NVIDIA path
without changing CPU check/audit detection or defaults. Keep existing JSON and
HTML compatibility. No publishing or changes to shared branches.

## Design decisions

- `gpu_audit` returns `AuditReport` and runs semantic matching with a CUDA
  SentenceTransformer. Default batch size 256; explicit 64 remains supported.
  Detect CUDA before loading data or models. Accept `cuda` and `cuda:N` only.
- Prefer a FAISS GPU inner-product index on normalized float32 vectors. If GPU
  index creation, insertion, or search fails (including a CPU-only FAISS build),
  warn and rebuild/search a CPU index. Embeddings always stay on CUDA. Fail on
  missing CUDA or embedding OOM with actionable messages.
- GPU training dataset names load the full train split unless limited. Optional
  positive `max_train` and `max_eval` bound rows before embedding; `max_eval`
  applies per benchmark so each requested benchmark remains represented.
  Pass HF split slices before loading when limits are set. Local files and
  dataset objects are materialized by existing loaders, then sliced.
- Retain top 10 neighbors per evaluation row (same semantic result policy as
  the CPU implementation). Batch training encoding/insertion and eval
  encoding/search to avoid holding the full embedding matrix in host RAM.
- Show CUDA device name, CUDA runtime, total/free memory at CLI start. Expose
  model, threshold, split/config, limits, batch size, and progress controls.
- Add `to_pdf` to the base report, inherited by audit reports. ReportLab builds
  a summary cover, paginated match table, and full highlighted word differences.
  Dataset text is escaped. Paragraphs split across pages for long examples.
  ReportLab is a base dependency so all report objects can export PDF.
- Add optional Streamlit dashboard extra and `verascan dashboard report.json`.
  Read existing JSON without changing its schema; present global contamination,
  method/benchmark charts, filters, literal text search, table and paginated
  collapsible diffs. Streamlit imports remain lazy. Launch on localhost.
- Add PDF output handling to existing CLI commands while preserving all
  existing CPU detection behavior and legacy output behavior. New GPU CLI
  defaults to PDF and supports a separate JSON sidecar for the dashboard.
- Keep Hatch's dynamic version configuration pointing at `_version.py`, bump
  to 0.6.0, and verify built metadata. Document CUDA PyTorch setup and FAISS-GPU
  wheel/platform constraints rather than promising universally available wheels.

## Verification

Mock hardware/model boundaries; exercise real FAISS CPU searches for fallback
correctness. Test missing CUDA, selected device, GPU index success and failures,
OOM, input limits, scores/indices, CLI output and progress controls. Parse PDFs,
test long/multi-page and empty reports, and visually render a representative PDF.
Use Streamlit AppTest to verify filters, charts, empty reports and escaped diffs.
Run targeted tests after each stage, then the full suite, Ruff and strict mypy.
Build wheel/sdist and inspect version and packaged dashboard/report resources.
