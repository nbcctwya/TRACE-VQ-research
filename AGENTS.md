# TRACE-VQ research workspace

This is an independent research copy of `../TRACE-VQ`. The user requested that
the original TRACE-VQ state be preserved. Make research changes in this copy.
Treat `../TRACE-VQ` as a read-only reference, including its Git metadata, data,
checkpoints, results, logs and caches.

- The original `research-baseline-20261026` tag and history are preserved only
  in the ignored local Git backup under `artifacts/research/history_reset_*/`.
  The active repository starts with a new root commit on `main`, as requested
  by the user. Keep the evidence under `research/baseline/` unchanged.
- Use `research/run.sh` to select the existing Python environment and this
  copy's import paths, datasets, checkpoints and caches. It sets process-local
  environment variables; it does not create an OS sandbox.
- Put new experiment outputs under `artifacts/research/<run_name>/`, with an
  explicit `artifact_root` or `--output-dir`. Keep imported results available
  for comparison. Use new filenames when changing datasets.
- Keep model implementations under `trace_vq/`. Root scripts and the old
  `module/`, `trainer/`, `dataset/*.py` and `utils/` paths are compatibility shims.
- Reuse the existing environment without installing or upgrading its packages.
  Create a separate environment inside this research copy if dependencies must
  change.
- Verify the original's state with `research/run.sh research/verify_baseline.py`.
  Add `--full` to hash all original data, model and cache files as well.
- Git history and branches are local to this copy. Its remote push URL is
  disabled. Do not publish research changes without a user request.

## Git synchronization scope

- Track source, configs and documentation. Curated aggregate results, experiment
  descriptions and portable effective configs belong under `experiments/`.
- Keep raw artifacts, datasets, weights, per-day/per-stock results, logs, caches,
  local credentials and the original baseline evidence on disk and gitignored.
  Do not edit `research/baseline/` to sanitize it; keep its bytes unchanged.
- Begin code/result analysis with `experiments/README.md`,
  `experiments/protocol.md` and `docs/analysis-guide.md`. Per-seed summaries are
  independent portfolio metrics; they are not prediction-ensemble results.
- Preserve reused-reference labels, weak/negative results and metric/quote-policy
  distinctions. Do not sum overlapping batch rows as new training runs.
- Keep the historical baseline tag/history in the ignored local backup; do not
  reintroduce old refs into the active repository. Publishing or rewriting
  history again requires a user request.
