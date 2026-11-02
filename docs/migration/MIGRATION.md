# TRACE-VQ migration record

This is the historical migration audit for commit `f4d962c`. File paths and
content hashes describe that snapshot, before the project was reorganized.
For current locations and architecture, see `../restructure_map.json` and
`../architecture.md`.

## Source and repository boundary

- Source: HVQ-Stock, branch `exp/034-combine-025-028-temporal-attention-stage1`.
- Fixed source commit: `2b1f576e4a7d46b098b6278b2fe76672f5881a23`.
- Target: existing independent TRACE-VQ repository on `main`, extending its
  initial commit `039a6d5`. Its existing remote configuration is retained;
  no source history was imported and nothing was pushed.
- Only committed Git objects were read for extraction. No source checkout,
  fetch, worktree, stash, reset or cleanup was performed; no links into the
  source repository were created.

## Included and excluded content

50 source files were imported: 40 remain byte-identical and 10 have migration
adjustments. All retained model modules, both trainers, and utility algorithms
are byte-identical to the source. `migration_manifest.json` contains the exact
whitelist, source/target SHA-256 hashes and all 46 excluded source paths.

The migration excludes experiment orchestration, prompts/rules, queues,
notifications, records, historical smoke scripts tied to old weights,
baseline-result reports, unrelated model/trainer variants, machine-specific
sync scripts, and the unsupported CSI500 generation configuration. Untracked
data, artifacts, checkpoints, logs, environments and source Git metadata were
never exported. Generic data-generation settings and tests are retained.

## Necessary adjustments

- Configuration: removed historical checkpoint filenames; Stage 2 requires
  `predictor.saved_model`. Removed checkpoint selection from market presets so
  explicit weights work for either market. All architecture, optimizer, seed,
  split and early-stopping values remain unchanged. Project logging is named
  TRACE-VQ and W&B defaults to offline.
- Entrypoints: resolve relative training data paths against the project,
  keep Hydra logs/snapshots and default caches local, make resolver registration
  safe when both stages are imported, and save the effective Stage 2 config
  alongside its checkpoint. Model mathematics and checkpoint state keys are
  unchanged.
- Data/backtest/ensemble: added an explicit Qlib provider path argument to the
  dataset generator; default backtest, ensemble and Qlib run outputs remain in
  the target project. Existing evaluation metrics are unchanged.
- Standalone evaluation: added `evaluate.py` to load a Stage 2 checkpoint with
  its compatible Stage 1 weights and effective config, then export validation
  or test predictions and IC/RankIC metrics without training.
- Dependencies: pinned the direct packages to the existing Python 3.11
  validation environment, including PyTorch 2.8.0, Lightning 2.6.4 and Qlib
  0.9.7. This replaces the source's older core pins and `pyqlib==0.9.6.99`;
  obsolete pickle5 and unused notebook/development/plotting dependencies were
  omitted. Fresh environment installation is not claimed as verified.
- Tests: removed sibling AlphaMaster file comparisons and checks of excluded
  trainer variants; retained behavioral architecture/data tests. Replaced
  historical-weight-dependent checks with temporary generated Stage 1
  checkpoints, and added explicit-path/config and standalone integration tests.
- Documentation: rewrote the initial README with preparation, training,
  evaluation and validation commands, and recorded inherited attribution in
  `NOTICE.md`. No source license file was present.

The original 034 experiment reused experiment 028's trained Stage 1 weights.
Those weights are intentionally absent here. Training Stage 1 anew enables
standalone use but does not establish equivalence to that historical run.

## Verification

- `python -m unittest discover -s tests -v`: **89 passed**, no skips.
- All six CLI help paths pass, including when invoked from outside the project.
- Synthetic CPU runs through the actual Stage 1 and Stage 2 commands each
  complete one epoch, save checkpoints, and generate predictions/metrics.
  Independent evaluation of the trained Stage 2 checkpoint using its saved
  effective configuration matches the training entrypoint's predictions.
- Integration tests cover both validation and test splits, forward/backward
  passes, adapter gradients, codebook freezing, and strict checkpoint loading.
- All Python files parse; no source links or copied data/weight artifacts are
  present. The manifest records exact source-to-target content comparisons.
- Before/after SHA-256 comparisons cover **9,114 source files / 17,804,357,138
  bytes**, including `.git`: **zero content changes**. Source HEAD, branch and
  clean worktree state are unchanged. The manifest contains the aggregate
  snapshot digest and recorded Git state.

Validation used the existing `prism-vq` environment, CPU, and temporary
synthetic data/weights under `/tmp`; no old experimental weights or datasets
were used. Real-data training, GPU execution, historical metrics reproduction,
raw-source acquisition and a full financial backtest remain unverified.
