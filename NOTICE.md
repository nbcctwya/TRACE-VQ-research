# Code provenance

TRACE-VQ is extracted from HVQ-Stock branch
`exp/034-combine-025-028-temporal-attention-stage1`, commit
`2b1f576e4a7d46b098b6278b2fe76672f5881a23`.

The source identifies PRISM-VQ as its base. Its temporal-attention Stage 1
encoder was brought from experiment 028, and its shared-routed MoE and
quantization-confidence adapter from experiment 025. The source README
attributes `PositionalEncoding`, `TAttention` and `TemporalAttention` in
`module/layers/encoder.py` (now `trace_vq/models/temporal/encoder.py`) to AlphaMaster. The algorithms and existing attribution are preserved through the structural
refactor.

No LICENSE or separate attribution file was present in that source commit.
This migration does not introduce a new license or change the attribution of
inherited code. See `docs/migration/migration_manifest.json` for exact retained source files
and their content hashes at migration commit `f4d962c`. Current locations are
recorded in `docs/restructure_map.json`.

The full-model/ablation import additionally reads frozen HVQ experiments 025,
034, 037 and 038. Experiment 025 supplies the original GRU encoder for w/o TA;
037 removes the confidence adapter; 038 removes the always-on shared expert.
Their pinned commits, original configuration files, Stage 1 weight provenance
and artifact checksums are recorded in `docs/migration/experiments_manifest.json`.

Generated manifests and file maps are retained locally and excluded from Git.
