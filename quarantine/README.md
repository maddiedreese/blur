# Rejected artifacts

This local-only directory holds models, calibration files, and packages rejected by release evaluation. It is ignored by Git and is never read by build or runtime code. Moving an artifact here records that it is diagnostic evidence only, not a release candidate.

The release packager independently requires `artifacts/release-test-scores.jsonl` to pass `scripts/release-gate.mjs`; moving or renaming a rejected artifact cannot bypass that gate.
