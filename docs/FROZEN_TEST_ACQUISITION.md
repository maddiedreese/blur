# Frozen test acquisition plan

This plan prepares evidence for the current frozen checkpoint. It does not
authorize checkpoint, preprocessing, threshold, calibration, or fusion changes
after acquisition starts. Images and labels remain sealed from model selection;
only acquisition integrity, license fields, corruption, and duplicates may be
inspected before the final deployed-path score.

## Quotas and pins

`tools/frozen_test_sources.json` is the machine-readable source contract. Its
selection totals exactly 500 unique real and 552 unique AI base images.

| Source | Pin | Rows | Required columns | Role and license |
|---|---|---:|---|---|
| `nyuuzyou/pxhere` | `9a0820f476ea1ba00f6053e63bc81f24c5eb605f` | 200 | WebDataset key/image plus JSON `image_id`, `download_url`, `tags`, `uploaded`, `exif_info` | Broad camera/stock photographs. Repository and dataset card declare CC0-1.0. Prefer rows with camera EXIF, while retaining product, people, landscape, and low-light diversity. |
| `Mitsua/art-museums-pd-440k` | `fba945da78b36262eb9272067197cc28d06cffbf` | 200 | WebDataset key/image, English caption/metadata, originating museum/object locator | Human art, sculpture, decorative objects, and render-like hard negatives. The assembled dataset is CC-BY-4.0; its card states underlying images/text are CC0 or public domain. Preserve both dataset attribution and the originating museum locator. Treat the four contributing museums as one acquisition source for source-count policy. |
| `naver-clova-ix/cord-v2` | `7f0115a4b758a71d6473b8d085751692da2fef98` | 100 | viewer row index, `image`, `ground_truth.meta.image_id` | Text-heavy photographed receipts. CC-BY-4.0; preserve dataset attribution. The official test split has exactly 100 rows. |
| `Coxy7/X-AIGD` (`labeled_test`) | `92180f32030507ab54a40d6f1b88f39d6cec8178` | 552 | direct `image`, exact `generator`, `uid`, `original_prompt`, dimensions/format | AI-only labeled benchmark rows: 181 `HYDiT_1.2-raw`, 198 `Infinity-raw`, and 173 `Lumina_Next-raw`. The repository card declares CC-BY-4.0. No paired comparison side is present or downloaded. |

The pinned X-AIGD `default/labeled_test` split contains 2,419 direct-image rows.
The three allowlisted families provide 552 immutable Viewer-row image bases.
Upstream `uid` values are unique within each selected generator, but reused
across generators; UID is metadata, not base identity. The pinned metadata scan
contains **375 globally unique UIDs** across 552 rows (and 552 unique
`(generatorModel, uid)` pairs). The ledger reports the observed `uniqueUids`
instead of using it as a quota. There are also **375 globally unique normalized
prompts** because the benchmark
intentionally evaluates multiple generators on some shared prompts. Every row
stores `promptGroup = SHA256(NFKC(casefold(collapse_whitespace(prompt))))`.
Same-prompt rows remain together in this one frozen `test` split; a prompt group
must never appear in training or calibration, and prompt reuse does not inflate
the reported base-image count.

## Acquisition protocol

The real-source fetcher is intentionally separate from scoring:

```sh
.venv311/bin/python tools/acquire_real_frozen_candidates.py \
  --output artifacts/frozen-test/real-candidates \
  --margin 0.2 \
  --seed blur-real-acquisition-v1
```

At the default margin it emits exactly 240 PxHere, 240 museum, and 120 CORD
candidates (the configured quota plus the larger of ten rows or 20%). PxHere
and museum bytes stream from tar shards at the commit-pinned `resolve` URL.
CORD rows use Dataset Viewer only while the repository head equals the pinned
revision; the pin is rechecked after fetching because Viewer has no revision
parameter. The signed image transport URL is never written to provenance.
CORD exhausts the 100-row official test split before using validation rows for
the acquisition margin; every emitted evaluation row still has `split: test`
because it belongs to this project's held-out test, while `upstreamLocator`
retains the original upstream split and row.

Every image is decoded and dimension-checked before a `.part` file is renamed.
Byte duplicates inside a source are skipped, cross-source byte duplicates fail
the complete run, and an error removes the staged directory. The final output
directory is installed by one same-filesystem rename and is never overwritten.
`candidates.jsonl` includes byte hash, source, commit revision, license,
attribution/public-domain statement, stable origin locator, upstream shard or
Viewer row, and decoded dimensions. `provenance.json` records the seed, margin,
exact source counts, and manifest hash. This fetch step does not inspect labels
beyond the pinned source contract and never loads or calls a detector.

1. Record the complete source contract, current checkpoint SHA-256,
   preprocessing version, identity calibration, threshold `0.65`, and selection
   seed before fetching candidate rows.
2. Query only the pinned Hub revisions or dated institutional API snapshot.
   Save raw response pages and their SHA-256 hashes in an acquisition ledger.
3. For X-AIGD, assert the Hub API still reports the exact revision and
   `cc-by-4.0`, paginate exactly `default/labeled_test`, and download only the
   direct `image` field for the three exact generator identifiers. Decode every image, reject corrupt files,
   compute the byte SHA-256, and emit one candidate row per unique original.
   Signed Dataset Viewer asset URLs are transport URLs and must not be retained
   as provenance; retain the dataset/revision/config/split/row/column locator.
4. Run exact byte, `id`, `baseId`, and origin-locator deduplication against all
   training, calibration, and existing protected manifests. Before freezing,
   run perceptual-hash and embedding near-duplicate clustering across every
   split and manually adjudicate collisions without reading model scores.
5. Inspect real labels and source licensing, not detector output. Reject images
   with uncertain provenance, synthetic edits, or an absent public-domain or
   attribution statement. Content review must be blinded to model response.
6. Acquire into a new directory (the command stages all bytes and atomically
   renames only after every quota, row identity, generator/UID pair,
   prompt-group, byte-hash, and image check
   succeeds):

   ```sh
   .venv311/bin/python tools/acquire_xaigd_frozen_test.py \
     artifacts/frozen-test/xaigd-ai
   ```

   Base identity is the immutable `(revision, config, split, rowIndex)` locator;
   IDs and filenames include `rowIndex`, not bare UID. The expected ledger is
   552 unique row bases and byte hashes, exact family counts `181/198/173`, and
   375 prompt groups. It separately reports observed global `uniqueUids` (375
   at the audited revision) and
   requires all `(generatorModel, uid)` pairs to be unique. The command is
   documented here but must not be run until acquisition is authorized.

7. Merge the independently audited ledgers without byte rewriting. The merger
   rebases every image path relative to the combined ledger, sorts rows
   deterministically, recomputes every referenced image hash, confines image
   paths to their acquisition directories, and rejects duplicate IDs, bases,
   hashes, row identities, and `(generatorModel, uid)` pairs while allowing a
   UID reused by a different generator. It atomically installs a new output and reports the
   combined manifest SHA-256:

   ```sh
   .venv311/bin/python tools/merge_frozen_test_candidates.py \
     artifacts/frozen-test/real-candidates/candidates.jsonl \
     artifacts/frozen-test/xaigd-ai/candidates.jsonl \
     --output artifacts/frozen-test/candidates.jsonl
   ```

8. Build once:

   ```sh
   .venv311/bin/python tools/build_frozen_test_manifest.py \
     artifacts/frozen-test/candidates.jsonl \
     --output artifacts/frozen-test/frozen-test.jsonl
   ```

   The command fails on missing local bytes, hash disagreement, unpinned source
   metadata, protected-source tokens, duplicates, or overlap with current
   training/calibration/protected manifests. It will not overwrite an existing
   frozen manifest.
9. Generate deterministic full-resolution and browser-stress derivatives from
   each selected base. Derivatives inherit `baseId`, split, source, license, and
   generator family and do not increase sample counts. Only after this manifest
   and its hash are sealed should the deployed browser pipeline score it.

## Legal and coverage caveats

- Repository-level licenses are not automatically image copyright grants.
  Institutional rows therefore require their public-domain flag or statement;
  attributed rows require attribution text. Keep a third-party notices ledger.
- CC-BY-4.0 is the repository-level license declared by X-AIGD. Preserve the
  dataset attribution and immutable row locator. It may not resolve every
  underlying generated image, trademark, personality, or prompt right.
  Redistributing image bytes deserves a separate legal review; publishing only
  hashes, locators, aggregate metrics, and a reproducible acquisition script is
  lower risk.
- The Open Images mirror was deliberately excluded: its schema exposes only
  `index` and `url`, insufficient to preserve creator attribution required by
  CC BY 2.0.
- RICO screenshot mirrors found during audit did not expose an explicit license
  in their repository metadata, so screenshots are not yet in the default
  contract. CORD supplies text-heavy camera images, but a separately licensed
  UI/screenshot source remains an important coverage gap.
- Museum sources are unusually clean public-domain hard negatives but are not a
  substitute for contemporary human digital art, CGI/game renders, memes, or
  edited photographs. Add those only through new pinned, row-level audited
  sources; do not silently relabel web search results.
- This plan satisfies the numerical source floor, not representativeness. Report
  each real source, generator family, and content group separately and avoid a
  universal “95% accurate” claim from one 1,000-base evaluation.
