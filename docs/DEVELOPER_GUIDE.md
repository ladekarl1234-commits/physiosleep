# Developer guide

[Documentation index](README.md) · [Pipeline](PIPELINE.md) · [Job ledger](EXPERIMENTS.md)

## Clone and verify

Use Python 3.11 and run from the repository root:

```bash
git clone https://github.com/ladekarl1234-commits/physiosleep.git
cd physiosleep
python -m venv .venv
```

Activate on Windows PowerShell with `.venv\Scripts\Activate.ps1`, or Linux/macOS with `source .venv/bin/activate`, then:

```bash
python -m pip install numpy==1.26.4 pyedflib==0.1.42 xlrd==2.0.2 PyYAML==6.0.3
python tools/verify_public_metrics.py
python tools/verify_exploratory_metrics.py
python -m unittest discover -s tests -v
python tools/check_publication.py
```

Expected: both metric replays report PASS for the development, Sleepyland/YASA and exploratory A/B pooled confusion matrices; the current public suite contains **99 synthetic tests**. Publication validation checks tracked paths, source hashes and links. No GPU, EDF, authentication or private model is needed. The [hosted workflow](../.github/workflows/verify.yml) ran these commands successfully on [commit 76b4452](https://github.com/ladekarl1234-commits/physiosleep/actions/runs/36096920665). The [older dated CI receipt](../evidence/hosted-ci.json) covers an earlier 92-test commit; do not treat it as a receipt for the current tree. Public checks do not reproduce protected training or participant-level bootstrap intervals.

## Rebuild the pictures

```bash
python -m pip install matplotlib==3.11.2
python tools/build_research_figures.py --from-summary reports/publication-figures/summary.json
python tools/plot_score_extension.py reports/score-extension/summary.json reports/score-extension/08_score_extension
python tools/build_judge_figures.py
python tools/build_evaluation_overview.py
python tools/build_comparison_figures.py
```

Use one OpenMP/BLAS thread for plots. `build_judge_figures.py` builds the pipeline, all-nine-system local comparison and literature context. `build_evaluation_overview.py` builds the Dev/train → Audit 1 → Audit 2 chart from sanitized public aggregates. `build_comparison_figures.py` builds the control-relative and stage-recall comparisons from sanitized public aggregates and records input/output hashes. Output is PNG and SVG. Numeric data are authoritative if platform fonts change image bytes; changes to hash-bound artifacts need review and deliberate manifest updates.

## Replay the generated hardware cases

The [desk evidence receipt](../evidence/hardware-desk-replay.json) identifies historical private outputs and a public numeric coefficient file. Install NumPy 1.26.4 and SciPy 1.13.1 in a separate Python 3.11 environment; use Node 24.16.0 with OpenSSL 3.5.6. From the repository root run:

```bash
python -B -m tools.verify_hardware_desk
```

The command checks source, runtime and coefficient hashes, generates six packet/clock fixtures, executes the original numerical integration in a disposable directory and compares the resulting masks, counts, transport states and waveform errors with the historical aggregate. The original private archival-manifest presence check is replaced by explicit public source/coefficient checks; the numerical qualification function itself is unchanged. This is generated-data replay, not physical hardware, recorder security, EDF inference or clinical validation. The historical 15/21/21 individual filter, clock and packet groups are identified by hash but are not all rerun by this command.

## Repository map

| Directory | Ownership |
|---|---|
| `sleepedf/` | Data/timing contracts, splits, adapters, predictions, evaluation, gates and summaries |
| `tests/` | Synthetic fixtures and negative contract checks; no patient data |
| `tools/` | Aggregate replay, figure generation, publication checks and selected diagnostic helpers |
| `requirements/` | Pinned descriptions for isolated research runtimes |
| `evidence/` | Aggregate metrics, registry, simulated reviews and public-safe receipts |
| `reports/` | Figure inputs and generated scientific PNG/SVG output |
| `literature/` | Workbook transcription, row identities, source URLs and CSV exports |
| `docs/` | Pipeline, jobs, results, judge route, scientific report and ADRs |

## What a clone cannot do by itself

The source is inspectable, but **exact training is not a one-command public replay**. Original EDFs, participant-linked split/truth manifests, prepared data, vendor sources/weights and private checkpoints are excluded. Training modules intentionally reject absent manifests, source identities or incompatible environments. Do not fake manifests or disable checks to make a command appear to work.

For authorized research, obtain the dataset under applicable terms, verify publisher checksums, resolve model-specific permissions, restore the exact controlled protocol/source artifacts, create isolated runtimes and use the validated preparation/training entrypoints. The [reproduction guide](REPRODUCIBILITY.md) documents this boundary; [pipeline](PIPELINE.md) and [baseline registry](BASELINES.md) identify modules and dependencies. Local commands in the job ledger describe observed work; they require inputs absent from this clone.

## Making a scientific change

Preserve original signals and old runs. Group every person's nights. Version source/config/preprocessing changes, fit learned components only on training people, emit complete grids and evaluate with the common mask. Add a focused synthetic failure case when changing those contracts. Record failures and distinguish software correctness from model adequacy. Do not consume audits to debug or select models.

## Common failures

| Symptom | Meaning / action |
|---|---|
| Missing dependency | Activate the intended environment and install the four pinned public packages above |
| Missing model manifest or private path | Protected-training input is absent; do not bypass integrity checks |
| Hash mismatch | Check whether source, output or line endings changed; restore or explicitly version before proceeding |
| Plot font differences | Inspect numeric inputs and layout; do not promise identical rendering across platforms |
| Passing public checks | These validate software and aggregate arithmetic; inspect the exploratory audit report separately for measured model performance |

No license to third-party models or original data follows from this guide. See [RIGHTS](../RIGHTS.md).
