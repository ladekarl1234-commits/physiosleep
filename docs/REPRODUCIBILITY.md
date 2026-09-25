# Reproduction and verification guide

## Public, offline checks

Use Python 3.11. The synthetic code checks use NumPy 1.26.4, pyEDFlib 0.1.42, xlrd 2.0.2 and PyYAML 6.0.3. Create an isolated environment, install the listed packages and run from the repository root:

```bash
python -m venv .venv
# Windows: .venv/Scripts/python.exe; POSIX: .venv/bin/python
python -m pip install numpy==1.26.4 pyedflib==0.1.42 xlrd==2.0.2 PyYAML==6.0.3
python tools/verify_public_metrics.py
python tools/verify_exploratory_metrics.py
python -m unittest discover -s tests -v
```

Activate the environment or replace `python` with its executable. Tests construct synthetic local fixtures; they do not need original EDFs, credentials, network, trained models or GPU. The tested surface covers source identities, timing, participant splits, full-grid predictions, exact metrics, gate rejection, audit access, temporal decoding, precision sampling and sleep summaries. It is not the entire original workspace suite.

For figures, install `numpy==1.26.4` and `matplotlib==3.11.2`, then run `python tools/build_research_figures.py --from-summary reports/publication-figures/summary.json`. This writes the same seven PNG/SVG figures to `reports/publication-figures/`; use one BLAS/OpenMP thread. The exact Matplotlib version is recorded in the verification receipt. Numeric output remains the authority if a different font/rendering platform changes image bytes. Numeric replay recalculates accuracy, kappa and Macro-F1 from published confusion counts rather than trusting printed scores. Distribution plots require the supplied non-identifying aggregate figure data. The figures do not reproduce training.

## Original-data research

Acquire [Sleep-EDF Expanded 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/) under its applicable terms. Obtain [the published checksum manifest](https://physionet.org/files/sleep-edfx/1.0.0/SHA256SUMS.txt), hash the download and verify every file against it before extraction or modeling. Keep originals immutable. The local project used 197 PSG/Hypnogram pairs; inspect channels, native sample rates, calibration, participant identity and exact annotation timing before projecting epochs.

This public snapshot intentionally excludes the participant-linked frozen manifests, truth store, private checkpoints and run trees. Consequently it cannot reproduce the exact full training run from a single clone alone. Such a replay requires those access-controlled artifacts, the original source-hash chain and authorized data access. A new split on the public dataset is a new experiment, not a replay or independent confirmation of this project. Do not claim fresh-host training reproduction from the synthetic test suite.

The supplied `requirements/research.hashed.txt` pins the original scientific environment; install with `python -m pip install --require-hashes -r requirements/research.hashed.txt` into a separate Python 3.11 environment. Preserve the original `uv.lock` audit setup. Legacy PyTorch/TensorFlow baselines require separate runtimes and source pins in the registry; they are not all installable in this environment. GPU resource requirements must be profiled rather than inferred from code availability.

Original modeling modules in `sleepedf/` are included for inspection. Their CLIs validate bound manifests and refuse missing private inputs. Prediction takes signal inputs and declared channels without reference Hypnograms; evaluator truth is separate. Each training job must hold the single local lease, initially four CPU threads and 10 GiB combined RSS, with at least 4 GiB free RAM and 20 GiB disk. Save optimizer/RNG/sampler state for native deep models. Pauses are not convergence.

## Confirmation sequence and retired cohorts

On 25 September 2026 the owner authorized immediate exploratory evaluation of both original audit cohorts; see [ADR 0016](adr/0016-exploratory-audits.md). Those cohorts are retired from confirmation. The sequence below describes the unchanged scientific requirements for a future protocol using **fresh confirmation participants**, not permission to reopen the original A/B as untouched tests.

1. Resolve all required baseline rights, lineage and native-route adequacy.
2. Complete development fits and registered tuning/finalist seeds, all-night grouped.
3. Meet the development readiness and precision planning requirements.
4. Freeze candidates, comparators, checkpoints, masks and confirmation manifests.
5. Open Audit A once through the evaluator; independently recompute the unchanged gate.
6. Only after A passes, finalize the reduced model and score charter on the original development people, then freeze and evaluate B.

No public replay command opens protected participant data. The repository does not include released weights or silently download data. See the [ADR index](adr/README.md) and [baseline gaps](BASELINES.md).

## Completed release checks

A fresh local clone passed **92 synthetic tests**, exact aggregate arithmetic, and the tracked-file/link/source-hash scan in a separately created minimal Python 3.11 environment. The initial clone exposed Windows line-ending conversion; repository attributes now preserve exact source bytes. The corrected fresh clone passed all checks. See [the receipt](../evidence/clean-clone-verification.json).

This is same-host verification, not independent external full-training replay. A pinned GitHub Actions workflow repeats public checks on a hosted runner after publication; its live status remains separate from this local receipt.

The eighth figure replays the aggregate score extension: `python tools/plot_score_extension.py reports/score-extension/summary.json reports/score-extension/08_score_extension`. Its independent private-artifact verification receipt is published, but aggregate replay cannot independently reconstruct participant bootstrap draws without protected participant-linked records.
