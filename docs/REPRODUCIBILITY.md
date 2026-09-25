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

## Generated hardware replay

The [public desk evidence](../evidence/hardware-desk-replay.json) records source and coefficient identities, six generated integration cases, and hashes of the separate historical filter/clock/packet results and independent review. With Python 3.11, NumPy 1.26.4, SciPy 1.13.1, Node 24.16.0 and OpenSSL 3.5.6, run `python -B -m tools.verify_hardware_desk` from the repository root. The command regenerates authenticated synthetic packets and replays the six packet-to-clock numerical cases in a temporary directory. It checks transport failures, valid-sample counts, epoch support and waveform error. It substitutes explicit public source/coefficient hashes for the private archival-manifest presence guard; it does not replay every historical 15/21/21 filter/clock/packet test group. No physical instrument or sleep-stage model participates.

## Original-data research

Acquire [Sleep-EDF Expanded 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/) under its applicable terms. Obtain [the published checksum manifest](https://physionet.org/files/sleep-edfx/1.0.0/SHA256SUMS.txt), hash the download and verify every file against it before extraction or modeling. Keep originals immutable. The local project used 197 PSG/Hypnogram pairs; inspect channels, native sample rates, calibration, participant identity and exact annotation timing before projecting epochs.

A separate read-only [25 September content crosswalk](../evidence/publisher-content-crosswalk-20260925.json) fetched that publisher manifest anew, found it byte-identical to the local 39,799-byte manifest, and freshly hashed all **398/398** listed files (**8,715,189,781** bytes). All **394/394 EDFs** matched both publisher checksums and the stored hashes of the **197 evaluated pairs**; no file was missing or mismatched. The exact source script and a receipt with executed arguments, exit status and aggregate results are retained privately with hashes in the public summary. To reproduce the *local-manifest* part from an authorized clone with the dataset present, run `python -m sleepedf audit --data-root <local-dataset-root> --files-only --verify-checksums --max-hash-bytes 9000000000 --output <private-report.json>`; verify the local manifest's bytes separately against the freshly obtained publisher URL. Do not publish the resulting per-file report. This checks source bytes, not annotation correctness or clinical validity, and the publisher HTTPS manifest was not digitally signed for this review.

A second, read-only [reader and annotation crosscheck](../evidence/reader-annotation-crosscheck-20260925.json) covered all **197 PSG/Hypnogram pairs**. It independently decoded **28,529 EDF+ annotation intervals** and compared each record's projected full epoch grid, five-class labels, validity mask and onset positions against the protected evaluator truth: zero differences. The seven Hypnograms rejected by pyEDFlib were read through MNE 1.8.0; the other 190 used pyEDFlib 0.1.42. The public JSON is an aggregate, with hashes of the private source and receipt; it contains no participant-level truth. Replaying it requires authorized access to those files, the local Sleep-EDF source bytes and the separate research environment. The crosscheck does not re-read PSG waveforms or establish calibrated-signal equivalence, model validity or independent confirmation.

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
