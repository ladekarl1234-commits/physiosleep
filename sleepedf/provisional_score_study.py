"""Development-only recording-score comparison under the explicit owner revision."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from math import lcm
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from .contracts import content_id, read_json
from .development_evidence import index_yasa
from .evaluation import macro_f1_fraction, score_aligned_record
from .predictions import load_prediction
from .protocol import development_records, load_development_truth
from .research import atomic_json, compute_lease, file_sha256, utc_now
from .sleep_summary import experimental_score, summarize

RUNS = {'fpz_eeg': 'yasa-eeg-s17-lr1-016e83b2f3',
        'fpz_eeg_eog': 'yasa-eeg_eog-s17-lr1-493ef41680'}
ENDPOINTS = ('score', 'tst_minutes', 'waso_minutes')
SEED, DRAWS = 2026092405, 2000


def _authorization(root: Path) -> dict:
    record = read_json(root / 'research/provisional-phase2-authorization-v1.json')
    if (record['authorization_id'] != content_id({k: v for k, v in record.items() if k != 'authorization_id'})
            or record['audit_access'] is not False or record['scientific_superiority_established'] is not False
            or 'experimental_sleep_summary_and_score' not in record['allowed_work']
            or record['original_protocol_sha256'] != file_sha256(root / 'research/protocol-v1.json')):
        raise ValueError('Explicit provisional development authorization does not verify')
    return record


def _output(root: Path, path: Path) -> Path:
    if path.resolve() != path or not any(path.is_relative_to(root / parent) for parent in ('runs', 'research')):
        raise ValueError('Provisional output must remain in canonical runs or research')
    return path


def _guard(root: Path, started: float) -> None:
    import psutil
    if (time.monotonic() - started > 300 or psutil.Process().memory_info().rss > 2 * 1024**3
            or psutil.virtual_memory().available < 4 * 1024**3 or shutil.disk_usage(root).free < 20 * 1024**3):
        raise RuntimeError('Provisional score study reached its five-minute/2GiB resource bound')


def _eligibility_rows(records: list[dict]) -> dict:
    rows = {}
    for rec in records:
        reasons = []
        if rec['duration_seconds'] < 25200:
            reasons.append('short_observation')
        if rec['n_epochs'] * 30 != rec['duration_seconds']:
            reasons.append('unscored_partial_tail')
        if rec['timing_summary']['valid_epochs'] != rec['n_epochs']:
            reasons.append('reference_invalid_or_unscored_intervals')
        rows[rec['recording_id']] = {'participant_id': rec['participant_id'], 'cohort': rec['cohort'],
            'start_seconds': 0, 'end_seconds': rec['duration_seconds'], 'n_epochs': rec['n_epochs'],
            'source_psg_sha256': rec['psg_sha256'], 'truth_binding': rec['frozen_truth'],
            'structurally_eligible': not reasons, 'reasons': reasons, 'whole_night_eligible': False,
            'window_provenance': 'verified raw PSG acquisition span; not a night or time-in-bed boundary'}
    return rows


def _validate_eligibility_rows(declared: dict, records: list[dict]) -> None:
    if declared != _eligibility_rows(records):
        raise ValueError('Eligibility differs from original raw-window provenance and eligibility rules')


def freeze(root: Path) -> dict:
    root = root.resolve()
    authorization = _authorization(root)
    protocol, split, records = development_records(root)
    config = experimental_score(summarize(np.zeros(840, dtype=np.int8), window_start_seconds=0,
                               window_end_seconds=25200, window_provenance='synthetic parameter identity only'))
    charter = {'artifact_type': 'provisional_recording_score_charter',
               'authorization_id': authorization['authorization_id'],
               'document_sha256': file_sha256(root / 'docs/PROVISIONAL_SCORE_CHARTER.md'),
               'score_definition': config['score_definition'], 'score_config_id': config['score_config_id'],
               'parameters': config['parameters'], 'gate_A': 'NOT_RUN', 'gate_B': 'NOT_RUN',
               'whole_night_eligible': False, 'clinical_validation': False}
    charter['charter_id'] = content_id(charter)
    cp = _output(root, root / 'research/provisional-score-charter-v0.1.json')
    atomic_json(cp, charter, immutable=True)
    rows = _eligibility_rows(records)
    eligibility = {'artifact_type': 'preprediction_provisional_score_eligibility',
        'charter_sha256': file_sha256(cp), 'protocol_hash': protocol['protocol_hash'],
        'split_id': split['split_id'], 'records': rows,
        'all_wake_rule': 'reference Q unavailable; apply before reading predictions',
        'prediction_outputs_used_for_selection': False, 'whole_night_eligible': 0}
    eligibility['eligibility_id'] = content_id(eligibility)
    ep = _output(root, root / 'research/provisional-score-eligibility-v0.1.json')
    atomic_json(ep, eligibility, immutable=True)
    return {'charter_id': charter['charter_id'], 'eligibility_id': eligibility['eligibility_id'],
            'structural_candidates': sum(row['structurally_eligible'] for row in rows.values())}


def weighted_quantile(values, weights, q: float) -> float:
    """Inverse weighted CDF with exact integer or supplied decimal mass."""
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=object)
    if (values.ndim != 1 or values.shape != weights.shape or not len(values)
            or not np.isfinite(values).all() or not 0 <= q <= 1):
        raise ValueError('Finite values, nonnegative nonempty weights and a valid quantile required')
    order = np.argsort(values, kind='stable')
    mass = []
    for weight in weights:
        if isinstance(weight, (int, np.integer)) and not isinstance(weight, (bool, np.bool_)):
            value = int(weight)
        elif isinstance(weight, (float, np.floating)) and np.isfinite(float(weight)):
            value = Fraction(str(float(weight)))
        else:
            raise ValueError('Weights must be finite real numbers')
        if value < 0:
            raise ValueError('Weights must be nonnegative')
        mass.append(value)
    if not any(mass):
        raise ValueError('Weights must have positive total mass')
    total, cumulative = sum(mass), 0
    target = Fraction(str(q))
    for index in order:
        if mass[index] == 0:
            continue
        cumulative += mass[index]
        if cumulative * target.denominator >= total * target.numerator:
            return float(values[index])
    raise ArithmeticError('Positive exact quantile mass did not reach its threshold')


def error_metrics(rows: list[tuple[str, float]], *, draws: int = DRAWS) -> dict:
    if not rows:
        return {'status': 'NO_ELIGIBLE_REFERENCE', 'nights': 0, 'participants': 0}
    participants = sorted({pid for pid, _ in rows})
    values = np.asarray([value for _, value in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Errors must be finite; unavailable candidate values cannot be dropped')
    counts = Counter(pid for pid, _ in rows)
    owner = np.asarray([participants.index(pid) for pid, _ in rows])
    within = np.asarray([1 / counts[pid] for pid, _ in rows])
    multiple = lcm(*counts.values())
    unit_mass = np.asarray([multiple // counts[pid] for pid, _ in rows], dtype=object)
    weights = within / len(participants)
    absolute = np.abs(values)
    mae, bias = float(weights @ absolute), float(weights @ values)
    sd = float(np.sqrt(weights @ (values - bias) ** 2))
    rng = np.random.Generator(np.random.PCG64(SEED))
    selected = []
    for cohort in ('SC', 'ST'):
        indices = np.asarray([i for i, pid in enumerate(participants) if pid.startswith(cohort + ':')])
        if len(indices):
            selected.append(rng.choice(indices, size=(draws, len(indices)), replace=True))
    sampled = np.concatenate(selected, axis=1)
    if sampled.shape[1] != len(participants):
        raise ValueError('Every participant needs a declared SC or ST cohort')
    bootstrap_weights = np.stack([np.bincount(row, minlength=len(participants))[owner] * within /
                                  len(participants) for row in sampled])
    mae_samples, bias_samples = bootstrap_weights @ absolute, bootstrap_weights @ values
    p90_samples = np.asarray([weighted_quantile(absolute,
        [int(value) for value in np.bincount(row, minlength=len(participants))[owner] * unit_mass], .9)
        for row in sampled])
    return {'status': 'DESCRIPTIVE_DEVELOPMENT', 'nights': len(rows), 'participants': len(participants),
            'mae': mae, 'bias': bias, 'p90_absolute_error': weighted_quantile(absolute, list(map(int, unit_mass)), .9),
            'approximate_limits_of_agreement': [bias - 1.96 * sd, bias + 1.96 * sd],
            'limits_assumption': 'normal-error descriptive approximation; not clinical agreement limits',
            'cluster_ci95': {name: np.quantile(sample, [.025, .975]).tolist() for name, sample in
                             (('mae', mae_samples), ('bias', bias_samples), ('p90_absolute_error', p90_samples))},
            'bootstrap': {'draws': draws, 'seed': SEED, 'rng': 'PCG64', 'strata': ['SC', 'ST'],
                          'unit': 'participant_all_eligible_nights', 'scope': 'conditional development only'}}


def _summarize(rec: dict, labels, epoch_index, valid_mask=None) -> dict:
    expected = int(rec['duration_seconds'] // 30)
    if rec['n_epochs'] != expected or not np.array_equal(epoch_index, np.arange(expected)) or len(labels) != expected:
        raise ValueError('Summary input differs from full complete raw-PSG epoch grid')
    result = summarize(labels, epoch_index=epoch_index, valid_mask=valid_mask, window_start_seconds=0,
                       window_end_seconds=rec['duration_seconds'],
                       window_provenance='verified raw PSG acquisition span; not a night or TIB boundary')
    result['experimental_score'] = experimental_score(result)
    return result


def _endpoint(summary: dict, name: str):
    return summary['experimental_score']['score'] if name == 'score' else summary[name]


def _completion(summaries: dict) -> dict:
    incomplete = {name: {endpoint: summary['endpoints'].get(endpoint, {}).get('status', 'MISSING_ENDPOINT')
                        for endpoint in ENDPOINTS
                        if summary['endpoints'].get(endpoint, {}).get('status') != 'DESCRIPTIVE_DEVELOPMENT'}
                  for name, summary in summaries.items()}
    incomplete = {name: endpoints for name, endpoints in incomplete.items() if endpoints}
    return {'status': 'INCOMPLETE_PROVISIONAL_COMPARISON' if incomplete or not summaries
                     else 'PROVISIONAL_DEVELOPMENT_COMPARISON',
            'required_endpoints_complete': bool(summaries) and not incomplete,
            'incomplete_endpoints': incomplete}


def _registered_runs(root: Path, eog_run: Path | None = None) -> dict:
    systems = {name: root / 'runs' / run_dir for name, run_dir in RUNS.items()}
    if eog_run is not None:
        path = eog_run.resolve()
        if not path.is_relative_to(root / 'runs/provisional-eog') or path == root / 'runs/provisional-eog':
            raise ValueError('EOG comparison requires a canonical provisional EOG run directory')
        systems['eog'] = path
    return systems


def _index(root: Path, name: str, run_dir: Path) -> dict:
    if name == 'eog':
        from .eog_evidence import index_eog
        return index_eog(root, run_dir)
    return index_yasa(root, run_dir)


def run(root: Path, *, eog_run: Path | None = None) -> dict:
    root = root.resolve()
    started = time.monotonic()
    systems = _registered_runs(root, eog_run)
    _authorization(root)
    _guard(root, started)
    charter_path = root / 'research/provisional-score-charter-v0.1.json'
    eligibility_path = root / 'research/provisional-score-eligibility-v0.1.json'
    charter, eligibility = read_json(charter_path), read_json(eligibility_path)
    if (charter['charter_id'] != content_id({k: v for k, v in charter.items() if k != 'charter_id'})
            or eligibility['eligibility_id'] != content_id({k: v for k, v in eligibility.items() if k != 'eligibility_id'})
            or eligibility['charter_sha256'] != file_sha256(charter_path)
            or charter['document_sha256'] != file_sha256(root / 'docs/PROVISIONAL_SCORE_CHARTER.md')):
        raise ValueError('Frozen score charter/eligibility has changed')
    protocol, split, records = development_records(root)
    if (eligibility['protocol_hash'] != protocol['protocol_hash'] or eligibility['split_id'] != split['split_id']
            or set(eligibility['records']) != {rec['recording_id'] for rec in records}):
        raise ValueError('Frozen eligibility is not the original development cohort')
    _validate_eligibility_rows(eligibility['records'], records)
    extra_sources = ('sleepedf/eog_evidence.py', 'sleepedf/eog_experiment.py', 'sleepedf/eog_features.py') if eog_run is not None else ()
    config = {'charter_sha256': file_sha256(charter_path), 'eligibility_sha256': file_sha256(eligibility_path),
              'sources': {name: file_sha256(root / name) for name in
                          ('sleepedf/provisional_score_study.py', 'sleepedf/sleep_summary.py',
                           'sleepedf/development_evidence.py', 'sleepedf/evaluation.py',
                           'sleepedf/predictions.py', 'sleepedf/protocol.py', 'sleepedf/contracts.py',
                           'sleepedf/research.py', 'sleepedf/splits.py') + extra_sources},
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__},
              'runs': {name: {'path': str((path / 'result.json').resolve()),
                              'sha256': file_sha256(path / 'result.json')} for name, path in systems.items()},
              'bootstrap_draws': DRAWS, 'bootstrap_seed': SEED}
    base = _output(root, root / 'runs' / ('provisional-score-study-' + content_id(config)[-12:]))
    if (base / 'result.json').exists():
        raise ValueError('This completed study is immutable; inspect its saved result')
    with compute_lease(root, 'provisional-score-study'):
        atomic_json(_output(root, base / 'config.json'), config, immutable=True)
        refs, truths = {}, {}
        for rec in records:
            _guard(root, started)
            rid = rec['recording_id']
            truth = load_development_truth(rec, split)
            truths[rid] = truth
            refs[rid] = _summarize(rec, truth['reference_label'], truth['epoch_index'], truth['valid_mask'])
            if refs[rid]['experimental_score']['score_config_id'] != charter['score_config_id']:
                raise ValueError('Current score implementation differs from frozen mathematical parameters')
        atomic_json(_output(root, base / 'reference-availability.json'), {
            'created_before_prediction_reads': True, 'charter_sha256': config['charter_sha256'],
            'records': {rid: {'score_available': value['experimental_score']['score'] is not None,
                              'reasons': value['experimental_score']['unavailable_reasons']} for rid, value in refs.items()}}, immutable=True)
        rows = {rec['recording_id']: {'participant_id': rec['participant_id'],
                'window_minutes': rec['duration_seconds'] / 60., 'reference': refs[rec['recording_id']],
                'models': {}} for rec in records}
        summaries, binding = {}, {}
        for name, run_dir in systems.items():
            _guard(root, started)
            index = _index(root, name, run_dir)
            if index['result']['sha256'] != config['runs'][name]['sha256']:
                raise ValueError('Saved model result changed after study freeze')
            indexed = {entry['recording_id']: entry for entry in index['records']}
            matrix = np.zeros((5, 5), dtype=np.int64)
            binding[name] = {'index_id': index['index_id'], 'result_sha256': index['result']['sha256'],
                             'prediction_bindings': {rid: entry['prediction'] for rid, entry in indexed.items()}}
            for rec in records:
                _guard(root, started)
                rid = rec['recording_id']
                meta, prediction = load_prediction(Path(indexed[rid]['prediction']['path']))
                if (prediction['recording_id'] != rid or prediction['participant_id'] != rec['participant_id']
                        or meta['payload_sha256'] != indexed[rid]['prediction']['sha256']):
                    raise ValueError('Prediction identity differs from fixed development record')
                confusion, _ = score_aligned_record(truths[rid], prediction)
                matrix += confusion
                rows[rid]['models'][name] = _summarize(rec, prediction['hard_label'], prediction['epoch_index'])
            f1 = float(macro_f1_fraction(matrix))
            original = read_json(run_dir / 'result.json')
            if f1 != original['metrics']['macro_f1']:
                raise ValueError('Original full-grid staging metric no longer reproduces')
            channels = {'fpz_eeg': ['EEG Fpz-Cz'], 'fpz_eeg_eog': ['EEG Fpz-Cz', 'EOG horizontal'],
                        'eog': ['EOG horizontal']}[name]
            summaries[name] = {'staging_macro_f1': f1, 'channel_count': len(channels), 'channels': channels}
        for rec in records:
            fold = next(f for f in split['folds'] if rec['participant_id'] in f['validation'])
            constants = {}
            for endpoint in ENDPOINTS:
                training = [(r['participant_id'], _endpoint(refs[r['recording_id']], endpoint)) for r in records
                            if r['participant_id'] in fold['train'] and _endpoint(refs[r['recording_id']], endpoint) is not None]
                counts = Counter(pid for pid, _ in training)
                constants[endpoint] = weighted_quantile([v for _, v in training], [1 / counts[pid] for pid, _ in training], .5) if training else None
            rows[rec['recording_id']]['train_only_constant'] = {'fold_id': fold['fold_id'], 'values': constants,
                                                             'fitted_participants': sorted(fold['train'])}
        summaries['train_only_constant'] = {'staging_macro_f1': None, 'channel_count': 0,
                                           'description': 'fold-specific participant-balanced reference median; no signal'}
        for name in summaries:
            summaries[name]['endpoints'] = {}
            for endpoint in ENDPOINTS:
                errors, missing = [], []
                for rid, row in rows.items():
                    reference = _endpoint(row['reference'], endpoint)
                    if reference is None:
                        continue
                    predicted = row['train_only_constant']['values'][endpoint] if name == 'train_only_constant' else _endpoint(row['models'][name], endpoint)
                    if predicted is None:
                        missing.append(rid)
                    else:
                        errors.append((row['participant_id'], predicted - reference))
                summaries[name]['endpoints'][endpoint] = ({'status': 'INCOMPLETE_REQUIRED_SCORES', 'missing_recordings': missing}
                                                         if missing else error_metrics(errors))
        reference_scores = [value['experimental_score']['score'] for value in refs.values()
                            if value['experimental_score']['score'] is not None]
        completion = _completion(summaries)
        result = {**completion, 'created_at': utc_now(),
            'config_id': content_id(config), 'charter_id': charter['charter_id'],
            'scope': 'conditional raw-PSG-window computational agreement only; adaptive development cohort',
            'recordings': len(records), 'participants': len(split['participants']['development']),
            'reference_score_available': len(reference_scores), 'whole_night_eligible': 0,
            'reference_score_range': [min(reference_scores), max(reference_scores)] if reference_scores else None,
            'reference_unavailable_reasons': dict(Counter(reason for value in refs.values()
                for reason in value['experimental_score']['unavailable_reasons'])),
            'models': summaries, 'input_bindings': binding, 'records': rows,
            'se_comparison': {'status': 'UNAVAILABLE', 'reason': 'no independent time-in-bed boundaries'},
            'confirmatory': False, 'gate_A': 'NOT_RUN', 'gate_B': 'NOT_RUN', 'clinically_validated': False,
            'elapsed_seconds': time.monotonic() - started}
        if config['sources'] != {name: file_sha256(root / name) for name in config['sources']}:
            raise ValueError('Study source changed during execution')
        if (file_sha256(charter_path) != config['charter_sha256']
                or file_sha256(eligibility_path) != config['eligibility_sha256']):
            raise ValueError('Frozen study definitions changed during execution')
        for name, run_dir in systems.items():
            if _index(root, name, run_dir)['index_id'] != binding[name]['index_id']:
                raise ValueError('Prediction, checkpoint or ancestry bytes changed during study')
        _guard(root, started)
        atomic_json(_output(root, base / 'result.json'), result, immutable=True)
    return {**completion, 'result_path': str(base / 'result.json'),
            'reference_score_available': len(reference_scores), 'models': summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    parser.add_argument('--provisional', action='store_true', required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--eog-run', type=Path)
    args = parser.parse_args()
    from .contracts import json_text
    if args.action == 'freeze' and args.eog_run is not None:
        parser.error('--eog-run is only supported for run')
    result = freeze(args.root) if args.action == 'freeze' else run(args.root, eog_run=args.eog_run)
    print(json_text(result))
    if args.action == 'run' and not result['required_endpoints_complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
