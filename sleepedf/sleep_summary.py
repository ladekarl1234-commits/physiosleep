"""Observation-window sleep arithmetic with explicit coverage and timing inputs."""
from __future__ import annotations

import math
import numpy as np
from .contracts import content_id


def _number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def summarize(stages, *, window_start_seconds: float, window_end_seconds: float,
              window_provenance: str, epoch_index=None, valid_mask=None,
              independent_night_boundaries: bool = False,
              attempt_to_sleep_seconds: float | None = None,
              attempt_provenance: str | None = None,
              time_in_bed: tuple[float, float] | None = None,
              time_in_bed_provenance: str | None = None) -> dict:
    """Summarize fixed 30-second stages; gaps and uncovered tails remain missing."""
    labels = np.asarray(stages)
    if labels.ndim != 1 or labels.dtype.kind not in 'iu' or np.any((labels < -1) | (labels > 4)):
        raise ValueError('stages must be an integer vector using -1 and classes 0..4')
    indices = np.arange(len(labels), dtype=np.int64) if epoch_index is None else np.asarray(epoch_index)
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(indices) != len(labels)
            or np.any(indices < 0) or np.any(indices > np.iinfo(np.int64).max)
            or np.any(np.diff(indices.astype(np.int64)) <= 0)):
        raise ValueError('epoch_index must be nonnegative, unique, increasing original indices')
    valid = labels >= 0 if valid_mask is None else np.asarray(valid_mask)
    if valid.dtype.kind != 'b' or valid.shape != labels.shape or np.any(valid & (labels < 0)):
        raise ValueError('valid_mask must be a matching boolean vector with valid class values')
    start = _number(window_start_seconds, 'window_start_seconds')
    end = _number(window_end_seconds, 'window_end_seconds')
    if start < 0 or end <= start:
        raise ValueError('observation window must be a positive half-open PSG-relative interval')
    if not isinstance(window_provenance, str) or not window_provenance.strip():
        raise ValueError('observation-window provenance is required')
    if type(independent_night_boundaries) is not bool:
        raise ValueError('independent_night_boundaries must be an explicit boolean')
    left = indices.astype(np.float64) * 30.
    right = left + 30.
    duration = np.where((left >= start) & (right <= end), 30., 0.)
    covered = float(duration[valid].sum())
    total = end - start
    missing = max(0., total - covered)
    complete = covered == total
    sleep = valid & (labels > 0) & (duration > 0)
    sleep_seconds = float(duration[sleep].sum())
    reasons = {}
    result = {
        'schema_version': '1.0', 'summary_definition': 'physiosleep-window-summary-v1',
        'scope': 'conditional_observation_window',
        'window': {'start_seconds': start, 'end_seconds': end, 'provenance': window_provenance,
                   'independent_night_boundaries': independent_night_boundaries},
        'coverage': {'window_seconds': total, 'covered_seconds': covered, 'missing_seconds': missing,
                     'fraction': covered / total, 'complete': complete},
        'observed_tst_minutes': sleep_seconds / 60.,
        'tst_minutes': sleep_seconds / 60. if complete else None,
        'spt_minutes': None, 'waso_minutes': None, 'sleep_period_start_seconds': None,
        'sleep_period_end_seconds': None, 'continuity_fraction': None,
        'sleep_touches_window_start': False if complete else None,
        'sleep_touches_window_end': False if complete else None,
        'awakening_count': None, 'sleep_stage_fractions': None,
        'sol_minutes': None, 'se_percent': None, 'time_in_bed_minutes': None,
        'unavailable_reasons': reasons,
    }
    if not complete:
        reasons.update({key: 'insufficient_coverage' for key in
                        ('tst_minutes', 'spt_minutes', 'waso_minutes', 'continuity_fraction',
                         'sleep_period_start_seconds', 'sleep_period_end_seconds',
                         'awakening_count', 'sleep_stage_fractions')})
    elif not np.any(sleep):
        reasons.update({key: 'no_sleep' for key in
                        ('spt_minutes', 'waso_minutes', 'continuity_fraction',
                         'sleep_period_start_seconds', 'sleep_period_end_seconds',
                         'awakening_count', 'sleep_stage_fractions')})
    else:
        first, last = float(left[sleep][0]), float(right[sleep][-1])
        inside = np.maximum(np.minimum(right, last) - np.maximum(left, first), 0.)
        waso = float(inside[valid & (labels == 0)].sum())
        wake = valid & (labels == 0) & (inside > 0)
        positions = np.flatnonzero(wake)
        awakenings = int(sum(i == 0 or positions[i] != positions[i - 1] + 1 or
                              indices[positions[i]] != indices[positions[i - 1]] + 1
                              for i in range(len(positions))))
        result.update(spt_minutes=(last - first) / 60., waso_minutes=waso / 60.,
                      sleep_period_start_seconds=first, sleep_period_end_seconds=last,
                      sleep_touches_window_start=first == start,
                      sleep_touches_window_end=last == end,
                      continuity_fraction=sleep_seconds / (last - first), awakening_count=awakenings,
                      sleep_stage_fractions={name: float(duration[valid & (labels == stage)].sum()) / sleep_seconds
                                             for stage, name in ((1, 'N1'), (2, 'N2'), (3, 'N3'), (4, 'REM'))})
    if attempt_to_sleep_seconds is None:
        if attempt_provenance is not None:
            raise ValueError('attempt provenance requires an attempt-to-sleep time')
        reasons['sol_minutes'] = 'missing_attempt_to_sleep_anchor'
    else:
        anchor = _number(attempt_to_sleep_seconds, 'attempt_to_sleep_seconds')
        if not isinstance(attempt_provenance, str) or not attempt_provenance.strip():
            raise ValueError('independent attempt-to-sleep provenance is required')
        result['attempt_to_sleep'] = {'seconds': anchor, 'provenance': attempt_provenance}
        if anchor < start or anchor >= end:
            reasons['sol_minutes'] = 'attempt_anchor_outside_observation_window'
        elif anchor % 30 != 0:
            reasons['sol_minutes'] = 'attempt_anchor_not_epoch_aligned'
        elif not complete:
            reasons['sol_minutes'] = 'insufficient_coverage'
        elif not np.any(sleep):
            reasons['sol_minutes'] = 'no_sleep'
        else:
            after = sleep & (right > anchor)
            if not np.any(after):
                reasons['sol_minutes'] = 'no_sleep_after_attempt_anchor'
            else:
                result['sol_minutes'] = (max(float(left[after][0]), anchor) - anchor) / 60.
    if time_in_bed is None:
        if time_in_bed_provenance is not None:
            raise ValueError('time-in-bed provenance requires both boundaries')
        reasons['se_percent'] = 'missing_independent_time_in_bed'
        reasons['time_in_bed_minutes'] = 'missing_independent_time_in_bed'
    else:
        if not isinstance(time_in_bed, (tuple, list)) or len(time_in_bed) != 2:
            raise ValueError('time_in_bed must contain start and end seconds')
        bed_start, bed_end = (_number(value, 'time_in_bed') for value in time_in_bed)
        if bed_start < 0 or bed_end <= bed_start:
            raise ValueError('time_in_bed must have a positive duration')
        if not isinstance(time_in_bed_provenance, str) or not time_in_bed_provenance.strip():
            raise ValueError('independent time-in-bed provenance is required')
        result['time_in_bed'] = {'start_seconds': bed_start, 'end_seconds': bed_end,
                                 'provenance': time_in_bed_provenance}
        result['time_in_bed_minutes'] = (bed_end - bed_start) / 60.
        bed_overlap = np.maximum(np.minimum(right, bed_end) - np.maximum(left, bed_start), 0.)
        if bed_start % 30 != 0 or bed_end % 30 != 0:
            reasons['se_percent'] = 'time_in_bed_boundaries_not_epoch_aligned'
        elif bed_start < start or bed_end > end or abs(float(bed_overlap[valid & (duration > 0)].sum()) - (bed_end - bed_start)) > 1e-7:
            reasons['se_percent'] = 'insufficient_time_in_bed_coverage'
        else:
            result['se_percent'] = float(bed_overlap[sleep].sum()) / (bed_end - bed_start) * 100.
    return result


def experimental_score(summary: dict, *, target_minutes: float = 420.,
                       duration_weight: float = .5) -> dict:
    """Two-component research index; defaults are the frozen provisional charter."""
    target = _number(target_minutes, 'target_minutes')
    weight = _number(duration_weight, 'duration_weight')
    if target <= 0 or not 0 < weight < 1:
        raise ValueError('positive duration target and weight strictly between zero and one required')
    coverage = summary['coverage']
    reasons = []
    duration = continuity = score = arithmetic = None
    if not summary['window'].get('provenance'):
        reasons.append('missing_window_boundary')
    if coverage['window_seconds'] < 420 * 60:
        reasons.append('short_observation')
    if not coverage['complete'] or coverage['missing_seconds'] != 0:
        reasons.append('insufficient_coverage')
    else:
        tst = summary['tst_minutes']
        if tst is None or not math.isfinite(tst) or tst < 0:
            raise ValueError('complete coverage requires a finite nonnegative TST')
        duration = min(tst / target, 1.)
        continuity = summary['continuity_fraction']
        if continuity is None:
            if tst != 0:
                raise ValueError('positive complete TST requires continuity')
            reasons.append('no_sleep')
        elif not 0 < continuity <= 1:
            raise ValueError('continuity must lie in (0,1] when sleep exists')
    if not reasons:
        score = 100 * duration ** weight * continuity ** (1 - weight)
        arithmetic = 100 * (weight * duration + (1 - weight) * continuity)
    definition = 'experimental_duration_continuity_psg_window_v0.1'
    parameters = {'formula': '100*D**w*C**(1-w)', 'duration_target_minutes': target,
                  'duration_weight': weight, 'minimum_window_minutes': 420,
                  'required_coverage': 1.0, 'all_wake_composite': None,
                  'partial_epoch_policy': 'only_complete_30_second_epochs'}
    config_id = content_id(parameters)
    sensitivity = target != 420 or weight != .5
    return {
        'score_definition': definition + ('/sensitivity/' + config_id[7:] if sensitivity else ''),
        'score_config_id': config_id, 'parameters': parameters,
        'scope': summary['scope'], 'status': 'available' if score is not None else 'unavailable',
        'score': score, 'duration_component': None if duration is None else 100 * duration,
        'continuity_component': None if continuity is None else 100 * continuity,
        'arithmetic_sensitivity_score': arithmetic, 'duration_target_minutes': target,
        'duration_weight': weight, 'minimum_observation_minutes': 420,
        'whole_night_score': None,
        'whole_night_unavailable_reason': 'outside_provisional_recording_window_charter',
        'clinically_validated': False, 'unavailable_reasons': reasons,
        'sensitivity_configuration': sensitivity,
    }
