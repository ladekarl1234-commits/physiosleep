"""Synthetic qualification of a nominal-clock conversion design, not a device adapter.

Run only generated fixtures. No EDF reader, model, labels or acquisition interface.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import firwin, freqz, upfirdn


def convert_design(x, valid, h):
    """Nominal 250-to-100 Hz design; unsupported samples stay NaN with a mask."""
    x = np.asarray(x, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if x.ndim != 1 or valid.shape != x.shape or not np.isfinite(x[valid]).all():
        raise ValueError('Expected finite valid samples on one nominal 250-Hz grid')
    count = len(x) * 2 // 5
    half = (len(h) - 1) // 2
    if half % 5:
        raise ValueError('Design delay must align with the output grid')
    y = upfirdn(2 * h, np.where(valid, x, 0.0), up=2, down=5)[half // 5:half // 5 + count]
    center = np.arange(count, dtype=np.int64) * 5
    low = -np.floor_divide(-(center - half), 2)
    high = np.floor_divide(center + half, 2)
    inside = (low >= 0) & (high < len(x))
    missing = np.concatenate(([0], np.cumsum(~valid)))
    covered = missing[np.clip(high + 1, 0, len(x))] == missing[np.clip(low, 0, len(x))]
    output_valid = inside & covered
    y[~output_valid] = np.nan
    return y, output_valid


def main():
    if scipy.__version__ != '1.13.1' or np.__version__ != '1.26.4':
        raise RuntimeError('Requalification requires a new dependency identity')
    h = firwin(201, 42.5, window=('kaiser', 8.6), fs=500, scale=True)
    f, response = freqz(h, worN=262144, fs=500)
    pass_error = float(np.max(np.abs(20 * np.log10(np.abs(response[f <= 35])))))
    stop_peak = float(np.max(20 * np.log10(np.maximum(np.abs(response[f >= 50]), 1e-300))))
    assert pass_error <= 0.05 and stop_peak <= -80
    assert np.allclose(h, h[::-1], rtol=0, atol=1e-16)
    checks = ['passband_target', 'stopband_target', 'symmetric_filter']
    n = 250 * 12
    t = np.arange(n) / 250
    valid = np.ones(n, dtype=bool)
    constant, mask = convert_design(np.ones(n), valid, h)
    assert mask[:20].sum() == 0 and mask[-20:].sum() == 0 and mask[20:-20].all()
    assert np.max(np.abs(constant[mask] - 1)) < 1e-5
    checks += ['dc_preserved', 'unsupported_edges_flagged']
    tones = []
    for hz in (0.3, 1, 15, 30, 35, 50, 60, 100):
        x = 100 * np.sin(2 * np.pi * hz * t + 0.3)
        y, mask = convert_design(x, valid, h)
        target_t = np.arange(len(y)) / 100
        if hz <= 35:
            basis = np.column_stack((np.sin(2*np.pi*hz*target_t[mask] + 0.3),
                                     np.cos(2*np.pi*hz*target_t[mask] + 0.3),
                                     np.ones(mask.sum())))
            fit = np.linalg.lstsq(basis, y[mask], rcond=None)[0]
            gain = float(np.hypot(fit[0], fit[1]) / 100)
            phase = float(np.arctan2(fit[1], fit[0]))
            assert abs(gain-1) < 0.001 and abs(phase) < 0.001
            tones.append({'input_hz':hz, 'gain':gain, 'phase_radians':phase})
        else:
            peak = float(np.max(np.abs(y[mask])) / 100)
            assert peak < 1e-4
            tones.append({'input_hz':hz, 'maximum_output_to_input_peak':peak})
    checks += ['passband_tone_amplitude_and_phase', 'out_of_band_tone_rejection']
    x = np.random.default_rng(20260924).normal(size=n)
    y, mask = convert_design(x, valid, h)
    reference = []
    for m in np.flatnonzero(mask):
        # Direct defining sum, independent of SciPy's polyphase implementation.
        total = 0.0
        for source in range(max(0, (5*m-100)//2), min(n, (5*m+100)//2+2)):
            k = 100 + 5*m - 2*source
            if 0 <= k < len(h):
                total += 2*h[k]*x[source]
        reference.append(total)
    direct_error = float(np.max(np.abs(y[mask] - reference)))
    assert direct_error < 1e-12
    impulse = np.zeros(n); impulse[1500] = 1
    yi, mi = convert_design(impulse, valid, h)
    assert np.nanargmax(yi) == 600
    checks += ['direct_sum_matches_polyphase', 'impulse_retains_time_origin']
    # One missing input at index 1500 affects exactly outputs 580..620 inclusive.
    gap_mask = valid.copy(); gap_mask[1500] = False
    gap_x = x.copy(); gap_x[1500] = np.nan
    yg, mg = convert_design(gap_x, gap_mask, h)
    assert np.array_equal(np.flatnonzero(mask & ~mg), np.arange(580,621))
    assert np.all(np.isnan(yg[580:621])) and np.array_equal(yg[mg], y[mg])
    checks += ['gap_support_expanded_without_time_compression', 'unaffected_samples_unchanged']
    flat, mf = convert_design(np.zeros(n), valid, h)
    assert np.all(flat[mf] == 0)
    short, ms = convert_design(np.ones(40), np.ones(40, dtype=bool), h)
    assert len(short) == 16 and not ms.any()
    partial, mp = convert_design(np.ones(251), np.ones(251, dtype=bool), h)
    assert len(partial) == 100  # Four milliseconds remain outside full target cells.
    # One-second captured guards permit a predeclared 60-second observation.
    guard, gm = convert_design(np.ones(250*62), np.ones(250*62, dtype=bool), h)
    assert len(guard[100:6100]) == 6000 and gm[100:6100].all()
    assert len(guard[100:6100]) // 3000 == 2
    checks += ['flat_signal', 'short_capture_unavailable', 'partial_tail_not_promoted', 'predeclared_guarded_epoch_window']
    out = Path(__file__).resolve().parents[1] / 'reports/hardware-filter-design-v1'
    out.mkdir(exist_ok=True)
    coeff = out / 'coefficients.json'
    result = out / 'result.json'
    if coeff.exists() or result.exists():
        raise FileExistsError('Design evidence is immutable; use a new version')
    coeff.write_text(json.dumps({'design':'nominal-250-to-100-kaiser201', 'up':2,'down':5,
        'design_rate_hz':500,'cutoff_hz':42.5,'beta':8.6,'coefficients':h.tolist()},indent=2,allow_nan=False),encoding='utf-8')
    value = {'status':'SYNTHETIC_DESIGN_CHECKS_PASSED', 'created_at':datetime.now(timezone.utc).isoformat(),
        'scope':'generated signals at exactly nominal clock rates; no hardware or model execution',
        'device_qualified':False,'product_route_enabled':False,'scipy':scipy.__version__,'numpy':np.__version__,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'coefficient_file_sha256':hashlib.sha256(coeff.read_bytes()).hexdigest(),
        'coefficient_float64_little_endian_sha256':hashlib.sha256(h.astype('<f8').tobytes()).hexdigest(),
        'passband_max_abs_dB':pass_error,'stopband_peak_dB':stop_peak,
        'frequency_grid_spacing_hz':float(f[1]-f[0]),'frequency_response_is_grid_evaluated':True,
        'direct_sum_max_abs_error':direct_error,'tone_checks':tones,'checks':checks,
        'clock_correction_qualified':False,'analog_frontend_qualified':False,
        'hardware_montage_equivalence_qualified':False}
    result.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':value['status'],'checks':len(checks),'result':str(result),
                      'passband_max_abs_dB':pass_error,'stopband_peak_dB':stop_peak}))


if __name__ == '__main__':
    main()
