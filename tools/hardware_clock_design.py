"""Generated-fixture clock conversion study; no device or model input route."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import freqz


RADIUS = 50
BETA = 8.6
CUTOFF = 42.5
ROUND_SECONDS = 1e-8


@dataclass(frozen=True)
class ClockFit:
    offset: float
    slope: float
    error_seconds: float
    slope_lower: float
    device_start: float
    device_end: float
    maximum_residual: float


def fit_clock(device_seconds, reference_seconds, event_error_seconds):
    """Conditional bound under affine time and independently bounded event error."""
    d = np.asarray(device_seconds, dtype=np.float64)
    r = np.asarray(reference_seconds, dtype=np.float64)
    eps = float(event_error_seconds)
    if (d.ndim != 1 or r.shape != d.shape or not 3 <= len(d) <= 4096
            or not np.isfinite(d).all() or not np.isfinite(r).all()
            or not np.isfinite(eps) or eps < 0 or eps > .01
            or np.any(np.diff(d) <= 0) or np.any(np.diff(r) <= 0)
            or d[0] < 0 or d[-1] > 86400 or np.max(np.abs(r)) > 172800):
        raise ValueError('Expected ordered local synchronization events and finite bounds')
    centered = d - d.mean()
    spread = float(centered @ centered)
    if not np.isfinite(spread) or spread <= 0:
        raise ValueError('Synchronization span cannot support a finite affine fit')
    slope = float(centered @ (r-r.mean()) / spread)
    offset = float(r.mean() - slope*d.mean())
    residual = r - (offset+slope*d)
    if not np.isfinite(slope) or not np.isfinite(offset) or not np.isfinite(residual).all():
        raise ValueError('Nonfinite affine estimate')
    maximum = float(np.max(np.abs(residual)))
    if maximum > eps + ROUND_SECONDS:
        raise ValueError('Synchronization events do not meet the declared affine fit bound')
    e0, e1 = eps + abs(residual[[0, -1]]) + ROUND_SECONDS
    error = float(max(e0, e1))
    slope_error = float((e0+e1)/(d[-1]-d[0]))
    if (not np.isfinite(error) or not np.isfinite(slope_error)
            or error > .01 or slope-slope_error < .999
            or slope+slope_error > 1.001):
        raise ValueError('Clock uncertainty or rate falls outside this design envelope')
    return ClockFit(offset, slope, error, slope-slope_error,
                    float(d[0]), float(d[-1]), maximum)


def weights(positions, slope):
    """Continuous Kaiser-windowed sinc; normalize every fractional phase to DC=1."""
    u = np.asarray(positions, dtype=np.float64).reshape(-1)
    if (not np.isfinite(u).all() or np.any(np.abs(u) > 1e8)
            or not np.isfinite(slope) or not .999 <= slope <= 1.001):
        raise ValueError('Kernel positions or slope outside the synthetic design range')
    native = np.floor(u).astype(np.int64)[:, None] + np.arange(-50, 52)
    distance = u[:, None]-native
    inside = np.abs(distance) <= RADIUS
    scaled = 2*CUTOFF*slope/250
    window = np.i0(BETA*np.sqrt(np.maximum(0, 1-(distance/RADIUS)**2)))/np.i0(BETA)
    kernel = np.where(inside, scaled*np.sinc(scaled*distance)*window, 0.0)
    denominator = kernel.sum(axis=1, keepdims=True)
    if not np.isfinite(denominator).all() or np.any(np.abs(denominator) < .5):
        raise ValueError('Unsafe fractional-phase DC normalization')
    kernel /= denominator
    return native, kernel


def convert(x, valid, clock, origin, count, first_sample=0):
    """Evaluate a declared 100-Hz reference grid; unsupported positions remain NaN."""
    x = np.asarray(x, dtype=np.float64)
    valid = np.asarray(valid)
    if (x.ndim != 1 or valid.shape != x.shape or valid.dtype != np.bool_
            or not np.isfinite(x[valid]).all() or type(count) is not int
            or not 0 <= count <= 100000 or type(first_sample) is not int
            or not 0 <= first_sample <= 250*86400 or not np.isfinite(origin)
            or abs(origin) > 172800 or not isinstance(clock, ClockFit)):
        raise ValueError('Expected bounded calibrated fixture, boolean mask and local time grid')
    if (not all(np.isfinite(v) for v in asdict(clock).values())
            or not .999 <= clock.slope_lower <= clock.slope <= 1.001
            or not 0 <= clock.error_seconds <= .01
            or not 0 <= clock.device_start < clock.device_end <= 86400):
        raise ValueError('Invalid clock design parameters')
    y = np.full(count, np.nan)
    supported = np.zeros(count, dtype=bool)
    missing = np.concatenate(([0], np.cumsum(~valid)))
    clean = np.where(valid, x, 0.0)
    displacement = 250*clock.error_seconds/clock.slope_lower
    for start in range(0, count, 512):
        target = origin + np.arange(start, min(start+512, count))/100
        u = 250*(target-clock.offset)/clock.slope
        low = np.ceil(u-RADIUS-displacement).astype(np.int64)
        high = np.floor(u+RADIUS+displacement).astype(np.int64)
        lo, hi = low-first_sample, high-first_sample
        covered = (missing[np.clip(hi+1, 0, len(x))]
                   == missing[np.clip(lo, 0, len(x))])
        ok = ((lo >= 0) & (hi < len(x)) & covered
              & (low/250 >= clock.device_start) & (high/250 <= clock.device_end))
        if ok.any():
            native, kernel = weights(u[ok], clock.slope)
            samples = clean[np.clip(native-first_sample, 0, len(x)-1)]
            values = np.sum(samples*kernel, axis=1)
            y[start:start+len(u)][ok] = values
        supported[start:start+len(u)] = ok
    return y, supported


def qualify():
    if scipy.__version__ != '1.13.1' or np.__version__ != '1.26.4':
        raise RuntimeError('Use a new qualification identity for changed dependencies')
    checks = []
    pass_max, stop_max = 0.0, -999.0
    for slope in (.999, .9999, 1., 1.0001, 1.001):
        for phase in np.arange(128)/128:
            _, kernel = weights([phase], slope)
            f, response = freqz(kernel[0], worN=32768, fs=250/slope, include_nyquist=True)
            db = 20*np.log10(np.maximum(np.abs(response), 1e-300))
            pass_max = max(pass_max, float(np.max(np.abs(db[f <= 35]))))
            stop_max = max(stop_max, float(np.max(db[f >= 50])))
            _, endpoints = freqz(kernel[0], worN=np.array([35.,50.]), fs=250/slope)
            pass_max = max(pass_max, float(abs(20*np.log10(abs(endpoints[0])))))
            stop_max = max(stop_max, float(20*np.log10(abs(endpoints[1]))))
    assert pass_max < .05 and stop_max < -80
    checks += ['fractional_phase_passband_grid', 'fractional_phase_stopband_grid']
    anchors = np.array([0., 10800., 21600., 32400., 43200.])
    tone_results = []
    for ppm in (-100., -20., 0., 20., 100.):
        slope = 1+ppm/1e6
        for offset in (-.37, .23):
            clock = fit_clock(anchors, offset+slope*anchors, 0.)
            assert abs(clock.slope-slope) < 1e-14
            assert abs(clock.offset-offset) < 1e-10
            first = 250*43190
            times = offset+slope*(first+np.arange(2500))/250
            origin = offset+slope*43191+.0017
            for hz in (.3, 1., 15., 30., 35., 50., 60., 100.):
                source = 100*np.sin(2*np.pi*hz*times+.3)
                y, ok = convert(source, np.ones(2500, dtype=bool), clock, origin, 700, first)
                assert ok.all()
                target = origin+np.arange(700)/100
                if hz <= 35:
                    expected = 100*np.sin(2*np.pi*hz*target+.3)
                    error = float(np.max(np.abs(y-expected))/100)
                    assert error < .0001
                else:
                    error = float(np.max(np.abs(y))/100)
                    assert error < .0001
                tone_results.append({'ppm':ppm, 'offset_seconds':offset,
                                     'hz':hz, 'relative_peak_error_or_leakage':error})
    checks += ['known_clock_offset_and_slope', '12_hour_position_alignment',
               'reference_grid_passband_tones', 'reference_grid_stopband_tones']
    uncorrected = float((1.00002-1)*43200)
    assert abs(uncorrected-.864) < 1e-10
    checks += ['uncorrected_20ppm_drift_visible']
    jitter = np.array([.0004, -.0006, .0007, -.0004, .0002])
    truth = -.2+1.00002*anchors
    clock = fit_clock(anchors, truth+jitter, .001)
    dense = np.linspace(0, 43200, 10001)
    discrepancy = np.abs((-.2+1.00002*dense)-(clock.offset+clock.slope*dense))
    assert discrepancy.max() <= clock.error_seconds
    checks += ['bounded_event_error_covers_affine_truth']
    clean_clock = fit_clock([0., 6., 12.], [0., 6., 12.], 0.)
    n = 3000
    rng = np.random.default_rng(2026092402)
    x = rng.normal(size=n)
    valid = np.ones(n, dtype=bool)
    y, mask = convert(x, valid, clean_clock, 0., 1200)
    assert not mask[:20].any() and not mask[-20:].any()
    checks += ['edges_remain_unsupported']
    for gap in (1473, 1500, 1527):
        v = valid.copy(); v[gap] = False
        xx = x.copy(); xx[gap] = np.nan
        yg, mg = convert(xx, v, clean_clock, 0., 1200)
        delta = 250*clean_clock.error_seconds/clean_clock.slope_lower
        expected = mask & (np.abs(np.arange(1200)*2.5-gap) > RADIUS+delta)
        assert np.array_equal(mg, expected)
        assert np.array_equal(yg[mg], y[mg]) and np.isnan(yg[~mg]).all()
    checks += ['gaps_preserve_grid_and_full_support', 'unaffected_samples_unchanged']
    uncertain = fit_clock([0., 6., 12.], [.0003, 5.9995, 12.0002], .001)
    v = valid.copy(); v[1500] = False
    _, um = convert(x, v, uncertain, .0017, 1100)
    u = 250*(.0017+np.arange(1100)/100-uncertain.offset)/uncertain.slope
    delta = 250*uncertain.error_seconds/uncertain.slope_lower
    assert not um[np.abs(u-1500) <= 50+delta].any()
    checks += ['uncertainty_expands_gap_support']
    nominal_file = (Path(__file__).resolve().parents[1]/
                    'reports/hardware-filter-design-v1/coefficients.json')
    nominal_coeff = json.loads(nominal_file.read_text())
    h = np.array(nominal_coeff['coefficients'])
    parity = 0.
    for phase in (0., .5):
        native, kernel = weights([phase], 1.)
        indices = (100+2*phase-2*native[0]).astype(int)
        expected = np.zeros(102)
        inside = (indices >= 0) & (indices <= 200)
        expected[inside] = 2*h[indices[inside]]
        expected /= expected.sum()
        parity = max(parity, float(np.max(np.abs(kernel[0]-expected))))
    assert parity < 1e-14
    checks += ['nominal_phase_normalized_kernel_parity']
    for value in (0., 123.4):
        yy, mm = convert(np.full(n, value), valid, clean_clock, 0., 1200)
        assert np.max(np.abs(yy[mm]-value)) < 1e-12
    short, sm = convert(np.ones(20), np.ones(20, dtype=bool), clean_clock, 0., 8)
    assert len(short) == 8 and not sm.any()
    empty, em = convert([], np.array([], dtype=bool), clean_clock, 0., 0)
    assert len(empty) == len(em) == 0
    checks += ['constant_and_flat', 'short_and_empty']
    guard_clock = fit_clock([0.,31.,62.], [0.,31.,62.], 0.)
    yy, mm = convert(np.ones(15500), np.ones(15500, dtype=bool), guard_clock, 1., 6000)
    assert len(yy) == 6000 and mm.all()
    # A caller's original target grid cannot shrink to erase a missing guard.
    yy, mm = convert(np.ones(15000), np.ones(15000, dtype=bool), guard_clock, 1., 6000)
    assert len(yy) == 6000 and not mm[-100:].any()
    checks += ['guarded_two_epoch_grid', 'missing_guard_does_not_shrink_output']
    rejection_cases = [([0,1],[0,1],.001), ([0,1,1],[0,1,2],.001),
        ([0,1,2],[0,2,1],.001), ([0,6,12],[0,6.03,12],.001),
        ([0,6,12],[0,6.1,12.2],0.), ([0,1,2],[0,1,float('nan')],0.),
        ([0,6,12],[0,6,12],-.001), ([0,6,12],[0,6,12],.02),
        ([0,1e-200,2e-200],[0,1e-200,2e-200],0.)]
    for args in rejection_cases:
        try:
            fit_clock(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid synchronization fixture was accepted')
    partial_clock = fit_clock([1.,6.,11.], [1.,6.,11.], 0.)
    _, pm = convert(x, valid, partial_clock, 0., 1200)
    assert not pm[:120].any() and not pm[1081:].any()
    distant_clock = fit_clock([86388,86394,86400],[-172800,-172794,-172788],0.)
    distant, dm = convert(np.ones(100),np.ones(100,dtype=bool),distant_clock,
                          172800.,1,21597000)
    assert len(distant) == 1 and np.isnan(distant[0]) and not dm[0]
    checks += ['invalid_events_resets_nonlinearity_drift_rejected', 'no_anchor_extrapolation',
               'underflow_fit_rejected', 'distant_unsupported_target_preserved']
    return {'status':'SYNTHETIC_AFFINE_CLOCK_CHECKS_PASSED', 'checks':checks,
        'passband_max_abs_dB':pass_max, 'stopband_peak_dB':stop_max,
        'spectral_grid':{'slopes':[.999,.9999,1.,1.0001,1.001],
                         'phases':128,'frequencies_per_phase':32768,
                         'nyquist_included':True,'additional_exact_hz':[35.,50.]},
        'grid_evidence_not_continuous_proof':True, 'tone_checks':tone_results,
        'uncorrected_20ppm_12h_drift_seconds':uncorrected,
        'bounded_jitter_fit':asdict(clock),
        'bounded_jitter_max_observed_error_seconds':float(discrepancy.max()),
        'normalized_nominal_coefficient_max_difference':parity,
        'nominal_coefficient_file_sha256':hashlib.sha256(nominal_file.read_bytes()).hexdigest(),
        'numpy':np.__version__, 'scipy':scipy.__version__,
        'hardware_clock_qualified':False, 'product_route_enabled':False,
        'bound_conditional_on_affine_clock_and_independently_bounded_events':True}


if __name__ == '__main__':
    target = Path(__file__).resolve().parents[1]/'reports/hardware-clock-design-v1/result.json'
    if target.exists():
        raise FileExistsError('Evidence is immutable; use a new qualification version')
    result = qualify()
    result.update(created_at=datetime.now(timezone.utc).isoformat(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target.parent.mkdir(exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','passband_max_abs_dB','stopband_peak_dB')}))
    print(f'{len(result["checks"])} checks; {target}')
