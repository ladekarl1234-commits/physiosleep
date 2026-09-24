"""Integrate only generated authenticated packets with the frozen clock design."""
import base64
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from sleepedf.contracts import json_text, read_json
from tools.hardware_clock_design import convert, fit_clock, weights


ROOT = Path(__file__).resolve().parents[1]
EDGE = 64
NAMES = ('clean_plus', 'clean_minus', 'faults', 'signal_faults', 'missing_guard', 'missing_footer')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer_string(value):
    if not isinstance(value, str) or str(int(value)) != value:
        raise ValueError('Expected canonical integer timestamp or counter')
    return int(value)


def verify_frozen_inputs():
    for name in ('hardware-packet-design-v0.3.json', 'hardware-clock-design-v0.2.json'):
        manifest = read_json(ROOT/'research'/name)
        for item in manifest['artifacts']:
            if sha(ROOT/item['path']) != item['sha256']:
                raise ValueError(f'Frozen input changed: {item["path"]}')


def integrate(fixture):
    if fixture.get('kind') != 'GENERATED_PACKET_CLOCK_FIXTURE_V1':
        raise ValueError('This bridge accepts its generated study fixture only')
    if fixture['node'] != 'v24.16.0' or fixture['openssl'] != '3.5.6':
        raise ValueError('Packet runtime requires separate qualification')
    replay = subprocess.run(['node',str(ROOT/'tools/hardware_packet_fixture.cjs'),'--verify'],
                            input=json.dumps(fixture,allow_nan=False),text=True,capture_output=True,
                            cwd=ROOT,timeout=30)
    if replay.returncode:
        raise ValueError('Authenticated packet replay disagrees with fixture assembly')
    assembly_hash = json.loads(replay.stdout)['assembly_sha256']
    native = fixture['assembled']
    metadata = fixture['metadata_json']
    if (metadata != json.dumps(fixture['config'], separators=(',', ':'), ensure_ascii=False)
            or hashlib.sha256(metadata.encode()).hexdigest() != native['config_sha256']):
        raise ValueError('Calibration metadata hash or representation changed')
    first = integer_string(native['first_sample'])
    count = native['sample_count']
    if type(count) is not int or not 1 <= count <= 100000 or not 0 <= first <= 21600000:
        raise ValueError('Unsupported native counter extent')
    raw = np.asarray(native['counts'])
    valid = np.asarray(native['valid'])
    reasons = np.asarray(native['reasons'])
    if (raw.shape != (count,4) or raw.dtype.kind not in 'iu'
            or np.any(raw < -8388608) or np.any(raw > 8388607)
            or valid.shape != raw.shape or valid.dtype != np.bool_
            or reasons.shape != raw.shape or reasons.dtype.kind not in 'iu'
            or np.any(reasons < 0) or np.any(reasons > 63)
            or not np.array_equal(valid, reasons == 0)):
        raise ValueError('Malformed native calibration/validity arrays')
    physical = np.asarray(native['values_uv'], dtype=np.float64)
    if (physical.shape != raw.shape or not np.isfinite(physical[valid]).all()
            or not np.isnan(physical[~valid]).all()):
        raise ValueError('Invalid values must remain unavailable')
    for ch, calibration in enumerate(fixture['config']['channels']):
        expected = calibration['scale_uv_per_count']*(raw[:,ch]-calibration['offset_count'])
        if not np.array_equal(physical[valid[:,ch],ch], expected[valid[:,ch]]):
            raise ValueError('Calibrated values changed across the language boundary')
    frames = fixture['frames_base64']
    for block in native['accepted_packets']:
        original = base64.b64decode(frames[block['packet_index']], validate=True)
        if hashlib.sha256(original).hexdigest() != block['packet_sha256']:
            raise ValueError('Original packet identity changed')
    origin_ns = integer_string(fixture['observation']['origin_ns'])
    duration_ns = integer_string(fixture['observation']['duration_ns'])
    if not 0 < duration_ns <= 1_000_000_000_000:
        raise ValueError('Unsupported observation duration')
    output_count, subcell_tail = divmod(duration_ns, 10_000_000)
    sync = fixture['synchronization']
    if sync['exact_generated_events'] is not True:
        raise ValueError('This study only qualifies exact generated clock events')
    clock = fit_clock([integer_string(n)/250 for n in sync['counters']],
                      [integer_string(n)/1e9 for n in sync['reference_ns']],
                      sync['event_error_seconds'])
    origin = origin_ns/1e9
    converted = np.full((output_count,4), np.nan)
    converted_valid = np.zeros((output_count,4), dtype=bool)
    for ch in range(4):
        converted[:,ch], converted_valid[:,ch] = convert(physical[:,ch],valid[:,ch],clock,
                                                        origin,output_count,first)
    u = 250*(origin+np.arange(output_count)/100-clock.offset)/clock.slope
    displacement = 250*clock.error_seconds/clock.slope_lower
    low = np.ceil(u-50-displacement).astype(np.int64)
    high = np.floor(u+50+displacement).astype(np.int64)
    output_reasons = np.zeros((output_count,4), dtype=np.uint16)
    for m, (a,b) in enumerate(zip(low, high)):
        inside = (a >= first and b < first+count and a/250 >= clock.device_start
                  and b/250 <= clock.device_end)
        if not inside:
            output_reasons[m] |= EDGE
        lo, hi = max(0,a-first), min(count,b-first+1)
        if lo < hi:
            output_reasons[m] |= np.bitwise_or.reduce(reasons[lo:hi],axis=0).astype(np.uint16)
    if not np.array_equal(converted_valid, output_reasons == 0):
        raise AssertionError('Clock mask and propagated reasons disagree')
    active = [i for i,c in enumerate(fixture['config']['channels']) if c['active']]
    epochs = []
    for index, start_ns in enumerate(range(0,duration_ns,30_000_000_000)):
        end_ns = min(start_ns+30_000_000_000,duration_ns)
        start, end = start_ns//10_000_000, end_ns//10_000_000
        complete = end_ns-start_ns == 30_000_000_000
        supported = bool(complete and active and converted_valid[start:end,active].all())
        epochs.append({'index':index,'onset_ns':start_ns,'end_ns':end_ns,
                       'complete':complete,'active_channels_supported':supported,
                       'reason':'partial_epoch' if not complete else None if supported else 'unsupported_signal'})
    # Transport completeness and signal support are distinct requirements.
    full_epochs = [x for x in epochs if x['complete']]
    ready = (bool(full_epochs) and native['status'] == 'AUTHENTICATED_TIMELINE_COMPLETE'
             and all(x['active_channels_supported'] for x in full_epochs))
    return {'values':converted,'valid':converted_valid,'reasons':output_reasons,
            'authenticated_assembly_sha256':assembly_hash,
            'origin_ns':origin_ns,'duration_ns':duration_ns,'output_count':output_count,
            'subcell_tail_ns':subcell_tail,'epochs':epochs,'active':active,'clock':clock,'u':u,
            'transport_status':native['status'],'engineering_full_epochs_supported':ready}


def qualify(folder):
    verify_frozen_inputs()
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    records, arrays = [], {}
    for name in NAMES:
        fixture_path = folder/f'{name}.json'
        with fixture_path.open('x',encoding='utf-8') as output:
            subprocess.run(['node',str(ROOT/'tools/hardware_packet_fixture.cjs'),name],
                           stdout=output,check=True,cwd=ROOT,timeout=30)
        fixture = read_json(fixture_path)
        result = integrate(fixture)
        arrays[name] = result
        assert result['output_count'] == 6001 and result['subcell_tail_ns'] == 7000000
        assert [x['index'] for x in result['epochs']] == [0,1,2]
        assert [x['complete'] for x in result['epochs']] == [True,True,False]
        assert result['epochs'][2]['end_ns']-result['epochs'][2]['onset_ns'] == 17000000
        assert not result['valid'][:,2:].any() and np.isnan(result['values'][:,2:]).all()
        checks = ['declared_grid_and_partial_tail','disabled_channels_remain_unavailable',
                  'calibration_and_packet_identity_preserved','support_reason_mask_agreement']
        tone_errors = []
        if name.startswith('clean_'):
            assert result['engineering_full_epochs_supported']
            t = np.arange(result['output_count'])/100
            expected = [50*np.sin(2*np.pi*1.3*t+.2)+25*np.sin(2*np.pi*13*t-.1),
                        40*np.cos(2*np.pi*.7*t+.3)+10*np.sin(2*np.pi*8.2*t+.6)]
            phase_l1 = np.empty(len(t))
            for start in range(0,len(t),512):
                _, w = weights(result['u'][start:start+512],result['clock'].slope)
                phase_l1[start:start+len(w)] = np.abs(w).sum(axis=1)
            for ch, amplitude_sum in enumerate((75.,50.)):
                lsb = fixture['config']['channels'][ch]['scale_uv_per_count']
                bound = .5*lsb*phase_l1 + 1e-4*amplitude_sum
                error = np.abs(result['values'][:,ch]-expected[ch])
                assert np.all(error <= bound)
                tone_errors.append({'channel':ch,'maximum_error_uv':float(error.max()),
                                    'maximum_bound_uv':float(bound.max()),
                                    'maximum_kernel_l1':float(phase_l1.max())})
            checks += ['exact_clock_analytic_waveform_with_quantization_bound']
            if name == 'clean_plus':
                partial = copy.deepcopy(fixture)
                partial['observation']['duration_ns'] = '17000000'
                small = integrate(partial)
                assert small['output_count'] == 1 and small['subcell_tail_ns'] == 7000000
                assert not small['engineering_full_epochs_supported']
                for field,value in [('values_uv',999),('valid',False)]:
                    changed = copy.deepcopy(fixture)
                    changed['assembled'][field][1000][0] = value
                    try:
                        integrate(changed)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('Changed calibration or validity accepted')
                changed = copy.deepcopy(fixture)
                changed['assembled']['counts'][1000][0] += 10000
                c = changed['config']['channels'][0]
                changed['assembled']['values_uv'][1000][0] = c['scale_uv_per_count']*(
                    changed['assembled']['counts'][1000][0]-c['offset_count'])
                try:
                    integrate(changed)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Jointly altered counts and physical values accepted')
                changed = copy.deepcopy(fixture)
                changed['assembled']['first_sample'] = str(2**53+1)
                try:
                    integrate(changed)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Out-of-domain clock counter accepted')
                checks += ['no_complete_epoch_cannot_pass','changed_arrays_and_clock_domain_rejected',
                           'counts_and_values_joint_tamper_rejected_by_authenticated_replay']
        else:
            assert not result['engineering_full_epochs_supported']
            clean = arrays['clean_plus']
            assert result['origin_ns'] == clean['origin_ns']
            for ch in range(2):
                mask = result['valid'][:,ch]
                assert np.array_equal(result['values'][mask,ch],clean['values'][mask,ch])
            checks += ['faults_do_not_shorten_grid','unaffected_samples_identical']
            if name == 'missing_footer':
                assert result['valid'][:,:2].all()
                assert result['transport_status'] == 'INCOMPLETE_CAPTURE'
                checks += ['transport_failure_veto_despite_full_signal_support']
            if name == 'signal_faults':
                assert result['transport_status'] == 'AUTHENTICATED_TIMELINE_COMPLETE'
                assert not result['engineering_full_epochs_supported']
                checks += ['complete_transport_does_not_override_signal_failures']
            if name in ('faults','signal_faults'):
                assert not result['epochs'][0]['active_channels_supported']
                assert not result['epochs'][1]['active_channels_supported']
                assert any(result['reasons'][:,0] & 4)
                if name == 'faults':
                    assert any(result['reasons'][:,0] & 1)
                assert any(result['reasons'][:,1] & 8) and any(result['reasons'][:,0] & 16)
                assert any(result['reasons'][:,0] & 32)
                checks += ['packet_and_channel_fault_reasons_survive_filtering']
        npz = folder/f'{name}.npz'
        with npz.open('xb') as stream:
            np.savez_compressed(stream,values_uv=result['values'],valid=result['valid'],
                                reasons=result['reasons'],output_index=np.arange(result['output_count']),
                                reference_onset_ns=result['origin_ns']+np.arange(result['output_count'])*10000000)
        with np.load(npz,allow_pickle=False) as saved:
            assert np.array_equal(saved['valid'],result['valid'])
            assert np.allclose(saved['values_uv'],result['values'],rtol=0,atol=0,equal_nan=True)
        records.append({'name':name,'fixture_sha256':sha(fixture_path),'arrays_sha256':sha(npz),
            'authenticated_assembly_sha256':result['authenticated_assembly_sha256'],
            'native_counter_origin':fixture['assembled']['first_sample'],
            'analysis_origin_ns':str(result['origin_ns']),'duration_ns':str(result['duration_ns']),
            'output_samples':result['output_count'],'subcell_tail_ns':result['subcell_tail_ns'],
            'epochs':result['epochs'],'transport_status':result['transport_status'],
            'engineering_full_epochs_supported':result['engineering_full_epochs_supported'],
            'valid_samples_per_channel':result['valid'].sum(axis=0).tolist(),
            'tone_errors':tone_errors,'checks':checks})
    return {'status':'GENERATED_PACKET_CLOCK_INTEGRATION_PASSED','cases':records,
            'product_route_enabled':False,'device_qualified':False,
            'clock_assumption':'exact affine generated events; physical delay and jitter unqualified',
            'class_predictions_created':False,'reference_annotations_used':False,
            'source_hashes':{p:sha(ROOT/p) for p in ['tools/hardware_packet_fixture.cjs',
                'tools/hardware_integration_design.py','tools/hardware_packet_design.cjs',
                'tools/hardware_clock_design.py','sleepedf/contracts.py']}}


if __name__ == '__main__':
    import sys
    if '--check' in sys.argv:
        with tempfile.TemporaryDirectory(prefix='physiosleep-integration-') as temporary:
            result = qualify(temporary)
        print(json.dumps({'status':result['status'],'cases':len(result['cases'])}))
    else:
        destination = ROOT/'reports/hardware-integration-design-v1'
        if destination.exists():
            raise FileExistsError('Evidence directory already exists; inspect or use a new version')
        result = qualify(destination)
        result['created_at'] = datetime.now(timezone.utc).isoformat()
        with (destination/'result.json').open('x',encoding='utf-8') as output:
            output.write(json_text(result))
        print(json.dumps({'status':result['status'],'cases':len(result['cases']),
                          'result':str(destination/'result.json')}))
