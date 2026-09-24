"""Reproduce provisional hardware sizing assumptions; no device measurements."""
from fractions import Fraction
import json


def budget():
    rate = 250
    block_samples = 20
    frame_bytes = 4 * 3 + 3
    block_bytes = block_samples * frame_bytes + 24 + 16
    bytes_per_second = Fraction(block_bytes * rate, block_samples)
    usable_wh = Fraction(37, 10) * 1 * Fraction(7, 10)
    storage = []
    for hours in (8, 12, 24):
        size = int(bytes_per_second * hours * 3600)
        storage.append({"hours": hours, "bytes": size,
                        "decimal_MB": size / 1_000_000, "MiB": size / 2**20,
                        "with_30_percent_allowance_bytes": int(size * Fraction(13, 10))})
    assert bytes_per_second == 4250
    assert storage[1]["bytes"] == 183600000
    assert storage[-1]["with_30_percent_allowance_bytes"] < 2**30
    assert usable_wh / Fraction(200, 1000) == Fraction(259, 20)
    return {"status": "PLANNING_ASSUMPTIONS_NOT_MEASURED", "adc_channels_logged": 4,
            "native_rate_hz": rate, "frame_bytes": frame_bytes,
            "samples_per_block": block_samples, "block_bytes": block_bytes,
            "bytes_per_second": int(bytes_per_second),
            "payload_bits_per_second": int(bytes_per_second * 8),
            "streaming_application_target_bits_per_second": int(bytes_per_second * 16),
            "storage": storage, "battery_nominal_volts": 3.7,
            "battery_nominal_Ah": 1, "usable_energy_factor": 0.7,
            "usable_Wh": float(usable_wh),
            "runtime_sensitivity": [{"battery_side_mW": mw,
                                     "hours": float(usable_wh / Fraction(mw, 1000))}
                                    for mw in (100, 200, 300)],
            "clock_20ppm_drift_seconds": {str(h): float(Fraction(20, 1_000_000) * h * 3600)
                                          for h in (8, 12)},
            "bench_unit_internal_USD_allowance": {"minimum": 135, "maximum": 325,
                                                   "supplier_quote": False}}


if __name__ == "__main__":
    print(json.dumps(budget(), indent=2, allow_nan=False))
