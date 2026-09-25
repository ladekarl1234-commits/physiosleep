"""Replay the six generated packet/clock cases against public aggregate evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import scipy

from tools import hardware_integration_design as integration


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "evidence/hardware-desk-replay.json"
COEFFICIENTS = ROOT / "reports/hardware-filter-design-v1/coefficients.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    runtime = expected["runtime"]
    require(sys.version_info[:2] == (3, 11), "Python runtime differs")
    require(np.__version__ == runtime["numpy"] and scipy.__version__ == runtime["scipy"],
            "Scientific runtime differs")
    node = subprocess.check_output(["node", "--version"], text=True).strip()
    openssl = subprocess.check_output(["node", "-p", "process.versions.openssl"], text=True).strip()
    require(node == runtime["node"] and openssl == runtime["openssl"], "Node/OpenSSL runtime differs")

    for name, expected_hash in expected["source_sha256"].items():
        require(digest(ROOT / name) == expected_hash, f"Source identity differs: {name}")
    require(digest(COEFFICIENTS) == expected["nominal_coefficient_sha256"],
            "Nominal filter coefficient identity differs")

    # The historical private manifests enumerate archived deliverables unavailable
    # in a public clone. Their guard is replaced by the source/coefficient checks
    # above; the copied numerical qualification and all six fixtures run unchanged.
    with tempfile.TemporaryDirectory(prefix="physiosleep-public-hardware-") as folder:
        with patch.object(integration, "verify_frozen_inputs", lambda: None):
            actual = integration.qualify(folder)
    require(actual["status"] == "GENERATED_PACKET_CLOCK_INTEGRATION_PASSED",
            "Generated integration did not pass")
    require(actual["source_hashes"] == expected["source_sha256"], "Result source identities differ")
    require(len(actual["cases"]) == len(expected["cases"]) == 6, "Case count differs")

    for got, wanted in zip(actual["cases"], expected["cases"], strict=True):
        for field in ("name", "fixture_sha256", "authenticated_assembly_sha256",
                      "transport_status", "engineering_full_epochs_supported"):
            require(got[field] == wanted[field], f"{wanted['name']}: {field} differs")
        require(got["output_samples"] == expected["fixed_grid"]["output_samples"],
                f"{wanted['name']}: grid length differs")
        require(got["subcell_tail_ns"] == expected["fixed_grid"]["subcell_tail_ns"],
                f"{wanted['name']}: trailing time differs")
        require(got["valid_samples_per_channel"] == wanted["valid_samples_per_channel"],
                f"{wanted['name']}: valid-sample counts differ")
        require([epoch["index"] for epoch in got["epochs"]] == expected["fixed_grid"]["epoch_indices"],
                f"{wanted['name']}: epoch indices differ")
        require([epoch["complete"] for epoch in got["epochs"]] == expected["fixed_grid"]["complete_epochs"],
                f"{wanted['name']}: complete-epoch mask differs")
        require([epoch["active_channels_supported"] for epoch in got["epochs"]] == wanted["epoch_support"],
                f"{wanted['name']}: supported-epoch mask differs")
        maximum = max((item["maximum_error_uv"] for item in got["tone_errors"]), default=None)
        reference = wanted["max_tone_error_uv"]
        require((maximum is None and reference is None) or
                (maximum is not None and reference is not None and abs(maximum-reference) < 1e-10),
                f"{wanted['name']}: generated waveform error differs")

    print(json.dumps({"status": "PASS", "scope": "six generated packet/clock cases only",
                      "cases": [item["name"] for item in actual["cases"]],
                      "physical_device_qualified": False, "model_predictions_created": False}))


if __name__ == "__main__":
    main()
