"""Owner-authorized retirement of reserved audits; never a confirmation pass."""
from pathlib import Path
import os
import socket

from .contracts import read_json
from .research import file_sha256
from .audit_access import read_ledger, append_audit_event
from .protocol import load_protocol


def append_exploratory_retirement(root: Path, selection_path: Path) -> dict:
    import psutil
    root, selection_path = root.resolve(), Path(selection_path)
    if selection_path != root / "research/exploratory-audits-v1.json" or selection_path.resolve() != selection_path:
        raise ValueError("Exploratory evaluation requires the canonical frozen selection")
    selection = read_json(selection_path)
    authorization_path = root / "research/exploratory-audits-authorization-v1.json"
    if (selection.get("authorization_path") != str(authorization_path) or
            selection.get("authorization_sha256") != file_sha256(authorization_path)):
        raise ValueError("Exploratory evaluation authorization differs")
    lease = read_json(root / "runs/compute.lock")
    if (lease.get("pid") != os.getpid() or lease.get("process_start") != psutil.Process().create_time()
            or lease.get("host") != socket.gethostname() or lease.get("run_id") != "exploratory-audits-v1"):
        raise ValueError("Exploratory retirement requires its live controller's exclusive lease")
    protocol, _, _ = load_protocol(root)
    ledger = root / "runs/audit-access.jsonl"
    evidence = {"authorization": {"path": str(authorization_path), "sha256": file_sha256(authorization_path)},
                "selection": {"path": str(selection_path), "sha256": file_sha256(selection_path)},
                "meaning": "Owner retired both holdouts for exploratory evaluation; no claim that labels have yet been read"}
    out = {}
    for phase in ("A", "B"):
        previous = [r for r in read_ledger(ledger, protocol["protocol_hash"]) if r["phase"] == phase]
        if previous:
            if len(previous) != 1 or previous[0]["event"] != "EXPLORATORY_RETIRED" or previous[0]["evidence"] != evidence:
                raise ValueError("Audit chronology differs from this exact exploratory selection")
            out[phase] = previous[0]
        else:
            out[phase] = append_audit_event(ledger, protocol["protocol_hash"], phase,
                                           "EXPLORATORY_RETIRED", evidence)
    return out
