"""Append-only procedural audit chronology, separate from metric calculations.

This is a shared-account audit trail, not an access-control security boundary.
The experiment controller must register an independently verified development
readiness receipt before opening an audit. Recomputations reuse the same frozen
selection; they do not create a new confirmatory attempt.
"""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import content_id, read_json
from .research import artifact_id, file_sha256, utc_now

CONTROL_KEYS = {"selection_manifest_path", "selection_manifest_sha256", "access_ledger_path"}
EVENTS = {"SELECTION_FROZEN", "AUDIT_OPENED", "EVALUATION_RECORDED", "VERIFICATION_RECORDED",
          "EXPLORATORY_RETIRED"}


def canonical_ledger(path: Path, protocol_hash: str) -> Path:
    """Anchor chronology to the original protocol's absolute workspace paths."""
    path = Path(path).resolve()
    root = path.parent.parent
    if path != root / "runs" / "audit-access.jsonl":
        raise ValueError("Audit history requires the canonical runs/audit-access.jsonl")
    protocol = read_json(root / "research" / "protocol-v1.json")
    if (protocol.get("protocol_hash") != protocol_hash or
            artifact_id(protocol, "protocol_hash") != protocol_hash or
            Path(protocol["split_path"]).resolve() != root / "research" / "split-v2.json"):
        raise ValueError("Audit ledger is not anchored to the original frozen protocol")
    return path


def _artifact(reference: dict) -> Path:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("An actual artifact path and digest are required")
    path = Path(reference["path"])
    if not path.is_absolute() or file_sha256(path) != reference["sha256"]:
        raise ValueError("Audit artifact path or digest differs")
    return path


def _verified_gate(evidence: dict, protocol_hash: str, phase: str, ledger: Path) -> dict:
    """A chronology entry cannot promote a claimed status into a verified gate."""
    from .gates import _verify_attestation, compute_gate, require_gate
    if set(evidence) != {"outcome", "report", "attestation", "recompute_arguments"}:
        raise ValueError("Verification requires bound report, invocation and recomputation artifacts")
    report_path = _artifact(evidence["report"])
    attestation_path = _artifact(evidence["attestation"])
    arguments = read_json(_artifact(evidence["recompute_arguments"]))
    if (arguments.get("phase") != phase or arguments.get("protocol_hash") != protocol_hash or
            arguments.get("synthetic_test", False) is not False or arguments.get("draws", 10_000) != 10_000 or
            Path(arguments.get("freeze_manifest", {}).get("access_ledger_path", "")).resolve() != ledger):
        raise ValueError("Verification arguments do not bind this real audit and its original ledger")
    if evidence["outcome"] == "VERIFIED_PASS":
        return require_gate(report_path, attestation_path=attestation_path, **arguments)
    if evidence["outcome"] != "VERIFIED_FAIL":
        raise ValueError("Unknown verification outcome")
    report = compute_gate(**arguments)
    if report["status"] != "FAILED" or read_json(report_path) != report:
        raise ValueError("Failed audit report differs from recomputation")
    _verify_attestation(attestation_path, report_path, report)
    return report


def _require_original_a(rows: list[dict], protocol_hash: str, ledger: Path) -> None:
    verified = [r for r in rows if r["phase"] == "A" and r["event"] == "VERIFICATION_RECORDED"]
    if len(verified) != 1 or verified[0]["evidence"].get("outcome") != "VERIFIED_PASS":
        raise ValueError("Audit B requires the original independently verified Gate A")
    report = _verified_gate(verified[0]["evidence"], protocol_hash, "A", ledger)
    if report["status"] != "PASSED":
        raise ValueError("The original Gate A has not passed recomputation")


def read_ledger(path: Path, protocol_hash: str) -> list[dict]:
    path = canonical_ledger(path, protocol_hash)
    if not path.exists():
        return []
    records = []
    previous = protocol_hash
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        digest = row.pop("event_hash", None)
        if (row.get("previous_hash") != previous or row.get("sequence") != len(records) or
            row.get("protocol_hash") != protocol_hash or row.get("event") not in EVENTS or
            row.get("phase") not in ("A", "B") or digest != content_id(row)):
            raise ValueError("Audit ledger content or chronology changed")
        row["event_hash"] = digest
        records.append(row)
        previous = digest
    return records


def append_audit_event(path: Path, protocol_hash: str, phase: str, event: str, evidence: dict) -> dict:
    """Serialize under the controller's compute lease; refuse second selections/openings."""
    import os
    path = canonical_ledger(path, protocol_hash)
    rows = read_ledger(path, protocol_hash)
    same = [row for row in rows if row["phase"] == phase]
    if phase not in ("A", "B") or event not in EVENTS:
        raise ValueError("Unknown audit phase/event")
    if event == "EXPLORATORY_RETIRED":
        if same:
            raise ValueError("An already registered audit cannot be retired a second time")
        authorization = read_json(_artifact(evidence.get("authorization")))
        selection = read_json(_artifact(evidence.get("selection")))
        if (authorization.get("artifact_type") != "owner_exploratory_audit_authorization" or
                authorization.get("owner_reply") != "Run both now as exploratory evaluations" or
                authorization.get("protocol_hash") != protocol_hash or
                authorization.get("fresh_data_required_for_confirmation") is not True or
                selection.get("artifact_type") != "physiosleep_exploratory_audits_selection" or
                selection.get("scope") != "EXPLORATORY_A_AND_B_NO_GATE_PASS" or
                selection.get("protocol_hash") != protocol_hash or
                selection.get("selection_id") != artifact_id(selection, "selection_id")):
            raise ValueError("Exploratory retirement requires the owner's decision and frozen selection")
    if event == "SELECTION_FROZEN" and same:
        raise ValueError("Audit selection already registered; a failed audit cannot be redrawn")
    if phase == "B" and event in ("SELECTION_FROZEN", "AUDIT_OPENED"):
        _require_original_a(rows, protocol_hash, path)
    if event == "AUDIT_OPENED":
        if [row["event"] for row in same] != ["SELECTION_FROZEN"]:
            raise ValueError("Audit can be opened exactly once after selection freeze")
        if evidence.get("selection_sha256") != same[0]["evidence"].get("selection_sha256"):
            raise ValueError("Audit selection changed before opening")
        receipt = evidence.get("readiness_receipt")
        if not isinstance(receipt, dict) or set(receipt) != {"path", "sha256"}:
            raise ValueError("Independent development-readiness receipt required")
        receipt_path = Path(receipt["path"])
        if file_sha256(receipt_path) != receipt["sha256"]:
            raise ValueError("Readiness receipt digest changed")
        ready = read_json(receipt_path)
        required = {"schema_version", "artifact_type", "protocol_hash", "selection_sha256",
                    "seed_evaluations", "precision_study", "verifier_receipt"}
        if (set(ready) != required or ready["schema_version"] != "1.0" or
            ready["artifact_type"] != "verified_development_readiness" or
            ready["protocol_hash"] != protocol_hash or ready["selection_sha256"] != evidence["selection_sha256"]):
            raise ValueError("Boolean status files cannot establish development readiness")
        if sorted(ready["seed_evaluations"]) != ["101", "17", "43"]:
            raise ValueError("Three fixed-seed evaluations are required before opening")
        for item in list(ready["seed_evaluations"].values()) + [ready["precision_study"], ready["verifier_receipt"]]:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"} or file_sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("Readiness receipt must bind all actual study/verifier artifacts")
        verification = read_json(Path(ready["verifier_receipt"]["path"]))
        if (verification.get("artifact_type") != "development_readiness_verification" or
            verification.get("protocol_hash") != protocol_hash or
            verification.get("selection_sha256") != evidence["selection_sha256"] or
            verification.get("outcome") != "VERIFIED_READY" or
            verification.get("recomputed_seed_evaluations") != ready["seed_evaluations"] or
            verification.get("recomputed_precision_study") != ready["precision_study"]):
            raise ValueError("Independent readiness verification is absent or unbound")
    if event in ("EVALUATION_RECORDED", "VERIFICATION_RECORDED"):
        expected = "AUDIT_OPENED" if event == "EVALUATION_RECORDED" else "EVALUATION_RECORDED"
        if not same or same[-1]["event"] != expected:
            raise ValueError("Audit result/verification is out of sequence")
        if event == "EVALUATION_RECORDED":
            report = read_json(_artifact(evidence.get("report")))
            if (report.get("phase") != phase or report.get("protocol_hash") != protocol_hash or
                    report.get("status") not in ("PASSED", "FAILED")):
                raise ValueError("Evaluation event lacks this audit's actual report")
        else:
            if evidence.get("report") != same[-1]["evidence"].get("report"):
                raise ValueError("Verification changed the original evaluated report")
            _verified_gate(evidence, protocol_hash, phase, path)
    row = {"sequence": len(rows), "previous_hash": rows[-1]["event_hash"] if rows else protocol_hash,
           "protocol_hash": protocol_hash, "phase": phase, "event": event,
           "recorded_at": utc_now(), "evidence": evidence}
    row["event_hash"] = content_id(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return row


def validate_audit_opening(freeze_manifest: dict) -> None:
    """Gate hook: bind calculation to the one already-opened frozen selection."""
    if any(not isinstance(freeze_manifest.get(key), str) for key in CONTROL_KEYS):
        raise ValueError("Audit calculation requires its original selection and consumption ledger")
    selection_path = Path(freeze_manifest["selection_manifest_path"])
    digest = file_sha256(selection_path)
    if digest != freeze_manifest["selection_manifest_sha256"]:
        raise ValueError("Frozen selection bytes changed")
    selected = read_json(selection_path)
    if selected != {k: v for k, v in freeze_manifest.items() if k not in CONTROL_KEYS}:
        raise ValueError("Audit recomputation changed its frozen selection")
    ledger = canonical_ledger(Path(freeze_manifest["access_ledger_path"]), freeze_manifest["protocol_hash"])
    if Path(freeze_manifest.get("protocol_path", "")).resolve() != ledger.parent.parent / "research" / "protocol-v1.json":
        raise ValueError("Audit selection relocated the original protocol")
    rows = read_ledger(ledger, freeze_manifest["protocol_hash"])
    same = [row for row in rows if row["phase"] == freeze_manifest["phase"]]
    opening = [row for row in same if row["event"] == "AUDIT_OPENED"]
    selection = [row for row in same if row["event"] == "SELECTION_FROZEN"]
    if (len(opening) != 1 or len(selection) != 1 or
        opening[0]["evidence"].get("selection_sha256") != digest or
        selection[0]["evidence"].get("selection_sha256") != digest):
        raise ValueError("Audit has not been opened once for this exact selection")
    if freeze_manifest["phase"] == "B":
        _require_original_a(rows, freeze_manifest["protocol_hash"], ledger)


def confirmation_launch_status(root: Path) -> dict:
    """No launcher is exposed until actual mandatory runs and preflight exist."""
    registry = read_json(root / "research" / "baseline_registry.json")
    runs = list((root / "research" / "baseline_results").glob("*.json"))
    required = [slot["id"] for repo in registry["repositories"] for slot in repo["execution_slots"] if slot["required"]]
    evidence = {read_json(path).get("system_id"): str(path) for path in runs}
    return {"launch_enabled": False, "missing_system_artifacts": sorted(set(required) - set(evidence)),
            "reason": "Full native baseline adequacy and independently recomputed three-seed/precision preflight are required; no audit launch command is exposed."}
