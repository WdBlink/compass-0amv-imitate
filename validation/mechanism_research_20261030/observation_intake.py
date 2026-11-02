"""Offline acquisition readiness only. Never compare predictions with labels."""
import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .display_measurement import _exact


_IDENTITY = ("symbol", "date", "indicator", "period")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
# Count nested object/array containers, including the root container.
_MAX_JSON_DEPTH = 100


@dataclass
class _ValidatedPlan:
    plan: dict
    expected: dict
    planned_at: datetime
    frozen_at: datetime
    check_at: datetime
    root: Path
    sources: dict
    prediction: dict
    contracts: dict
    digest: str


def _negative_flags():
    return {"ready_for_independent_review": False, "blindness_verified": False,
            "label_truth": "not_verified", "official_formula_replication": "not_established",
            "mechanism_parity": "not_evaluated"}


def _input_error(exc):
    if isinstance(exc, OSError):
        cause = os.strerror(exc.errno) if exc.errno is not None else "filesystem operation failed"
        return f"{type(exc).__name__}: {cause} (errno {exc.errno})"
    if isinstance(exc, UnicodeError):
        return "input JSON is not valid UTF-8"
    return str(exc)


def _keys(value, required, name):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(f"{name} has missing or unexpected fields")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _known_text(value, name):
    _text(value, name)
    if value.strip().casefold() in ("unknown", "unavailable", "ambiguous"):
        raise ValueError(f"{name} must be known; retain unknown metadata explicitly")
    return value


def _decimal(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an exact decimal string")
    return _exact(value, name)


def _timestamp(value, name):
    if not isinstance(value, str) or "T" not in value:
        raise ValueError(f"{name} must be an aware ISO timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an aware ISO timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{name} must be an aware ISO timestamp")
    return result


def _digest(value, name):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _identity(row):
    if not isinstance(row, dict) or not set(_IDENTITY) <= set(row):
        raise ValueError("case identity is incomplete")
    _text(row["symbol"], "symbol")
    if row["indicator"] not in ("CYF", "CYC"):
        raise ValueError("indicator must be CYF or CYC")
    if type(row["period"]) is not int or row["period"] <= 0:
        raise ValueError("period must be a positive integer")
    try:
        valid_date = isinstance(row["date"], str) and date.fromisoformat(row["date"]).isoformat() == row["date"]
    except ValueError:
        valid_date = False
    if not valid_date:
        raise ValueError("date must be YYYY-MM-DD")
    return tuple(row[key] for key in _IDENTITY)


def plan_sha256(plan):
    """Bind the full declared plan; expected-case ordering is immaterial."""
    canonical = dict(plan)
    canonical["expected"] = sorted(plan["expected"], key=_identity)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _evidence(reference, root, sources):
    _keys(reference, ("source_id", "path", "sha256"), "evidence reference")
    identity = reference["source_id"]
    if not isinstance(identity, str) or not _SOURCE_ID.fullmatch(identity):
        raise ValueError("source_id must be a short path-free identifier")
    digest = _digest(reference["sha256"], "evidence sha256")
    relative = reference["path"]
    if (not isinstance(relative, str) or "\\" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError("evidence path must be relative without traversal")
    try:
        path = root
        for part in relative.split("/"):
            path = path / part
            if path.is_symlink():
                raise ValueError("evidence symlinks are not allowed")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not stat.S_ISREG(resolved.stat().st_mode):
            raise ValueError("evidence must be a regular file inside the evidence root")
        with resolved.open("rb") as stream:
            checksum = hashlib.sha256()
            for chunk in iter(lambda: stream.read(256 * 1024), b""):
                checksum.update(chunk)
            actual = checksum.hexdigest()
    except OSError as exc:
        raise ValueError(f"evidence reference is missing or unreadable ({_input_error(exc)})") from exc
    if actual != digest:
        raise ValueError("evidence hash does not match local bytes")
    binding = (relative, digest)
    if identity in sources and sources[identity] != binding:
        raise ValueError("source_id has conflicting path or hash bindings")
    sources[identity] = binding
    return {"source_id": identity, "sha256": digest}


def _contract(contract, planned_at, root, sources):
    if not isinstance(contract, dict):
        raise ValueError("display_contract must be an object")
    status = contract.get("status")
    if status in ("unknown", "ambiguous"):
        _keys(contract, ("status", "reason", "quantum", "origin", "rounding"), "unknown display contract")
        _text(contract["reason"], "contract reason")
        if contract["quantum"] is not None or contract["origin"] is not None or contract["rounding"] != "unknown":
            raise ValueError("unknown contracts must keep quantum/origin null and rounding unknown")
        return False
    _keys(contract, ("status", "quantum", "origin", "rounding", "allows_cyc_zero", "basis",
                     "evidence", "established_at"), "known display contract")
    if status != "known" or contract["basis"] not in ("documentation", "independent_calibration"):
        raise ValueError("known contract requires a declared independent basis")
    quantum = _decimal(contract["quantum"], "quantum")
    origin = _decimal(contract["origin"], "origin")
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    if contract["rounding"] not in ("nearest", "floor", "ceiling", "truncate"):
        raise ValueError("known rounding must be explicit; otherwise use an unknown or ambiguous contract")
    if type(contract["allows_cyc_zero"]) is not bool:
        raise ValueError("allows_cyc_zero must be boolean")
    if _timestamp(contract["established_at"], "contract established_at") > planned_at:
        raise ValueError("contract must be independently established no later than planned_at")
    _evidence(contract["evidence"], root, sources)
    return origin == 0 and contract["rounding"] == "nearest"


def _observation(row, expected, planned_at, frozen_at, assembled_at, root, sources):
    status = row.get("status")
    if status in ("unavailable", "ambiguous"):
        _keys(row, (*_IDENTITY, "status", "reason"), "uncollected observation")
        _text(row["reason"], "observation reason")
        return {"status": status, "reason": row["reason"], "metadata_checked": False,
                "evidence_checked": False, "usable": False}
    _keys(row, (*_IDENTITY, "status", "value", "display_text", "source", "software", "field", "captured_at"),
          "observed observation")
    if status != "observed":
        raise ValueError("observation status must be observed, unavailable or ambiguous")
    value = _decimal(row["value"], "value")
    if row["display_text"] != row["value"]:
        raise ValueError("display_text must preserve the exact original numeric token in value")
    _keys(row["software"], ("name", "version"), "software")
    _known_text(row["software"]["name"], "software name")
    _known_text(row["software"]["version"], "software version")
    _keys(row["field"], ("label", "period", "unit"), "actual field")
    _known_text(row["field"]["label"], "actual field label")
    _known_text(row["field"]["unit"], "actual field unit")
    if type(row["field"]["period"]) is not int or row["field"]["period"] != expected["period"]:
        raise ValueError("actual field period does not match the planned period")
    for observed_key, planned_key in (("label", "field_label"), ("unit", "unit")):
        if expected[planned_key] is not None and row["field"][observed_key] != expected[planned_key]:
            raise ValueError(f"actual field {observed_key} does not match the planned metadata")
    captured_at = _timestamp(row["captured_at"], "captured_at")
    if not max(planned_at, frozen_at) <= captured_at <= assembled_at:
        raise ValueError("capture must follow planning/freeze and not exceed assembled_at")
    contract = expected["display_contract"]
    if expected["indicator"] == "CYF" and not 0 <= value <= 100:
        raise ValueError("CYF must be in [0, 100]")
    if expected["indicator"] == "CYC" and (value < 0 or (value == 0 and not (
            contract["status"] == "known" and contract["allows_cyc_zero"]
            and contract["origin"] is not None and _decimal(contract["origin"], "origin") == 0
            and contract["rounding"] == "nearest"))):
        raise ValueError("CYC must be positive unless an explicit nearest-grid contract permits displayed zero")
    if contract["status"] == "known":
        grid_index = (value - _decimal(contract["origin"], "origin")) / _decimal(contract["quantum"], "quantum")
        if grid_index.denominator != 1:
            raise ValueError("displayed value is not on the independently declared display grid")
    source = _evidence(row["source"], root, sources)
    metadata_checked = expected["unit"] is not None and expected["field_label"] is not None
    return {"status": status, "value": row["value"], "display_text": row["display_text"],
            "source": source, "software": dict(row["software"]), "field": dict(row["field"]),
            "captured_at": row["captured_at"], "metadata_checked": metadata_checked,
            "evidence_checked": True, "usable": False}


def validate_intake(plan, packet, evidence_root, *, as_of=None):
    """Return every planned case, including missing/invalid cases, without scoring.

    Malformed plans raise ValueError because no trustworthy denominator exists.
    Packet errors produce an invalid report with zero usable coverage.
    """
    return _validate_packet(packet, _validate_plan(plan, evidence_root, as_of=as_of))


def _validate_plan(plan, evidence_root, *, as_of=None):
    _keys(plan, ("planned_at", "prediction_frozen_at", "prediction", "purpose", "expected"), "plan")
    planned_at = _timestamp(plan["planned_at"], "planned_at")
    frozen_at = _timestamp(plan["prediction_frozen_at"], "prediction_frozen_at")
    check_at = _timestamp(datetime.now(timezone.utc).isoformat() if as_of is None else as_of, "as_of")
    if max(planned_at, frozen_at) > check_at:
        raise ValueError("planning and freeze timestamps must not be later than as_of")
    if plan["purpose"] not in ("contract_calibration", "blind_validation"):
        raise ValueError("purpose must be contract_calibration or blind_validation")
    if not isinstance(plan["expected"], list) or not plan["expected"]:
        raise ValueError("expected must contain the nonempty predeclared planned set")
    expected = {}
    for case in plan["expected"]:
        _keys(case, (*_IDENTITY, "field_label", "unit", "display_contract"), "planned case")
        key = _identity(case)
        if key in expected:
            raise ValueError("planned set contains a duplicate identity")
        for field in ("field_label", "unit"):
            if case[field] is not None:
                _known_text(case[field], f"planned {field}")
        expected[key] = case
    try:
        root = Path(evidence_root).resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("evidence root must be a local directory") from exc
    except RuntimeError as exc:
        raise ValueError("evidence root cannot be resolved") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError("evidence root cannot be resolved") from exc
        raise
    if not root.is_dir():
        raise ValueError("evidence root must be a local directory")
    sources = {}
    prediction = _evidence(plan["prediction"], root, sources)
    contracts = {key: _contract(case["display_contract"], planned_at, root, sources) for key, case in expected.items()}
    digest = plan_sha256(plan)
    return _ValidatedPlan(plan, expected, planned_at, frozen_at, check_at, root, sources, prediction, contracts, digest)


def _validate_packet(packet, validated, *, packet_error=None):
    plan, expected = validated.plan, validated.expected
    planned_at, frozen_at, check_at = validated.planned_at, validated.frozen_at, validated.check_at
    root, sources = validated.root, validated.sources
    prediction, contracts, digest = validated.prediction, validated.contracts, validated.digest
    errors = []
    rows = []
    indexed = {}
    assembled_at = None
    try:
        if packet_error is not None:
            raise ValueError(packet_error)
        _keys(packet, ("plan_sha256", "prediction_sha256", "prediction_frozen_at", "purpose", "assembled_at",
                       "observations"), "packet")
        if _digest(packet["plan_sha256"], "plan_sha256") != digest:
            raise ValueError("packet plan_sha256 does not match the supplied plan")
        if _digest(packet["prediction_sha256"], "prediction_sha256") != prediction["sha256"]:
            raise ValueError("packet prediction_sha256 does not match the frozen prediction")
        if packet["prediction_frozen_at"] != plan["prediction_frozen_at"]:
            raise ValueError("packet prediction_frozen_at does not match the declared freeze")
        if packet["purpose"] != plan["purpose"]:
            raise ValueError("packet purpose differs from plan; calibration must not be relabeled blind")
        assembled_at = _timestamp(packet["assembled_at"], "assembled_at")
        if not max(planned_at, frozen_at) <= assembled_at <= check_at:
            raise ValueError("assembled_at must follow planning/freeze and not exceed as_of")
        if not isinstance(packet["observations"], list):
            raise ValueError("observations must be a list")
        for row in packet["observations"]:
            key = _identity(row)
            indexed.setdefault(key, []).append(row)
        if set(indexed) - set(expected):
            errors.append("packet contains unexpected case identities")
    except ValueError as exc:
        errors.append(str(exc))
    for key, case in sorted(expected.items()):
        result = {field: case[field] for field in _IDENTITY}
        result.update(status="missing", metadata_checked=False, evidence_checked=False, usable=False,
                      contract_status=case["display_contract"]["status"], helper_contract_supported=contracts[key])
        matches = indexed.get(key, [])
        if packet_error is not None:
            result.update(status="invalid", reason=packet_error)
        elif not matches:
            result["reason"] = "planned case missing from packet"
            errors.append(result["reason"])
        elif len(matches) != 1:
            result.update(status="invalid", reason="duplicate observations for planned case")
            errors.append(result["reason"])
        else:
            try:
                if assembled_at is None:
                    raise ValueError("packet chronology could not be validated")
                result.update(_observation(matches[0], case, planned_at, frozen_at, assembled_at, root, sources))
                result["usable"] = result["status"] == "observed" and result["metadata_checked"] and contracts[key]
            except ValueError as exc:
                result.update(status="invalid", reason=str(exc))
                errors.append(str(exc))
        rows.append(result)
    counts = Counter(row["status"] for row in rows)
    if errors:
        status = "invalid"
        for row in rows:
            row["usable"] = False
    elif counts["unavailable"] or counts["ambiguous"]:
        status = "collection_incomplete"
    elif any(not row["metadata_checked"] for row in rows):
        status = "needs_metadata"
    elif not all(contracts.values()):
        status = "needs_contract"
    else:
        status = "ready_for_independent_review"
    total = len(expected)
    usable = sum(row["usable"] for row in rows)
    return {**_negative_flags(), "status": status,
            "ready_for_independent_review": status == "ready_for_independent_review",
            "plan_sha256": digest, "prediction": prediction,
            "prediction_frozen_at": plan["prediction_frozen_at"], "purpose": plan["purpose"],
            "as_of": check_at.isoformat(),
            "coverage": {"planned": total, "observed": counts["observed"], "usable": usable,
                         "usable_fraction": f"{usable}/{total}", "unavailable": counts["unavailable"],
                         "ambiguous": counts["ambiguous"], "missing": counts["missing"], "invalid": counts["invalid"]},
            "errors": errors, "cases": rows}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON contains duplicate object keys")
        result[key] = value
    return result


def _load_json(filename):
    with Path(filename).open(encoding="utf-8") as stream:
        try:
            value = json.load(stream, object_pairs_hook=_unique_object)
        except RecursionError as exc:
            raise ValueError("JSON nesting is too deep") from exc
    # Do not depend on the interpreter's JSON parser recursion limit.
    pending = [(value, 1)]
    while pending:
        node, depth = pending.pop()
        if isinstance(node, (dict, list)):
            if depth > _MAX_JSON_DEPTH:
                raise ValueError("JSON nesting is too deep")
            children = node.values() if isinstance(node, dict) else node
            pending.extend((child, depth + 1) for child in children if isinstance(child, (dict, list)))
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--as-of", help="Caller-declared aware checking time; defaults to local UTC now")
    args = parser.parse_args(argv)
    validated = None
    try:
        validated = _validate_plan(_load_json(args.plan), args.evidence_root, as_of=args.as_of)
        report = _validate_packet(_load_json(args.packet), validated)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        message = _input_error(exc)
        report = ({**_negative_flags(), "status": "invalid", "errors": [message]} if validated is None
                  else _validate_packet(None, validated, packet_error=message))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 2 if report["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
