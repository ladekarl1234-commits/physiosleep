"""Project EDF annotation intervals onto the original PSG 30-second grid.

All arithmetic is integer microseconds.  An epoch is usable only when one
continuous segment and nonoverlapping, recognized intervals cover it fully.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN

from .contracts import map_annotation


MICROSECONDS = 1_000_000
EPOCH_US = 30 * MICROSECONDS
TOLERANCE_US = 1


def to_microseconds(value: object) -> int:
    """Convert an EDF second value, allowing at most 1 us quantization error."""
    if isinstance(value, bool):
        raise ValueError("Boolean is not a time")
    try:
        exact = Decimal(str(value)) * MICROSECONDS
        rounded = exact.to_integral_value(rounding=ROUND_HALF_EVEN)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Time must be finite and numeric") from exc
    if not exact.is_finite() or abs(exact - rounded) > TOLERANCE_US:
        raise ValueError("Time cannot be represented within 1 us")
    return int(rounded)


def project_annotations(
    recording_id: str,
    psg_duration_seconds: object,
    intervals: list[tuple[object, object, str]],
    *,
    offset_seconds: object = 0,
    segments: list[tuple[object, object, str]] | None = None,
) -> dict:
    """Return original-location labels/masks; interval onsets use Hypnogram time.

    ``offset_seconds`` is Hypnogram start minus PSG start, established outside
    this pure function.  Segment intervals use PSG-relative seconds.
    """
    if not isinstance(recording_id, str) or not recording_id:
        raise ValueError("Recording identity is required")
    duration = to_microseconds(psg_duration_seconds)
    offset = to_microseconds(offset_seconds)
    if duration <= 0:
        raise ValueError("PSG duration must be positive")
    if not isinstance(intervals, list):
        raise ValueError("Intervals must be a list")
    if segments is None:
        parsed_segments = [(0, duration, "continuous-0")]
    else:
        if not isinstance(segments, list) or not segments:
            raise ValueError("Explicit segments must be a nonempty list")
        parsed_segments = []
        for item in segments:
            if not isinstance(item, (tuple, list)) or len(item) != 3 or not isinstance(item[2], str) or not item[2]:
                raise ValueError("Segment must have start, end, and identity")
            start, end = to_microseconds(item[0]), to_microseconds(item[1])
            if start < 0 or end > duration or start >= end:
                raise ValueError("Segment lies outside PSG or has no duration")
            parsed_segments.append((start, end, item[2]))
        parsed_segments.sort()
        if any(a[1] > b[0] for a, b in zip(parsed_segments, parsed_segments[1:])):
            raise ValueError("Continuous segments overlap")

    parsed = []
    off_record = []
    for index, item in enumerate(intervals):
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError("Annotation must have onset, duration, description")
        start = to_microseconds(item[0]) + offset
        length = to_microseconds(item[1])
        if length <= 0:
            raise ValueError("Annotation duration must be positive")
        end = start + length
        mapped = map_annotation(item[2])
        parsed.append((start, end, mapped["label"], mapped["reason"]))
        if start < 0 or end > duration:
            off_record.append({"source_index": index, "start_us": start, "end_us": end})
    parsed.sort(key=lambda item: (item[0], item[1]))

    epochs = []
    cursor = 0
    for index in range(duration // EPOCH_US):
        left, right = index * EPOCH_US, (index + 1) * EPOCH_US
        while cursor < len(parsed) and parsed[cursor][1] <= left:
            cursor += 1
        reasons = set()
        covering_segments = [s for s in parsed_segments if s[0] <= left and s[1] >= right]
        if len(covering_segments) != 1:
            reasons.add("segment_gap_or_boundary")
        overlaps = []
        position = cursor
        while position < len(parsed) and parsed[position][0] < right:
            if parsed[position][1] > left:
                overlaps.append(parsed[position])
            position += 1
        expected = left
        label = None
        for start, end, current_label, invalid_reason in overlaps:
            clip_start, clip_end = max(left, start), min(right, end)
            if clip_start > expected + TOLERANCE_US:
                reasons.add("annotation_gap")
            elif clip_start < expected - TOLERANCE_US:
                reasons.add("annotation_overlap")
            if invalid_reason:
                reasons.add("invalid_label_" + invalid_reason)
            elif label is None:
                label = current_label
            elif current_label != label:
                reasons.add("label_conflict")
            expected = max(expected, clip_end)
        if expected < right - TOLERANCE_US:
            reasons.add("annotation_gap")
        epochs.append({"recording_id": recording_id, "index": index,
                       "onset_us": left, "end_us": right,
                       "segment_id": covering_segments[0][2] if len(covering_segments) == 1 else None,
                       "label": label if not reasons else None,
                       "valid": not reasons,
                       "invalid_reasons": sorted(reasons)})
    trailing = duration % EPOCH_US
    return {"recording_id": recording_id, "offset_us": offset,
            "psg_duration_us": duration, "epoch_seconds": 30,
            "timing_tolerance_us": TOLERANCE_US,
            "epochs": epochs, "off_record_intervals": off_record,
            "incomplete_trailing_epoch": ({"index": len(epochs),
                                           "onset_us": len(epochs) * EPOCH_US,
                                           "duration_us": trailing,
                                           "reason": "incomplete_trailing_epoch"}
                                          if trailing else None)}
