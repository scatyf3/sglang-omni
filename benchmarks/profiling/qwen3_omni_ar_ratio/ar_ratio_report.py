# SPDX-License-Identifier: Apache-2.0
"""Compute the autoregressive share of end-to-end latency from request event JSONL files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from sglang_omni.profiler.views import reconstruct_timelines

AR_STAGES = ("thinker", "talker_ar")
NS_PER_MS = 1e6


def first_ts(events: list[dict], name: str, stage: str | None = None) -> int | None:
    for ev in events:
        if ev["event_name"] == name and (stage is None or ev.get("stage") == stage):
            return ev["timestamp_ns"]
    return None


def last_ts(events: list[dict], name: str, stage: str | None = None) -> int | None:
    found = None
    for ev in events:
        if ev["event_name"] == name and (stage is None or ev.get("stage") == stage):
            found = ev["timestamp_ns"]
    return found


def span_ms(start: int | None, end: int | None) -> float | None:
    if start is None or end is None or end < start:
        return None
    return (end - start) / NS_PER_MS


def union_ms(intervals: list[tuple[int, int]]) -> float:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged) / NS_PER_MS


def analyze_request(request_id: str, events: list[dict]) -> dict:
    admission = first_ts(events, "request_admission")
    terminal = last_ts(events, "terminal_response")
    if admission is None:
        admission = events[0]["timestamp_ns"]
    if terminal is None:
        terminal = events[-1]["timestamp_ns"]
    row: dict = {"request_id": request_id, "e2e_ms": span_ms(admission, terminal)}

    row["preprocess_ms"] = span_ms(first_ts(events, "preprocess_start"), last_ts(events, "preprocess_end"))
    for stage in ("image_encoder", "audio_encoder"):
        row[f"{stage}_ms"] = span_ms(
            first_ts(events, "encoder_start", stage), last_ts(events, "encoder_end", stage)
        )

    ar_intervals: list[tuple[int, int]] = []
    for stage in AR_STAGES:
        queue_enter = first_ts(events, "scheduler_queue_enter", stage)
        prefill_start = first_ts(events, "scheduler_prefill_start", stage)
        prefill_end = first_ts(events, "scheduler_prefill_end", stage)
        complete = last_ts(events, "stage_complete", stage)
        row[f"{stage}_build_ms"] = span_ms(
            first_ts(events, "scheduler_request_build_start", stage),
            first_ts(events, "scheduler_request_build_end", stage),
        )
        row[f"{stage}_queue_ms"] = span_ms(queue_enter, prefill_start)
        row[f"{stage}_prefill_ms"] = span_ms(prefill_start, prefill_end)
        row[f"{stage}_decode_ms"] = span_ms(prefill_end, complete)
        row[f"{stage}_ar_ms"] = span_ms(prefill_start, complete)
        row[f"{stage}_start_rel_ms"] = span_ms(admission, prefill_start)
        row[f"{stage}_end_rel_ms"] = span_ms(admission, complete)
        if prefill_start is not None and complete is not None:
            ar_intervals.append((prefill_start, complete))

    decode_pairs = [
        (ev["timestamp_ns"], ev["event_name"])
        for ev in events
        if ev["event_name"] in ("code2wav_decode_start", "code2wav_decode_end")
    ]
    busy = 0
    open_start: int | None = None
    for ts, name in decode_pairs:
        if name == "code2wav_decode_start":
            open_start = ts
        elif open_start is not None:
            busy += ts - open_start
            open_start = None
    row["code2wav_busy_ms"] = busy / NS_PER_MS if decode_pairs else None
    row["code2wav_span_ms"] = span_ms(
        first_ts(events, "code2wav_decode_start"), last_ts(events, "code2wav_decode_end")
    )
    row["code2wav_windows"] = sum(1 for _, name in decode_pairs if name == "code2wav_decode_start")
    talker_complete = last_ts(events, "stage_complete", "talker_ar")
    row["after_talker_tail_ms"] = span_ms(talker_complete, terminal)

    row["ttft_ms"] = span_ms(admission, first_ts(events, "stage_first_stream_chunk_sent", "thinker"))
    row["ttfa_ms"] = span_ms(admission, first_ts(events, "code2wav_first_audio"))

    ar_sum = sum(row[f"{stage}_ar_ms"] or 0.0 for stage in AR_STAGES)
    ar_union = union_ms(ar_intervals)
    e2e = row["e2e_ms"] or 0.0
    row["ar_sum_ms"] = ar_sum
    row["ar_union_ms"] = ar_union
    row["non_ar_exclusive_ms"] = e2e - ar_union
    row["ar_sum_ratio"] = ar_sum / e2e if e2e else None
    row["ar_union_ratio"] = ar_union / e2e if e2e else None
    for stage in AR_STAGES:
        row[f"{stage}_ratio"] = (row[f"{stage}_ar_ms"] or 0.0) / e2e if e2e else None
    return row


def aggregate(rows: list[dict]) -> dict:
    summary: dict = {}
    keys = [k for k in rows[0] if k != "request_id"]
    for key in keys:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if not values:
            continue
        summary[key] = {
            "mean": statistics.fmean(values),
            "p50": statistics.median(values),
            "max": max(values),
            "n": len(values),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_dir")
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument("--skip-first", type=int, default=0, help="Drop the first N requests by admission time")
    args = parser.parse_args()

    timelines = reconstruct_timelines(args.event_dir)
    ordered = sorted(timelines.values(), key=lambda t: t.t0_ns or 0)
    rows = [analyze_request(t.request_id, t.events) for t in ordered[args.skip_first :]]
    rows = [r for r in rows if r["e2e_ms"]]
    if not rows:
        raise SystemExit(f"no complete requests found in {args.event_dir}")
    summary = aggregate(rows)
    Path(args.out).write_text(json.dumps({"summary": summary, "requests": rows}, indent=2))

    def fmt(key: str) -> str:
        entry = summary.get(key)
        return f"{entry['mean']:9.1f} {entry['p50']:9.1f}" if entry else f"{'-':>9} {'-':>9}"

    print(f"requests={len(rows)}  (columns: mean  p50, ms)")
    for key in (
        "e2e_ms", "ttft_ms", "ttfa_ms", "preprocess_ms", "audio_encoder_ms", "image_encoder_ms",
        "thinker_queue_ms", "thinker_prefill_ms", "thinker_decode_ms", "thinker_ar_ms",
        "talker_ar_build_ms", "talker_ar_queue_ms", "talker_ar_prefill_ms", "talker_ar_decode_ms", "talker_ar_ar_ms",
        "thinker_start_rel_ms", "thinker_end_rel_ms", "talker_ar_start_rel_ms", "talker_ar_end_rel_ms",
        "code2wav_busy_ms", "code2wav_span_ms", "after_talker_tail_ms",
        "ar_sum_ms", "ar_union_ms", "non_ar_exclusive_ms",
    ):
        print(f"  {key:26s} {fmt(key)}")
    for key in ("thinker_ratio", "talker_ar_ratio", "ar_sum_ratio", "ar_union_ratio"):
        entry = summary.get(key)
        if entry:
            print(f"  {key:26s} {entry['mean']:9.3f} {entry['p50']:9.3f}")


if __name__ == "__main__":
    main()
