# SPDX-License-Identifier: Apache-2.0
"""Split a served Nemotron Transcribe request into the model's three parts.

Reads the request-event JSONL that /start_request_profile writes and reports, per request, the time in the
Parallel Expert Encoder (audio to embeddings, encoder_start to encoder_end), the NemotronH decoder (the
autoregressive part: prefill of the prompt plus token-by-token decode, with any encoder time nested inside the
prefill span taken out) and the CTC timestamp heads (not served by this integration, so always 0), plus the
remainder (request build, queue, dispatch and response). Shares are over the coordinator's end-to-end span.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from sglang_omni.profiler.views import reconstruct_timelines

NS_PER_MS = 1e6
REPORTED = (
    "encoder_ms",
    "decoder_prefill_ms",
    "decoder_decode_ms",
    "decoder_ms",
    "ctc_timestamp_ms",
    "other_ms",
)


def first_ts(events: list[dict], name: str) -> int | None:
    for event in events:
        if event["event_name"] == name:
            return event["timestamp_ns"]
    return None


def last_ts(events: list[dict], name: str) -> int | None:
    found = None
    for event in events:
        if event["event_name"] == name:
            found = event["timestamp_ns"]
    return found


def span_ms(start: int | None, end: int | None) -> float:
    if start is None or end is None or end < start:
        return 0.0
    return (end - start) / NS_PER_MS


def analyze(events: list[dict]) -> dict[str, float | None]:
    admission = first_ts(events, "request_admission") or events[0]["timestamp_ns"]
    terminal = last_ts(events, "terminal_response") or events[-1]["timestamp_ns"]
    e2e_ms = span_ms(admission, terminal)
    encoder_start, encoder_end = first_ts(events, "encoder_start"), last_ts(
        events, "encoder_end"
    )
    prefill_start, prefill_end = first_ts(events, "scheduler_prefill_start"), last_ts(
        events, "scheduler_prefill_end"
    )
    complete = last_ts(events, "stage_complete")
    encoder_ms = span_ms(encoder_start, encoder_end)
    prefill_ms = span_ms(prefill_start, prefill_end)
    # With the pre-LM encoder off the encoder runs inside the prefill forward; its span then overlaps the prefill span.
    encoder_inside_prefill = (
        encoder_start is not None
        and prefill_start is not None
        and prefill_start <= encoder_start <= (prefill_end or 0)
    )
    decoder_prefill_ms = (
        prefill_ms - encoder_ms if encoder_inside_prefill else prefill_ms
    )
    decoder_decode_ms = span_ms(prefill_end, complete)
    decoder_ms = decoder_prefill_ms + decoder_decode_ms
    row: dict[str, float | None] = {
        "e2e_ms": e2e_ms,
        "encoder_ms": encoder_ms,
        "decoder_prefill_ms": decoder_prefill_ms,
        "decoder_decode_ms": decoder_decode_ms,
        "decoder_ms": decoder_ms,
        "ctc_timestamp_ms": 0.0,
        "other_ms": e2e_ms - encoder_ms - decoder_ms,
        "ttft_ms": span_ms(admission, first_ts(events, "stage_first_stream_chunk_sent"))
        or None,
    }
    for key in REPORTED:
        row[key.replace("_ms", "_ratio")] = row[key] / e2e_ms if e2e_ms else None
    row["ar_ratio"] = row["decoder_ratio"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("event_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-first", type=int, default=0)
    args = ap.parse_args()

    timelines = reconstruct_timelines(args.event_dir)
    ordered = sorted(timelines.values(), key=lambda t: t.t0_ns or 0)
    inventory = collections.Counter(
        (ev.get("stage"), ev["event_name"]) for t in ordered for ev in t.events
    )
    rows = [analyze(t.events) for t in ordered[args.skip_first :]]
    rows = [r for r in rows if r["e2e_ms"]]
    if not rows:
        raise SystemExit(f"no complete requests in {args.event_dir}")
    keys = sorted({k for r in rows for k in r})
    summary = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            summary[k] = {
                "mean": statistics.fmean(vals),
                "p50": statistics.median(vals),
                "n": len(vals),
            }
    Path(args.out).write_text(
        json.dumps(
            {
                "summary": summary,
                "event_inventory": {
                    f"{s}:{n}": c for (s, n), c in sorted(inventory.items(), key=str)
                },
                "requests": rows,
            },
            indent=2,
        )
    )
    print(f"requests={len(rows)}  (mean  p50)")
    for k in keys:
        if k in summary:
            e = summary[k]
            fmt = "{:9.3f} {:9.3f}" if k.endswith("ratio") else "{:9.1f} {:9.1f}"
            print(f"  {k:40s} " + fmt.format(e["mean"], e["p50"]))
    print("event inventory:")
    for (s, n), c in sorted(inventory.items(), key=str):
        print(f"  {s}:{n} {c}")


if __name__ == "__main__":
    main()
