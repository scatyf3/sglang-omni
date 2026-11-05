# SPDX-License-Identifier: Apache-2.0
"""Join each case's SeedTTS speed, WER, similarity, UTMOS and AR ratio results into one Markdown table."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def cell(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main() -> None:
    run_dir = Path(sys.argv[1])
    cases = [entry.split(":") for entry in sys.argv[2:]]
    columns = [
        "case", "c", "latency_mean_s", "latency_p95_s", "rtf_mean", "throughput_qps",
        "wer_corpus", "wer_below_50", "sim_mean", "utmos_mean",
        "e2e_ms", "ttft_ms", "ttfa_ms", "thinker_ratio", "talker_ratio", "ar_sum_ratio", "ar_union_ratio",
    ]
    rows = []
    for run_id, conc in cases:
        case_dir = run_dir / run_id
        speed = load(case_dir / "seedtts" / "speed_results.json").get("summary", {})
        wer = load(case_dir / "seedtts" / "wer_results.json").get("summary", {})
        sim = load(case_dir / "seedtts" / "similarity_results.json").get("summary", {})
        utmos = load(case_dir / "seedtts" / "utmos_results.json").get("summary", {})
        ar = load(case_dir / "ar_ratio.json").get("summary", {})

        def mean(key: str) -> float | None:
            entry = ar.get(key)
            return entry["mean"] if entry else None

        rows.append([
            run_id, conc,
            cell(speed.get("latency_mean_s")), cell(speed.get("latency_p95_s")),
            cell(speed.get("rtf_mean"), 3), cell(speed.get("throughput_qps"), 3),
            cell(wer.get("wer_corpus"), 4), cell(wer.get("wer_below_50_corpus"), 4),
            cell(sim.get("speaker_similarity_mean")), cell(utmos.get("utmos_mean"), 3),
            cell(mean("e2e_ms"), 0), cell(mean("ttft_ms"), 0), cell(mean("ttfa_ms"), 0),
            cell(mean("thinker_ratio"), 3), cell(mean("talker_ar_ratio"), 3),
            cell(mean("ar_sum_ratio"), 3), cell(mean("ar_union_ratio"), 3),
        ])
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")
    (run_dir / "summary.json").write_text(json.dumps([dict(zip(columns, row)) for row in rows], indent=2))


if __name__ == "__main__":
    main()
