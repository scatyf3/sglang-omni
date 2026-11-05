# SPDX-License-Identifier: Apache-2.0
"""Drive a speech-mode Qwen3-Omni server under the request event recorder."""

from __future__ import annotations

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

TEXT_PROMPTS = [
    "Explain in three sentences why the sky is blue.",
    "Give me a short recipe for scrambled eggs.",
    "What is the capital of France and what is it famous for?",
    "Describe a sunset over the ocean in two sentences.",
    "Tell me a short fun fact about octopuses.",
    "How do airplanes stay in the air? Keep it brief.",
    "Summarize the plot of Romeo and Juliet in a few sentences.",
    "What should I pack for a weekend hiking trip?",
    "Why do leaves change color in autumn?",
    "Give me three tips for learning a new language.",
    "What is machine learning, explained to a child?",
    "Describe how to make a cup of tea.",
    "What causes thunder and lightning?",
    "Recommend a book for a long flight and say why.",
    "How does a refrigerator keep food cold?",
    "What are the benefits of regular exercise?",
]

AUDIO_PROMPTS = ["tests/data/query_to_cars.wav", "tests/data/query_to_draw.wav"]


def build_request(index: int, mode: str, max_tokens: int) -> dict:
    body = {
        "model": "qwen3-omni",
        "modalities": ["text", "audio"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if mode == "text":
        body["messages"] = [{"role": "user", "content": TEXT_PROMPTS[index % len(TEXT_PROMPTS)]}]
    else:
        body["messages"] = [{"role": "user", "content": ""}]
        body["audios"] = [AUDIO_PROMPTS[index % len(AUDIO_PROMPTS)]]
        if mode == "audio_image":
            body["images"] = ["tests/data/cars.jpg"]
    return body


def send_one(base_url: str, index: int, body: dict) -> dict:
    started = time.perf_counter()
    response = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=600)
    wall_s = time.perf_counter() - started
    record = {"index": index, "status": response.status_code, "client_wall_s": wall_s}
    if response.status_code != 200:
        record["error"] = response.text[:500]
        return record
    payload = response.json()
    choice = payload["choices"][0]["message"]
    record["text"] = choice.get("content", "")
    record["usage"] = payload.get("usage")
    audio = choice.get("audio") or {}
    if audio.get("data"):
        audio_bytes = base64.b64decode(audio["data"])
        # 24 kHz mono PCM16 WAV with a 44-byte header.
        record["audio_seconds"] = max(len(audio_bytes) - 44, 0) / (24000 * 2)
    return record


def wait_for_sidecars(event_dir: Path, timeout_s: float) -> list[dict]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        sidecars = sorted(event_dir.glob("run_status_*.json"))
        statuses = []
        for path in sidecars:
            try:
                statuses.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                break
        else:
            if sidecars and all("complete" in status for status in statuses):
                return statuses
        time.sleep(1.0)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--event-dir", required=True)
    parser.add_argument("--mode", choices=("text", "audio", "audio_image"), default="text")
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--no-profile", action="store_true", help="Warmup only; do not record events")
    args = parser.parse_args()

    event_dir = Path(args.event_dir)
    if not args.no_profile:
        event_dir.mkdir(parents=True, exist_ok=False)
        requests.post(
            f"{args.base_url}/start_request_profile",
            json={"run_id": args.run_id, "event_dir": str(event_dir), "event_buffer_capacity": 65536},
            timeout=60,
        ).raise_for_status()

    bodies = [build_request(i, args.mode, args.max_tokens) for i in range(args.num_requests)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records = list(pool.map(lambda pair: send_one(args.base_url, *pair), enumerate(bodies)))
    elapsed_s = time.perf_counter() - started

    statuses: list[dict] = []
    if not args.no_profile:
        requests.post(f"{args.base_url}/stop_request_profile", json={}, timeout=60).raise_for_status()
        statuses = wait_for_sidecars(event_dir, timeout_s=120)

    summary = {
        "run_id": args.run_id,
        "mode": args.mode,
        "concurrency": args.concurrency,
        "num_requests": args.num_requests,
        "max_tokens": args.max_tokens,
        "elapsed_s": elapsed_s,
        "failed": sum(1 for r in records if r["status"] != 200),
        "sidecars_complete": [s.get("complete") for s in statuses],
        "records": records,
    }
    out_path = event_dir.parent / f"client_{args.run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"run_id={args.run_id} mode={args.mode} c={args.concurrency} n={args.num_requests} "
        f"failed={summary['failed']} elapsed={elapsed_s:.1f}s sidecars={summary['sidecars_complete']}"
    )
    for record in records:
        print(
            f"  #{record['index']} status={record['status']} wall={record['client_wall_s']:.2f}s "
            f"audio={record.get('audio_seconds', 0):.2f}s usage={record.get('usage')}"
        )


if __name__ == "__main__":
    main()
