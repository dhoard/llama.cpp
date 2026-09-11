#!/usr/bin/env python3
"""Measure llama-server prompt and decode throughput for Ornith."""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


UNIT = "Ornith ROCm benchmark line {index}: compare the stable response path. "


def request_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def token_count(url, prompt, timeout):
    response = request_json(
        url + "/tokenize",
        {"content": prompt, "add_special": False},
        timeout,
    )
    return len(response["tokens"])


def build_prompt(count):
    return "".join(UNIT.format(index=index) for index in range(count))


def make_prompt(url, budget, timeout, reserve):
    target = max(1, budget - reserve)
    high = max(1, target // 12)
    while token_count(url, build_prompt(high), timeout) < target:
        high *= 2

    low = 0
    while low + 1 < high:
        middle = (low + high) // 2
        if token_count(url, build_prompt(middle), timeout) >= target:
            high = middle
        else:
            low = middle

    prompt = build_prompt(high)
    return prompt, token_count(url, prompt, timeout)


def completion(url, prompt, n_predict, timeout, seed):
    started = time.monotonic()
    response = request_json(
        url + "/completion",
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0,
            "seed": seed,
            "cache_prompt": False,
        },
        timeout,
    )
    elapsed = time.monotonic() - started
    timings = response.get("timings", {})
    return {
        "elapsed_s": elapsed,
        "prompt_n": timings.get("prompt_n", 0),
        "prompt_per_second": timings.get("prompt_per_second", 0.0),
        "predicted_n": timings.get("predicted_n", 0),
        "predicted_per_second": timings.get("predicted_per_second", 0.0),
    }


def median(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.median(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--contexts", default="8192,32768,65536,131072,196608,262144")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--metadata", default="")
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    url = args.url.rstrip("/")
    contexts = [int(value) for value in args.contexts.split(",") if value]
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.warmups < 0:
        parser.error("--warmups must not be negative")

    result = {
        "metadata": json.loads(args.metadata) if args.metadata else {},
        "server_args": args.server_arg,
        "url": url,
        "contexts": [],
    }

    for budget in contexts:
        prompt, prompt_tokens = make_prompt(url, budget, args.timeout, args.n_predict)
        print(f"context={budget} prompt_tokens={prompt_tokens}", flush=True)

        for index in range(args.warmups):
            print(f"  warmup={index + 1}", flush=True)
            completion(url, prompt, args.n_predict, args.timeout, args.seed)

        rows = []
        for index in range(args.repetitions):
            row = completion(url, prompt, args.n_predict, args.timeout, args.seed)
            rows.append(row)
            print(
                f"  run={index + 1} prompt_t/s={row['prompt_per_second']:.2f} "
                f"decode_t/s={row['predicted_per_second']:.2f}",
                flush=True,
            )

        result["contexts"].append(
            {
                "context_tokens_requested": budget,
                "prompt_tokens": prompt_tokens,
                "repetitions": rows,
                "median_prompt_per_second": median(rows, "prompt_per_second"),
                "median_predicted_per_second": median(rows, "predicted_per_second"),
                "median_elapsed_s": median(rows, "elapsed_s"),
            }
        )

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError, urllib.error.HTTPError) as error:
        raise SystemExit(f"benchmark failed: {error}")
