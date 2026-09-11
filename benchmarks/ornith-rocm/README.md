# Ornith ROCm benchmark outputs

Run `scripts/ornith/bench-rocm.sh` from the repository root. The script starts
the server in a ROCm container, measures prompt and decode throughput at several
context sizes, and writes `results.json` in this directory by default.

Override `CONTEXTS`, `REPETITIONS`, `WARMUPS`, `N_PREDICT`, and `OUTPUT` when a
short smoke run or a different report location is needed. The JSON records the
image ID, source revision, server arguments, prompt token count, and each raw
measurement so results remain tied to the binary that produced them.
