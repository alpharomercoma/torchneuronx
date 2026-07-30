"""fallback_client vs a mock SSE server: schema parity + timing plausibility.

The client's whole reason to exist is that summarize.py must not be able to
tell it apart from `vllm bench serve --save-result`. So this test asserts
(a) every key summarize.parse_serve reads is present, (b) the standard vllm
bench key families are all there, and (c) TTFT/TPOT roughly match the delays
the mock server was programmed with.

    python3 -m pytest tests/test_fallback_client.py -q   # 3 passed
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared", "serve"))
import fallback_client  # noqa: E402

TTFT_DELAY_S = 0.10       # mock: first token after 100 ms
ITL_DELAY_S = 0.02        # mock: 20 ms between tokens
N_TOKENS = 12

# Keys summarize.parse_serve actually reads (verified against the source).
SUMMARIZE_KEYS = [
    "output_throughput", "total_token_throughput", "mean_tpot_ms", "duration",
    "total_output_tokens", "max_concurrency", "p50_ttft_ms", "p99_ttft_ms",
    "p50_tpot_ms", "p99_tpot_ms", "p99_itl_ms", "completed", "failed",
]
# Full vllm-bench-compatible families.
FAMILY_KEYS = [f"{s}_{m}_ms"
               for m in ("ttft", "tpot", "itl", "e2el")
               for s in ("mean", "median", "std", "p50", "p90", "p99")]


class MockSSE(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(TTFT_DELAY_S)
        for i in range(N_TOKENS):
            if i:
                time.sleep(ITL_DELAY_S)
            chunk = json.dumps({"choices": [{"text": f"tok{i} "}]})
            self.wfile.write(f"data: {chunk}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):  # keep pytest output clean
        pass


def run_bench(tmp_path, num_prompts=4, conc=2):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockSSE)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        args = fallback_client.argparse.Namespace(
            base_url=f"http://127.0.0.1:{srv.server_port}", model="mock-model",
            num_prompts=num_prompts, max_concurrency=conc,
            input_len=64, output_len=N_TOKENS, seed=42,
            out=str(tmp_path / "r.json"))
        return fallback_client.Bench(args).run()
    finally:
        srv.shutdown()


def test_schema_superset(tmp_path):
    r = run_bench(tmp_path)
    for k in SUMMARIZE_KEYS + FAMILY_KEYS + [
            "date", "backend", "model_id", "num_prompts",
            "total_input_tokens", "max_output_tokens_per_s",
            "max_concurrent_requests"]:
        assert k in r, f"missing key: {k}"
    assert r["completed"] == 4 and r["failed"] == 0
    assert r["total_output_tokens"] == 4 * N_TOKENS


def test_timings_match_programmed_delays(tmp_path):
    r = run_bench(tmp_path, num_prompts=2, conc=1)
    # TTFT ~100 ms (plus a little overhead), TPOT ~20 ms.
    assert 90 <= r["p50_ttft_ms"] <= 400
    assert 15 <= r["p50_tpot_ms"] <= 60
    assert r["max_concurrent_requests"] == 1


def test_failure_counted_not_raised(tmp_path):
    args = fallback_client.argparse.Namespace(
        base_url="http://127.0.0.1:1", model="m", num_prompts=2,
        max_concurrency=2, input_len=8, output_len=4, seed=0,
        out=str(tmp_path / "r.json"))
    r = fallback_client.Bench(args).run()
    assert r["failed"] == 2 and r["completed"] == 0


def test_poisson_arrivals_pure():
    arr = fallback_client.draw_poisson_arrivals(10.0, 5.0, seed=42)
    assert all(b > a for a, b in zip(arr, arr[1:]))       # strictly increasing
    assert all(0 <= t < 5.0 for t in arr)
    assert 25 <= len(arr) <= 85                            # ~50 expected, wide CI
    assert arr == fallback_client.draw_poisson_arrivals(10.0, 5.0, seed=42)
    assert len(fallback_client.draw_poisson_arrivals(10.0, 5.0, seed=42, cap=7)) == 7


def test_goodput_pure():
    rs = [{"ttft": 100, "tpot": 20, "ntok": 10},
          {"ttft": 5000, "tpot": 20, "ntok": 10},   # ttft SLO miss
          {"ttft": 100, "tpot": 200, "ntok": 10},   # tpot SLO miss
          {"ttft": 100, "tpot": None, "ntok": 10}]  # 1-token: tpot undefined = pass
    g = fallback_client.compute_goodput(rs, 10.0, 3000, 80)
    assert g["slo_attainment_pct"] == 50.0
    assert g["goodput_output_tokens_per_s"] == 2.0
    assert g["slo_ttft_ms"] == 3000 and g["slo_tpot_ms"] == 80


def test_poisson_mode_against_mock(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockSSE)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        args = fallback_client.argparse.Namespace(
            base_url=f"http://127.0.0.1:{srv.server_port}", model="mock-model",
            num_prompts=100, max_concurrency=8,
            input_len=32, output_len=N_TOKENS, seed=7,
            out=str(tmp_path / "p.json"),
            arrival_mode="poisson", request_rate=6.0, duration_s=2.0,
            slo_ttft_ms=3000.0, slo_tpot_ms=80.0)
        r = fallback_client.Bench(args).run()
    finally:
        srv.shutdown()
    assert r["arrival_mode"] == "poisson"
    assert r["requests_offered"] == r["completed"] + r["failed"]
    assert 3 <= r["requests_offered"] <= 30          # ~12 expected
    assert r["slo_attainment_pct"] == 100.0          # mock: 100ms/20ms beats SLOs
    assert r["goodput_output_tokens_per_s"] > 0
