"""load_telemetry contract: busy-window rule, refusal rules, Neuron columns.

    python3 -m pytest tests/test_summarize.py -q   # 4 passed
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
import summarize  # noqa: E402

HEADER = ("t_s,gpu_util_pct,mem_used_mib,power_w,temp_c,sclk_mhz,mclk_mhz,"
          "util_nc0,util_nc1,host_mem_used_mib\n")


def write_csv(tmp_path, rows):
    p = tmp_path / "x.telemetry.csv"
    p.write_text(HEADER + "".join(rows))
    return str(p)


def test_missing_file_refused():
    assert summarize.load_telemetry("/nonexistent/never.csv") is None


def test_busy_window_excludes_idle_prologue(tmp_path):
    # 3 idle samples (client tokenizing), then 3 busy ones. The busy-window
    # rule must judge saturation over samples 3..5 only.
    rows = [
        "0.0,0.0,10,,,,,0.0,0.0,100\n",
        "1.0,0.0,10,,,,,0.0,0.0,100\n",
        "2.0,5.0,10,,,,,5.0,5.0,100\n",
        "3.0,95.0,16000,,,,,94.0,96.0,2000\n",
        "4.0,100.0,16100,,,,,99.0,100.0,2000\n",
        "5.0,98.0,16050,,,,,97.0,99.0,2000\n",
    ]
    out = summarize.load_telemetry(write_csv(tmp_path, rows))
    assert out is not None
    assert out["busy_window"] == [3, 6]
    assert out["busy_samples"] == 3
    assert out["gpu_util_pct"]["mean"] == 97.67
    # whole-window mean is dragged down by the prologue -- that is WHY the
    # busy window exists; assert the two genuinely differ.
    assert out["whole_window"]["gpu_util_pct"]["mean"] < 60


def test_power_absent_on_neuron_is_none_not_zero(tmp_path):
    rows = ["0.0,95.0,16000,,,,,95.0,95.0,2000\n"]
    out = summarize.load_telemetry(write_csv(tmp_path, rows))
    assert out is not None
    assert out["power_w"] is None       # absent sensor reported absent


def test_all_empty_columns_refused(tmp_path):
    # A CSV with neither utilization nor power carries no evidence.
    rows = ["0.0,,,,,,,,,\n", "1.0,,,,,,,,,\n"]
    assert summarize.load_telemetry(write_csv(tmp_path, rows)) is None
