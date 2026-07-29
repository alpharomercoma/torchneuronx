"""NeuronSampler parsing against the committed neuron-monitor fixture.

No hardware needed: NeuronSampler.parse is a pure function precisely so this
test can exist. The fixture starts life synthetic (shaped from the documented
neuron-monitor format) and is replaced by a real captured line during the
first on-box smoke run.

    python3 -m pytest tests/test_telemetry.py -q   # 4 passed
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
import telemetry  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "neuron_monitor_inf2.json")


def load_fixture():
    with open(FIXTURE) as fh:
        return json.load(fh)


def test_parse_fixture_maps_all_columns():
    util, dev_mib, nc0, nc1, host_mib = telemetry.NeuronSampler.parse(load_fixture())
    assert util == 89.38            # mean of 87.5 and 91.25, rounded
    assert dev_mib == 16384.0       # 16 GiB of device HBM in MiB
    assert nc0 == 87.5
    assert nc1 == 91.25
    assert host_mib == 2048.0       # 2 GiB host-side runtime memory


def test_parse_idle_box_is_zero_not_missing():
    # No attached runtime => idle is a measurement (0.0), never None.
    assert telemetry.NeuronSampler.parse({}) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert telemetry.NeuronSampler.parse(
        {"neuron_runtime_data": []}) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_parse_sums_across_multiple_runtimes():
    obj = load_fixture()
    obj["neuron_runtime_data"].append(
        json.loads(json.dumps(obj["neuron_runtime_data"][0])))
    util, dev_mib, nc0, nc1, host_mib = telemetry.NeuronSampler.parse(obj)
    # Utilization adds per core across runtimes sharing a core; memory sums.
    assert dev_mib == 32768.0
    assert host_mib == 4096.0
    assert nc0 == 175.0


def test_schema_and_extra_columns():
    assert telemetry.SCHEMA[0] == "t_s"
    assert telemetry.SCHEMA[1] == "gpu_util_pct"   # name kept for cross-repo parsing
    assert telemetry.NeuronSampler.extra == [
        "util_nc0", "util_nc1", "host_mem_used_mib"]
