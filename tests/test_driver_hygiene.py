"""Structural gates for the shell drivers.

These exist because helper drift caused real defects: five different answers to
"is this lane recorded?" and a deadline calculation that compared local time to
UTC and reported -479 minutes, silently disabling its own alarm.
"""
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXTRAS = ROOT / "extras"
ATTIC = EXTRAS / "attic"


def _active_drivers():
    """Drivers that are maintained. The attic is frozen evidence, not code."""
    return [p for p in EXTRAS.glob("*.sh") if ATTIC not in p.parents]


def test_every_shell_driver_parses():
    """A driver that does not parse cannot fail honestly -- it fails confusingly."""
    bad = []
    for p in list(EXTRAS.rglob("*.sh")) + list((ROOT / "shared").rglob("*.sh")):
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True)
        if r.returncode != 0:
            bad.append((p.relative_to(ROOT), r.stderr.decode()[:120]))
    assert not bad, f"shell syntax errors: {bad}"


def test_common_lib_exists_and_parses():
    lib = EXTRAS / "lib" / "common.sh"
    assert lib.is_file(), "extras/lib/common.sh is the single source of driver helpers"
    assert subprocess.run(["bash", "-n", str(lib)], capture_output=True).returncode == 0


def test_attic_is_frozen_and_hashes_match():
    """The attic holds evidence. If a byte changes, a report citation is no
    longer traceable to the script that produced it."""
    manifest = json.loads((ATTIC / "trn2-window" / "MANIFEST.json").read_text())
    drifted = []
    for entry in manifest["scripts"]:
        p = ATTIC / "trn2-window" / entry["name"]
        assert p.is_file(), f"frozen script missing: {entry['name']}"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != entry["sha256"]:
            drifted.append(entry["name"])
    assert not drifted, f"attic scripts were EDITED (they must not be): {drifted}"


def test_attic_is_never_sourced_by_maintained_code():
    """Frozen evidence must not become a live dependency."""
    offenders = []
    for p in _active_drivers() + list((ROOT / "shared").rglob("*.sh")):
        if "attic" in p.read_text():
            offenders.append(p.relative_to(ROOT))
    assert not offenders, f"maintained code references the attic: {offenders}"


def test_converted_driver_uses_the_shared_library():
    """The proof-of-concept conversion must not silently regress."""
    src = (EXTRAS / "run_dataloader_isolation.sh").read_text()
    assert "lib/common.sh" in src
    assert "np_recorded" in src
    for local_def in ("\nhave() {", "\nstep() {"):
        assert local_def not in src, "driver re-declared a shared helper"
