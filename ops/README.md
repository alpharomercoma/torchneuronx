# ops — how this account's runs were operated

Nothing in this directory is needed to reproduce the study. It is the record
of *running* it on one AWS account under a non-refundable capacity window:
what was chased, what was preserved, and what was thrown away.

Reproduction lives in [`docs/runbook/`](../docs/runbook/); the numbers live in
[`REPORT.md`](../REPORT.md) and [`REPORT-EXTENSIONS.md`](../REPORT-EXTENSIONS.md).

| path | what it is |
|---|---|
| `preservation/` | The 2026-08-26 teardown: seven EBS snapshots, AMI pins, KMS keys, restore steps, and the persistent-spot-request trap that spawned a replacement instance mid-teardown. Snapshot and bucket IDs are specific to account 600627330911 and are useless elsewhere. |
| `capacity/` | Trainium2 capacity hunting. `trn2_capacity_watch.sh` polls for capacity unattended, `trn2_block_launch.sh` waits for a Capacity Block to go `active` before deploying, `trn2_preserve_state.sh` snapshots on-box state before a window closes. Cited by [runbook 12](../docs/runbook/12-trainium2.md). |
| `attic/trn2-window/` | **Frozen.** Twelve one-off drivers written against a closing trn2 window, plus `MANIFEST.json` holding their sha256s. Report citations trace to these exact bytes, so `tests/test_driver_hygiene.py` fails if one is edited or if maintained code ever sources them. |
| `PHASE3-STATE.md` | The live working-state log kept during Phases 3-5. Superseded by the report; kept because it records what was *believed* at each point, including the wrong beliefs. |

## Why these are separated

The repo is the companion to the talk *Beyond GPUs: Production LLMs on AWS
Trainium and Inferentia*. Someone arriving from the last slide wants the
harness, the infrastructure and the receipts — not the scaffolding that a
deadline and an empty capacity pool forced into existence. Deleting it would
have been dishonest (several report sections cite it); leaving it at the top
level made the reproduction path harder to find. So it lives here.

One exception stays behind in `extras/`: `trn2_deadline_push.sh` is invoked by
`cdk/user_data/trn2_autorun.sh` at boot and asserted by
`cdk/tests/test_trainium2_stack.py`. It is operational in nature but wired into
a launch path, and moving it would edit live infrastructure that no longer has a
box to run on.
