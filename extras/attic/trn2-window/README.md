# Attic — the Trainium2 capacity window, 2026-08-05/06

Eleven orchestrators written in one day to drive a **single non-refundable
24-hour Capacity Block** (`cr-08dc8b22d254cd3da`, trn2.3xlarge, sa-east-1,
$53.64). The instance was terminated on schedule at 11:00Z on 2026-08-06.

## These are evidence, not library code

They are frozen **verbatim**, with sha256 in `MANIFEST.json`, so any claim in
the report can be traced to the exact script that produced it. They are:

- never sourced by maintained code
- never edited
- never depended on

An independent review made the case plainly: for evidence-bearing scripts,
preserve the record rather than tidying it. Refactoring them would improve
nothing — the hardware they drove no longer exists — and would break the chain
between a published number and the code that generated it.

## Why there are eleven of them

Each is a queue re-order forced by something going wrong, and the sequence is
the story of the window:

| script | why it was created |
|---|---|
| `followon` | move the quality gate onto the box so it did not depend on a laptop |
| `followon2` | stage 1 was mid-execution; **bash reads a running script lazily by byte offset**, so appending to it would have executed garbage |
| `followon3` | isolate the `ctx_16384` retry, which had already taken one instance down |
| `followon4` | demote the optional passes behind the symmetry items |
| `recover` | replace stages 3–4 after killing lanes left `EADDRINUSE` collateral; added the port guard |
| `final` | new-information-first: frontier and maxutil before redundant seed lanes |
| `optional_last`, `ladder_last` | further demotions as the window shrank |
| `tail`, `lastwindow` | fill the last idle hours with the two open questions |
| `cifar_vit_retry` | on-demand capacity watcher for the one lane that never passed |

## The two durable lessons

1. **Never append to a script that is currently executing.** bash reads lazily
   by byte offset and resumes at a stale position. This is why stage 2 is a
   separate file rather than three more lines in stage 1.
2. **A receipt written during a forced shutdown is an artifact, not a result.**
   `have()` cannot tell the difference, so three lanes were silently skipped on
   receipts that recorded operator action rather than measurement (§29.4).

Both lessons are now encoded in `extras/lib/common.sh`, which is where new
drivers get their helpers.
