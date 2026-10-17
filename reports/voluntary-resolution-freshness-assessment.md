# Voluntary-resolution pilot: semantic freshness assessment

The existing 24-world selection passes the prespecified **zero exact complete-world repeats** criterion within the available local execution history. It is a development transfer sample, **not an independent holdout**: all 12 object worlds reuse a two-property projection of a table in an earlier, structurally different task. No corresponding machine overlap was found. Selection and assignments were not changed by this audit.

| Proposed family | Selected worlds | Exact complete-world repeats | Relational projected table + realized truth | Target projected table, ignoring unused truth | Any projected exposure |
|---|---:|---:|---:|---:|---:|
| Objects | 12 | 0 | 3 | 11 | 12 |
| Machines | 12 | 0 | 0 | 0 | 0 |

The three relational overlaps are object slots **1, 4, 6**. Target-table overlaps cover **all object slots except 4**. The machine finding concerns the actual named input/output tables, rather than an assertion that their abstract four-row structure is novel.

## Meaning and coverage

Exact identity uses `voluntary_resolution.canonical_world`: the named two-property candidate table and realized true row, invariant to row/column display order, delivery, question, menu and sampling seed. The original two-property information-state collections are directly comparable. Earlier legacy object tasks used incompatible shape vocabulary and four distinct color/shape pairings; legacy machine inputs were 0–6 with outputs modulo 7, whereas this pilot uses inputs 11–19 and outputs 20–39. Their two-column projections were also screened and produced no matches.

Relational and target-relevant tasks contain a third property and different inferential questions; they are not exact complete-task repeats. The secondary screen projects each pair of named columns that retains four distinct rows. Relational projections retain realized truth. Target tasks never reveal or use their internally generated truth, so their projection screen deliberately removes truth and compares the candidate table itself. Altering that unused truth therefore cannot manufacture freshness.

The registry contains **75 live manifests**: 62 with persisted individual/cohort record files, four parent manifests with executed children, one reference-check collection with a result-file presence, and eight without persisted inference evidence. Every assigned world from an evidenced collection is included conservatively, including incomplete or failed runs; no correctness or preference filter is applied. Parent/child/replay duplicates are unioned semantically. All **44 manifests inside local historical archives** resolve to this live manifest inventory. The concept corpus is recorded as executed but semantically incomparable to finite candidate worlds.

Frozen source-hash-matched archives reconstruct 65 evidenced finite-task manifests. The initial `local-smoke-v1` lacks a matching frozen source archive; its **48 recorded task and assignment inputs cover all 48 assigned trials**, and those input tables were screened directly. Their envelopes and input hashes are bound in the receipt. No answer, correctness, probability, activation or preference fields were used; the opaque record checksum was verified without interpreting its outcome fields. Thus this missing source archive does not leave an uncovered local input table.

## Consequences for this pilot

Fresh model calls use separate context and do not update model weights. Reusing a table across studies therefore does not itself transmit an earlier answer to a new call. Within-world randomized delivery comparisons remain interpretable for these selected worlds. Prior investigator exposure and repeated table structure nevertheless limit claims about novel-table generalization and cross-study independence. The accepted scope is a development test of transfer and optional behavior, with all projection reuse disclosed; future confirmation should use a prospectively separate vocabulary/world space.

A descriptive scan of candidate indices `[0, 10000)` under seed 2026101432 finds 64 unique object worlds and 9,988 unique machine worlds. All are unused under the exact complete-world criterion. Under the stronger rule that also excludes relational realized-truth projections and target table exposure, only **7 object worlds** remain, versus 9,988 machine worlds. This explains why a 12-object sample cannot meet that stronger criterion in the present vocabulary. These counts were not used to replace any selected world.

Coverage is limited to persisted local history. Unexported remote calls, external/pretraining exposure, and equivalence under arbitrary renaming are not certified. The eight manifests with no persisted execution evidence remain explicitly listed rather than silently declared never executed.

## Reproduction and bindings

Run from the project root:

```bash
.venv/bin/python scripts/audit_voluntary_freshness.py --output /tmp/voluntary-freshness-recheck.json
```

The script reconstructs source-bound task inputs, rescans the live execution inventory, and fails if it finds an archived manifest absent from the inventory or an unsupported historical generator. A later run may legitimately have a larger registry; compare selection, overlapping worlds and the full registry, not only the boolean. The pilot receipt binds the exact inventory snapshot used here.

- Receipt: `reports/voluntary-resolution-v1-freshness.json`
- Receipt byte SHA-256: `417879d0a83cb34ea8c93f22bd2889cba117bbcc33e607695e2c42b320047586`
- Selection SHA-256: `4e00adb378f2fd584c53cfd58f5e0e54068fc8cac52533ffeb26b78b36b1344a`
- Assignment SHA-256: `1ce1ec3ed878afdbde28ed75809df10141337b164a64f6fce1c950b075b83724`
- Historical registry SHA-256: `d903d4a2d505abab7cc27abf5ca09815d80dce71786e74868ff4474e2a93660e`
- Audit script byte SHA-256: `33436b8ed2948f1e68455341e6fcb6c8ae9d46efcd55cbc1d4fea5fd5462b606`

The receipt includes each manifest and assignment binding, execution-evidence status, frozen archive hashes, all selected canonical worlds, and per-world collision source runs. Targeted tests verify the truth-sensitive versus truth-irrelevant projection distinction, frozen-oracle reconstruction, incompatible legacy domains, and the voluntary task's complete diagnostic inventory. This audit authorizes no compute and makes no psychological or phenomenal claim.
