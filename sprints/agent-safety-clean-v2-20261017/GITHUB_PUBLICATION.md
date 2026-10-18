# GitHub research archive

Start with [FINAL_RESULTS_FOR_REVIEW.md](FINAL_RESULTS_FOR_REVIEW.md) and
[STAGE0_HYGIENE_REPORT.md](STAGE0_HYGIENE_REPORT.md). The research program is
**incomplete**: the primary Phase 1 scientific qualification failed, later
attempts failed mandatory hygiene gates, and Phases 2 and 3 were not completed.
Publishing this archive does not change any gate outcome or completion claim.

This archive includes all locally present research files in this sprint:
specifications, source code, tests, frozen execution inputs, scenario designs,
raw transcripts, numerical arrays, receipts, failed attempts, provenance,
logs, and the verified Modal closeout package. Local environments, caches,
downloaded utilities, and credentials are excluded. The repository also
contains separately maintained artifacts from earlier sprints; they were not
used as source code for this clean-room sprint.

Forty large artifacts are stored with Git LFS at their original paths. Install
[Git LFS](https://git-lfs.com/) before cloning, or run these commands in an
existing clone:

```sh
git lfs install
git lfs pull
git lfs fsck
```

`GITHUB_PUBLICATION_MANIFEST.json` records the size and SHA-256 of each included
sprint file except the manifest itself. Hashes describe the original file
bytes, including the content represented by Git LFS pointers. This publication
manifest supplements the existing experiment provenance; it does not replace
or modify any frozen receipt or source closure.

Full model checkpoints remain in their original Modal weight-volume namespace,
`curiosity-study-weights:/agent-safety-clean-v2-20261017/`. They are not local
repository files and are not copied into GitHub. The verified closeout package
is also persisted at
`curiosity-study-artifacts:/agent-safety-clean-v2-20261017/closeout/20261018T082021Z`.
Exact artifact paths and provenance are recorded in the research reports and
infrastructure receipts.
