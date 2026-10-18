# Exp2.3 author v3: terminal failure audit

The author finished all48 full requests after its five passing pilots, but its
all-record authoring contract gate failed. This is a complete preserved failure,
not a passing author receipt or Phase2 authorization.

- 47/48 full records and5/5 pilots are strict VALID; one full record is malformed.
- There are0 capped outputs,0 unclosed reasoning sections, and0 logical
 truncations. H1 and the five-trial H5 pilot pass.
- The malformed record is `exp23:author:exp23_wetlab_07`:9,639 generated tokens,
 normal `stop`, closed reasoning, and `MULTIPLE_TOOL_CALLS`. Its post-think text
 has7 opening and3 closing tool-call tags. All3 non-greedy complete spans fail
 strict JSON decoding. No span was salvaged or substituted.
- Both parser paths reject this record. The saved parser-agreement flag is false
 because no accepted independent call exists for comparison; this is not a
 discrepancy in acceptance between the two parsers.

All53 raw records,53 writeahead requests, and four pre-parser return batches
(5,16,16,16) are retained and hash-linked. Independent stream recomputation
matches the worker status exactly. Fresh isolated CPU replay matches all53
exact prompt strings and recorded prompt-token counts. Central source/input
admission and the actual successful `modal_exp23_author.author` FunctionCall
were verified. FunctionCall success describes execution completion; the returned
scientific-input gate remains failed.

Modal reports author app `ap-WWcFWmAVhCF7Z1lX2cZiLx` stopped with0 tasks at
07:35:52 UTC. No retry or GPU launch was performed. Primary qualification remains
separately gated; no author result overrides it.

Tokenizer/package capture is explicitly POST_LAUNCH. Original generation-time
prompt token IDs and per-generation package attestations were not recorded and
are not claimed. Container-to-app association is operator-supplied; actual
backend call and immutable admission were checked independently.

Detailed errors, chain references, actual backend identity, source hashes, and
recomputation entrypoints are in
[the failure audit JSON](exp23_author_v3_cap16384_terminal_failure_provenance.json).
