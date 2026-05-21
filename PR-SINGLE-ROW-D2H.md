# simplify MTP verify_h: replace bulk CPU buffer with single-row lazy reads

## Summary

The MTP speculative implementation stored ALL target hidden states in a
per-sequence `verify_h` CPU buffer (~N × 20 KB per batch), populated eagerly
via `llama_get_embeddings_pre_norm_ith` in a loop. However, only one row
is ever used — the accepted-position row in `accept()`.

## What changed

- **Removed** `verify_h` vector (per-sequence float buffer) — no longer needed
- **Removed** `verify_h_rows` vector — derived from `i_batch_beg`/`i_batch_end`
- **`process()`**: reads only the last row for `pending_h` default (single D2H)
- **`accept()`**: reads the accepted-position row lazily from GPU at point of use

## Diff

1 file: `common/speculative.cpp`, -22 lines, +13 lines.

## Testing

Verified on Qwen3.6 27B (dense) + Qwen3.6 35B-A3B (MoE), CUDA backend:
- Single GPU (RTX 3090, 24 GB): MTP n=2 works, draft acceptance unchanged
- Dual GPU layer split (3090 + 5070 Ti): MTP n=2 works, speedup preserved
- Generated text identical to baseline
- No new API, no context changes, no performance regression
