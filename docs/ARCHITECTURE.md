# Architecture

Filled in progressively. The authoritative design is in `HNSWForge_AGENT_BUILD_SPEC.md` sections 3-10.

## Deletes (Phase 3)

Deletes are tombstones: `delete(id)` flips a boolean flag and decrements `liveCount()`. Deleted nodes stay in the layered graph and are still traversed during search; they are filtered only when the final result list is built. This keeps the graph structure stable and makes re-upsert cheap (tombstone the old id, insert a new one), but it has limitations:

- `size()` counts all inserted vectors while `liveCount()` counts only live ones.
- Tombstoned nodes consume memory and can crowd the `efSearch` result set, so recall degrades as the tombstone fraction grows.
- There is no graph repair: a deleted node is never physically removed and its edges are never rewired. A future phase could add background compaction.
