# Architecture

Filled in progressively. The authoritative design is in `HNSWForge_AGENT_BUILD_SPEC.md` sections 3-10.

## Deletes (Phase 3)

Deletes are tombstones: `delete(id)` flips a boolean flag and decrements `liveCount()`. Deleted nodes stay in the layered graph and are still traversed during search; they are filtered only when the final result list is built. This keeps the graph structure stable and makes re-upsert cheap (tombstone the old id, insert a new one), but it has limitations:

- `size()` counts all inserted vectors while `liveCount()` counts only live ones.
- Tombstoned nodes consume memory and can crowd the `efSearch` result set, so recall degrades as the tombstone fraction grows.
- There is no graph repair: a deleted node is never physically removed and its edges are never rewired. A future phase could add background compaction.

## Concurrency model (Phase 5)

- `size` and `liveCount` are `AtomicInteger`s; `add` reserves an id with `getAndIncrement()` and publishes the node with a release store to an `AtomicIntegerArray` after its vector, level and upper-layer arrays are initialized.
- Searches take no locks. They read `entryPoint`/`maxLevel` from volatile fields, acquire-read each degree word via a `VarHandle`, validate neighbor ids against the current size and skip unpublished nodes. Stale-but-valid neighbor reads are tolerated; a node published later becomes visible without a second lookup.
- Neighbor-list writes are guarded by 4096 striped `ReentrantLock`s. At most one node lock is held at a time, so lock ordering is trivially acyclic; the entry-point lock is never acquired while a node lock is held.
- Random level generation is serialized on a small monitor to keep the seeded sequence deterministic; all other insertion work is parallel. `addAll` uses a fixed thread pool and strided id assignment.
- Deletes use an `AtomicIntegerArray` of tombstones and never mutate the graph.
