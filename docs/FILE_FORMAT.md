# File format (v1 and v2)

All integers and floats are little-endian. The file is a 128-byte header followed by four sections.

## Header

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 8 | magic | bytes `48 4E 53 57 46 52 47 00` (`"HNSWFRG\0"`) |
| 8 | 4 | version | int32: `1` (legacy) or `2` (with tombstones) |
| 12 | 4 | endianness | int32, `1` = little-endian |
| 16 | 4 | dimension | int32, positive |
| 20 | 4 | metric | int32: `0` = L2, `1` = COSINE, `2` = INNER_PRODUCT |
| 24 | 4 | m | int32, in [2, 128] |
| 28 | 4 | efConstruction | int32 |
| 32 | 4 | efSearch | int32 |
| 36 | 4 | maxElements | int32 |
| 40 | 4 | size | int32, actual number of vectors |
| 44 | 4 | entryPoint | int32 |
| 48 | 4 | maxLevel | int32 |
| 52 | 8 | seed | int64 |
| 60 | 8 | vectorsOff | int64 byte offset, 64-byte aligned |
| 68 | 8 | levelsOff | int64 byte offset |
| 76 | 8 | layer0Off | int64 byte offset |
| 84 | 8 | upperOff | int64 byte offset |
| 92 | 8 | upperLen | int64 count of int32 values in the upper section |
| 100 | 4 | crc32 | CRC-32 over every byte after the header |
| 104 | 8 | deletedOff | int64 byte offset of the tombstone section; zero in v1 |
| 112 | 4 | deletedLen | int32 count of tombstone ids; zero in v1 |
| 116 | 12 | reserved | zero padding to 128 bytes |

## Sections

- **vectors**: `size * dimension` float32 values in id order. Cosine vectors are already normalized.
- **levels**: `size` int32 values, the highest layer of each node.
- **layer0**: `size * (1 + maxM0)` int32 values; for node `i` the slot at `i * (1 + maxM0)` is its degree, followed by neighbor ids.
- **upper**: for each node with `level >= 1`, in ascending node order, `level * (1 + m)` int32 values using the same degree-then-neighbors layout per layer. Nodes at level 0 contribute nothing; their offsets are derivable by accumulating the levels.
- **deleted** (v2 only): `deletedLen` int32 ids in ascending order, all in `[0, size)`. Version 1 files have `deletedLen = 0` and no section.

## Error handling

`HnswIndex.load` accepts both v1 and v2 and throws `IndexFormatException` (an `IOException`) for a wrong magic, unsupported version, wrong endianness, CRC mismatch, truncated file, inconsistent size/capacity, invalid offsets, or unsorted/duplicate tombstone ids. `save` writes to a sibling `.tmp` file and renames atomically where the filesystem supports it.

`HnswIndex.loadMapped` opens the same file read-only through a shared arena. The returned index supports search but throws `UnsupportedOperationException` from `add` and `save`.
