# LinkedIn Post

I built an approximate nearest-neighbor index from scratch in Java, and the number I am proudest of is not a speed figure. It is 0.9995 recall@10 at 3,026 QPS on a 10,000-vector benchmark, measured against exact brute-force ground truth.

HNSW (Hierarchical Navigable Small World) powers much of modern semantic search. Most engineers use a library. I implemented the layered graph, the diversity-based neighbor selection, SIMD distance kernels, concurrent inserts, and memory-mapped persistence myself, with zero runtime dependencies, so that every design choice was one I could defend.

The honest part: I did not beat hnswlib. At equal recall it is 3.0 to 4.8 times faster, and builds 3.1 times faster. My first version had a bug in the neighbor-selection heuristic that produced sparse, fast graphs at 0.9265 recall; fixing it raised recall to 0.9995 but cost throughput. Reporting that tradeoff mattered more to me than hiding it.

The full recall-versus-QPS picture across hnswforge, hnswlib, Lucene, and pgvector is in the best chart from the project: docs/charts/recall_vs_qps_all.png

The write-up covers what worked, what failed, and the four hypotheses for where the performance gap comes from.

TODO(human): add your personal call to action here (what you are looking for, or what you would like readers to do).
