# TODO(human) Checklist

Every placeholder introduced in the Phase 8 launch documents, plus the outstanding real-data and interview-prep work. Check each item off once completed with your own words or a real measurement.

- [ ] **BLOG** — Why did you personally decide a from-scratch build was worth the time? (`docs/BLOG_POST.md`, section 1)
- [ ] **BLOG** — How would you explain the diversity heuristic to a colleague who does not work on retrieval? (`docs/BLOG_POST.md`, section 2)
- [ ] **BLOG** — Describe a specific moment when the brute-force baseline caught or ruled out a problem. (`docs/BLOG_POST.md`, section 3)
- [ ] **BLOG** — What surprised you most about the heuristic bug? (`docs/BLOG_POST.md`, section 4)
- [ ] **BLOG** — What was your reaction when you first saw the 0.4330 figure, and how did you confirm it was the data? (`docs/BLOG_POST.md`, section 4)
- [ ] **BLOG** — Did the build-time result change where you would invest optimization effort next? (`docs/BLOG_POST.md`, section 4)
- [ ] **BLOG** — Which of the four loss hypotheses do you believe matters most, and how would you test that? (`docs/BLOG_POST.md`, section 6)
- [ ] **BLOG** — What would you measure next to show the value of mmap on a real, fully built index? (`docs/BLOG_POST.md`, section 7)
- [ ] **BLOG** — Which single next step would you actually take first, given a free weekend? (`docs/BLOG_POST.md`, section 8)
- [ ] **LINKEDIN** — Add your personal call to action: what you are looking for, or what you would like readers to do. (`docs/LINKEDIN_POST.md`)
- [ ] **RESUME** — If a specific job posting needs an additional metric (real SIFT1M/GloVe QPS, production RepoPrism hit@6, index size in MB per million vectors), replace the placeholder with a measured value only after the real-data run. (`docs/RESUME_BULLETS.md`)
- [ ] **DATA** — Download SIFT1M and GloVe and rerun the Phase 6 baseline comparison on real data; all current numbers are synthetic uniform vectors (`docs/RESULTS.md`, Phase 6 dataset note).
- [ ] **DATA** — Run the deferred real SIFT1M 10k subset recall check against exact ground truth (`docs/RESULTS.md`, Phase 1).
- [ ] **DATA** — Obtain a real RepoPrism export and run the real 72-case evaluation for hit@6, latency, throughput, and memory (`docs/RESULTS.md`, Phase 7; project plan Week 7).
- [ ] **DATA** — Produce pgvector-only RepoPrism numbers on the real export (`docs/RESULTS.md`, Phase 7).
- [ ] **INTERVIEW** — Read the HNSW paper and `docs/ARCHITECTURE.md`, and explain search and insert on a whiteboard (build spec section 15, item 1).
- [ ] **INTERVIEW** — Re-implement `searchLayer` and `selectNeighborsHeuristic` from memory and compare them with the agent's version (build spec section 15, item 2).
- [ ] **INTERVIEW** — Rerun the benchmarks on your own machine and understand each chart (build spec section 15, item 3).
- [ ] **INTERVIEW** — Be able to answer without notes: why layers exist and the expected layer distribution; what `M`, `efConstruction`, and `efSearch` control; why the heuristic improves recall; what happens on insert when a neighbor exceeds `M`; how deletes work and why they are hard; why SIMD helped and where it stopped; how concurrent inserts stay safe and what races remain; where HNSWForge loses to hnswlib and why; how to scale beyond one machine; and what you would do differently (build spec section 15, item 4; project plan section 10).
- [ ] **INTERVIEW** — Fill in every `TODO(human)` marker in all three launch documents with your own words (build spec section 15, item 5).
