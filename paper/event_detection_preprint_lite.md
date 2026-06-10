# bioRxiv preprint "lite" — frame-rate-adaptive, high-coverage calcium event detection

**Date:** 2026-06-10
**Goal:** cheap, fast priority-establishing preprint for DRIADA's event-detection improvements over Neugornet 2021 — using **only in-house + synthetic data** (no external ground-truth corpus). Full cross-method benchmark (Path C) comes later and supersedes this.

---

## 0. What this is / is NOT

- **IS:** a methods/software validation note documenting two distinctive engineering contributions — **frame-rate invariance** and **coverage via cascading** — with synthetic ground truth + real-data self-consistency, plus an open Python implementation.
- **IS NOT:** a head-to-head accuracy benchmark vs OASIS/CASCADE/MLspike/FluoroSNNAP (that needs ephys ground truth → Path C). State this explicitly as scope.
- **Honesty:** the frame-rate-invariance claim is *tested here, not assumed* — if FPS-adaptive scaling does not clearly beat fixed scales, the headline must change. Run E2 first as a go/no-go.

## 1. Working title

> *Frame-rate-adaptive wavelet event detection for calcium imaging: a robust, high-coverage open-source implementation*

## 2. Positioning (1 paragraph)

Event detection answers a different question than spike inference — robust, F0-independent identification of discrete transients for downstream feature extraction, not single-spike reconstruction. The base method (wavelet ridgewalking, Neugornet et al. 2021) has no maintained open implementation and uses fixed wavelet scales. We provide an open Python implementation (DRIADA) that (i) makes detection **invariant to acquisition frame rate** and (ii) **maximizes coverage** through a cascading fallback, and we validate both.

## 3. Experiments (both need NO external data)

### E2 — Frame-rate invariance (headline; two parts)

**E2a — Synthetic, true ground truth.**
- Generator: traces = Σ transients with double-exponential kernel `(1−e^{−t/τr})·e^{−t/τd}`, τr,τd fixed in **physical time** (e.g. 0.1 s / 0.6 s), random onsets, amplitudes ~ lognormal; add noise to target SNR ∈ {2, 4, 8}.
- Sample the SAME underlying signal at fps ∈ {5, 7.5, 10, 15, 20, 30, 60} Hz.
- Run **DRIADA (FPS-adaptive scales)** vs **fixed-scale ablation** (Neugornet-style, tuned at 20 Hz).
- True onsets known → **event F1** (match within physical Δ=0.3 s) and **onset error** vs fps.
- **Expected:** adaptive ≈ flat F1 across rates; fixed degrades away from 20 Hz. This is the whole point.

**E2b — Real data, self-consistency (no GT needed).**
- Take native-rate recordings (your GCaMP6s miniscope traces); decimate each to lower rates.
- Metric: cross-rate **agreement** of detected events — event-count stability, onset Jaccard/F1 (native as reference), and stability of estimated kinetics (t_rise, t_off). Adaptive should stay self-consistent; fixed should drift.

### E3 — Coverage via cascading (real data, descriptive)

- On your labeled candidates (subset of the 92,242; report on KEEP-labeled genuine neurons).
- "Characterizable" = detector yields valid event features (≥ N events + converged kinetics fit).
- **Single-pass (tier-1 wavelet only)** vs **full 5-tier cascade** → fraction characterizable (target replicate ≈ 42% → 80%), with per-tier contribution.
- **Sanity:** among neurons rescued by tiers 2–5, fraction labeled KEEP is high → cascade rescues *genuine* neurons, not artifacts (uses your existing labels; no new GT).
- Optional: rescued-neuron event-feature distributions overlap tier-1 → features are plausible.

### (optional) E4 — Speed

- Runtime/neuron: Numba ridge extraction + MAD fallback vs naive. One bar plot. Cheap.

## 4. Figures (3, maybe 4)

1. **Fig 1** — Frame-rate invariance: F1 & onset-error vs fps, adaptive vs fixed (synthetic, by SNR). + inset: real-data cross-rate agreement.
2. **Fig 2** — Coverage: characterizable fraction, single-pass vs cascade, per-tier stacked bar; + rescued-neuron genuineness.
3. **Fig 3** — Method illustration: example trace, CWT scalogram, ridges, detected events at two frame rates (schematic + real).
4. (opt) **Fig 4** — Speed bar.

## 5. Data & code (all in-house)

- **Synthetic:** ~50–100-line generator (check DRIADA for an existing simulator to reuse).
- **Real:** existing DRIADA-processed sessions / autoinspect traces; a representative subset (a few sessions across paradigms) is enough.
- **Code:** the detector already exists in DRIADA; only the experiment harness (decimation loop, synthetic gen, matching/metrics, coverage tally) is new. Release as a `reproduce/` folder → strengthens the open-implementation claim.

## 6. Effort (~3–5 days)

| Step | Days |
|---|---|
| Synthetic generator + E2a | 1 |
| E2b decimation on real data | 1 |
| E3 coverage (largely from existing kinetics_source/opt + reruns) | 0.5 |
| (opt) E4 speed | 0.25 |
| Figures | 1 |
| Write 4–6 pp + post to bioRxiv | 1–1.5 |

## 7. Section skeleton

1. **Intro** — event vs spike question; Neugornet base + its two practical gaps (fixed scales, coverage); open implementation.
2. **Methods** — wavelet ridgewalking recap (cite Neugornet 2021); FPS-adaptive scaling (scale = time·fps); 5-tier cascade; event features; synthetic generator.
3. **Results** — E2a, E2b, E3 (+E4).
4. **Discussion** — scope (no external benchmark yet; Path C forthcoming); when to prefer event detection; availability (DRIADA, reproduce/).

## 8. Risks / pitfalls

- **Headline depends on E2a** — run it first; if adaptive doesn't beat fixed, pivot the framing to coverage+openness only.
- **Coverage stat must reproduce** cleanly on data (42%→80% is currently an internal report number).
- **Reviewer "so what vs OASIS/CASCADE"** — acceptable for a preprint; set expectation that the accuracy benchmark is Path C.
- **Self-overlap with the autoinspect DCNA paper** — keep distinct: autoinspect = QC system using events; this = the detector's robustness/coverage validation. Cross-cite, don't repeat figures.

## 9. Why do this (payoff)

- Establishes **priority/credit** for FPS-adaptation + cascade cheaply, with real evidence (not release notes).
- Produces the **reusable harness** that Path C extends (synthetic gen + metrics + decimation) — not throwaway work.
- Gives a citable artifact to reference from the JOSS DRIADA paper and the autoinspect paper.
