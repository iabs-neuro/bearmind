# Path C — Benchmark-driven short paper: wavelet event detection for calcium imaging

**Date:** 2026-06-10
**Status:** Plan for a standalone short methods communication (decision pending).

---

## 1. Thesis / contribution (one line)

> An open, frame-rate-adaptive implementation of wavelet-ridgewalking calcium **event detection**, benchmarked for the first time against modern spike-inference methods (CASCADE, MLspike, OASIS) on a large simultaneous-ephys ground-truth corpus — establishing *when event detection beats spike inference*, and quantifying frame-rate invariance, coverage, and speed.

The paper is **not** "we beat everyone on accuracy." It is "here is the missing benchmark for the event-detection niche, plus an open robust implementation."

## 2. Why this is publishable (the gap)

- The core wavelet-ridgewalking method (Neugornet, O'Donovan, Ortinski 2021, *Front. Neurosci.* 15:620869) is published and shown to beat dF/F0 thresholding — but has **no maintained open-source implementation**, and was **never benchmarked against modern spike-inference methods** (CASCADE/MLspike/OASIS) on ground-truth ephys.
- Spike-inference benchmarks (Rupprecht 2021) evaluate **spike-rate** accuracy — a *different task* from **event detection** (transient identification, F0-independent). Nobody has done a head-to-head at the **event level**.
- DRIADA's genuine additions (frame-rate-adaptive scales, 5-tier cascade → 42%→80% coverage, MAD fallback, Numba speed, NNLS overlap handling) are **engineering**; they only become a *paper* when validated on ground truth. The benchmark is the scientific contribution; the implementation is the artifact.

## 3. Datasets (all public)

| Dataset | What | Use |
|---|---|---|
| **Rupprecht 2021 ground-truth DB** (Nat. Neurosci.; github.com/HelmchenLabSoftware/Cascade + Zenodo) | 298 neurons, >35 h, **simultaneous juxtacellular ephys + 2P imaging**, zebrafish+mouse, indicators OGB-1 / GCaMP6f,s / jGCaMP7 / jRGECO / **GCaMP8** (added 2024), broad SNR | **Primary** ground truth |
| **spikefinder** (Berens et al. 2018) | simultaneous ephys+imaging, older/smaller | Cross-check / external validity |
| **Synthetic transients** (own generator) | known kinetics, controllable SNR & overlap | Frame-rate & overlap experiments (E2, E5) |
| Lab's own GCaMP6s CA1 data | the autoinspect corpus | Qualitative case study (optional) |

## 4. Methods compared (all open-source, run at their authors' recommended settings)

1. **dF/F0 + n·MAD threshold** — naive baseline (Neugornet's comparison target).
2. **OASIS** deconvolution (CaImAn / Suite2p) → events.
3. **MLspike** (model-based; MATLAB — Python wrapper or drop if time-boxed).
4. **CASCADE** (pretrained GCaMP6/GCaMP8 models) → spike-rate → events.
5. **Wavelet ridgewalking — DRIADA** (this work): FPS-adaptive scales, cascade, MAD fallback, Numba.
   - If feasible, a **"vanilla wavelet"** ablation (fixed scales, no cascade) to isolate DRIADA's deltas vs the base Neugornet method.

> Fairness rule: each method tuned per its own best practice; report both default and tuned. Never strawman a competitor.

## 5. Ground-truth **event** definition (the crux — pre-register this)

Spike-inference GT gives spike *times*; we need *event* GT.

- **GT events:** cluster true spike times into transient-generating events — spikes separated by < `τ_merge` merge into one event (`τ_merge` ∝ indicator decay). GT event onset = first spike of each cluster.
- **Detected events:** every method (incl. OASIS/MLspike/CASCADE) emits an inferred trace; threshold + cluster into events with the **same** rule, so all are compared at the event level.
- **Matching:** one-to-one (Hungarian) within tolerance window `Δ` (≈ max(rise-time, 0.3 s)). TP=matched, FP=spurious, FN=missed.
- **Robustness:** report sensitivity to `τ_merge` and `Δ` (the make-or-break that reviewers will probe).

## 6. Metrics

- **Detection:** event-level precision, recall, **F1**.
- **Timing:** onset jitter = median |Δt| of matched events.
- **Specificity:** false events / minute on **verified-silent** epochs (no spikes).
- **Bias (links to Neugornet):** event-count bias, event-duration bias vs GT.
- Stratify everything by **indicator** and by **imaging SNR tertile**.

## 7. Experiments (each maps to a claimed DRIADA contribution)

- **E1 — Accuracy** (primary table): F1 / jitter / FP-rate per method × indicator × SNR. Honest headline: where wavelet/DRIADA wins (low SNR, slow/astrocytic signals, F0-independence) and where CASCADE/MLspike win (high-SNR fast spiking).
- **E2 — Frame-rate invariance** (DRIADA's signature result): decimate native recordings to {7.5, 15, 30, 60} Hz; plot F1 vs rate. Hypothesis: FPS-adaptive scales → flat F1; fixed-scale / rate-assuming methods → degrade. **This is the cleanest novel result.**
- **E3 — Coverage** (the 42%→80% story): fraction of neurons yielding usable event features, esp. low-SNR/sparse cells; cascade lifts coverage without precision collapse.
- **E4 — Speed:** runtime/neuron (Numba + MAD fallback) vs OASIS/MLspike/CASCADE(GPU).
- **E5 — Overlap recovery** (synthetic): overlapping transients at varying ISI; onset/amplitude recovery (NNLS vs others).

## 8. Expected outcome & honest framing

Likely: CASCADE/MLspike win raw spike-timing at high SNR; **wavelet/DRIADA is competitive-to-better at event-level detection on low-SNR, slow, or astrocytic signals, and clearly wins on frame-rate invariance, coverage, speed, and F0-independence.** Framing = *decision guidance* ("detect events vs infer spikes — when and why") + an open implementation, not a leaderboard sweep. A method that loses E1 but wins E2–E4 is still a solid short paper.

## 9. Venue

- **First choice:** *eNeuro* (Methods/New Research) or *Journal of Neuroscience Methods* — short methods report, benchmark-friendly.
- **Natural fit:** *Frontiers in Neuroscience* (Neugornet's venue, receptive to this exact topic).
- **Also:** *Neuroinformatics*; **bioRxiv preprint first** (priority + feedback).
- Pairs well with the **DRIADA JOSS** software paper (software artifact ↔ this validation paper cross-cite).

## 10. Effort & timeline (~1–1.5 person-weeks)

| Step | Days |
|---|---|
| Download Rupprecht DB + loader | 1 |
| GT-event converter + matching + metrics harness | 2 |
| Wrap methods (DRIADA ✓, OASIS, CASCADE pip, dF/F0 trivial; MLspike optional) | 2 |
| Run E1–E5 + figures | 2–3 |
| Write 4–6 pp short report | 2 |

**Risks:** (a) MLspike is MATLAB → integration friction (drop or Python port); (b) event-definition scrutiny → pre-register + sensitivity; (c) method may lose E1 → frame honestly around E2–E4.

## 11. Minimal viable version (if time-boxed)

E1 (Rupprecht, GCaMP subset) + **E2 (frame-rate invariance)** + E4 (speed). Drop E5 and MLspike. Still a publishable short note, with E2 as the distinctive result.

## 12. Attribution (do this regardless)

- Cite **Neugornet et al. 2021** (*Front. Neurosci.*) as the base method everywhere; state DRIADA's additions explicitly (FPS-adaptive scaling, cascade, MAD fallback, Numba, NNLS).
- Note wavelet-ridgewalking's deeper origin in mass-spectrometry signal analysis (the technique predates calcium imaging) — pre-empts "is the core novel?" reviews by being upfront.
