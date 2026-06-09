# DCNA 2026 Paper — Automated ML Neuron Screening: Design / Spec

**Date:** 2026-06-09
**Target venue:** DCNA 2026 (https://nctech-lab.ru/dcna2026)
**Status:** Design approved in brainstorming; awaiting spec review.

---

## 1. Venue constraints (hard requirements)

| Item | Value |
|---|---|
| Submission deadline | **June 15, 2026** (6 days out) |
| Acceptance notice | July 15, 2026 |
| Final paper | August 1, 2026 |
| Conference | September 10–12, 2026 |
| Length | **2–4 pages** (we target **4**) |
| Language | **English only** |
| Template | **IEEE conference (IEEEtran `[conference]`)** |
| Submission | Yandex form: https://forms.yandex.ru/cloud/69ef8e111f1eb518b4d3a12b |
| Relevant tracks | AI & Machine Learning; Biology and Medical Applications; Brain–Computer Interfaces |

Original work only; no duplicate/parallel submission.

## 2. Framing decision

**Tool / method paper.** Lead message: *"An automated, interpretable quality-control system for calcium-imaging neuron curation that reaches expert-level accuracy (97.6% precision / 93.3% recall) while remaining fully transparent."* Emphasis on the system, its design choices, metrics, and practical impact — a sharpened, compressed translation of the existing English report (`docs/autoinspect_report_draft.md`).

## 3. Authorship & front matter

- **Authors (as in `paper/eff_dim/main_ru_passive.tex`):** Никита Поспелов, Ольга Рогожникова, Виктор Плюснин, Анна Иванова, Ксения Торопова, Ольга Ивашкина, Константин Анохин — all at *Laboratory of Neuronal Intelligence, Institute for Advanced Brain Studies, Lomonosov Moscow State University*. Names/affiliations rendered in **English** for this submission.
- **Funding `\thanks`:** Non-commercial Foundation for support of science and education "INTELLECT"; N.P. acknowledges the Brain program of the IDEAS Research Center. (Translate the Russian `\thanks` to English.)

## 4. Structure (IEEEtran two-column, ~4 pages, 2 figures)

1. **Title + Abstract (~150 words) + Keywords** — new.
   - Keywords: calcium imaging, quality control, machine learning, interpretable models, explainable boosting machine, neuron curation.
2. **I. Introduction** (report §1–2): manual curation bottleneck; CNMF-E/CaImAn over-segments for sensitivity → false candidates; need for automated *and interpretable* QC. State contribution explicitly.
3. **II. Methods**
   - **A. Data & calcium imaging** — *detailed* paragraph (reuse surgery/ethics/protocol prose from example papers: C57Bl/6, CA1, Miniscope V4.4, GCaMP6s, 1 mm GRIN lens, Bonsai sync, MSU bioethics approval, BEARMIND/CaImAn + NoRMCorre + CellReg preprocessing). Add dataset composition: **92,242 neurons / 187 sessions / 4 paradigms** (LNOF 49,767; NOF 30,398; RFC 7,505; FOF 4,572).
   - **B. Feature space** — 35 features in 5 categories (spatial/morphological, temporal/trace, event-based, reconstruction, algorithm-provided). Condensed; name the categories and 2–3 representative metrics each.
   - **C. Cascading kinetics estimation** — the 5-tier fallback (wavelet strict → wavelet relaxed → threshold strict → threshold relaxed → defaults); tier tracked as a feature; coverage 42% → 80%. DRIADA for event detection.
   - **D. Classifier & objective** — Explainable Boosting Machine (glass-box GAM, 35 features + 20 interactions); F-β objective with β=0.577 (FP 3× costlier than FN); brief deployment modes (rule / ML / hybrid).
4. **III. Results** (report §7)
   - Performance: F-β 0.965±0.004, precision 0.976±0.003, recall 0.933±0.007, AUC 0.978±0.005; stratified CV by session.
   - **Fig 1** — feature importance: trace skewness/kurtosis dominate; traditional CaImAn SNR ranks low while event-based SNR ranks high.
   - **Fig 2** — threshold/asymmetric-cost analysis (F-β vs threshold; FP/FN trade-off).
5. **IV. Discussion & Conclusion** (report §8–9): hours→minutes (~30–40×), consistency/reproducibility, interpretability + active-learning loop, limitations (modality/silent neurons/~20% kinetics defaults), future directions.
6. **Acknowledgments** — V.A. Avetisov & A.S. Gorsky for discussions (from example) — confirm relevance; funding already in `\thanks`.
7. **References** — IEEEtran style.

## 5. Figures

- **Fig 1:** `docs/figures/figure1_feature_importance.png` (English version).
- **Fig 2:** `docs/figures/figure2_threshold_analysis.png` (English version).
- Fig 3 (`figure3_model_comparison.png`) held in reserve only if space permits — default **excluded**.

## 6. Reuse vs. new

- **Reuse directly:** IEEEtran preamble, author block, `\thanks` funding, preprocessing bib entries (`lopes2015bonsai`, `Pnevmatikakis2016`/CaImAn, `Pnevmatikakis2017`/NoRMCorre, `Sheintuch2017`/CellReg) from `paper/eff_dim/references.bib`.
- **Add bib entries:** Giovannucci 2019 (CaImAn, eLife 8:e38173); Zhou 2018 (CNMF-E, eLife 7:e28728); Sych 2019 (DRIADA / multi-fiber photometry, Nat Methods 16:1104); an EBM/InterpretML reference (Nori et al. 2019 / Lou et al. 2013); optionally an F-β reference.
- **Write fresh:** all body prose, directly in **English** (no Russian-draft → translate step, since the source report is already English).

## 7. Deliverables / file layout

- New folder `paper/autoinspect/` with `main.tex` and `references.bib`.
- Figures referenced from `docs/figures/` (or copied into `paper/autoinspect/figs/`).
- Compile target: IEEEtran, English, ≤4 pages.

## 8. Open items to confirm during drafting

- Exact English spelling of author names / affiliation (use a consistent transliteration).
- Whether the Avetisov/Gorsky acknowledgment line applies to this paper.
- Final figure count if page budget gets tight (drop to 1 figure before cutting Methods detail).
