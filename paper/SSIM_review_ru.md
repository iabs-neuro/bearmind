# Рецензия — «Structural similarity of neuronal activity maps: a complementary stability criterion for spatial coding in calcium imaging»

*Объём: это рецензия только про письмо и фрейминг. Наука, анализы и фигуры — добротные и интересные; ни одно из замечаний ниже не требует новых экспериментов. Всё — про то, как уже имеющиеся результаты поданы.*

## В целом

Хорошая работа: ясная идея, честная позиция («дополнение, а не замена») и продуманный контроль — пересчёт обоих критериев на одном и том же сигнале — ровно тот ход, которого ждёт скептичный читатель, и вы его сделали. Фигуры аккуратные и рассказывают свою историю. Единственное содержательное замечание: рукопись сейчас читается как тщательный технический отчёт, а не как журнальная статья, и этот разрыв — почти целиком вопрос фрейминга. Разницу стоит назвать прямо, потому что она объясняет всё остальное:

> Отчёт документирует, что было сделано, чтобы это можно было воспроизвести и проверить; его добродетели — полнота и точность, и каждое число оправдано тем, что оно истинно. Статья выстраивает аргумент, который читатель должен запомнить; её добродетели — отбор и нарратив, и число оправдано, только если оно *и есть* мысль. У них разные задачи, и абстракт, написанный с инстинктами отчёта — сначала проблема, всё квантифицировано, процедуры перечислены, — оставляет читателя с множеством истинных фактов и без истории.

Хорошая новость: ваше Введение уже написано как статья — оно открывается place-клетками и когнитивной картой, а затем приходит к проблеме коротких сессий. Абстракту и началам разделов Results нужно лишь перенять тот же инстинкт.

## Абстракт — приоритет

Сейчас он открывается *внутри* технической проблемы («Identifying spatially-tuned neurons … classically relies on the Skaggs spatial-information criterion, which requires dense arena sampling …») и дальше читается как лист параметров — *(1000 cells; 80% …)*, *AUC = 0.95*, *(16 mice, 64 sessions)*, shuffle-тест, *ρ*, *event-count bias*. Из этих чисел право на место имеет только AUC; остальное — в Methods и Results. Сам же хедлайн — *два общепринятых способа назвать клетку place-клеткой расходятся, и мы показываем почему* — спрятан в последнем 40-словном предложении. Переписанная версия — в блоке «Готовые абзацы» ниже.

## В Results — сначала находка, потом процедура

§III-B сейчас открывается машинерией («We applied both criteria to 30,105 CA1 neurons pooled across 64 sessions …»), а смысл (критерии расходятся) приходит предложением позже. Переверните: сначала — что разошлись, потом — как считали. Готовый вариант — ниже.

## Откалибруйте две формулировки под то, что данные уже показывают — чистое словоупотребление, без нового анализа

- Ваш же same-signal контроль говорит, что меры ближе к *независимым*, чем к *антикоррелированным* (самое отрицательное число — это кросс-сигнальная пара). Опираться на «complementary / different facets», а не на «negatively correlated», и безопаснее под рецензию, и, по-моему, интереснее — а стоит это только акцента.
- SSIM меряет, *повторяется* ли карта, что связано, но не тождественно пространственной *селективности*. Фрейминг вклада как «map stability» (а «place cell» пусть следует из неё) снимает очевидный вопрос и заостряет новизну.

## Ремесло на уровне предложений

- *Числа в нарративных местах.* В бегущем тексте — словами («about a third», «substantially different»), а точные значения (*≈33%*, *ρ ≈ −0.46*) — в Results по существу и в подписях, где их и ищут.
- *Ритм.* Почти все предложения длинные, на тире и запятых. Короткое предложение время от времени делает настоящую работу: «The two criteria disagree. Here is why.»
- *Хедж плюс сверхточность.* «≈» при числах с тремя значащими цифрами звучит одновременно неуверенно и придирчиво; выберите один регистр.

## Мелочи, быстро

- Заголовок: текущий точный, но длинный; из вашего же списка #1 («Map reproducibility versus spatial information …») или #5 («… complementary to, not a replacement for …») схватывают результат кратко.
- Порядок ссылок нарушен (в тексте [1‑2] → [6],[7],[8] → [3],[5],[4]); IEEE нумерует по первому появлению — перенумеровать.
- Не забыть удалить блок «ALTERNATIVE TITLES».
- Несколько дефисов вместо тире («populations - a divergence»); «MiniScope v4.4» vs написание в других статьях группы — проход на консистентность.

## Итого

Наука и фигуры готовы; рукописи в основном нужно сменить голос с отчётного на статейный — переписать абстракт вокруг вопроса, вынести находки вперёд процедур, откалибровать пару формулировок под контроли и проредить числа в нарративе. Данных это не касается.

---

# Готовые переписанные абзацы (вставить в статью)

## Abstract

> A hippocampal place cell is known by what it reveals about where an animal is, and the standard test asks how much spatial information its activity carries. That test, however, needs long recordings with many firing events; on the short, sparsely sampled sessions typical of freely-moving calcium imaging it becomes unreliable, and genuinely tuned cells are missed. We ask a different question: not whether a cell is *informative* about space, but whether it draws the *same map* each time the animal explores. We quantify this map stability with the Structural Similarity Index (SSIM), an image-comparison measure that, unlike pixelwise correlation, is sensitive to spatial layout. On simulated data with known ground truth, the stability criterion recovers stable place fields with high accuracy (AUC 0.95), using a threshold calibrated against a matched null rather than chosen by hand. Applied to CA1 recordings in freely-moving mice, stability and spatial information identify substantially different cells — a divergence that survives our controls and traces to a known bias of the information measure toward rarely-firing cells. Map stability is therefore not a replacement for spatial information but a complementary view of place coding, one that remains usable precisely where the classical measure breaks down.

## Results — III-A (Synthetic validation)

> SSIM compares spatial patterns, not pixel values. Two independent noise maps score low, and so do two place fields at different locations; only two maps sharing a field at the same place score high (Fig. 1).
>
> On a simulated population with known ground truth — most cells untuned, the rest carrying stable fields of varying size and amplitude — the mean adjacent-trial SSIM cleanly separates the two groups (AUC 0.95; Fig. 2A,B). We set the decision threshold without tuning it by hand: the untuned cells define a null centred near zero, and a cell counts as tuned if its stability exceeds the null's 95th percentile, fixing the false-positive rate at 5% by construction — the direct analogue of the shuffle test used for spatial information. At this operating point the criterion is accurate and highly specific, with sensitivity limited mainly by small single-field cells (full metrics in Fig. 2D).
>
> A sweep over field size and noise marks the regime where the criterion can be trusted: detection is reliable for all but the smallest, faintest fields, which smoothing washes into the background (Fig. 2C). The same map recovers the previously used fixed cutoff of 0.3 as a much stricter operating point — near the 99th percentile of the null — explaining why it under-detects.

## Results — III-B (Real data)

> On real CA1 data the two criteria part ways. Across neurons pooled from 16 mice, the cells flagged by map stability and by spatial information overlap only weakly (Jaccard ≈ 0.2), and within the population a high stability score tends to accompany a low information score rather than a high one (Fig. 3A,B); about a third of cells pass the stability test and about a fifth the information test. Because a fixed SSIM threshold over-detects when an animal samples the arena only briefly — sparse maps are trivially similar — we assessed stability per neuron with a circular-shift shuffle test, mirroring the classical significance test.
>
> This split is not an artefact of feeding the two measures different signals (dF/F maps for stability, event maps for information). Computing both on the same signal leaves the relationship essentially unchanged — at most weakly negative, and near zero on dF/F — whereas swapping the signal under a fixed measure barely moves its scores. What drives the divergence is a known property of the information rate: expressed per event, it rewards rarely-firing cells, whose sparse maps are the least reproducible. Stability and spatial information are thus better seen as largely independent readouts — map reproducibility on one side, per-event selectivity on the other — than as two estimates of the same thing.
>
> Both criteria, once significance is properly tested, detect fewer cells as the analysed session shrinks toward two minutes; only the naive fixed threshold moves the other way, inflating as coverage drops. Significance testing is therefore not optional on short sessions — it is what keeps either criterion honest.

## Discussion

> We have proposed and validated a stability criterion for spatial coding: a cell counts as place-tuned if it redraws the same map each time the animal explores, measured by structural similarity. On synthetic ground truth the criterion is accurate, and its threshold is set by a matched null rather than by hand, failing only for the smallest, faintest fields. On real CA1 data it does not reproduce the classical spatial-information criterion; it complements it. The two flag largely independent populations, because map reproducibility and per-event selectivity are different things, and because the information rate is biased toward sparse, poorly-reproducible cells. This independence is the point: stability stays usable in exactly the short, sparsely-sampled regime where spatial information becomes unreliable, provided significance is tested per cell rather than by a fixed cutoff. We therefore see map stability not as a replacement for the classical measure but as a second axis of evidence — and a natural next step is to combine selectivity and stability into a single classifier.
