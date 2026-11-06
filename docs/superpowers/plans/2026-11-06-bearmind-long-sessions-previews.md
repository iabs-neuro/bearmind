# BEARMiND волна A2: часовые сессии и превью по флажку — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Превью-видео делаются только по флажку и сразу, без повторных MC и CNMF-E. Часовая сессия проходит батч без человека: кроп сам находит область с клетками, CNMF-E сам понижает разрешение фона, если не влезает в память, карты порогов считаются параллельно.

**Architecture:** Превью уходят из этапов MC и CNMF-E в два побочных этапа runner (`mc_preview` после `mc`, `cnmf_preview` после `cnmf`). Хэш побочного этапа — настройки превью плюс хэш родителя; основная цепочка от них не зависит, поэтому выключенное или упавшее превью ничего не держит. Понижение для CNMF-E — чистая функция от размеров фильма, модели памяти и бюджета машины; её результат (`ssub_B`) входит в параметры этапа `cnmf`. Кроп ищет поле по избытку карт CNMF-E над их зеркальным нулём внутри освещённой области; после каждой расстановки рамок runner рисует обзорную картинку серии.

**Tech Stack:** Python 3.12, CaImAn 1.13.2, NumPy 2, PyAV, OpenCV, SciPy, matplotlib, PySide6 (только задача 7), pytest. Окружение conda `bearmind`.

**Spec:** раздел «Decisions» ниже — решения автора 2026-11-06, дословно. Замеры, на которые план опирается: `docs/superpowers/specs/2026-11-01-e2e-report.md` («Что решить» п. 1–2, «Куда уходит время MC», «series_lnof: часовая сессия»), `docs/superpowers/specs/2026-10-31-caiman-memory.md` (модель памяти CNMF-E, `compute_W`). Соглашения ядра: `docs/superpowers/plans/2026-10-31-bearmind-batch-core.md`.

**Выполняется после** волны A1 (`docs/superpowers/plans/2026-11-06-bearmind-author-fixes.md`: ExamineCells, группировка слияний, одно правило экспорта с переименованным `bearmind.stages.COMPONENT_RULE`, `autoinspect.seed`, пул MC 16) и фазы 2 (`docs/superpowers/plans/2026-10-31-bearmind-phase2-monitor.md`: `bearmind/monitor.py`, `bearmind/actions.py`, `bearmind/gui/`). Этот план не трогает правило экспорта, автоинспекцию и размер пула MC. A1 добавляет в `defaults.yaml` раздел `pool` (`mc_processes: 16`) и `mc_pool_size(cfg)` в `mc.py`; пул карт порогов (задача 5) туда не кладётся, его размер считается по памяти. Номера строк в тексте — по коду до A1; исполнитель ищет места по именам функций.

## Decisions

Решения автора (спецификация этого плана), дословно:

1. Preview videos (the four MC previews raw / raw_dff / mc / mc_dff, and the CNMF-E QC video) take most of the MC time (41%) and disk. They are made ONLY when a checkbox is set (a config flag, default off), and when it is set they are made right away, without re-running MC or CNMF-E. There is a setting for which downsample to use; default: x2 in time, full size (the current `preview.tbin: 2`). The residual-maps PNG is cheap: decide whether it stays always on and say why. Design how "right away without re-running" works with the runner (for example a separate optional stage after mc / after cnmf whose params are the preview settings, or another way that fits the runner), how a disabled preview shows in the registry/summary without blocking downstream stages, and how a session series already processed gets its previews when the flag is turned on later.
2. Long (hour) sessions: CNMF-E automatic downsampling. When a session would not fit the RAM budget, the batch itself applies the WEAKEST downsampling that fits (spatial ssub and/or temporal tsub; decide the order and justify it from CaImAn's use of these params and the memory model), records it in the stage params/provenance, and adds a warning anomaly (no waiting for a human). Measured example: LNOF_J01_1D auto-crop 567x520, ~148 GiB predicted for compute_W (float64) vs 153 GiB budget. Unit-test the decision function with the memory model on synthetic sizes; no real CNMF-E run.
3. The crop stage must be fully automatic (no human step), and tighter on long sessions (LNOF_J01_1D: auto box 567x520 vs a manual 458x458; current FOV rule p1 + 0.2*(p99 - p1), `fov_threshold_frac`). Investigate why the box is loose on that session using only cheap reads (mean/max projections of a few thousand frames from H:\Projects\_bearmind_test\series_lnof, read-only; do not run MC), propose the method change with numbers, and add a crop overview image per series (every session's box drawn on its projection, one PNG or a small grid) that the author will look at once to validate the method.
4. Threshold maps for hour sessions take ~2.2 h (extrapolated); the e2e report's recommendation 4 says how to speed them up (parallel computation): include it if it is a contained change, otherwise list it as out of scope with the reason.

## Решения плана (моё решение, автор может отменить)

Метки: **[изм]** — измерено при написании плана (2026-11-06, скрипты в scratch, исходные данные только читались); **[код]** — прочитано в исходниках CaImAn 1.13.2 в окружении `bearmind`; **[вывод]** — расчёт или экстраполяция.

**П1. Превью — два побочных этапа runner.** `mc_preview` (raw, raw_dff, mc, mc_dff) читает сырые куски сессии и `3_MC/<SID>_MC.avi`; `cnmf_preview` (QC-видео) читает `4_Estimates/<SID>_estimates.pickle` и то же видео MC. Параметры побочного этапа — раздел `preview` конфига; хэш — эти параметры плюс хэш родителя. Хэши основной цепочки превью не включают (как и сейчас), поэтому флажок и настройки превью никогда не перезапускают MC и CNMF-E.
- «Сразу»: runner перечитывает конфиги на каждом витке; включённый побочный этап становится готов, как только его родитель актуален, и идёт следующим после родителя (в порядке `crop, mc, mc_preview, thresholds, cnmf, cnmf_preview, gate, autoinspect, export`). Батч, который ждёт человека, подхватывает флажок не позже чем через `poll_s` (30 с).
- Выключено (по умолчанию): этап не предлагается, процесс не запускается, в реестр ничего не пишется. В `summary.csv` столбцы `mc_preview` и `cnmf_preview` показывают `off`, в `status` — столбец `pv` со значением `off`. Основные этапы от побочных не зависят ни хэшем, ни готовностью.
- Упало: состояние FAILED у побочного этапа, остальная сессия идёт дальше; `run` и монитор показывают сбой, `retry` перезапускает только превью.
- Уже обработанная серия: `python -m bearmind previews SERIES on` (или флажок в окне) и обычный `run`. Для каждой сессии с актуальным MC (CNMF-E) делается `mc_preview` (`cnmf_preview`); MC и CNMF-E не перезапускаются. Старые файлы превью, сделанные внутри этапа MC, перезаписываются.
- Настройка понижения: `preview.tbin` (кадров в одном кадре превью, по умолчанию 2) и `preview.ssub` (блок пикселей, по умолчанию 1 = полный размер). Действуют на все пять видео. Скорость воспроизведения и `crf` у каждого видео прежние.
- Выигрыш: на NOF_H03_3D превью и dF/F-видео — 365.6 с из 742 с этапа MC (49%) [изм, таблица «Куда уходит время MC»]; без флажка этап MC короче примерно вдвое [вывод]. QC-видео: 41.1 с из 77.1 с QC на клипе 2000 кадров 463×608 и 132 МБ [изм]; на часовой сессии ~37 мин и ~7 ГБ [вывод, линейно по T].

**П2. PNG карт остатков остаётся всегда.** Он не дешёвый по времени: на том же клипе карты остатков — 46.3 с из 77.1 с QC (из них 12.1 с — карты самого фильма для сравнения) [изм]. Остаётся, потому что это не просмотр, а измерение: из того же прохода берётся `missed_cells` (проверка `cnmf_missed_cells` в реестре, сводке и экспорте), а файл весит 2.5 МБ на сессию [изм]. Флажок выключает только видео. Экономия 12 с из 46 с (брать карты фильма из карт порогов) возможна, но меняет `valid_mask` и число `missed_cells`; в этот план не входит.

**П3. Понижение для CNMF-E — `ssub_B`, а не `ssub` / `tsub`.** Спецификация называет `ssub` и `tsub`, но в CaImAn 1.13.2 они не уменьшают тот шаг, который не влезает:
- `compute_W` вызывается в `CNMF.fit` на всём фильме: `compute_W(Yr, A.toarray(), C, dims, ring_size_factor * gSiz[0], ssub=ssub_B)` (`cnmf.py:497-501`), где `Yr` — memmap в полном разрешении, а `A`, `C` уже возвращены к полному размеру и полному T [код]. `ssub` и `tsub` работают только внутри воркеров патчей: `downscale(Y, (ssub, ssub, tsub))` (`initialization.py:315-324`), затем `C = np.repeat(C, tsub, 1)[:, :T]` и `A` повторяется по `ssub` (`initialization.py:1307-1323`) [код].
- Значит, `tsub=2` и `ssub=2` уменьшают только память воркера патча (245 → 163 ГБ на миллион кадров для `tsub`), а число воркеров и так подбирается под бюджет (`patch_workers`). Пик `compute_W` (5.05 × T·d) от них не меняется.
- В `compute_W` все большие массивы имеют размер (d / ssub_B²) × T (`ds(Y)`, `ds(A).dot(C)`, их разность; `initialization.py:1966-1969`) [код]. Пик по замеру 5.05 × T·d при `ssub_B=2` [изм, задача 0.3], отсюда модель `5.05 × T·d × (2/ssub_B)²`.
- По ущербу `ssub_B` — самое слабое понижение: оно огрубляет только модель фона, а фон 1p гладкий на масштабе кольца (радиус 1.5 × gSiz ≈ 25 px при gSig 4). `A` и `C` остаются в полном разрешении и полной частоте; пороги, которые человек выбрал по картам, не теряют смысла. `ssub` ищет зерна на уменьшенной картинке с `gSig/ssub` (`initialization.py:298-299`); `tsub` — на усреднённом по времени фильме. Обе меняют то, что видел человек в окне порогов.
- Лестница: настроенное `cnmf.ssub_B` (2), затем 3, затем 4 (`memory.max_ssub_B`); берётся первое значение, при котором предсказание укладывается в `machine.cnmf_ram_budget_gb`. Предсказание = `5.05 × T·h·w × (2/ssub_B)² / 2³⁰ + memory.cnmf_base_gib` ГиБ. Константа 8 ГиБ складывается из процесса этапа после `import caiman` (4.65 ГиБ), простаивающих воркеров (0.67 ГиБ каждый) и плотных `A` и `C` [вывод из задачи 0.3]. Бюджет в `machine.yaml` записан в ГиБ (`psutil total / 1024³`).
- Если не влезает даже при 4, берётся 4, этап идёт, аномалия уровня error говорит, что памяти может не хватить. Падать заранее не стал: множитель `compute_W` в настоящем прогоне между 3.0 и 5.05 [изм, задача 0.3], модель консервативна.
- Числа (T = 107 107, бюджет 152.96 ГиБ) [вывод]:

| Рамка | T·d, ГБ | ssub_B 2 | 3 | 4 | Решение |
|---|---|---|---|---|---|
| авто сейчас 567×520 | 31.6 | 156.5 | 74.0 | 45.1 | 3 |
| новая авто (П4) 472×436 | 22.0 | 111.7 | 54.1 | 33.9 | 2 |
| ручная 458×458 | 22.5 | 113.7 | 55.0 | 34.4 | 2 |
| весь кадр 608×608 | 39.6 | 194.2 | 90.8 | 54.6 | 3 |

- Запись: действующее `ssub_B` входит в параметры этапа `cnmf` (`StageRecord.params`, `provenance.stage_params` экспорта, `est.cnmf_dict`); метрики `ssub_B` и `need_gib`; аномалия `cnmf_downsampled` (warning; error, если не влезает) с числами в `data`. Человека батч не ждёт.
- Следствие: бюджет машины стал косвенным входом хэша `cnmf`, но только у сессий, которые не влезают при настроенном `ssub_B`. Перенос такой сессии на машину с другим объёмом ОЗУ может пересчитать её CNMF-E; у остальных сессий хэши не меняются.
- Монитор: аномалия `cnmf_downsampled` — седьмой вид записи в панели предупреждений (задача 7). Автор просил «показать предупреждение»; событие редкое (часовые сессии с большим полем), панель оно не засорит. Альтернатива — только подсказка в клетке CNMF-E.

**П4. Кроп по активности внутри освещённой области.** Почему рамка LNOF_J01_1D рыхлая [изм, 3000 кадров: три окна по 1000 в начале, середине и конце сессии, MC не запускался]: круг линзы освещён почти целиком (правило яркости 0.2 даёт поле 547×500 от верхнего левого угла кадра), а кальциевые транзиенты есть только в его верхней левой части. Нижняя правая треть круга светлая, но клеток там нет. Правило яркости находит свет, а не клетки. Повышение порога яркости не помогает. При 0.3 поле LNOF почти то же (493×496), при 0.4 — 426×489, но тогда у NOF_H03_3D вне поля остаются 19 зерен против 11 шумовых, то есть около 8 клеток; при 0.3 — 3 против 0 [изм, пороги зерен 0.9 / 12, как в таблице ниже].

Новый метод:
1. Прочитать до 3000 кадров сырого фильма тремя окнами (начало, середина, конец), без кропа.
2. Освещённая область — прежнее правило `p1 + fov_threshold_frac * (p99 - p1)` по средней проекции. Это внешняя граница: рамка никогда не выходит за неё.
3. Карты CNMF-E `bm_qc.compute_init_maps(gSig=4, tsub=1, mirror=True)` и карта избытка `E = gauss(cn·pnr − cn_neg·pnr_neg, σ = 10 px)`. Транзиенты кальция положительные, зеркальная карта — шум. Обод линзы (10 px внутрь освещённой области) обнуляется: на нём артефакты. `E` обрезается сверху по 99-му перцентилю своих положительных значений: одна мигающая точка не может нести массу. На семи тестовых сессиях это меняет рамки не больше чем на 1 px [изм].
4. Поле — прямоугольник, вне которого с каждой стороны остаётся не больше 0.5% положительной массы `E` (`activity_mass_cut`), расширенный на 20 px (`activity_margin_px`) и обрезанный освещённой областью. Если он меньше 25% освещённой области, берётся освещённая область.
5. Дальше как сейчас: размер рамки мыши — максимум по сессиям плюс `margin_px` с каждой стороны, рамка ставится по центру поля.

Проверка на тестовых сессиях [изм]. Зерна считались по картам тех же окон внутри освещённой области при порогах corr / PNR 0.85 / 10, 0.9 / 12, 0.95 / 15; «нуль» — те же зерна на зеркальных картах, то есть шум.

| Сессия | Рамка сейчас | Новое поле | Новая рамка мыши | Зерна вне поля (реальные / нуль), 0.9 / 12 |
|---|---|---|---|---|
| LNOF_J01_1D | 567×520 | 452×416 | 472×436 (ручная была 458×458) | 7 / 5 |
| NOF_H01_1D | 608×608 | 514×498 | 534×518 | 0 / 0 |
| NOF_H03_3D, NOF_H03_4D | 461×608 | 391×536, 392×537 | 412×557 | 0 / 0, 0 / 0 |
| RFC_F01_1D | 463×485 | 409×387 | 429×407 | 0 / 0 |
| RFC_F20_1D | 440×562 | 403×450 | 423×470 | 0 / 0 |
| LT1_K02_1D | 532×608 | 506×608 | 526×608 | 0 / 0 |

Вне нового поля реальных зерен почти столько же, сколько шумовых: разница не больше 2 при всех трёх порогах. При 0.95 / 15 вне поля нет ни одного зерна ни в одной сессии. У LT1 активность слабая, и поле почти равно освещённой области: метод не сжимает рамку, если не уверен. Цена: этап кроп ~1–2 мин на сессию вместо 1–14 с [изм: 51–104 с на чтение и карты одной сессии 608×608], против ~10 мин MC.

Метод меняет параметры кропа, поэтому на существующих сериях кроп и всё после него пересчитаются один раз. Это только тестовые серии: батч ещё не отдан студентам. Обзор серии: `_bearmind/crop_overview.png`, по две панели на сессию (средняя проекция | карта избытка) с освещённой областью, полем и рамкой мыши. Рисуется после каждой расстановки рамок. Автор смотрит его один раз на LT1 (24 сессии) и на тестовых сериях (задача 8).

**П5. Карты порогов — параллельно по gSig.** Изменение замкнуто в `bearmind/stages/thresholds.py`, поэтому входит в план. Пять значений gSig считаются в пуле CaImAn со сторожем (`pool.start_pool`), каждый воркер открывает тот же uint8 memmap на диске. Число воркеров = `min(число gSig, n_cpus − 1, бюджет // пик одного расчёта)`. Пик одного расчёта — полоса `compute_init_maps` из ~6 float32-копий высотой не меньше 4·gSig + 10 строк. Бюджет одного расчёта прежний (`mem_budget_gb`), поэтому полосы и карты те же, что в одном процессе. Хэш этапа не меняется: параллельность — настройка машины.

| Сессия | Пик самого большого gSig | Воркеров | Кругов | Время [вывод] |
|---|---|---|---|---|
| 10 мин, 17 861 × 461 × 608 | 19.1 ГБ | 5 | 1 | ~1/5 от 21 мин |
| час, 567×520 | 50.8 ГБ | 3 | 2 | ~53 мин вместо ~2.2 ч |
| час, 472×436 (П4) | 42.6 ГБ | 3 | 2 | ~37 мин |

Время [вывод]: замер масштабирования MC показал предел машины около 8 процессов; для 3–5 процессов ускорение должно быть близко к линейному, но это не измерено.

## Global Constraints

- Env `bearmind` (`C:\Users\User\anaconda3\envs\bearmind\python.exe`, CaImAn 1.13.2, Py 3.12).
- Fast suite `python -m pytest -q -m "not slow and not realdata" tests` stays green.
- NO heavy MotionCorrection or CNMF-E runs (author's rule; quick 2000-frame clips only, e.g. `H:\Projects\_bearmind_test\series_short`, read-only, or skip and say so).
- New packages may be installed into env `bearmind` if they do not conflict with CaImAn.
- Program/console text in English, console output ASCII only.
- Code comments explain why, no edit history.
- Commits only with `git commit -- <paths>` (new files `git add <path>` first), no AI attribution.
- Source data (G:, `H:\Dataset`, `H:\Projects\LT1`, 1_Raw folders) read-only.
- Ядро: хранимое видео только `3_MC` (FFV1); временные файлы — в `machine.temp_dir`, удаляются и при ошибке; `fr` CNMF-E = точный FPS при любом `tsub`; поля npz и mat экспорта не меняются, в json только добавляются.

## Review Focus

1. Флажок включили на серии, обработанной до этого плана: MC и CNMF-E записаны с превью в `outputs`, побочных этапов в реестре нет. Ожидается: MC и CNMF-E не перезапускаются, превью делаются побочными этапами. Тест: задача 3, `test_old_records_with_preview_outputs_rerun_only_previews`.
2. Превью упало (кончился диск, ошибка кодека): сессия доходит до экспорта, сбой виден в `status`, `summary.csv` и панели, `retry` перезапускает только превью. Тесты: задача 3, `test_preview_failure_does_not_block` и `test_retry_resets_only_the_preview`; задача 7, `test_failed_preview_in_panel_with_retry`.
3. Серию открыли на машине с другим объёмом ОЗУ: у часовой сессии может смениться `ssub_B`, и её CNMF-E пересчитается; у сессий, которые влезают на обеих машинах, хэш тот же. Тест: задача 4, `test_budget_changes_the_hash_only_near_the_limit`.
4. Сессия без заметной активности в прочитанных окнах (молчит, плохой день) или с одной мигающей точкой: поле не схлопывается на шум или на точку, остаётся освещённая область. Тест: задача 6, `test_activity_fov_does_not_collapse_without_cells` (варианты `noise`, `hot_spot`).
5. Видео MC подменили руками, и оно не совпадает с записью MC: CNMF-E падает сразу с понятной ошибкой, а не считает с чужими размерами. Тест: задача 4, `test_run_rejects_an_mc_video_that_differs_from_its_record`.

## Карта файлов

| Файл | Что меняется |
|---|---|
| `bearmind/registry.py` | `SIDE_STAGES`, `ALL_STAGES` |
| `bearmind/runner.py` | побочные этапы; `stage_params("cnmf")` с размером фильма; обзор кропа после расстановки рамок; столбцы сводки |
| `bearmind/stages/previews.py` (новый) | `preview_params`, `mc_previews`, `cnmf_preview` |
| `bearmind/stages/mc.py`, `bearmind/stages/cnmf.py` | без видео; в `cnmf` — решение о `ssub_B` |
| `bearmind/stages/thresholds.py` | параллельные карты |
| `bearmind/crop.py`, `bearmind/stages/crop.py` | поле по активности, проекция сессии на диск |
| `bearmind/crop_overview.py` (новый) | `_bearmind/crop_overview.png` |
| `bearmind/video.py` | `ssub` у превью, `chunks_to_memmap`, `SessionReader.windows` |
| `bearmind/paths.py` | `crop_projection(sid)`, `crop_overview` |
| `bm_qc.py` | `block_mean`; `ssub` у dF/F-видео, `video_ssub` у QC-видео (по умолчанию поведение прежнее) |
| `bearmind/defaults.yaml` | разделы `preview`, `memory`, `crop` |
| `bearmind/cli.py`, `bearmind/actions.py` | команда `previews`; `retry` по `ALL_STAGES`; столбец `pv` |
| `bearmind/monitor.py`, `bearmind/gui/monitor_tab.py` | побочные этапы, седьмой вид панели, флажок «Preview videos» |
| `docs/superpowers/specs/2026-11-01-operator-notes.md`, новый отчёт проверки | инструкции и итог |

---

### Task 1: Понижение превью в писателях видео

**Files:**
- Modify: `bearmind/video.py` (`write_preview_mp4`, `_write_mp4`), `bm_qc.py` (`block_mean`, `make_dff_video`, `cnmf_qc`), `bearmind/defaults.yaml` (раздел `preview`)
- Test: `tests/bearmind/test_video.py`, `tests/test_bm_qc.py`

**Interfaces:**
- Produces:
```python
# bearmind/video.py
def write_preview_mp4(chunks, path, fps, crf=23, speed=4, tbin=2, ssub=1) -> None
# bm_qc.py
def block_mean(img: np.ndarray, s: int) -> np.ndarray   # 2-D float32, mean of s x s blocks; rows/cols that do not fill a block are dropped; s == 1 returns img as float32
def make_dff_video(movie, ..., tbin=2, ..., ssub=1)      # panels shrunk by block_mean before color mapping
def cnmf_qc(estimates, ..., video_tbin=2, ..., video_ssub=1)  # video panels shrunk; contours scaled by 1 / video_ssub
```
- `defaults.yaml`, раздел `preview` (комментарии на английском, «почему»):
```yaml
preview:
  enabled: false   # the checkbox: MC previews and the CNMF-E QC video
  tbin: 2          # frames averaged per preview frame
  ssub: 1          # pixels per side averaged per preview pixel (1 = full size)
  crf: 23          # crf and speed: raw and mc previews; the dF/F and QC videos keep their own
  speed: 4
```

- [ ] **Step 1: Тесты**
```python
# tests/bearmind/test_video.py
def test_preview_mp4_ssub_halves_the_frame(tmp_path):
    frames = np.full((10, 64, 80), 120, np.uint8)
    write_preview_mp4([frames], tmp_path / "a.mp4", 20.0, tbin=2, ssub=2)
    got = _decode(tmp_path / "a.mp4")      # helper: av decode to gray arrays
    assert got.shape[1:] == (32, 40) and len(got) == 5
    assert abs(float(got.mean()) - 120) <= 2
    write_preview_mp4([frames], tmp_path / "b.mp4", 20.0, tbin=2)
    assert _decode(tmp_path / "b.mp4").shape[1:] == (64, 80)

# tests/test_bm_qc.py
def test_block_mean():
    a = np.arange(35, dtype=np.float32).reshape(5, 7)
    b = bm_qc.block_mean(a, 2)
    assert b.shape == (2, 3) and b[0, 0] == a[:2, :2].mean() and b[1, 2] == a[2:4, 4:6].mean()
    assert np.array_equal(bm_qc.block_mean(a, 1), a)

def test_dff_video_ssub(tmp_path):        # synthetic_movie(T=40, d=(64, 80)); f0_window_s=5; 3 panels side by side, 2 px gaps
    assert frame_shape(ssub=1) == (64, 3 * 80 + 4) and frame_shape(ssub=2) == (32, 3 * 40 + 4)

def test_cnmf_qc_video_ssub(tmp_path):    # the estimates of test_cnmf_qc_finds_omitted_cell, residual_maps=False
    # 2 x 2 grid of 64 x 80 panels -> 130 x 162; video_ssub=2 -> 66 x 82
```
- [ ] **Step 2:** `C:\Users\User\anaconda3\envs\bearmind\python.exe -m pytest tests/bearmind/test_video.py tests/test_bm_qc.py -q -p no:cacheprovider` → FAIL (`ssub` не принимается, `block_mean` нет).
- [ ] **Step 3:** Реализовать. В `_write_mp4` — `block_mean` после усреднения по `tbin`, затем обрезка до чётных размеров, как сейчас. В `make_dff_video` уменьшать каждое изображение панели после `render` до `_to_rgb`, и в выборке для шкал тоже. В `cnmf_qc` уменьшать `Yb[k], YBb[k], ACb[k], Rb[k]`; контуры для `drawContours` — `(c / video_ssub).astype(np.int32)`, посчитать один раз. Карты остатков от `video_ssub` не зависят.
- [ ] **Step 4:** Тот же запуск → PASS; быстрый набор PASS.
- [ ] **Step 5:** Commit: `git commit -m "feat(preview): spatial downsampling of preview videos (ssub), preview settings" -- bearmind/video.py bm_qc.py bearmind/defaults.yaml tests/bearmind/test_video.py tests/test_bm_qc.py`

### Task 2: Этапы превью; MC и CNMF-E без видео

**Files:**
- Create: `bearmind/stages/previews.py`, `tests/bearmind/test_stage_previews.py`
- Modify: `bearmind/stages/mc.py` (убрать превью, `_preview_chunks` и `import bm_qc`), `bearmind/stages/cnmf.py` (QC без видео), `bearmind/video.py` (`chunks_to_memmap`)
- Test: `tests/bearmind/test_stage_mc.py`, `tests/bearmind/test_stage_cnmf.py`, `tests/bearmind/test_video.py`

**Interfaces:**
- Consumes: задача 1 (`write_preview_mp4(..., ssub)`, `make_dff_video(..., ssub)`, `cnmf_qc(..., video_ssub)`); `SessionReader`, `decode_to_memmap`, `require_temp_space`, `paths.preview(sid, kind)`, `paths.temp_dir`.
- Produces:
```python
# bearmind/video.py
def chunks_to_memmap(chunks, out: Path, shape: tuple[int, int, int]) -> np.memmap   # uint8 (T, h, w) written sequentially, returned read-only; file removed on error; frame count checked
# bearmind/stages/previews.py  (light at import: bm_qc and CaImAn only inside the functions; the runner imports preview_params)
TEMP_BYTES_PER_PX_FRAME = 1      # one uint8 copy at a time: the cropped raw movie, then the MC movie
def preview_params(cfg) -> dict  # {"enabled": False} when off; else {"enabled": True, "tbin", "ssub", "crf", "speed"} (ints)
def mc_previews(paths, rec, cfg) -> StageResult   # outputs preview_raw, preview_raw_dff, preview_mc, preview_mc_dff; metrics {"tbin", "ssub", "bytes"}
def cnmf_preview(paths, rec, cfg) -> StageResult  # outputs preview_cnmf_qc; metrics {"tbin", "ssub", "bytes"}
```
- `mc_previews`: кроп — `rec.stages["mc"].metrics["crop"]` (рамка, с которой реально шёл MC; `None` = без кропа). Порядок: raw-превью из `SessionReader(crop=None)` потоком; обрезанный сырой фильм → `chunks_to_memmap` → `make_dff_video` (raw_dff) → удалить; видео MC → `decode_to_memmap` → mc-превью и `make_dff_video` (mc_dff) → удалить. Временная папка — `paths.temp_dir(...)`, удаляется в `finally`, как в `mc.run`. Перед работой `require_temp_space(tmp, T*h*w, "MC previews")`.
- `cnmf_preview`: `pickle.load` оценок, видео MC → `decode_to_memmap`, `bm_qc.cnmf_qc(est, movie=u8, out_base=<tmp>/<sid>, fps=rec.fps, tmp_dir=..., make_video=True, residual_maps=False, video_tbin=tbin, video_ssub=ssub, verbose=False)`, затем `_qc.mp4` переносится в `paths.preview(sid, "cnmf_qc")` (как `cnmf._move`). Фильм здесь uint8, а у CNMF-E был uint8 + 1e-4: разница на панели raw невидима.
- `cnmf.run`: `bm_qc.cnmf_qc(..., make_video=False)`; `outputs = {"estimates", "residual_png"}`; `metrics` и `missed_cells` прежние. `mc.run`: `outputs = {"mc_video", "mc_shifts"}`, метрики прежние. Параметры и хэши этапов MC и CNMF-E не меняются.

- [ ] **Step 1: Тесты**
```python
# tests/bearmind/test_stage_previews.py
def test_preview_params_off_and_on():
    cfg = config.load_defaults()
    assert previews.preview_params(cfg) == {"enabled": False}
    cfg["preview"]["tbin"] = 4
    assert previews.preview_params(cfg) == {"enabled": False}    # settings of a disabled preview do not matter
    cfg["preview"]["enabled"] = True
    assert previews.preview_params(cfg) == {"enabled": True, "tbin": 4, "ssub": 1, "crf": 23, "speed": 4}

def test_mc_previews_writes_four_videos(tmp_path, make_avi):
    # raw AVI 300 x 72 x 72 (as test_stage_mc._raw_frames), MC video = crop [4, 4, 64, 64] of it,
    # rec.stages["mc"] = StageRecord(state=DONE, metrics={"crop": [4, 4, 64, 64], "n_frames": 300})
    res = previews.mc_previews(paths, rec, cfg)
    assert set(res.outputs) == {"preview_raw", "preview_raw_dff", "preview_mc", "preview_mc_dff"}
    assert frame_shape(preview_raw) == (72, 72) and frame_shape(preview_mc) == (64, 64)
    assert frame_shape(preview_raw_dff) == frame_shape(preview_mc_dff)   # both of the 64 x 64 crop
    assert temp folder empty and res.metrics["tbin"] == 2
    # cfg["preview"]["ssub"] = 2 -> preview_raw 36 x 36, preview_mc 32 x 32

def test_mc_previews_failure_cleans_temp(tmp_path, make_avi, monkeypatch):   # make_dff_video raises -> raised, temp folder gone

def test_cnmf_preview_writes_qc_video(tmp_path):
    # synthetic_movie(T=200) written as the MC video; estimates pickle = SimpleNamespace as in
    # tests/test_bm_qc.py::test_cnmf_qc_finds_omitted_cell (b, f background, idx_components)
    assert res.outputs == {"preview_cnmf_qc": paths.rel(paths.preview(SID, "cnmf_qc"))} and file size > 0
    assert not paths.residual_png(SID).exists() and temp folder empty

# tests/bearmind/test_video.py
def test_chunks_to_memmap_round_trip(tmp_path): ...       # two chunks -> equal array, read-only
def test_chunks_to_memmap_short_input_removes_file(tmp_path): ...   # fewer frames than shape -> ValueError, no file
```
- [ ] **Step 2:** Обновить ожидания медленных тестов: `test_mc_stage` — `set(res.outputs) == {"mc_video", "mc_shifts"}` и ни одного файла в `Preview/`; `test_cnmf_on_synthetic` — `{"estimates", "residual_png"}` и `cnmf_qc` вызван с `make_video=False` (обёртка-шпион вокруг настоящей функции). Удалить `test_mc_stage_dff_failure_releases_temp_tif`: dF/F больше не в MC; его смысл переходит в `test_mc_previews_failure_cleans_temp`.
- [ ] **Step 3:** `... -m pytest tests/bearmind/test_stage_previews.py tests/bearmind/test_video.py -q -p no:cacheprovider` → FAIL (модуля нет).
- [ ] **Step 4:** Реализовать `previews.py`, `chunks_to_memmap`, правки `mc.py` и `cnmf.py`.
- [ ] **Step 5:** Новые тесты PASS; `... -m pytest tests/bearmind/test_stage_mc.py tests/bearmind/test_stage_cnmf.py -q -m slow` PASS (синтетика, ~2–4 мин, это не реальные данные); быстрый набор PASS.
- [ ] **Step 6:** Commit: `git add bearmind/stages/previews.py tests/bearmind/test_stage_previews.py` и `git commit -m "feat(preview): preview stage functions; MC and CNMF-E write no videos" -- bearmind/stages/previews.py bearmind/stages/mc.py bearmind/stages/cnmf.py bearmind/video.py tests/bearmind/test_stage_previews.py tests/bearmind/test_stage_mc.py tests/bearmind/test_stage_cnmf.py tests/bearmind/test_video.py`

### Task 3: Побочные этапы в runner, команда `previews`

**Files:**
- Modify: `bearmind/registry.py`, `bearmind/runner.py`, `bearmind/cli.py`, `bearmind/actions.py` (фаза 2)
- Test: `tests/bearmind/test_runner.py`, `tests/bearmind/test_cli.py`, `tests/bearmind/test_actions.py`, `tests/bearmind/test_e2e_real.py` (только ожидания превью)

**Interfaces:**
- Consumes: `previews.preview_params(cfg)`; `"bearmind.stages.previews:mc_previews"`, `"bearmind.stages.previews:cnmf_preview"`.
- Produces:
```python
# bearmind/registry.py
SIDE_STAGES = {"mc_preview": "mc", "cnmf_preview": "cnmf"}   # optional stage -> the stage whose outputs it reads
ALL_STAGES = ("crop", "mc", "mc_preview", "thresholds", "cnmf", "cnmf_preview", "gate", "autoinspect", "export")
# bearmind/runner.py
def side_enabled(stage: str, cfg: dict) -> bool
def stage_hash(rec, stage, cfg, paths) -> str          # also for side stages
def reset_from(paths, sid, stage, who) -> None          # stage from ALL_STAGES; a side stage resets only itself
# bearmind/actions.py
def set_previews(paths, enabled: bool, tbin: int | None = None, ssub: int | None = None) -> None   # series config, section preview
```
- CLI: `previews SERIES on|off [--tbin N] [--ssub N]`; печатает `[OK] previews on (tbin 2, ssub 1); the next run makes them for every session whose MC / CNMF-E is done` или `[OK] previews off; existing preview files are kept`. `--tbin` и `--ssub` — целые ≥ 1, иначе `CliError`.

Правила:
1. Хэш побочного этапа: `config.params_hash({"params": preview_params(cfg), "upstream": [h[parent]]})`, где `h` — хэши основной цепочки (`_chain` без изменений). Хэши основной цепочки остаются прежними: после этой задачи ни один основной этап не пересчитывается.
2. Выключенный побочный этап не предлагается и не пишется в реестр; старая запись остаётся как была.
3. Готов: включён; все основные этапы до родителя включительно актуальны; сам не актуален (DONE, тот же хэш, файлы `REQUIRED` на месте) и не заблокирован (FAILED / SKIPPED с тем же хэшем).
4. `next_ready` идёт по `ALL_STAGES`, затем по sid. При `until` побочный этап в диапазоне, если в диапазоне его родитель.
5. Запуск побочного этапа: RUNNING только у него, основные этапы не сбрасываются. Ошибка → FAILED (и `stage_crashed`, если процесс упал), дальше ничего не пропускается; проверок нет.
6. Основной этап стал RUNNING или WAITING, или `reset_from` с основного этапа: побочные этапы, у которых индекс родителя ≥ индекса этого этапа, → PENDING. Основной этап упал: такие побочные этапы → SKIPPED и входят в счёт `skipped`.
7. `REQUIRED["mc_preview"] = ("preview_raw", "preview_raw_dff", "preview_mc", "preview_mc_dff")`, `REQUIRED["cnmf_preview"] = ("preview_cnmf_qc",)`.
8. `series_state.failed` — первый FAILED по `ALL_STAGES`: `run` даёт код 1 и подсказку `retry`, если упало превью. `cmd_retry` и `actions.retry_failed` ищут первый FAILED по `ALL_STAGES`.
9. Пульс: `counts` и `by_stage` — только основные этапы (полосы фазы 2 не меняются); `current` может назвать побочный этап.
10. `summary.csv`: столбец `mc_preview` после `mc_max_shift_px`, `cnmf_preview` после `cnmf_n_components`; значение — состояние этапа (`pending`, если записи нет) или `off`, если разрешённый конфиг сессии превью выключает. FAILED побочный этап попадает в `worst_check` и `error`, как основной.
11. `status`: столбец `pv` после этапов: `off`, или две буквы состояний (`mc_preview`, `cnmf_preview`) теми же буквами, что у этапов.

- [ ] **Step 1: Тесты** (фикстура `series`; заглушки побочных этапов добавить в `Stubs`: пишут свои файлы превью; `monkeypatch.setitem(runner.STAGE_FUNCS, ...)` для обоих; `_on(s)` = `actions.set_previews(s.paths, True)`; заглушки `mc` и `cnmf` больше не пишут файлов превью):
```python
def test_registry_stage_order():
    assert registry.ALL_STAGES == ("crop", "mc", "mc_preview", "thresholds", "cnmf", "cnmf_preview", "gate",
                                   "autoinspect", "export")

def test_previews_off_by_default(series):            # calls == ALL (no side stage); summary columns mc_preview, cnmf_preview == "off"; no side record in the registry
def test_previews_on_run_right_after_parent(series): # _on, _run -> calls in ALL_STAGES order: crop x3, mc x3, mc_preview x3, thresholds x3, cnmf x3, cnmf_preview x3, gate, autoinspect, export
def test_previews_turned_on_later_rerun_nothing_else(series):
    _run(s); finished = {(sid, st): finished time for every main stage}
    _on(s); s.stubs.calls.clear(); res = _run(s)
    assert s.stubs.calls == [(sid, "mc_preview") for sid in SIDS] + [(sid, "cnmf_preview") for sid in SIDS]
    assert res["done"] == 6 and finished times of every main stage unchanged
def test_old_records_with_preview_outputs_rerun_only_previews(series):
    # after _run, add preview_raw ... keys and files to outputs of every mc / cnmf record (the pre-plan format), then _on, _run
    assert s.stubs.calls == side stages only
def test_preview_settings_change_remakes_only_previews(series):   # _on, _run; set_previews(tbin=4); _run -> side stages only
def test_preview_failure_does_not_block(series):
    s.stubs.fail[(S1, "mc_preview")] = RuntimeError("disk full"); _on(s); res = _run(s)
    assert _states(s.paths, S1) == [State.DONE] * 7                       # every main stage of S1
    assert load_record(s.paths, S1).stages["mc_preview"].state is State.FAILED
    assert res["failed"] == 1 and res["skipped"] == 0
    assert runner.series_state(s.paths).failed[S1][0] == "mc_preview"
def test_retry_resets_only_the_preview(series):      # after the failure: reset_from(S1, "mc_preview") -> _run -> calls == [(S1, "mc_preview")]
def test_mc_failure_skips_its_preview(series):       # mc of S1 fails -> mc_preview and cnmf_preview of S1 SKIPPED
def test_mc_rerun_resets_mc_preview(series):         # _on, _run; change cfg mc -> mc, mc_preview, thresholds, cnmf, cnmf_preview, gate... of every session again, mc_preview right after mc
def test_missing_preview_file_remakes_only_it(series):   # delete preview_mc of S2 -> calls == [(S2, "mc_preview")]
def test_until_includes_side_stages_of_stages_in_range(series):   # _on; run(until="thresholds") -> mc_preview ran, cnmf_preview not
def test_flag_off_keeps_files_and_runs_nothing(series):   # _on, _run, set_previews(False), _run -> ZERO; preview files still there; summary "off"
def test_heartbeat_counts_main_stages_only(series):  # _on, _run: heartbeat by_stage keys == list(STAGES)

# tests/bearmind/test_cli.py
def test_previews_command(series, capsys):           # "previews SERIES on --tbin 4" -> series.yaml preview == {"enabled": True, "tbin": 4}; "[OK] previews on"; "--tbin 0" -> exit 2 and nothing written
def test_status_previews_column(series, capsys):     # "pv" header; "off" before, "DD" after an on-run
def test_retry_all_failed_covers_a_failed_preview(series, capsys):

# tests/bearmind/test_actions.py
def test_retry_failed_preview_only(series): ...      # actions.retry_failed resets only mc_preview
```
- [ ] **Step 2:** `... -m pytest tests/bearmind/test_runner.py tests/bearmind/test_cli.py tests/bearmind/test_actions.py -q -p no:cacheprovider` → новые тесты FAIL.
- [ ] **Step 3:** Реализовать правила 1–11. В `tests/bearmind/test_runner.py::test_missing_required_output_reruns` удалить файлы превью из заглушек (их больше нет); проверка «удаление превью не перезапускает основные этапы» переходит в `test_flag_off_keeps_files_and_runs_nothing`. В `test_e2e_real.py` ожидать `residual_png` у `cnmf`, а файлы превью — только если в записи есть DONE `mc_preview` / `cnmf_preview`.
- [ ] **Step 4:** Тесты PASS; быстрый набор PASS.
- [ ] **Step 5:** Commit: `git commit -m "feat(runner): optional preview stages after mc and cnmf; previews command" -- bearmind/registry.py bearmind/runner.py bearmind/cli.py bearmind/actions.py tests/bearmind/test_runner.py tests/bearmind/test_cli.py tests/bearmind/test_actions.py tests/bearmind/test_e2e_real.py`

### Task 4: CNMF-E сам понижает разрешение фона (`ssub_B`)

**Files:**
- Modify: `bearmind/stages/cnmf.py`, `bearmind/runner.py` (`stage_params("cnmf")`), `bearmind/defaults.yaml` (раздел `memory`)
- Test: `tests/bearmind/test_stage_cnmf.py`, `tests/bearmind/test_runner.py`

**Interfaces:**
- Produces:
```python
@dataclass(frozen=True)
class BackgroundDownsample:
    ssub_B: int                 # the value CNMF-E runs with
    configured: int             # cnmf.ssub_B of the config
    need_gib: float             # predicted peak at ssub_B
    need_configured_gib: float  # predicted peak at the configured value
    budget_gib: float           # machine.cnmf_ram_budget_gb (GiB: machine.yaml stores total / 1024**3)
    fits: bool

def cnmf_need_gib(T: int, h: int, w: int, ssub_B: int, cfg) -> float
def background_downsample(T: int, h: int, w: int, cfg) -> BackgroundDownsample
def cnmf_movie_shape(rec) -> tuple[int, int, int] | None
    # (T, h, w) of the DONE MC record: metrics n_frames (absent -> rec.n_frames) and crop (None -> rec.parts[0].shape);
    # None without a DONE MC record
def cnmf_params(cfg, fps, movie_shape=None) -> dict          # with movie_shape: ssub_B = background_downsample(...).ssub_B
def downsample_anomaly(d: BackgroundDownsample) -> Anomaly | None
```
- `defaults.yaml`, раздел `memory` (ни один этап не хэширует этот раздел, поэтому новые ключи ничего не перезапускают):
```yaml
memory:
  patch_gb_per_mframe: {1: 245, 2: 163}
  # compute_W of CaImAn 1.13.2 (docs/superpowers/specs/2026-10-31-caiman-memory.md): peak bytes per uint8
  # pixel-frame at ssub_B 2; every large array in it is (pixels / ssub_B^2) x T, so the peak scales as (2 / ssub_B)^2
  compute_w_bytes_per_px_frame: 5.05
  # held besides compute_W: the stage process after import caiman (4.65 GiB), idle patch workers (0.67 GiB each),
  # dense A and C
  cnmf_base_gib: 8
  # strongest background downsampling the batch applies by itself
  max_ssub_B: 4
```
- Правило: `need(s) = compute_w_bytes_per_px_frame * T*h*w * (2/s)**2 / 2**30 + cnmf_base_gib`; кандидаты `s = configured, configured+1, ..., max(configured, max_ssub_B)`; первый с `need(s) <= budget`, иначе `max(configured, max_ssub_B)` и `fits=False`.
- `downsample_anomaly`: `None`, если `ssub_B == configured` и `fits`. Иначе `Anomaly("cnmf_downsampled", "warning" if fits else "error", message, data=asdict(d))`. Тексты:
  - `background downsampled to fit RAM: ssub_B {s} instead of {s0} (predicted {need0:.0f} GiB at {s0}, {need:.0f} GiB at {s}, budget {budget:.0f} GiB)`
  - `predicted {need:.0f} GiB even at ssub_B {s} (budget {budget:.0f} GiB): CNMF-E may run out of memory; a tighter crop helps`
- `runner.stage_params("cnmf")` = `cnmf_params(cfg, rec.fps, cnmf_movie_shape(rec))`; ожидание порогов — как сейчас.
- `cnmf.run`: `shape = cnmf_movie_shape(rec)`; если `shape` есть и не равен `video_info(mc)[:3]` → `ValueError("<mc video> has TxHxW frames, its MC record says ...: run MC again")` до временной папки. `params = cnmf_params(cfg, rec.fps, shape)` — те же параметры, что в записи RUNNING. Метрики `ssub_B`, `need_gib` (1 знак); аномалия из `downsample_anomaly`. `check()` без изменений.

- [ ] **Step 1: Тесты**
```python
HOUR = 107107

def _cfg(budget=152.96, ssub_B=2):
    cfg = load_defaults(); cfg["cnmf"]["ssub_B"] = ssub_B
    cfg["machine"] = {"n_cpus": 32, "ram_gb": 191.2, "cnmf_ram_budget_gb": budget}
    return cfg

def test_downsample_lnof_auto_box():
    d = background_downsample(HOUR, 567, 520, _cfg())
    assert (d.ssub_B, d.configured, d.fits) == (3, 2, True)
    assert d.need_configured_gib == pytest.approx(156.5, abs=0.3) and d.need_gib == pytest.approx(74.0, abs=0.3)

def test_downsample_keeps_the_config_when_it_fits():
    for T, h, w in ((HOUR, 472, 436), (HOUR, 458, 458), (17861, 461, 608), (2000, 608, 608)):
        assert background_downsample(T, h, w, _cfg()).ssub_B == 2

def test_downsample_takes_the_weakest_that_fits():
    assert background_downsample(HOUR, 567, 520, _cfg(budget=60)).ssub_B == 4     # 74.0 > 60 >= 45.1
    assert background_downsample(HOUR, 608, 608, _cfg()).ssub_B == 3              # 194.2, 90.8

def test_downsample_none_fits():
    d = background_downsample(HOUR, 567, 520, _cfg(budget=20))
    assert (d.ssub_B, d.fits) == (4, False)

def test_downsample_starts_at_the_configured_value():
    assert background_downsample(HOUR, 458, 458, _cfg(ssub_B=3)).ssub_B == 3

def test_cnmf_params_change_only_ssub_b():
    cfg = _cfg(); cfg["cnmf"].update(gSig=4, min_corr=0.8, min_pnr=8)
    base = cnmf_params(cfg, 29.75)
    assert cnmf_params(cfg, 29.75, (2000, 608, 608)) == base          # sessions that fit: same params, same hash
    big = cnmf_params(cfg, 29.75, (HOUR, 567, 520))
    assert big["ssub_B"] == 3 and {k: v for k, v in big.items() if k != "ssub_B"} == {k: v for k, v in base.items() if k != "ssub_B"}

def test_downsample_anomaly_levels():   # changed+fits -> warning, "ssub_B 3 instead of 2" in message, data["need_gib"]; not fits -> error; unchanged -> None

def test_movie_shape_from_the_mc_record():   # mc DONE {"n_frames": HOUR, "crop": [0, 0, 567, 520]} -> (HOUR, 567, 520); crop None -> (HOUR, *parts[0].shape); no MC -> None

def test_run_rejects_an_mc_video_that_differs_from_its_record(tmp_path):
    # MC video of 50 frames, MC record n_frames 60 -> ValueError matching "run MC again"; temp folder not created

# tests/bearmind/test_runner.py
def test_cnmf_stage_params_take_the_downsampling(series):   # S1 record with mc metrics {"n_frames": HOUR, "crop": [0, 0, 567, 520]}, budget 152.96 -> stage_params("cnmf")["ssub_B"] == 3
def test_budget_changes_the_hash_only_near_the_limit(series):
    # hour-size record: stage_hash("cnmf") at budget 152.96 != at 200; 16 x 16 record after _run: equal at 12.8 and 200
```
- [ ] **Step 2:** FAIL (`background_downsample` нет).
- [ ] **Step 3:** Реализовать; `test_stage_hash_inputs` остаётся без правок (там MC не сделан, решения нет).
- [ ] **Step 4:** PASS; быстрый набор PASS. Медленный `test_cnmf_on_synthetic` (синтетика) PASS, у него `metrics["ssub_B"] == 2` и аномалии `cnmf_downsampled` нет.
- [ ] **Step 5:** Commit: `git commit -m "feat(cnmf): weakest background downsampling (ssub_B) that fits the RAM budget, recorded with an anomaly" -- bearmind/stages/cnmf.py bearmind/runner.py bearmind/defaults.yaml tests/bearmind/test_stage_cnmf.py tests/bearmind/test_runner.py`

### Task 5: Карты порогов параллельно по gSig

**Files:**
- Modify: `bearmind/stages/thresholds.py`
- Test: `tests/bearmind/test_stage_thresholds.py`

**Interfaces:**
- Consumes: `pool.start_pool(n)`, `pool.stop_pool(dview)`, `mem_budget_gb(cfg)`, `bm_qc.compute_init_maps`.
- Produces:
```python
def map_peak_gb(T: int, h: int, w: int, g: int, tsub: int, mem_gb: float) -> float
    # one compute_init_maps call: 6 float32 copies of a strip of ceil(T / tsub) frames x w columns x
    # min(h, max(budget rows, 4 * g + 10)) rows, budget rows = mem_gb * 1e9 / (24 * ceil(T / tsub) * w)
def map_workers(T: int, h: int, w: int, gsigs: list[int], tsub: int, cfg) -> int
    # min(len(gsigs), n_cpus - 1, floor(cnmf_ram_budget_gb / max peak)), at least 1
def _maps_job(args: tuple) -> tuple[int, dict]
    # (u8 path, shape, g, tsub, mem_gb) -> (g, compute_init_maps result); opens the uint8 memmap read-only in the worker
```
- `run`: `n = map_workers(...)`. При `n == 1` — нынешний цикл в этом процессе. Иначе `OPENBLAS_NUM_THREADS=1` на время пула (сохранить и вернуть, как в `cnmf.run`), `dview = pool.start_pool(n)`, `dview.map(_maps_job, jobs)` в порядке `gsigs`, `pool.stop_pool` в `finally` до освобождения memmap и удаления файла. Метрика `map_workers`. Параметры и хэш этапа прежние.

- [ ] **Step 1: Тесты**
```python
def test_map_peak_follows_compute_init_maps_strips():
    assert map_peak_gb(17861, 461, 608, 7, 1, 19.12) == pytest.approx(19.12, rel=0.02)   # 73 budget rows > 38
    assert map_peak_gb(107107, 567, 520, 7, 1, 19.12) == pytest.approx(50.8, abs=0.2)  # 38-row floor
    assert map_peak_gb(100, 20, 30, 7, 1, 19.12) == pytest.approx(24 * 100 * 30 * 20 / 1e9)   # whole frame

def test_map_workers():
    cfg = {"machine": {"n_cpus": 32, "ram_gb": 191.2, "cnmf_ram_budget_gb": 152.96}}
    g = [3, 4, 5, 6, 7]
    assert map_workers(17861, 461, 608, g, 1, cfg) == 5 and map_workers(107107, 567, 520, g, 1, cfg) == 3
    cfg["machine"]["n_cpus"] = 2
    assert map_workers(17861, 461, 608, g, 1, cfg) == 1
    cfg["machine"].update(n_cpus=32, cnmf_ram_budget_gb=30)
    assert map_workers(107107, 567, 520, g, 1, cfg) == 1

@pytest.mark.slow
def test_parallel_maps_equal_serial(tmp_path):
    # synthetic_movie(T=300), gsig_range [3, 4, 5]; run 1: machine n_cpus 2 (serial); run 2: n_cpus 8,
    # cnmf_ram_budget_gb 64 (3 workers), separate series roots
    assert res2.metrics["map_workers"] == 3 and res1.metrics["map_workers"] == 1
    assert every array of both npz files is np.array_equal and res1.metrics["suggested"] == res2.metrics["suggested"]
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. `test_thresholds_stage` остаётся без правок: при `n_cpus: 2` путь последовательный, шпион видит оба вызова.
- [ ] **Step 4:** Быстрые тесты PASS; `... -m pytest tests/bearmind/test_stage_thresholds.py -q -m slow` PASS; быстрый набор PASS.
- [ ] **Step 5:** Commit: `git commit -m "perf(thresholds): gSig maps in parallel within the machine RAM budget" -- bearmind/stages/thresholds.py tests/bearmind/test_stage_thresholds.py`

### Task 6: Кроп по активности и обзор кропа серии

**Files:**
- Create: `bearmind/crop_overview.py`, `tests/bearmind/test_crop_overview.py`
- Modify: `bearmind/crop.py`, `bearmind/stages/crop.py`, `bearmind/video.py` (`SessionReader.windows`), `bearmind/paths.py`, `bearmind/runner.py` (`_run_finalize`), `bearmind/defaults.yaml` (раздел `crop`)
- Test: `tests/bearmind/test_crop.py`, `tests/bearmind/test_video.py`, `tests/bearmind/test_runner.py`

**Interfaces:**
- Produces:
```python
# bearmind/video.py
def SessionReader.windows(self, n_windows: int, length: int) -> np.ndarray
    # one pass; n_windows contiguous windows of `length` frames at the start, the middle ((T - length) // 2) and the end
    # (T - length); a session shorter than n_windows * length: all frames
# bearmind/crop.py
RIM_PX = 10          # the lens rim carries edge artefacts in the seed maps
ACTIVITY_SIGMA_PX = 10
MIN_FRAC_OF_LIT = 0.25
def lit_fov(mean: np.ndarray, min_fov_frac=0.05, threshold_frac=0.2) -> tuple[tuple, np.ndarray] | None   # (box, mask of the largest lit component); detect_fov(frames, ...) stays = lit_fov(frames.mean(0))[0]
def activity_map(maps: dict, lit_mask: np.ndarray) -> np.ndarray
    # gauss(cn*pnr - cn_neg*pnr_neg, ACTIVITY_SIGMA_PX), 0 outside the lit mask eroded by RIM_PX, clipped from above at the
    # 99th percentile of its positive values inside (a hot spot cannot carry the mass); scipy imported inside, as in detect_fov
def activity_fov(activity: np.ndarray, lit_box: tuple, mass_cut: float, margin: int) -> tuple | None
# bearmind/paths.py
def crop_projection(self, sid) -> Path   # _bearmind/crop/<sid>_projection.npz (mean, activity as float16)
@property crop_overview -> Path          # _bearmind/crop_overview.png
# bearmind/crop_overview.py
def write_overview(paths, records) -> Path | None   # None (and no matplotlib import) when no record has a projection
```
- `defaults.yaml`, раздел `crop` (`n_sample_frames` заменяется на `sample_frames`):
```yaml
crop:
  # frames read for detection: three windows of sample_frames / 3 at the start, middle and end of the session
  sample_frames: 3000
  margin_px: 10
  min_fov_frac: 0.05
  fov_threshold_frac: 0.2
  # activity: the field covers where calcium transients are (CNMF-E seed maps against their sign-inverted null)
  # inside the lit area; brightness: the lit area only (the rule before 2026-11)
  fov_method: activity
  activity_mass_cut: 0.005
  activity_margin_px: 20
```
- `stages.crop.detect`: окна → средняя проекция → `lit_fov`; при `fov_method: activity` — `bm_qc.compute_init_maps(frames, gSig=4, tsub=1, mirror=True, max_mem_gb=mem_budget_gb(cfg), verbose=False)` → `activity_map` → `activity_fov`. Метрики: `fov` (итоговое поле, его читает `finalize_mouse`), `fov_lit`, `fov_activity`, `fov_method` (`activity` | `brightness`), `frame_shape`. Выход `projection`. Освещённой области нет → `fov: None`, как сейчас. gSig 4 зашит: рамке нужно, где транзиенты, а не точный размер клетки.
- `activity_fov`: `w = max(E, 0)`; пусто → `None`. Строки и столбцы: первый индекс, где накопленная доля маргинали ≥ `mass_cut`, и первый, где ≥ `1 − mass_cut`. Расширить на `margin`, обрезать `lit_box`. Площадь < `MIN_FRAC_OF_LIT` × площадь `lit_box` → `None`.
- Обзор: по две панели на сессию (средняя проекция | карта активности), три сессии в ряд. Освещённая область — жёлтый пунктир, поле — зелёный пунктир, рамка мыши `metrics["box"]` — голубая сплошная; подпись `<sid>  box HxW  (<fov_method>)`. Запись через `.partial.png` и `fileio.replace`. Runner вызывает `write_overview` после записи рамок в `_run_finalize` (успешный путь). Ошибка — `log.warning("crop overview not written: ...")`: расстановку рамок она не роняет.

- [ ] **Step 1: Тесты** (сцена для `activity_fov`: кадр 200×240, освещённый диск радиуса 90 с центром (100, 120), шум σ=4; «клетки» — гауссианы σ=3 с редкими транзиентами, как в `synthetic_movie`, только при x < 110; 600 кадров; карты `compute_init_maps(gSig=3, mirror=True)`):
```python
def test_activity_fov_covers_the_active_part():
    box = activity_fov(activity_map(maps, lit_mask), lit_box, 0.005, 20)
    assert box lies inside lit_box and covers every cell centre with 5 px to spare
    assert box[1] + box[3] <= 110 + 20 + 3 * ACTIVITY_SIGMA_PX and box[3] <= 0.8 * lit_box[3]

def test_activity_fov_keeps_dim_active_tissue():   # cells at 25 % brightness, 20 px inside the lit edge, are inside the box

@pytest.mark.parametrize("scene", ["noise", "hot_spot"])   # no cells; hot_spot: one 3 x 3 spot blinking in 1 % of frames
def test_activity_fov_does_not_collapse_without_cells(scene):
    assert box is None or box area >= 0.9 * lit_box area
def test_detect_metrics_and_projection(tmp_path, make_avi):   # metrics keys {fov, fov_lit, fov_activity, fov_method, frame_shape}; projection npz has mean, activity of the frame shape
def test_detect_brightness_method(tmp_path, make_avi):        # fov_method brightness -> fov == fov_lit, no seed maps computed (compute_init_maps patched to raise)

# tests/bearmind/test_video.py
def test_session_reader_windows(tmp_path, make_avi):   # 1000 frames, frame i has value i % 251: windows(3, 100) -> frames 0-99, 450-549, 900-999; windows(3, 400) -> all 1000

# tests/bearmind/test_crop_overview.py
def test_overview_draws_every_session(tmp_path):   # 3 records with crop metrics, 2 with projections -> PNG exists, readable by cv2, height grows with rows
def test_overview_none_without_projections(tmp_path, monkeypatch):   # returns None; matplotlib not imported (sys.modules)

# tests/bearmind/test_runner.py
def test_finalize_writes_the_crop_overview(series):   # crop stub writes a projection npz -> after _run paths.crop_overview exists
def test_overview_failure_does_not_stop_the_batch(series, monkeypatch):   # write_overview raises -> run result as usual, "crop overview not written" in series.log
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. Прежние тесты `test_detect_fov_*` проходят без правок (`detect_fov` — прежнее правило яркости).
- [ ] **Step 4:** PASS; быстрый набор PASS.
- [ ] **Step 5:** Commit: `git add bearmind/crop_overview.py tests/bearmind/test_crop_overview.py` и `git commit -m "feat(crop): field of view from calcium activity inside the lit area; crop overview per series" -- bearmind/crop.py bearmind/stages/crop.py bearmind/crop_overview.py bearmind/video.py bearmind/paths.py bearmind/runner.py bearmind/defaults.yaml tests/bearmind/test_crop.py tests/bearmind/test_crop_overview.py tests/bearmind/test_video.py tests/bearmind/test_runner.py`

### Task 7: Монитор: побочные этапы, седьмой вид панели, флажок превью

**Files:**
- Modify: `bearmind/monitor.py`, `bearmind/gui/monitor_tab.py` (фаза 2)
- Test: `tests/bearmind/test_monitor.py`, `tests/bearmind/gui/test_monitor_tab.py`, `tests/bearmind/gui/test_actions_ui.py`

**Interfaces:**
- Consumes: `registry.SIDE_STAGES`, `ALL_STAGES`; аномалия `cnmf_downsampled` (задача 4); `actions.set_previews`, `actions.retry_failed` (задача 3); `previews.preview_params`.
- Produces:
  - `STAGE_TITLE["mc_preview"] = "MC preview"`, `STAGE_TITLE["cnmf_preview"] = "CNMF-E preview"`.
  - `WarningItem.kind` получает седьмое значение `"cnmf_downsampled"`: аномалия с этим кодом, её `level` и `message`; `HINTS["cnmf_downsampled"] = "Background downsampled to fit RAM; a tighter crop avoids it."`; `retry=False`.
  - `stage_failed` — и для FAILED побочного этапа: `stage` = имя побочного этапа, `retry=True`; Retry вызывает `actions.retry_failed(paths, [sid], who)`, который сбрасывает только превью.
  - Клетки MC и CNMF-E: в подсказку добавляется строка `preview: off | pending | running | done | failed`. Столбцов в таблице по-прежнему семь (макет).
  - `MonitorTab.previews_box: QCheckBox` с текстом «Preview videos» в шапке (часть 1), справа от «Open folder». Отмечен = `preview.enabled` разрешённого конфига серии. Переключение вызывает `actions.set_previews(paths, checked)` без подтверждения (обратимо, ничего не удаляет). Статусная строка: `Previews on: the batch makes them for sessions whose MC / CNMF-E is done` или `Previews off: existing files are kept`.
  - Оценка времени и полосы прогресса побочные этапы не считают.

- [ ] **Step 1: Тесты**
```python
# tests/bearmind/test_monitor.py
def test_warning_panel_has_only_seven_kinds(tmp_series): ...   # was ..._six_kinds (phase 2): kinds subset of the 6 + "cnmf_downsampled"; other checks and anomalies still absent
def test_downsampled_anomaly_in_panel(tmp_series): ...         # cnmf anomaly cnmf_downsampled warning -> WarningItem(kind="cnmf_downsampled", level="warning", retry=False), text has "ssub_B 3 instead of 2" and the hint
def test_failed_preview_in_panel_with_retry(tmp_series): ...   # mc_preview FAILED -> stage_failed entry, stage "mc_preview", retry True; MC cell stays "ok" with "preview: failed" in its tooltip
def test_side_stages_not_in_bars_or_eta(tmp_series): ...       # bars keys == STAGES; a pending mc_preview adds nothing to eta_s
def test_runner_current_side_stage_title(tmp_series): ...      # heartbeat current {"stage": "cnmf_preview"} -> header detail contains "CNMF-E preview"

# tests/bearmind/gui/test_monitor_tab.py
def test_previews_checkbox_reflects_and_writes_config(qapp, series4): ...   # unchecked by default; click -> series.yaml preview.enabled True, status message "Previews on"; again -> False
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать; переименовать тест фазы 2 `test_warning_panel_has_only_six_kinds` в `..._seven_kinds` с новым списком.
- [ ] **Step 4:** `QT_QPA_PLATFORM=offscreen` — тесты PASS; быстрый набор PASS.
- [ ] **Step 5:** Commit: `git commit -m "feat(monitor): preview stages, downsampling warning, preview checkbox" -- bearmind/monitor.py bearmind/gui/monitor_tab.py tests/bearmind/test_monitor.py tests/bearmind/gui/test_monitor_tab.py tests/bearmind/gui/test_actions_ui.py`

### Task 8: Проверка на реальных данных без тяжёлых прогонов, документы

**Files:**
- Create: `docs/superpowers/specs/<дата>-a2-long-sessions-check.md` (`<дата>` — день выполнения, `YYYY-MM-DD`), папка `docs/superpowers/specs/a2-check/` (картинки обзора кропа, каждая < 500 КБ)
- Modify: `docs/superpowers/specs/2026-11-01-operator-notes.md` (разделы «Превью», «Обзор кропа», «Если CNMF-E понизил фон»)

Правила: MC запускается только на клипах `series_short` (2000 кадров), CNMF-E — только там же. Исходные данные и папки `1_Raw` только читаются. Серии, которые пишет проверка, — только в `H:\Projects\_bearmind_test`.

- [ ] **Step 1:** Быстрый набор зелёный; медленные синтетические тесты задач 2, 5 зелёные.
- [ ] **Step 2: Кроп часовой сессии.** `run_bearmind.bat run H:\Projects\_bearmind_test\series_lnof --until crop` (в этой серии сделаны только скан и кроп; MC не запускается). Записать поле, рамку и время этапа. Ожидается рамка 472×436 ± 10 px и время ~2 мин. Вызвать `background_downsample` для новой рамки и `T = 107 107`, ожидается `ssub_B 2`, и `map_workers`, ожидается 3. Отклонение больше 10 px — разобраться до следующего шага: числа плана получены тем же методом.
- [ ] **Step 3: Обзор на шести сессиях series1.** Создать `H:\Projects\_bearmind_test\crop_check_series1` с junction `1_Raw` → `H:\Projects\_bearmind_test\series1\1_Raw` (`cmd /c mklink /J`), затем `scan` и `run ... --until crop`. Сама `series1` не трогается: её MC остаётся актуальным. Сверить рамки с таблицей П4.
- [ ] **Step 4: Обзор на LT1.** Проверить, что `H:\Projects\_bearmind_test\scan_lt1\1_Raw` — junction на `H:\Projects\LT1\CalciumData\1_Raw` и что реестр там есть. Выполнить `run H:\Projects\_bearmind_test\scan_lt1 --until crop` (24 сессии, ~30–50 мин; пишет только в `scan_lt1\_bearmind`). Сравнить список и mtime файлов `LT1\CalciumData` до и после.
- [ ] **Step 5: series_short целиком.** `run H:\Projects\_bearmind_test\series_short`. Новый кроп меняет рамку, поэтому MC, пороги и CNMF-E двух клипов по 2000 кадров пересчитаются один раз (~25 мин; это разрешённые короткие клипы). Ворота ждут новое решение (R28): `gate ... --all-waiting not-needed`, снова `run`. Ожидается: превью не сделаны, в `Preview/` нет новых mp4, в `summary.csv` стоит `off`. Время этапа MC записать рядом с прежними 149 / 134 с из отчёта e2e (тогда с превью и с прежней рамкой, поэтому сравнение грубое). Затем `previews ... on` и `run`: запускаются только `mc_preview` и `cnmf_preview`, `finished` у MC и CNMF-E не изменился. В конце `BEARMIND_TEST_SERIES=...series_short pytest tests/bearmind/test_e2e_real.py` → 4 passed.
- [ ] **Step 6: Параллельные карты на реальном клипе.** Скриптом в scratch вызвать `thresholds.run` для NOF_H01_1D из `series_short` на отдельном корне scratch (junction `3_MC` → `series_short\3_MC`; пишет только в scratch) дважды: `n_cpus 2` и машинный. Записать время обоих и `map_workers`; NPZ должны совпасть побайтно по массивам.
- [ ] **Step 7: Отчёт** `<дата>-a2-long-sessions-check.md` (метки [изм] / [вывод]). Содержание: таблица рамок «было / стало» по всем проверенным сессиям; обзоры кропа (series_lnof, crop_check_series1, LT1) с просьбой к автору посмотреть их один раз; время этапа MC без превью и времена побочных этапов; решение о `ssub_B` для LNOF; время карт порогов serial / parallel. Отдельно — что не проверено: CNMF-E и карты порогов на часовой сессии (правило автора: без тяжёлых прогонов), реальный пик памяти `compute_W` при `ssub_B 3`. В заметки оператора (по-русски, стиль STE, как разделы A1) добавить: как включить превью (`previews SERIES on`, флажок в окне), что значит `cnmf_downsampled`, где обзор кропа.
- [ ] **Step 8:** Commit: `git add docs/superpowers/specs/<дата>-a2-long-sessions-check.md docs/superpowers/specs/a2-check` и `git commit -m "docs: A2 check on real data (crop overview, previews, downsampling)" -- docs/superpowers/specs/<дата>-a2-long-sessions-check.md docs/superpowers/specs/a2-check docs/superpowers/specs/2026-11-01-operator-notes.md`

## Вне плана

- Out-of-core `compute_W` (подмена функции CaImAn): по-прежнему в TODO автора. С новым кропом часовая сессия LNOF влезает и без понижения (111.7 из 152.96 ГиБ) [вывод].
- Карты фильма для PNG остатков из карт порогов (−12 с из 46 с на клип 2000 кадров): меняет `missed_cells`, нужно решение автора.
- Понижение по `tsub` или `ssub` (CaImAn) для воркеров патчей: число воркеров уже подбирается под память. Отказ воркера патча при T > ~600 000 кадров (≈ 5.5 ч при 30 к/с) не рассматривается [вывод].
- Полоса `compute_init_maps` больше бюджета при длинном T (задача 10 ядра): параллельный расчёт её учитывает, но саму полосу не уменьшает.
