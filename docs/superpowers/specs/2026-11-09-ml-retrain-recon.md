# Разведка к переобучению классификатора автоинспекции (2026-11-09)

Только чтение: код, история git, обучающий CSV, файлы CNMF-E на H: и G:. Ни одного прогона CNMF-E или автоинспекции. Скрипты проверок лежат во временной папке `C:\Users\User\AppData\Local\Temp\claude\3fb66d12025b3383\8c6833f1-37e2-4703-88da-d2e058320a8f\scratchpad\ml_recon\` (`compare.py`, `gt_check.py`, `batch_gt.py`, `lnof_check.py`, `lnof_map.py`, `cthr_check.py`, `trace_feats.py`, `spatial_feats.py`, `eval_split.py`, `cv_overlap.py`, `corr_count.py`).

**Цель переобучения.** Модель EBM (объяснимая модель бустинга, interpret) должна учиться на правильных признаках и честной разметке. На этом всё держится. Судьбу работы решают три вещи:

1. Дефект сжатия испортил 5 из 35 признаков во ВСЕХ 187 сессиях, включая LNOF. Остальные 30 признаков пересчитываются здесь точно.
2. Разметка не однородна. В NOF/RFC/FOF метка — ручной результат (GT, эталонная разметка эксперта). В LNOF (54 % строк) метка — решение прежней модели. Плюс около 2000 правок меток в сторону модели.
3. Ручного GT для RFC (28 сессий) на этой машине нет. Без него RFC пересчитать нельзя.

## Итог по вопросам

| № | Вывод |
|---|---|
| 1 | NOF/RFC/FOF: `capcan_validation.batch_validate` -> `estimates_to_metrics` на `data/raw_compressed` (сжатые CNMF-E) и `data/final_compressed` (сжатый ручной результат) -> `experiments/create_training_dataset.py` (метка = центр ближе 3 px к центру GT). LNOF: `metrics_df` из `*_processed.pickle` автоинспекции, метка = `1 - delete`. |
| 2 | Испорчены 5 признаков: `caiman_snr`, `caiman_r_score`, `noise_level`, `baseline`, `tau_decay`. A и C при сжатии выбраны верно. 12 трейсовых и пространственных признаков проверены и совпадают точно. Контурные признаки зависят от `cthr` и версии CaImAn. Событийные признаки от дефекта не зависят, но не пересчитаны. |
| 3 | Да, сжаты. В 6 сессиях NOF/RFC/FOF эти 5 столбцов равны значению «первых N» в 94-100 % строк и своему значению в 0-10 % строк. В 2 сессиях LNOF — то же (100 % и 10 %). `component_idx` — не номер сырого CNMF-E. |
| 4 | Сырые CNMF-E есть для 187/187 сессий. Ручной GT есть для NOF 60/60 и FOF 11/11 (проверено: `distance_to_gt` воспроизводится в 14/14 сессий точно). Для RFC 0/28: `4_Estimates_refined` не тот GT. Для LNOF ручного GT нет по замыслу. |
| 5 | Просить у Никиты: папку `capcan_validation_127*`, `data/raw_compressed` и `data/final_compressed` (прежде всего 28 RFC), всю цепочку CSV и журналов правок, данные LNOF (`data/LNOF/inspection_artifacts_*`), командные строки, коммит и окружение. |
| 6 | Главные ловушки: метки LNOF сделаны моделью; правки меток всегда в сторону модели; «кросс-валидация» README и статьи оценивает модель на сессиях её же обучения (AUC 0,978 завышен); `cthr` 0,3 в обучении против 0,35 в бою; угловые артефакты исключены в обучении, но оцениваются в бою; новая модель перезапустит autoinspect и export всех серий. |

**Что нужно от автора.** (а) Отправить Никите список из раздела 5. (б) Решить судьбу LNOF: исключить, переразметить вручную или оставить только строки с feedback. (в) Решить, публикуются ли числа статьи (AUC 0,978) — см. риск 6.3.

## 1. Поток данных

```
NOF / RFC / FOF  (99 сессий, 42 475 строк)
  сырой CNMF-E  <SID>*_estimates.pickle              ручной результат <SID>*_final*_estimates.pickle
        |  compress_estimates_ultra_lightweight          |  то же сжатие
        |  (старое: C, YrA, A по idx_components,         |
        |   bl, c1, neurons_sn, SNR_comp, r_values, g     |
        |   -> arr[:N]; idx_components -> 0..N-1)         |
        v                                                 v
  data/raw_compressed/                              data/final_compressed/
        |   capcan_validation.batch_validate (capcan_validation.py:357-385)
        |   estimates_to_metrics(est, fps=round(fps), include_event_based=True,
        |       hybrid_kinetics=True, cthr=0.3 (умолчание), detect_corner_artifacts_flag=True)
        v                                                 v
  capcan_artifacts_<SID>/metrics_init.csv          capcan_artifacts_<SID>/metrics_gt.csv
        (ae_utils.save_validation_outputs, ae_utils.py:348-354)
        \___________ experiments/create_training_dataset.py ___________/
            1) убрать строки is_corner_artifact == 1
            2) ground_truth = 1, если min |center - center_GT| <= 3 px; distance_to_gt = этот минимум
                                   v
        training_dataset_v8.csv  (+ half_crossing_rate: experiments/add_hcr_to_dataset.py,
                                   из est.C[component_idx] файлов data/raw_compressed)
                                   v   правки v8 iter1..5 (experiments/v8_iterations/*)
        training_dataset_v8_corrected_iter5.csv ------------------------------+
                                                                              |
LNOF  (88 сессий, 49 767 строк)                                               |
  сырой CNMF-E (G:\Projects\LNOF\CalciumData\4_Estimates)                      |
     -> ? промежуточный файл: сжат старым сжатием, idx_components меньше     |
     -> ae_launch.run_auto_inspection (cthr=0.35, углы выкл., модель v7/v8?)  |
     -> data/LNOF/inspection_artifacts_LNOF_*/*_processed.pickle (est.metrics_df)
     +  *_feedback.csv (FP/FN эксперта; нет файла = «модель права»)            |
     -> experiments/create_lnof_from_processed.py -> LNOF_dataset_from_processed.csv
        метка: ground_truth = 1 - delete                                      |
                                   experiments/merge_v8_lnof_to_v9.py  <------+
                                   v
        training_dataset_v9.csv -> правки v9 iter1..7 (ml/apply_corrections.py)
                                   v
        ml/results/training_dataset_v9_corrected_iter7.csv  (92 242 x 52, в git)
                                   v   ml/retrain_iter.py, seed 46, признаки = get_feature_cols
        production_models/ebm_v9_iter8.pkl  (sha256 175c0af7...4216, 35 признаков, 20 пар)
```

Доказательства:

- `capcan_validation.py:21-23`: `init_path = data/raw_compressed`, `gt_path = data/final_compressed`, комментарий «use compressed data». До 2025-12-14 те же пути назывались `raw_ultra_lightweight` / `final_ultra_lightweight` (коммиты 6bdf1e5…5333e4b). Оба имени соответствуют функции `compress_estimates_ultra_lightweight`.
- Ошибка «первых N» есть в самой первой версии функции (f6ce19f, 2025-12-09, строки 95-111 диффа) и живёт до 6c73765 (2026-10-31).
- Метка GT: `experiments/create_training_dataset.py:86-108`. Центр — `CoM` из `caiman.utils.visualization.get_contours` (`polygon.py:7-12`), то есть центр масс всего футпринта (пространственной маски компоненты). Сопоставление не один-к-одному: два кандидата у одного GT-нейрона оба получают 1.
- `ml/data_utils.construct_gt_labels` (`data_utils.py:189-222`) повторяет то же правило (<= 3 px). Его используют `train_ebm.py` и `grid_search_ebm.py` в режиме папок. Модель v9 обучена из CSV через `retrain_iter.py`, не через этот путь.
- LNOF: `experiments/create_lnof_from_processed.py:68-115` (метки из `est.metrics_df['delete']` и feedback), `experiments/merge_v8_lnof_to_v9.py:96-115` (`ground_truth = 1 - delete`, `distance_to_gt = NaN`).
- `experiments/clean_v9_dataset_nans.py` к итоговому CSV не применён: NaN в событийных столбцах остались (0,19-0,25 % строк).
- `ellipse_r` у LNOF: скрипт слияния ставит NaN (`merge_v8_lnof_to_v9.py:109`), а в итоговом CSV значения есть у всех 49 767 строк. Скрипт, который их заполнил, в репозитории не найден.

## 2. Столбцы CSV: источник и риск

Признаки модели: 35 столбцов (`model.feature_names_in_` совпадает с `get_feature_cols(df)`). Остальные 17 — служебные.

Строки кода — `auto_inspector.estimates_to_metrics` (`auto_inspector.py:1040-1270`).

**Испорчены дефектом сжатия (5).** Проверено по данным, раздел 3.

| Столбец | Источник | Риск |
|---|---|---|
| `caiman_snr` | `est.SNR_comp[comps]`, :1132 (inf заменяется максимумом, :1137) | испорчен |
| `caiman_r_score` | `est.r_values[comps]`, :1138 | испорчен |
| `noise_level` | `est.neurons_sn[comps]`, :1141 | испорчен |
| `baseline` | `est.bl[comps]`, :1142 | испорчен |
| `tau_decay` | `est.g[comps]` -> `-1/log(g)/fps`, :610-641, :1143; fps округлён до целого | испорчен |

Доля важности модели в членах с этими признаками — 8,7 % (`caiman_snr` на 22-м месте из 55). Модель училась на шуме и потому мало на них опирается. В бою значения правильные, и форма функций для них не обучена.

**Безопасны: трейсы из C (C выбран при сжатии верно).** Пересчитаны кодом репозитория из сырого CNMF-E NOF_H01_1D. Совпали в 100 % строк (rtol 1e-3).

`trace_skewness`, `trace_kurtosis`, `bimodality` (:1149), `mean_time_at_peak` (:1152), `hurst_exponent` (:1155), `baseline_drift` (:1158), `half_crossing_rate` (:1165; у v8-части добавлен позже из `est.C[component_idx]`, тоже верно).

**Безопасны: пространство из A и центров.** NOF_H01_1D, совпали в 100 % строк. Набор соседей — все компоненты, принятые CaImAn.

`center` (точно, 0 px), `eccentricity` (:1119), `edge_distance`, `nn_distance_center`, `local_density` (:1122-1130), `ellipse_r` (:1245).

**Безопасны от дефекта, но зависят от `cthr` и версии CaImAn.** `area`, `circularity`, `max_edge`, `convexity`, `aspect_ratio`, `footprint_compactness` (:1097-1116, :1161).

- NOF/RFC/FOF посчитаны с `cthr = 0.3`. LNOF и боевой конвейер — с `cthr = 0.35`. Проверка `area`: при верном `cthr` медиана расхождения 1,2-1,8 %, при чужом — 20 % (NOF_H01_1D, RFC_F01_1D, LNOF_J21_4D).
- Остаток 1-2 % — эффект CaImAn 1.13 (`norm_nrg`, см. `2026-11-01-e2e-report.md`, шаг 6).

**Безопасны от дефекта, не пересчитаны: события и реконструкция (driada по нормированному C).** `event_r2_score`, `event_snr`, `events_fraction`, `events_per_min`, `peak_amplitude_cv`, `t_rise`, `t_off`, `kinetics_opt`, `nmae`, `nrmse`, `r2_score`, `snr_recon`.

- Зависят от `event_method`, `n_iter`, `hybrid_kinetics`, `include_heavy`, целого fps и шума driada без seed.
- Признак расхождения настроек: `kinetics_source = wavelet_standard` у LNOF в 99,2 % строк, у NOF в 83,7 %, у RFC в 58 %. Статус: неизвестно до пересчёта.

**Служебные (не признаки).**

| Столбец | Что это | Замечание |
|---|---|---|
| `component_idx` | NOF/RFC/FOF: номер k в сжатом файле, сырой номер = `idx_components_raw[k]`; LNOF: сквозной номер подмножества | НЕ номер сырого CNMF-E |
| `session`, `session_name`, `experiment` | метаданные | — |
| `ground_truth` | метка | см. риски 6.1, 6.2 |
| `distance_to_gt` | расстояние до ближайшего GT | NaN у LNOF; воспроизводится точно |
| `delete`, `decision` | v8-часть: `1 - ground_truth` (скрипт слияния); LNOF: решение автоинспекции | `delete` = метка; утечка, если попадёт в признаки (риск 6.9) |
| `merge`, `failed_*`, `is_corner_artifact`, `corr_groups` | v8-часть: заглушки; LNOF: решения автоинспекции | — |
| `ml_keep_probability` | NaN у v8-части; у LNOF — P прежней модели | — |
| `kinetics_source` | строка; отсекается по типу | есть в `FEATURE_COLS` (`data_utils.py:92`) |

Косвенной порчи признаков нет: другие признаки `caiman_snr` не используют. Метки LNOF испорчены косвенно: их ставила модель по испорченным признакам.

## 3. Были ли входы обучения сжаты

**Путь в коде.** Да: `capcan_validation.py:21-23` -> `data/raw_compressed`. Скрипты `experiments/add_hcr_to_dataset.py:12`, `experiments/ml_experiments/visualize_errors.py:21`, `experiments/inspect_estimates_temp.py:5` читают тот же каталог; пример имени — `NOF_H01_3D_gsig4_mincorr0.92_minpnr7_estimates.pickle`.

**Данные.** Скрипт `compare.py`:

1. Строка CSV с `component_idx = k` сопоставлена сырой компоненте `good[k]` (`good = idx_components` сырого файла).
2. Центр и `trace_skewness` совпали в 100 % строк. Значит, сопоставление верно, а k — сжатый номер. Гипотеза «k = сырой номер» даёт 0-10 %.
3. Каждый из 5 столбцов сравнён со значением своей компоненты `arr[good[k]]` и со сдвинутым `arr[k]` (ошибка «первых N»), rtol 1e-4.

| Сессия | сырых / принятых CaImAn / строк CSV | свой | сдвинутый `arr[k]` | строк без сдвига | rho (Спирмен) CSV-SNR / свой SNR |
|---|---|---|---|---|---|
| NOF_H01_1D | 228 / 185 / 176 | 0-0,6 % | 100 % | 0 % | -0,10 |
| NOF_H27_3D | 352 / 261 / 232 | 0 % | 100 % | 0 % | -0,16 |
| RFC_F01_1D | 184 / 159 / 134 | 9,7 % | 99,3-100 % | 9,7 % | 0,00 |
| RFC_F35_3D | 360 / 232 / 202 | 0 % | 100 % | 0 % | 0,19 |
| FOF_F53_1D | 160 / 155 / 37 | 0 % | 94,6-100 % | 0 % | 0,17 |
| FOF_F11_1D | 202 / 173 / 59 | 10,2 % | 98,3-100 % | 10,2 % | 0,26 |

- Неполные 100 % у `caiman_snr` — строки с бесконечным SNR (замена на максимум, :1137).
- «Свой» совпадает только у компонент до первой отбракованной CaImAn (9,7 % и 10,2 %).
- `tau_decay` совпал при fps, округлённом до целого (NOF_H01_1D: 19,76 -> 20).
- Сырые файлы RFC и FOF на H: — тот же прогон CNMF-E, что в обучении: центры совпали точно.

**LNOF тоже испорчен** (`lnof_check.py`, `lnof_map.py`, `lnof_tau.py`):

| Сессия | сырых / принятых / строк | центр = сырая компонента | своя SNR | `arr[component_idx]` |
|---|---|---|---|---|
| LNOF_J21_4D | 131 / 103 / 83 | 100 % | 0 % | 100 % (`caiman_snr`, `caiman_r_score`, `noise_level`, `baseline`, `tau_decay` при fps 20) |
| LNOF_J21_1D (1 слияние) | 144 / 106 / 94 | 98,9 % | 0 % | 9,7 % (после слияния номера сдвинуты ещё раз) |

- В LNOF_J21_4D строка k = 0 — сырая компонента 9. Её ранг среди принятых — 8, а не 0.
- 20 принятых CaImAn компонент в таблице отсутствуют.
- Вывод: вход автоинспекции LNOF был промежуточным файлом, сжатым старым сжатием, с меньшим `idx_components`. Этого файла на машине нет.
- 50 строк `from_merge` LNOF несут `caiman_snr` из неперенумерованного `SNR_comp` (до fbd1ad3).

**Цитата аудита main** (`docs/code_improvement_ideas.md`, раздел «COMPONENT ALIGNMENT AUDIT (2026-10-18)», коммит 8552e88 от 2026-11-01, niveousdragon):

> ### [HIGH] Training features come from mis-compressed files
> - **Location**: `capcan_validation.py` init_path = data/raw_compressed ->
>   metrics_init.csv -> `experiments/create_training_dataset.py` -> EBM training data
> - **Issue**: all 127 files in data/raw_compressed (and data/final_compressed) were made
>   by the old compression: stored SNR_comp vs SNR recomputed from each file's own traces
>   has Spearman rho ~0 on average (max 0.68; a correctly paired session gives 0.85).
>   Labels are matched by position and are fine; caiman_snr, caiman_r_score, noise_level,
>   baseline and tau_decay are partly another neuron's. On uncompressed sessions these
>   features are correct, so the model is trained on a different distribution than it
>   is applied to. Needs the original uncompressed CaImAn outputs to rebuild.

Сообщение коммита 6c73765 (2026-10-31):

> compress_estimates_ultra_lightweight sliced bl, c1, neurons_sn, SNR_comp,
> r_values and g to the first N entries instead of selecting the good
> components by ID, so every kept neuron carried another neuron's SNR,
> r_values, baseline and decay. lam and cnn_preds were not sliced at all.

`docs/superpowers/specs/ml-filter-2026-11-01/findings.md`, пункт 5 и обновление 2026-11-06: боевые входы автоинспекции (`4_Estimates_refined`, 9 из 9 файлов) не сжаты, то есть признаки данных статьи верны. Пункт «Обучение модели [не проверено мной]» повторяет вывод аудита.

Аудит прав в главном. Уточнения: (1) порча касается и LNOF — аудит говорит только о 127 файлах capcan; (2) «Labels are fine» верно только для NOF/FOF (проверено) — у LNOF метки ставила модель.

## 4. Можно ли пересчитать здесь

| Набор | Сессий | Сырой CNMF-E | Ручной GT | Проверка |
|---|---|---|---|---|
| NOF | 60 | 60/60 `H:\Dataset\CalciumData\NOF\4.1_EstimatesRaw\<SID>_gsig*_estimates.pickle` | 60/60 `H:\...\NOF\4_Estimates\<SID>*_final*_estimates.pickle` | 10/10 сессий: `distance_to_gt` и метка до правок совпали в 100 % строк |
| RFC | 28 | 28/28 `H:\...\RFC\4_Estimates\<SID>_estimates.pickle` | 0/28 | `4_Estimates_refined` (2026-04-25) — не тот GT: `distance_to_gt` совпал в 23-99 % строк, 28/28 сессий |
| FOF | 11 | 11/11 `H:\...\FOF\4_Estimates` и `G:\Projects\FOF\4_Estimates\<SID>_estimates.pickle` | 11/11 `G:\Projects\FOF\4_Estimates\<SID>_final_estimates.pickle` | 4/4 сессий: 100 % |
| LNOF | 88 | 88/88 `G:\Projects\LNOF\CalciumData\4_Estimates\<SID>_estimates.pickle` | нет по замыслу | `*_processed.pickle`, `*_feedback.csv` и промежуточный вход — 0/88 |

- Корпус 127 сессий = 60 NOF + 28 RFC + 11 FOF + 28 3DM. Сессии NOF_H36 (4 шт.) есть на H:, но в корпус не входили.
- `RFC/5_Filters` (2026-05-15) — выход боевого экспорта, не ручной результат.
- `FOF/5_Filters` и `G:\Projects\FOF\5_Filters` — футпринты ручного результата 2024 г.

**Что нужно для пересчёта `metrics_init`.**

1. Вызывайте `auto_inspector.estimates_to_metrics` прямо на НЕсжатом сыром файле. `comps_to_select` = `idx_components` сырого файла. `capcan_validation` не подходит: он ищет файлы по `name[:10]`, сам решает и сливает компоненты и требует пару каталогов.
2. Задайте те же параметры, что в обучении:
   - `fps = round(fps)` по `fps_data.csv`;
   - `cthr = 0.3` для сравнения с CSV;
   - `detect_corner_artifacts_flag = True`, потом убрать `is_corner_artifact == 1`;
   - `include_event_based = True`, `include_heavy = True`, `hybrid_kinetics = True`;
   - `event_method` и `n_iter` — спросить у Никиты (умолчание CLI: `threshold`, 2; бой: `wavelet`, 3).
3. Для новой модели задайте боевые параметры (`bearmind/defaults.yaml:78-97`): `cthr = 0.35`, `detect_corner_artifacts = False`, `seed = 0`. Это решение автора, риск 6.5.
4. Перенесите номера. Новый `component_idx` — сырой номер. Старый — k, сырой = `idx_components_raw[k]` (NOF/RFC/FOF). Для LNOF переносите только по центру (точное совпадение). Журнал правок индексирован старыми номерами.
5. Учтите версию CaImAn. Признаки обучения посчитаны на CaImAn 1.11.3 / NumPy 1.26, бой идёт на 1.13.2. В 1.13 контурные признаки сдвинуты (area: медиана 1,4 %; 72-134 из 156 строк, `2026-11-01-e2e-report.md`, шаг 6). Для обучения новой модели 1.13 — правильный выбор: так считает бой. Для сверки с артефактами Никиты нужен допуск или отдельное окружение 1.11.3.
6. Учтите шум driada. Без seed событийные признаки расходятся в 15-50 строках из 156 от прогона к прогону. Для обучения задайте `seed = 0`. Для сверки сравнивайте распределения.
7. Учтите ресурсы. Сырые FOF весят 0,3-3,9 ГБ, LNOF — 0,07-1,8 ГБ. Событийные признаки — самая долгая часть. Не запускайте пересчёт, пока идёт батч LT1.

## 5. Что просить у Никиты и как сверять

**Список запроса.**

1. Папку артефактов, из которой собран `training_dataset_v8.csv` (точное имя: `data/capcan_validation_127`, `_v7` или другая). Из каждой `capcan_artifacts_<SID>/`: `metrics_init.csv`, `metrics_gt.csv`, `metrics_auto_final.csv`, `decisions_with_criteria.csv`, `validation_metrics.csv`, `FCD.npy`, `FBD.npy`, `match_mtx.npy`. Все 127 сессий, включая 3DM.
2. `data/raw_compressed/` и `data/final_compressed/` (по 127 pickle). Обязательно — 28 файлов RFC из `final_compressed` или исходные несжатые ручные результаты RFC. Без них RFC не пересчитать.
3. Командную строку `capcan_validation.py` (`--event-method`, `--n-iter`, `--hybrid-kinetics`, `--include-heavy`, `--artifacts-path`), дату, коммит, `conda list` окружения (CaImAn, NumPy, driada, interpret).
4. Цепочку наборов и журналы: `training_dataset_v7.csv`, `training_dataset_v8.csv`, `training_dataset_v8_corrected*.csv` (iter1-5) с журналами, `training_dataset_v9.csv`, `training_dataset_v9_corrected_iter1..6.csv` с журналами, `v9_dataset_summary.json`. По каждой итерации: `top100_fp.csv`, `top100_fn.csv`, списки «REAL» эксперта.
5. Объяснение числа правок: README называет 1129, журнал iter7 содержит 165 строк. По данным в v8-части 1735 меток отличаются от правила «<= 3 px» (116 есть в журнале, 1619 нет). В LNOF 254 строки с `ground_truth != 1 - delete` (49 в журнале).
6. Данные LNOF: `LNOF_dataset_from_processed.csv`; 88 папок `data/LNOF/inspection_artifacts_LNOF_*` (`*_processed.pickle`, `*_feedback.csv`). Какой файл был входом автоинспекции LNOF и почему в нём меньше принятых компонент. Модель, порог и `autoinspection_config`. Кто давал feedback и как.
7. Скрипт, который заполнил `ellipse_r` у LNOF после слияния.
8. Почему исключены 3DM (с v6, `*_no3dm`), NOF_H36 и остальные сессии RFC/FOF.
9. Скрипт и выборки для чисел статьи (`paper/autoinspect/main.tex:82, 143, 164`: AUC 0,978, F-beta 0,965) и для пробы с разбиением по животным.

**Сверки, которые докажут правильность пересчёта** (на каждую сессию):

| Уровень | Что сравнить | Критерий |
|---|---|---|
| 0. Состав | после переноса k -> `idx_components_raw[k]` множества компонент | равны; центры совпадают до 1e-6 px |
| 1. Трейсы и пространство | `trace_skewness`, `trace_kurtosis`, `bimodality`, `hurst_exponent`, `baseline_drift`, `half_crossing_rate`, `mean_time_at_peak`, `eccentricity`, `edge_distance`, `nn_distance_center`, `ellipse_r`; `local_density` | rtol 1e-3, atol 1e-4 (C в сжатом файле float32) в 100 % строк; `local_density` — точно |
| 2. Контуры | `area`, `circularity`, `max_edge`, `convexity`, `aspect_ratio`, `footprint_compactness` при `cthr = 0.3` | CaImAn 1.11.3: точно; CaImAn 1.13: медиана относительного расхождения <= 2 % |
| 3. Дефект | 5 испорченных столбцов | старый CSV = `arr[k]` в 100 % конечных строк; новый = `arr[good[k]]`; совпадение нового со старым только у строк до первой отбракованной компоненты |
| 4. События | 12 событийных столбцов | в том же окружении с тем же seed — точно; иначе доля строк в rtol 1e-3 и медианы по сессии; нужен отдельный прогон двух seed для оценки шума |
| 5. Метки | `distance_to_gt`; метка до правок; метка после правок | `distance_to_gt` до 1e-3 px в 100 % строк (уже так для 14 сессий NOF/FOF); после переноса правок метка совпадает с CSV в 100 % строк |
| 6. Код обучения | `retrain_iter.py` на CSV Никиты, seed 46 | вероятности воспроизводят `ebm_v9_iter8.pkl` до 1e-9; при другой версии interpret допустимо расхождение — проверить на тестовой выборке seed 46 |

## 6. Риски и ловушки

**6.1. Метки LNOF ставила модель (утечка меток через самообучение).** `ground_truth` LNOF = `1 - delete` автоинспекции. Без feedback решение считается верным (`create_lnof_from_processed.py:116`). Совпадение `ground_truth` с `ml_keep_probability >= 0.75` — 98,5 % (49 418 строк). Тест seed 46: AUC на LNOF 0,989, на NOF 0,940, RFC 0,924, FOF 0,895. LNOF — 54 % строк и 47 % тестовых: он поднимает общий AUC до 0,966. Решение автора: исключить LNOF, переразметить, взвесить или оставить только строки с feedback.

**6.2. Правки меток смещены к модели.** `ml/apply_corrections.py:121-160` переворачивает ВСЕ топ-100 ошибок FP и FN к ответу модели, кроме отмеченных экспертом как «REAL». Итого 1129 правок (README), все по направлению модели. Ошибки брались с тестовой выборки, исправленные метки шли и в обучение, и в оценку следующих итераций. Против исходных ручных меток (до правок) тест seed 46 даёт AUC 0,874 (NOF), 0,816 (RFC), 0,838 (FOF). Истина где-то между: часть правок исправляет настоящие ошибки эксперта. Рекомендация: оценивать на сессиях с нетронутыми ручными метками; правки делать вслепую, без показа ответа модели.

**6.3. «Кросс-валидация» оценивает модель на её обучающих сессиях.** `experiments/ml_experiments/cross_validate_iters5_8.py:114-165` грузит одну обученную модель (seed 46) и считает метрики на 10 разбиениях seed 42-51. В среднем 67 % тестовых сессий этих разбиений были в обучении модели (`cv_overlap.py`). Отсюда README «AUC 0.9782 ± 0.0053» и статья (`main.tex:82, 164`: «candidates from one session never appear in both training and test folds», AUC 0,978). Честная отложенная выборка seed 46: AUC 0,9655 (`ebm_v9_iter8_summary.json`, повторено здесь). Это вопрос к тексту статьи, а не только к обучению.

**6.4. Разбиение по сессиям, не по животным.** `stratified_session_split` (`data_utils.py:332`) и `retrain_iter.py:139-155` держат сессию целиком в одной выборке, но дни одной мыши (NOF_H01_1D…4D) попадают в обе. Те же нейроны повторяются по дням. Рекомендация: группировать по животному (`GroupShuffleSplit` по `NOF_H01`).

**6.5. Расхождение обучения и боя помимо дефекта.**

- `cthr`: 0,3 в NOF/RFC/FOF, 0,35 в LNOF и в бою. Площадь отличается на 20 %. В обучающем наборе — две разные шкалы контурных признаков.
- Угловые артефакты: в обучении строки `is_corner_artifact == 1` выкинуты (FOF — не меньше 34 % принятых CaImAn компонент, RFC — 16 %, NOF — 10 %; FOF_F09_1D: 52 строки из 245). В бою `detect_corner_artifacts: false`. Модель оценивает краевые компоненты, которых не видела.
- События: `event_method`, `n_iter` в обучении неизвестны (бой: `wavelet`, 3). Распределение `kinetics_source` различается по наборам.
- CaImAn 1.11.3 против 1.13.2; driada без seed против `seed: 0`.

**6.6. Порог 0,72 и хэши этапов.** Порог выбран по завышенной «CV» старой модели. Для новой модели порог надо выбрать заново на честной отложенной выборке. Хэш этапа autoinspect включает весь раздел `autoinspect` конфига (там `ml_threshold`) и `model_sha256` (`bearmind/runner.py:279-285`); хэш export включает `ml_threshold` (`runner.py:286-297`). Новая модель ИЛИ новый порог перезапустит autoinspect и export во всех обработанных сериях при следующем `run_bearmind` (данные статьи, LT1). Смена одного порога тоже перезапустит тяжёлую автоинспекцию, хотя порог нужен только экспорту. Решение автора: хранить версии моделей и фиксировать старую модель для данных статьи.

**6.7. LNOF доминирует.** 49 767 из 92 242 строк (54 %), 88 из 187 сессий. Даже с ручными метками его надо взвешивать по сессиям, иначе модель подстроится под условия съёмки LNOF.

**6.8. История 3DM и состав корпуса.** 28 сессий 3DM (17 861 нейрон, `experiments/analysis_event_method_comparison_2025_12_15/FINDINGS.md:33`) выкинуты с v6 (`ml/ebm_grid_search_v6_no3dm`). Причина в репозитории не записана. Доля оставленных экспертом в FOF скачет от 0,10 до 0,90 (FOF_F09_1D: 5 из 52, FOF_F35_1D: 0,17). Возможно, у FOF другое правило разметки. Это стоит спросить у автора разметки до обучения.

**6.9. Ловушки в коде обучения.**

- `ml/grid_search_ebm.run_grid_search_from_csv` (`grid_search_ebm.py:449-450`) не исключает `delete`, `merge`, `failed_*`, `ml_keep_probability`. На CSV v9 столбец `delete` = `1 - ground_truth` попадёт в признаки — прямая утечка метки. `retrain_iter.py` использует `NON_FEATURE_COLS` и защищён (`retrain_iter.py:96-101`).
- `ml/plot_pr_tradeoff.py:56-57` — тот же неполный список.
- `FEATURE_COLS` содержит строковый `kinetics_source` (`data_utils.py:92`).
- README модели ссылается на файлы, которых нет в `ml/`: `cross_validate_iters5_8.py`, `optimize_threshold.py`, `visualize_errors.py` лежат в `experiments/ml_experiments/`; `cv_analysis_iter5_to_iter8.csv`, `cv_comparison_v8_iter5_vs_v9_iter8.csv`, `ml/ebm_v9_iter8/feature_importance.csv` не найдены.
- Метка «<= 3 px» не один-к-одному: дубль у GT-нейрона получает 1. Merge-случаи README называет 10 % ошибок FP.
- Гиперпараметры: в модели `min_samples_leaf = 5`, умолчание `retrain_iter.py` — 2. Брать из `--base-model`.

## Не проверено

- Событийные и реконструкционные признаки (12 столбцов) не пересчитаны: нужен прогон driada, а батч LT1 занимает машину.
- Сдвиг пяти столбцов проверен в 6 сессиях NOF/RFC/FOF и 2 сессиях LNOF из 187. Аудит main говорит о 127 файлах. Сессия без отбракованных CaImAn компонент сдвига не имела бы; такие не искались.
- Метки GT проверены в 10 сессиях NOF и 4 сессиях FOF; RFC — 28 сессий против `refined` (не совпадает).
- Параметры событий в обучении (`event_method`, `n_iter`) выведены косвенно, не установлены.
- Скрипт «CV» не запускался; вывод 6.3 сделан по коду и по пересечению разбиений.

## Термины

- EBM — объяснимая модель бустинга (interpret).
- GT — эталонная разметка, ручной результат эксперта.
- Утечка меток — признак или процедура подсказывает модели ответ.
- Футпринт — пространственная маска компоненты.
- `cthr` — порог энергии для контура футпринта.
