# Аудит чистки: приложение А против текущего кода (фаза 6, задача 1)

- Дата: 2026-11-09.
- Коммит, на котором сделан аудит: `53dece2` (HEAD ветки `bearmind-batch` на начало и на конец работы).
- Номера строк — на `53dece2` (`git grep` и `git show 53dece2:<файл>`). Для `bm_examinator.py` и `bearmind/inspect_server.py` рядом с номером стоит имя функции: эти файлы может менять другой агент.
- Источники: дизайн `2026-10-30-bearmind-batch-pipeline-design.md` (§2, §5.14, §5.16, приложение А), разведка `2026-10-30-bearmind-recon.md`, журналы волн `.superpowers/sdd/*/progress.md`, план фазы 6 (строки 1–69).
- Ноутбук `BEARMiND_full_pipeline.ipynb`: рабочая копия и `HEAD` сравнены по ячейкам. Тексты и типы всех 58 ячеек совпадают. Различаются только метаданные ячейки 11 и метаданные ноутбука. Решение P6-1 (задача 5, шаг 1) проходит.

## Статусы

| Статус | Что значит |
|---|---|
| `исправлено в <коммит>` | В текущем коде пункта нет; коммит, который его убрал. |
| `осталось` | Пункт есть в коде; задача этого плана его убирает (или «—», если аудит предлагает не трогать). |
| `уходит в history` | Код переезжает в `history/` (задача 3) без переписывания. |
| `уходит с ноутбуком` | Пункт живёт только в ноутбуке или нужен только ему (задача 5). |
| `удаляется` | Файл или блок удаляется. |
| `остаётся намеренно` | Расхождение известно и оставлено решением (автора, плана или этого аудита). |
| `не подтвердилось` | Подозрение разведки не подтвердил код текущего окружения. |

## 1. Итог

### Счёт по статусам (таблицы 2, 3, 4)

| Статус | Приложение А (2) | Разведка (3) | Отложенные (4) | Всего |
|---|---|---|---|---|
| `исправлено в <коммит>` | 6 | 21 | 0 | 27 |
| `осталось` | 22 | 5 | 11 | 38 |
| `уходит в history` | 6 | 3 | 2 | 11 |
| `уходит с ноутбуком` | 3 | 7 | 1 | 11 |
| `удаляется` | 8 | 0 | 5 | 13 |
| `остаётся намеренно` | 0 | 5 | 2 | 7 |
| `не подтвердилось` | 1 | 0 | 0 | 1 |
| **строк** | **46** | **41** | **21** | **108** |

Сверка ноутбука (таблица 5): 41 строка. Расхождение без решения было одно: `tsub` (Н2). Автор решил 2026-11-09: оставить 1.
Окружение (таблица 7): 19 строк. Новых пакетов фазы 3–5 не ставили; не закреплены 15 пакетов, которые код импортирует напрямую.

### Что меняет план задач 2–7 (план этого не предусмотрел)

Подробности и рекомендации — в разделе 8. Решающие пункты:

1. **`tsub` CNMF-E (Н2).** Ноутбук считал CNMF-E с `tsub=2` (ячейка 26 переписывает `tsub: 1` ячейки 24). В `defaults.yaml` стоит `tsub: 1`. Это второе расхождение кроме `fr`. Задача 5, шаг 2 на нём остановится. Нужно решение автора или контролёра до задачи 5. **Решено автором 2026-11-09: оставить `tsub` = 1, намеренно; данные статьи считались с `tsub` = 2.**
2. **Третий `os.chdir` и `config.json` (П-1, П-2).** `os.chdir(REPO_ROOT)` стоит ещё в `bearmind/runner.py:1583` (`run`). Через `config.json` текущей папки автоинспекция выбирает бэкенд `joblib` (`auto_inspector.py:497, 676, 753, 975` → `config.get_parallelization_backend`). Без `chdir` выбор станет зависеть от папки запуска. Задача 2 должна закрепить `config.json` за корнем репозитория.
3. **Тест «один источник параметров автоинспекции» в задаче 4 (П-7).** В текущем виде он невыполним: литералы `n_iter`, `ml_threshold`, `pxlthr_distance_boundary` стоят в умолчаниях функций `auto_inspector.py`, `capcan_validation.py`, `bm_examinator.py`, в `experiments/`, `ml/`, `tests/`. Ещё риск: 14 параметров `run_auto_inspection` этап берёт из умолчаний `ae_launch` (их нет в хэше этапа). Если задача 4 их поменяет, результаты изменятся молча.

Ещё 14 пунктов меньшего веса: П-3…П-6, П-8…П-17 (раздел 8).

### Что нужно от автора

- Н2 `tsub`: оставить 1 (умолчание всех прогонов фаз 1–5) или вернуть 2 (как данные статьи). Рекомендация — оставить 1 и записать как намеренное изменение, как `fr`.
- О4: копия старого `startup.py` в профиле IPython `C:\Users\User\.ipython\profile_default\startup\startup.py` (2025-09-30). Она работает при каждом запуске IPython. Аудит её не трогал.

## 2. Приложение А

| № | Пункт | Где сейчас (`53dece2`) | Статус | Задача |
|---|---|---|---|---|
| А1 | FPS из таблицы округляется при экспорте | `export_estimates_data.py:48-77` `get_fps_from_table` возвращает `float` без округления; этап пишет точный `rec.fps` (`bearmind/stages/export.py:166`). Округление в автоинспекции явное: `bearmind/stages/autoinspect.py:54`, метрика `autoinspection_fps` | `исправлено в 1b7b368` | — |
| А2 | `capcan_validation.load_fps_data` без `sep=';'` | `capcan_validation.py:49` — `sep=';'` есть | `исправлено в e64d693` | — |
| А3 | `ReDoCNMF` строит `opts` до seeds | `bm_batch_routines.py:621-626` без изменений | `уходит в history` | 3 (баг описать в `history/README.md`) |
| А4 | `fr=fps/2` при `tsub=2` | ячейка 26; новый код: `bearmind/stages/cnmf.py:96-112` `cnmf_params`, `fr=float(fps)` | `исправлено в 7cca587` | — (про `tsub` см. Н2) |
| А5 | `ExamineCells`: `end_frame=0` по умолчанию | `bm_examinator.py` `_FIRST_VIEW_DEFAULTS`, строка 758: `end_frame: None` — вся длина. Ячейка 39 с `end_frame: 90000` уходит с ноутбуком | `исправлено в 8f70a10` | — |
| А6 | Ручная инспекция пишет в текущую папку | Сервер пишет через хуки: `bm_examinator.py` `make_app` (строка 900, `final_save`, `progress_save`), `bearmind/inspect_server.py` (74b9d0a). Без хуков пишет в текущую папку: `make_app` → `bkapp` → `save_callback` (1768), `final_save_callback` (1785), `save_feedback_callback` (1856, запасной путь в cwd 1914-1922). Без хуков окно вызывают только ноутбук и тесты | `исправлено в 74b9d0a` | — (режим без хуков оставить: О19) |
| А7 | `update_config` подменяет `config.CONFIG` | `config.py:60-61` (`global CONFIG`) | `осталось` | 2 |
| А8 | Неизвестный `DATA_PATHWAY` → `UnboundLocalError` | `config.py:95-103` `get_session_name_from_path`. Режим съёмки конвейер определяет сам (`bearmind/scan.py`, bf753a9) | `осталось` | 2 |
| А9 | Ноутбук: неопределённые `folder_structure`, `events_to_csv`, `get_events_from_ridges`; `mc_dict` без `write_mp4`; запятые-кортежи | ячейки 11, 53, 55 (имена), 17 (`write_mp4`), 55 (запятые). `folder_structure` определяла старая копия `startup.py` в профиле IPython (О4) | `уходит с ноутбуком` | 5 |
| А10 | FPS по умолчанию 20 / 30 / 19.76 | `bm_examinator.py` `default_fps=20`: `LoadEstimates` 122, `get_timestamps` 149, `get_fps_from_timestamps` 175, `prepare_session` 837, `make_app` 900, `ExamineCells` 2184; `inspect_server.py` `DEFAULT_FPS=20` (строка 116; первый вид его не использует); `ae_launch.py:33` `get_fps_from_table(default_fps=30)` — молча 30, если сессии нет в таблице; `auto_inspector.py:983` (30), 2282 (20); `bm_qc.py:697`, 912 (20); `capcan_validation.py:153` (20); `collect_nof_aligned_events.py:178` (19.76). Конвейер берёт FPS из таймстемпов (`bearmind/timestamps.py:58`, ff66233) | `осталось` | 4: в `ae_launch` нет FPS в таблице → ошибка, не 30. Остальные умолчания — библиотечные, вне конвейера |
| А11 | `ml_threshold` 0.5 / 0.71 / 0.72 | источник `defaults.yaml` (0.72); `ae_launch.py:211` и CLI 672 (0.71); `export_estimates_data.py:35` `DEFAULT_ML_THRESHOLD` (0.72, запас для файлов без `autoinspection_config`; берёт этап export, `bearmind/stages/export.py:30`); `auto_inspector.py:1569, 1637, 1704` (0.5); `bm_examinator.py:80, 412, 758` (0.5, раскраска вида capcan) | `осталось` | 4 |
| А12 | Модель v6 / v9 | `ae_launch.py:210`, CLI 670: `ml/production_models/ebm_v6_no3dm.pkl`; `defaults.yaml`: `production_models/ebm_v9_iter8.pkl` | `осталось` | 4 |
| А13 | `brain` ml / thresholds / hybrid | `ae_launch.py:209`, CLI 668 (`ml`); `auto_inspector.py:1702` (`thresholds`); `defaults.yaml` (`hybrid`) | `осталось` | 4 |
| А14 | `n_iter` 2 / 3 | `ae_launch.py:194`, CLI 674 (2); `auto_inspector.py:134, 403, 461, 1035` (2); `bm_examinator.py:414, 758` (2); `capcan_validation.py:134, 313` (2); `defaults.yaml` (3) | `осталось` | 4 |
| А15 | `detect_corner_artifacts` | `ae_launch.py:190` (True); `bm_examinator.py:358, 413, 758` (True); `defaults.yaml` (false) | `осталось` | 4 |
| А16 | `corr_thr` | `ae_launch.py:182` и `auto_inspector.py:1032` (0.4 = `defaults.yaml`); `bm_examinator.py:356, 409, 517, 757` (0.6, вид ручной инспекции) | `осталось` | 4 (вид — см. Н30) |
| А17 | `cthr` | `ae_launch.py:181` (0.35 = `defaults.yaml`); `auto_inspector.py:1031, 2125` (0.3); `polygon.py:7` (0.3); `bm_examinator.py:324, 349, 356, 408, 517, 757` (0.3) | `осталось` | 4 (вид — см. Н29) |
| А18 | Метод корреляции | `ae_launch.py:193`, `auto_inspector.py:983, 1035` (spearman = `defaults.yaml`); `bm_examinator.py:358, 414, 517, 758` (pearson, вид) | `осталось` | 4 |
| А19 | `pxlthr_distance_boundary` и `d_snr_thr` в 5–7 местах | `defaults.yaml`; `ae_launch.py:204-205`; `auto_inspector.py:1698-1699, 1840`; `capcan_validation.py:223-224, 398-399`; `batch_autoinspect_all.py:44-45` (history); `test_export_e2e.py:287-288` (удаляется) | `осталось` | 4 (порог 15 не менять) |
| А20 | Абсолютные пути | `config.py:8-9`; `collect_nof_aligned_events.py:30-31, 174` (`D:/_Projects`, прямая косая); `experiments/testing/test_autoinspection_timing.py:25`, `experiments/testing/test_fix_verification.py:150`; `ml/train_logistic_lasso_capcan.py:250`, `ml/tune_logistic_lasso_capcan.py:333`; в history: `batch_autoinspect_all.py:60, 64, 68`, `resplit_fof.py:21-26`; в мёртвом блоке: `bm_examinator.py:2351` | `осталось` | 4 (П-9) |
| А21 | `n_processes=14` | `bm_batch_routines.py:408` (`DoMotionCorrection`) | `уходит в history` | 3 |
| А22 | `max_mem_gb=16` | только ячейки 15, 26, 28. Пакет считает бюджет сам: `bearmind/stages/cnmf.py` `mem_budget_gb` | `уходит с ноутбуком` | 5 |
| А23 | Путь к модели от текущей папки | Этап: от корня репозитория (`bearmind/stages/autoinspect.py:12, 51`, 143c78f). Осталось: `ae_launch.py:210, 670` (`ml/production_models/...` от cwd); 30+ скриптов `experiments/` открывают `production_models/...` от cwd | `осталось` | 4 (`ae_launch` — путь от файла модуля; `experiments/` не трогать) |
| А24 | Порт Bokeh | `startup.py:29-30` (`origin[7:-1]`), ячейка 31 (`localhost:8888`). Сервер берёт свободный порт (`defaults.yaml` `inspection.port: 0`) | `уходит с ноутбуком` | 5 |
| А25 | Срезы имён файлов | Вне history: `capcan_validation.py:54, 80` (`[:10]`); `collect_nof_aligned_events.py:93, 197-198` (`[:10]`); `config.py:98, 100` (`splt_path[-2]`, `[-5]`); `bm_examinator.py` `make_app` → `bkapp` 1147 (`rpartition('/')`), `SaveResults` 611, 615 (`partition('estimates')`), `ManualSeeds` → `bkapp` 2229. В history: `bm_batch_routines.py:232, 263-266, 388-390, 651`. Ячейки 12, 19, 26, 46 — с ноутбуком | `осталось` | 4 (П-10) |
| А26 | Обрезка 8 px и отброс кадра при 608×608 | `bm_batch_routines.py:291-317` (`DoCropAndRewrite`) | `уходит в history` | 3 |
| А27 | Списки наборов CC / RFC / FOF | `batch_autoinspect_all.py:58-71`, `batch_export_all.py:21-25`, `chain_run_export_after_autoinspect.py:29, 99`, `resplit_fof.py:28`; ячейка 24 | `уходит в history` | 3 |
| А28 | Широкие `except` с молчаливым продолжением | `auto_inspector.py:232, 245, 265, 286` (каскад кинетики), 725, 798, 880, 1111; `bm_spike_detection.py:45, 49`, `estimates_compression.py:256`, `table_routines.py:27` (голые); `table_routines.py:81-83` (`calc_fps` → `None`); `ae_launch.py:73`, `export_estimates_data.py:59` (ошибка чтения таблицы → умолчание) | `осталось` | 4 (П-11: в `auto_inspector` только запись в лог) |
| А29 | `sys.modules['torch'] = None` | `collect_nof_aligned_events.py:46` | `осталось` | 2 |
| А30 | `SSQ_GPU` ставится после импорта driada | `wavelet_backend.py:117` ставит переменную при вызове (`auto_inspector.py:473`). driada 1.2.2 импортирует ssqueezepy лениво (`driada/experiment/wavelet_event_detection.py:787`), ssqueezepy читает переменную при вызове (`ssqueezepy/configs.py:142` `USE_GPU`). Вывод по коду, не замер | `не подтвердилось` | — |
| А31 | `config.json` при импорте | `config.py:153-156` | `осталось` | 2 |
| А32 | Окно Tk при импорте | `startup.py:17-20` | `удаляется` | 5 |
| А33 | `ae_launch` тянет весь GUI ради `LoadEstimates` | `ae_launch.py:26` `from bm_examinator import LoadEstimates` | `осталось` | план не назначил (П-12) |
| А34 | `np.Inf` | в `*.py` нет ни одного (`np.Inf`, `np.NaN`, `np.float_` и др.) | `исправлено в e64d693` | — |
| А35 | Экспорт ×4 | ячейки 41, 42 (с ноутбуком); `batch_export.py` (history); CLI `export_estimates_data.py:480-585` `main`; этап `bearmind/stages/export.py`. Правило одно: `approved_components` (`export_estimates_data.py:222`) | `удаляется` | 4 (CLI, решение О5) |
| А36 | Цикл автоинспекции ×3 | ячейка 37 (с ноутбуком); `batch_autoinspect_all.py` (history); `batch_autoinspect.py` (удаляется, А43). Остаётся этап `bearmind/stages/autoinspect.py` | `уходит в history` | 3 |
| А37 | Набор параметров автоинспекции ×4 | `defaults.yaml` (источник); `ae_launch.py:172-227` (умолчания расходятся); `capcan_validation.py:182-226, 371-401` (свои правила валидации); `bm_examinator.py:758` (`_FIRST_VIEW_DEFAULTS`, вид capcan); `batch_autoinspect_all.py:20-46` (history); `test_export_e2e.py:256-290` (удаляется); ячейки 36, 39 | `осталось` | 4 (П-7) |
| А38 | Чтение таблицы FPS ×4 | `ae_launch.py:33-85` (округляет, умолчание 30); `export_estimates_data.py:48-77, 106-114` (точно); `capcan_validation.py:38-58` (округляет, ключ `[:10]`); `collect_nof_aligned_events.py:73-95` (точно) | `осталось` | 4 (П-8) |
| А39 | PNR → imax ×3 | `bearmind/stages/cnmf.py:171-193` `_imax` (`IMAX_PNR_CLIP = 70`, строка 36); `bm_batch_routines.py:551-558` (`DoCNMF`), 644-648 (`ReDoCNMF`) → history; `bm_examinator.py:2250-2258` `build_average_image` (вызов только из мёртвого блока) | `осталось` | 4 |
| А40 | Блок-строка в `bm_examinator.py` | сейчас 2303-2360 (`split_estimate`, `merge_estimates`, хвост с абсолютным путём). Ещё два мёртвых блока: `LoadEstimates` 130-133, `make_app` → `bkapp` 1158-1169 | `удаляется` | 3 |
| А41 | Код после `return` | `bm_batch_routines.py:586-590` (`DoCNMF`), 106-111 (`DrawCropper`) | `удаляется` | 3 |
| А42 | Скрытые кнопки кропа | `bm_batch_routines.py:123-205` (`DrawCropper`: три обработчика, `vbox1`-`vbox3`, закомментированный `display` в строке 205) | `удаляется` | 3 |
| А43 | Устаревший `batch_autoinspect.py` | его никто не импортирует | `удаляется` | 3 |
| А44 | Отслеживаемые `.DS_Store`, `.ipynb_checkpoints` | `.DS_Store`, `.ipynb_checkpoints/MoviePlay-checkpoint.ipynb`. В `.gitignore` они уже есть (строки 53-54, 77, 157) | `удаляется` | 5 |
| А45 | Скрипты MINI3 → история | `batch_autoinspect_all.py`, `batch_export_all.py`, `chain_run_export_after_autoinspect.py`, `resplit_fof.py` + `batch_export.py` | `уходит в history` | 3 (П-6) |
| А46 | Ноутбуки (§5.16) | `BEARMiND_full_pipeline.ipynb`, `MoviePlay.ipynb`. Значения ноутбука — таблица 5 | `удаляется` | 5 (после решения по Н2) |

## 3. Разведка: строки, которых нет в приложении А

Пункты разведки, которые повторяют приложение А, в таблицу не входят. Соответствие: стадии 8–9 → А6, А35; места касания 2, 3, 12 → А9, А42, А24; хардкоды (пути, FPS, ML, capcan, срезы, наборы, PNR) → А10–А27, А39; баги и костыли §4 → А2, А3, А7–А9, А28–А34; копии → А35–А39; мёртвое → А40–А44.

| № | Строка разведки | Где сейчас (`53dece2`) | Статус | Задача |
|---|---|---|---|---|
| Р1 | §1 этап 0: `startup` в профиле IPython, `update_config`, `config.json` в cwd, `MOUSE/SESSION_CONFIGS` | слои конфигов `bearmind/config.py`, раскладка `bearmind/paths.py`. Остаток — `config.py` (А7, А8, А31) | `исправлено в 6538d3e` | — |
| Р2 | §1 этап 1: кроп ползунками, `*_cropping.pickle` | автокроп `bearmind/crop.py`, `bearmind/stages/crop.py` | `исправлено в 9107826` | — |
| Р3 | §1 этап 2: `DoCropAndRewrite`, `extract_and_copy_ts` | чтение сессии `bearmind/video.py` `SessionReader`, таймстемпы `bearmind/timestamps.py` | `исправлено в ff66233` | — (старое — А26, О18) |
| Р4 | §1 этап 3: MC в TIF и mp4 | `bearmind/stages/mc.py`, FFV1 `3_MC` | `исправлено в 8d1393f` | — |
| Р5 | §1 этап 4 и место 6: пороги на глаз, массивы в ячейке 24 | `bearmind/stages/thresholds.py` (fec71a0), вкладка Thresholds (e108463) | `исправлено в fec71a0` | — |
| Р6 | §1 этап 5: CNMF-E по диалогу, FPS через `glob(name[:-10])`, конфиг по `name[-20:-10]`, абсолютный `tif_name` | `bearmind/stages/cnmf.py`; `est.mc_video` относительный (строка 266) | `исправлено в 7cca587` | — |
| Р7 | §1 этап 6: сжатие с `input()` | ячейка 33; в конвейере сжатие делает автоинспекция (умолчание `compress_estimates=True`) | `уходит с ноутбуком` | 5 |
| Р8 | §1 этап 7: пять путей автоинспекции, `artifacts_path='./output'` от cwd | этап `bearmind/stages/autoinspect.py` пишет в `paths.inspection_dir(sid)` | `исправлено в 143c78f` | — |
| Р9 | §1 этап 10: `resplit_fof.py` удаляет без подтверждения | `resplit_fof.py` | `уходит в history` | 3 (в README: не запускать на оригиналах) |
| Р10 | §1 этап 11: события (FitEvents, вейвлет-ячейки, сборщики) | `bm_spike_detection.py` (P6-2, О8); ячейки 47-56 — с ноутбуком; сборщики `collect_nof_aligned_events.py`, `collect_wavelet_event_stats.py`, `summarize_nof_events.py` остаются в корне как исследовательские | `остаётся намеренно` | — (их импорт — П-3) |
| Р11 | §1: нет манифеста и статуса этапов | реестр `bearmind/registry.py` (7cdc009), планировщик `bearmind/runner.py` (3cb7e6d) | `исправлено в 7cdc009` | — |
| Р12 | §2 место 1: `ROOT`/`DATA_PATHWAY` правкой ячейки | CLI и окно; раскладка определяется сканом | `исправлено в bf753a9` | — |
| Р13 | §2 место 4: битые AVI чинят ffmpeg вручную (ячейка 10) | скан пишет `broken_last_frame` (`bearmind/scan.py:191`), чтение без битого хвоста (`bearmind/video.py:43-53`) | `исправлено в bf753a9` | — |
| Р14 | §2 место 5: диалоги в 13 ячейках, `input()` | только ноутбук | `уходит с ноутбуком` | 5 |
| Р15 | §2 место 7: `mc_params` одной сессии (ячейка 20) | раздел `mc` в `defaults.yaml` | `исправлено в 6538d3e` | — |
| Р16 | §2 место 8: курация в Bokeh, `feedback.csv` | сервер инспекции; путь feedback отменён автором; кнопки FP/FN и Save Feedback скрыты в режиме сервера (b77a1a4) | `исправлено в 74b9d0a` | — |
| Р17 | §2 место 9: `residual_excess_outside` никто не читает | этап cnmf читает его (`bearmind/stages/cnmf.py:274`) → метрика `missed_cells`, проверка `cnmf_missed_cells` (`bearmind/checks.py:68`). `ManualSeeds` остаётся (`bm_examinator.py` строка 2198, тест), `ReDoCNMF` — А3 | `исправлено в 7cca587` | — |
| Р18 | §2 место 10: `fps_data.csv` руками; `get_fps_from_timestamps` не используется | FPS конвейера из таймстемпов. `get_fps_from_timestamps` (`bm_examinator.py` строка 175) вызывали ячейки 19, 26, 28 | `исправлено в ff66233` | — (остаток — А10, О19) |
| Р19 | §2 место 11: цепочка опрашивает лог, который скрипт не пишет | `chain_run_export_after_autoinspect.py:22-24` | `уходит в history` | 3 |
| Р20 | §2 место 13: параметры вейвлета правкой ячеек | ячейки 53-56; автоинспекция берёт умолчания driada | `уходит с ноутбуком` | 5 |
| Р21 | §3: `utils.py:21` Europe/Moscow | `utils.get_datetime` — только имя файла при сохранении без хуков (`bm_examinator.py` `make_app` → `bkapp` → `save_callback`, 1774) | `осталось` | — (вне конвейера) |
| Р22 | §3: `memory_fact=RAM/16` | выброшен из `mc` (комментарий в `defaults.yaml`); MotionCorrect его не использует | `исправлено в 6538d3e` | — |
| Р23 | §3: округление FPS из таблицы в `ae_launch` и `capcan_validation` | `ae_launch.py:78, 83`, `capcan_validation.py:55` — округление для автоинспекции и валидации (§3 дизайна) | `остаётся намеренно` | — |
| Р24 | §3: `match_threshold` 3 против 1 | `ae_launch.py:184`, `auto_inspector.py:1032`, `bm_examinator.py:409` (3); `bm_examinator.py:758`, `batch_autoinspect_all.py:27` (1). При `num_sessions=1` не влияет | `остаётся намеренно` | — (не менять: П-7) |
| Р25 | §3: полный `cnmf_dict` только в ячейке 24; `mc_dict` дважды; `images[::5]` | разделы `cnmf` и `mc` в `defaults.yaml`; `images[::5]` в конвейере нет (`bearmind/stages/cnmf.py:186`) | `исправлено в 6538d3e` | — |
| Р26 | §3: `maxframes` 20000 / 12000 / 10000 / 2000 | ячейки 12, 13, 23; функции `plot_*` в `bm_batch_routines.py` | `уходит с ноутбуком` | 5 |
| Р27 | §4: `get_timestamps` ищет файл от cwd | сервер передаёт `time=` из реестра (`bm_examinator.py` `LoadEstimates`, строки 122-145). Без сервера — `get_timestamps` (149) | `исправлено в 8f70a10` | — |
| Р28 | §4: `ROOT+'*traces.csv'` без разделителя | ячейки 48, 50 | `уходит с ноутбуком` | 5 |
| Р29 | §4: `'dview' in locals()` всегда False | `bm_batch_routines.py:406, 490`; в пакете такого нет | `уходит в history` | 3 |
| Р30 | §4: memmap `PermissionError` → 100 × 1 с | ячейка 26; пакет повторяет запись ограниченно (`bearmind/fileio.py`) | `уходит с ноутбуком` | 5 |
| Р31 | §4: предупреждения выключены при импорте | `bm_batch_routines.py:36` `warnings.filterwarnings('ignore')` — через `bm_examinator` действует в этапе autoinspect и в сервере; `test_env.py:3` | `осталось` | 2 (П-3) |
| Р32 | §4: `output_notebook()` при импорте `bm_examinator` | теперь только внутри `ExamineCells` (строка 2194) и `ManualSeeds` (2246) | `исправлено в 8f70a10` | — |
| Р33 | §4: `capcan_validation` грузит таблицу при импорте | `capcan_validation.py:62-63` (таблица и печать), 91-96 (папки `data/`) — только чтение | `осталось` | — (не мешает тесту задачи 2) |
| Р34 | §4: `auto_inspector` зависит от `ml.data_utils` | `auto_inspector.py:40` — список признаков модели | `остаётся намеренно` | — |
| Р35 | §4: конфликт имён `get_fps` | `table_routines.py:86` `get_fps(df)` и `export_estimates_data.py:106` `get_fps(session_name)`; `bm_examinator.py:53` `from table_routines import *` | `осталось` | 4 (при сведении читалок FPS) |
| Р36 | §4: две схемы папок артефактов | конвейер: `inspection_artifacts_` (`ae_launch.py:140`); `capcan_artifacts_` — валидация (`ae_utils.py:16-28`, `capcan_validation.py:423`) и данные обучения в `ml/` | `остаётся намеренно` | — |
| Р37 | §4: обработанный файл по stem, папка по sid | этап передаёт `session_name=sid` (`bearmind/stages/autoinspect.py:61`); файл `<sid>_processed.pickle` | `исправлено в 143c78f` | — |
| Р38 | §4: блок-строка `nb24.1-5` | ячейка 24 | `уходит с ноутбуком` | 5 |
| Р39 | §5: источники конфигурации вразнобой | слои `defaults.yaml` < серия < мышь < сессия; остаток — `config.py` и умолчания `ae_launch` | `исправлено в 6538d3e` | — |
| Р40 | §5: не записывались параметры MC, выбор значений CNMF, список AVI, путь входа, версия кода, хэш модели, источник FPS, правки GUI, решения FOF | `StageRecord.params` и provenance (`bearmind/provenance.py`), решения в реестре, `fps` и `autoinspection_fps` в метаданных | `исправлено в 8d2b990` | — |
| Р41 | §5: всё ещё не записываются `match_threshold`, `num_sessions`, `sf`/`ef`/`ds`, `comps_to_select`, раскрытые правила удаления, фактический бэкенд вейвлета | `ae_launch.py:545-566` `autoinspection_config` пишет `'DEFAULT_DELETION_RULES'` строкой и `wavelet_backend: 'auto'` | `осталось` | — (формат не меняется в фазе 6; П-7) |

Возможности §6 разведки 1–9 реализованы фазами 1–5 (реестр, CLI и планировщик, имена и пути, `defaults.yaml`, автоподбор порогов, автокроп, точный FPS, provenance, ядро без GUI). Возможность 10 реализована частично: ворота ручной инспекции и подсказка волны (фаза 5), проверка `missed_cells`. Очередь по полосам ML не делалась: ручная инспекция идёт до автоинспекции (решение автора).

## 4. Отложенные пункты

| № | Пункт | Где сейчас (`53dece2`) | Статус | Задача |
|---|---|---|---|---|
| О1 | `test_export_e2e.py:133` повторяет правило `>=` (A1, задача 3) | корень; `pytest tests` его не собирает (`pytest.ini` без `testpaths`: `pytest` из корня собрал бы). Покрытие: (1) `metrics_df` после сжатия — `tests/test_compression_transforms.py:60, 88`; (2) `asp_cache` и `reconstructions` после сжатия — `tests/test_compression_transforms.py:88`, `tests/test_compression_values.py:46`; (3) метаданные под выгруженные трейсы — `tests/test_export_merged.py:75`, но на сессии `output/inspection_artifacts_WAW_H02_1D` (в `.gitignore`; на этой машине папки `output/` нет — тест пропускается). Проверка «нет нулевых строк ASP» (`test_export_e2e.py:154-156`) не покрыта | `удаляется` | 5 (П-13: сначала синтетический тест на `approved_components`) |
| О2 | Ячейки экспорта ноутбука без защиты от старых файлов (A1) | ячейки 41, 42: `ml_approved_components` с `>=`, без `approved_components` и без проверки `merge_grouping` | `уходит с ноутбуком` | 5 |
| О3 | `startup.py` | импортирует только ноутбук (ячейка 1 копирует его в профиль IPython). При импорте: окно Tk (17-20), `BOKEH_ALLOW_WS_ORIGIN` (29-30), `jupyter_server` | `удаляется` | 5 |
| О4 | Копия `startup.py` вне репозитория | `C:\Users\User\.ipython\profile_default\startup\startup.py` (2025-09-30, старая версия: `folder_structure`, импорт `wavelet_event_detection`). Работает при каждом запуске IPython | `осталось` | 6 (П-14: строка в инструкции; удалять только с согласия автора) |
| О5 | Командная строка `export_estimates_data.py` | `main` 480-585, `import argparse` 21. Решение: удалить `main`, `argparse` и `if __name__`. Функции остаются: их импортируют `bearmind/stages/export.py:11`, `bearmind/runner.py:46`, `ae_launch.py:96`, `batch_export.py:33` (history), тесты. Причины: FPS только из `fps_data.csv` (новых сессий там нет — `main` падает); флаг `--incorporate-feedback` для отменённого пути feedback; конвейер экспортирует этапом | `удаляется` | 4 |
| О6 | Импорты `batch_export.py` | `batch_export_all.py:17`; `tests/bearmind/test_equivalence.py:200`; `tests/bearmind/test_stage_export.py:339, 385` | `уходит в history` | 3 |
| О7 | Импорты `batch_autoinspect.py` | нет. Следствие: `ae_utils.save_auto_inspection_outputs` теряет единственного вызывающего | `удаляется` | 3 |
| О8 | `bm_spike_detection.py` (P6-2) | `FitEvents`, `DrawSpEvents`, `DrawSpEvents2`; `output_notebook()` внутри функций (строки 20, 75, 90); голые `except` 45, 49; импорт устаревшего `scipy.ndimage.filters` (строка 2) | `остаётся намеренно` | — (инструкция о нём не пишет) |
| О9 | `os.chdir(REPO_ROOT)` (фаза 2 и план) | `bearmind/cli.py:140` (`cmd_gui`), 752 (`main`), **`bearmind/runner.py:1583` (`run`)** | `осталось` | 2 (П-1) |
| О10 | `.gitignore`: `env/` скрывает новые файлы `env/` (фаза 0) | строка 65 `env/`, 67 `ENV/`; нужно `!env/`. `.DS_Store`, `.ipynb_checkpoints`, `temp/`, `__pycache__/` уже есть | `осталось` | 5 |
| О11 | Lock: строки `pypi_0` не восстанавливаются `conda create --file` (фаза 0) | `env/bearmind-lock.txt`: 70 строк `pypi_0` (driada, scipy, interpret и др.); заметки в шапке нет | `осталось` | 7 |
| О12 | `python=3.12.16` — тесная привязка (фаза 0) | `environment.yml` | `осталось` | 7 |
| О13 | `OPENBLAS_CORETYPE` | комментарий `environment.yml` говорит, что её ставит переменная окружения conda. В файле нет `variables:`. Переменная есть только в окружении этой машины (`envs/bearmind/conda-meta/state`) | `осталось` | 7 (П-15) |
| О14 | Установщики | `install1_conda.bat` (годен: `env create -f environment.yml`); `install2_pip.bat:86` запускает `test_env.py`; `test_env.py:17` требует `tensorflow` (в `bearmind` его нет → FAILED); `bearmind2025.yml` — экспорт старого окружения `caiman` (py310, tensorflow), его ничто не использует | `осталось` | 7 (П-16) |
| О15 | Корневой `README.md` | установка через окружение CaImAn и список pip (строки 21-46), работа через ноутбук (48-100) | `осталось` | 6 (П-17) |
| О16 | Корневые тестовые скрипты | `test_event_stats.py` (бэкенд вейвлета при импорте, строка 14; нужна реальная сессия), `test_wavelet_backend_integration.py` (весь код на уровне модуля) | `осталось` | 5 (П-3: `git mv` в `experiments/testing/`) |
| О17 | `collect_nof_aligned_events.py` строит `Neuron` без seed (A1, задача 4) | `collect_nof_aligned_events.py:152` | `осталось` | — (исследовательский скрипт) |
| О18 | Функции ноутбука в `bm_batch_routines.py` без вызывающих после задачи 5 | `CleanMemmaps`, `DrawFrameAndBox`, `LoadSelectedVideos`, `DrawCropper`, `SaveCrops`, `get_file_num_id`, `get_bit_depth_tif`, `extract_number`, `DoCropAndRewriteTiff`, `extract_and_copy_ts`, `FindMaxima`, `Test_gSig_Range`, `plot_gsig_range`, `plot_min_corr_and_pnr_range`, `crop_tiff`, `convert_tiff_to_mp4` | `уходит в history` | 3 (П-4) |
| О19 | Функции ноутбука в `bm_examinator.py` | `ExamineCells` (2184) и `ManualSeeds` (2198) вызывают тесты; `get_fps_from_timestamps` (175), `build_average_image` (2250), `test_min_corr_and_pnr` (2262) без вызывающих | `остаётся намеренно` | — (как P6-2; можно удалить позже) |
| О20 | Корневые модули с побочными эффектами импорта кроме `config.py` | `collect_wavelet_event_stats.py:35` и `collect_nof_aligned_events.py:62` — `set_wavelet_backend` (переменные `SSQ_*`); `summarize_nof_events.py:14` — `matplotlib.use("Agg")` | `осталось` | 2 (П-3) |
| О21 | Мёртвые блоки `bm_examinator.py` кроме хвоста | `LoadEstimates` 130-133, `make_app` → `bkapp` 1158-1169 (там же `608` и голый `except`) | `удаляется` | 3 |

## 5. Ноутбук против `bearmind/defaults.yaml` (шаг 4)

Значения ноутбука взяты из `HEAD` и рабочей копии (тексты ячеек совпадают). «Решение» — что делает задача 5.

### `cnmf_dict` (ячейка 24; ячейка 26 переписывает `fr` и `tsub`)

| № | Ключ | Ноутбук | `defaults.yaml` / код | Решение |
|---|---|---|---|---|
| Н1 | `fr` | 30, затем `fps/2` | FPS сессии (`cnmf_params`, `bearmind/stages/cnmf.py:108`) | известное намеренное расхождение (§3 дизайна) |
| Н2 | `tsub` | 1 в ячейке 24, **2** в ячейке 26 (так считались данные статьи) | 1 (взято из ячейки 24 планом фазы 1) | **решено автором 2026-11-09: оставить 1, намеренно; данные статьи считались с `tsub` = 2.** `cnmf.tsub` входит в хэш этапа thresholds и в карты PNR (`bm_qc.compute_init_maps` делит PNR на √tsub). Смена на 2 пересчитает пороги и CNMF-E всех сессий. Рекомендация: оставить 1 и записать как намеренное, как `fr`. Решает автор |
| Н3 | `gSig`, `gSiz`, `min_corr`, `min_pnr` | массивы по сессиям в ячейке 24; `gSiz = 4*gSig+1` | конфиг сессии из вкладки Thresholds; `gSiz = 4*gSig+1` (`cnmf.py:107`) | то же правило; источник заменён (фаза 4) |
| Н4 | 23 ключа: `decay_time` 1, `method_init` corr_pnr, `K` None, `merge_thr` 0.8, `p` 1, `ssub` 1, `rf` 40, `stride` 25, `only_init` True, `nb` 0, `nb_patch` 0, `method_deconvolution` oasis, `low_rank_background` None, `update_background_components` True, `normalize_init` False, `center_psf` True, `ssub_B` 2, `ring_size_factor` 1.5, `del_duplicates` True, `border_pix` 5, `min_SNR` 2.5, `rval_thr` 0.95, `use_cnn` False | как слева | как слева | совпадают. `ssub_B` батч поднимает до 3–4 сам, если сессия не влезает в RAM (аномалия `cnmf_downsampled`, волна A2) |
| Н5 | `noise_ampl` (аргумент `DoCNMF`) | 1e-5 | `ADD_TO_MOVIE = 1e-4` (`cnmf.py:35`, умолчание `save_memmap` CaImAn 1.13.2) | оставить 1e-4. Сдвиг 9e-5 на кадре uint8 (0–255) пренебрежим. В CaImAn 1.11.3 ноутбук фактически прибавлял 1e-4 + 1e-5 |
| Н6 | Пул `DoCNMF` | `n_processes=None` (все ядра) | `patch_workers` по бюджету RAM; размер пула пишется в `est.patch_workers` | намеренно (память; пул меняет результат — записывается) |
| Н7 | QC | `qc=True`, видео QC всегда | карты остатка и `missed_cells` всегда; видео QC только с `preview.enabled` | намеренно (решение автора: превью по галочке) |
| Н8 | `imax` | `correlation_pnr(images[::5])`, обрезка 70 | PNR из карт порогов, обрезка 70 (`_imax`) | только картинка фона ручной инспекции; намеренно (память) |

### `mc_dict` (ячейки 12 и 17)

| № | Ключ | Ноутбук | `defaults.yaml` / код | Решение |
|---|---|---|---|---|
| Н9 | 8 ключей: `pw_rigid` False, `max_shifts` (35, 35), `gSig_filt` (8, 8), `strides` (48, 48), `overlaps` (24, 24), `max_deviation_rigid` 15, `border_nan` copy, `niter_rig` 1 | как слева | как слева | совпадают |
| Н10 | `use_cuda` | True | false | без влияния: CaImAn 1.13.2 его игнорирует (`caiman/motion_correction.py:204-205`, «cuda is no longer supported») |
| Н11 | `memory_fact` | `CONFIG['RAM']/16` | выброшен | без влияния: MotionCorrect его не использует (комментарий раздела `mc`) |
| Н12 | `splits_rig` | 20 | `ceil(T/500)` (`bearmind/stages/mc.py:161`) | намеренно (память; план фазы 1, задача 0.3) |
| Н13 | `num_splits_to_process_rig` | None | не хранится; умолчание CaImAn None | совпадает |
| Н14 | `write_mp4` | True (ячейка 12); нет в ячейке 17 (KeyError) | превью MC по `preview.enabled` (false) | намеренно (решение автора 2026-11-06) |
| Н15 | `n_processes` `DoMotionCorrection` | 14 | `pool.mc_processes: 16` (не параметр этапа) | сдвиги при пулах 31/16/8 одинаковы (комментарий `pool` в `defaults.yaml`) |

### Автоинспекция (ячейка 36)

| № | Ключ | Ноутбук | `defaults.yaml` / код | Решение |
|---|---|---|---|---|
| Н16 | 15 ключей: `brain` hybrid, модель `ebm_v9_iter8.pkl`, `ml_threshold` 0.72, `n_iter` 3, `detect_corner_artifacts` False, `correlation_method` spearman, `cthr` 0.35, `corr_thr` 0.4, `pxlthr_distance_boundary` 15, `d_snr_thr` 10, `enable_merge` True, `include_event_based` True, `include_heavy` True, `event_method` wavelet, `hybrid_kinetics` True | как слева | как слева | совпадают |
| Н17 | `seed` | не передаётся (умолчание `ae_launch` 0) | `seed: 0` | совпадает |
| Н18 | `fps` | None → таблица, округление | `round(rec.fps)` из таймстемпов (`autoinspect.py:54`) | намеренно (§3: источник — таймстемпы, округление сохранено) |
| Н19 | `ml_model_path` | от текущей папки | от корня репозитория (`autoinspect.py:51`) | исправлено (143c78f) |
| Н20 | `artifacts_path` | `./output` | `artifacts_dir = paths.inspection_dir(sid)` | исправлено (143c78f) |
| Н21 | `session_name` | None (stem файла) | `sid` | то же действие |
| Н22 | 14 ключей не из `defaults.yaml`: `comps_to_select` None, `num_sessions` 1, `match_threshold` 3, `sf` None, `ef` None, `ds` 1, `corner_artifact_params` None, `wavelet_backend` auto, `deletion_rules` None, `track_criteria_failures` True, `save_artifacts`, `save_estimates`, `save_matrices`, `save_corner_detection` True, `compress_estimates` True | как слева | этап их не передаёт; берёт умолчания `ae_launch.run_auto_inspection` (`ae_launch.py:172-227`), они равны ноутбуку | совпадают сейчас. **Риск:** их нет в хэше этапа. Задача 4 не должна менять эти умолчания (П-7) |

### Ручная инспекция (ячейка 39, `bkapp_kwargs`, `default_fps=30`)

Сервер строит окно с `_kwargs_of` (`bearmind/inspect_server.py`, строка 878): `mode` legacy, `end_frame` None, `enable_gpu_backend` из `inspection.webgl`, `trace_points`. Остальное — `_FIRST_VIEW_DEFAULTS` (`bm_examinator.py:756-762`).

| № | Ключ | Ноутбук | Сейчас | Решение |
|---|---|---|---|---|
| Н23 | `mode` | capcan | legacy | намеренно: ручная инспекция до автоинспекции (§5.9) |
| Н24 | `end_frame` | 90000 | None (вся длина) | исправлено (8f70a10) |
| Н25 | `start_frame` | 0 | 0 | совпадает |
| Н26 | `downsampling` | 5 | 1; точки трейса прореживает `trace_points: 2000` | намеренно (фаза 5) |
| Н27 | `enable_gpu_backend` | 1 | `inspection.webgl: true` | совпадает |
| Н28 | `default_fps` | 30 | `DEFAULT_FPS = 20` (`inspect_server.py:116`); время из реестра | не влияет: первый вид его не использует |
| Н29 | `cthr` | 0.35 | 0.3 | только контуры на экране. Оставить 0.3. Если автор хочет как в ноутбуке — один ключ в `_kwargs_of` |
| Н30 | `corr_thr` | 0.35 | 0.6 | только подсветка коррелированных соседей. Оставить (как Н29) |
| Н31 | `correlation_method` | spearman | pearson | только вид. Оставить (как Н29) |
| Н32 | `color_by_ml_probability`, `ml_threshold`, `ml_model_path`, `n_iter`, `include_*`, `event_method`, `detect_corner_artifacts`, `num_sessions`, `match_threshold` 1, `corner_artifact_params` | заданы | в режиме legacy не используются | не применимо |
| Н33 | `fill_alpha` 0.8, `ns_alpha` 0.2, `line_width` 0.5, `line_alpha` 0.5, `trace_line_width` 1, `trace_alpha` 0.7, `size` 450, `metrics_width` 200, `button_width` 50, `sort_order` up, `verbose` 0, `oh_shit` 0 | заданы | умолчания окна (`bm_examinator.py` `make_app` → `bkapp`, 1049-1071) | только вид; выбор фазы 5 |
| Н34 | `compress_estimates` | True | результат пишется без сжатия (`write_result`, фаза 5) | намеренно (фаза 5) |

### Прочие значения ноутбука (не параметры этапов)

| № | Что | Ноутбук | Сейчас | Решение |
|---|---|---|---|---|
| Н35 | Сетки порогов | ячейки 12, 13, 23: gSig 3–6, PNR и min_corr списками, `maxframes` | вкладка Thresholds: `thresholds.gsig_range` [3..7], ползунки, `fallback` 0.85 / 8 | заменено (фаза 4) |
| Н36 | `batch_suggest_thresholds` | ячейка 15: `gsig_range=[4]`, `tsub=2`, `target_fdr=0.1` | этап thresholds: `gsig_range` [3..7], `tsub = cnmf.tsub` (1), `target_fdr` 0.1 (умолчание `bm_qc.py:279`) | `tsub` — решено вместе с Н2 (оставить 1) |
| Н37 | Видео dF/F | ячейка 19: окно F0 60 с, 10-й перцентиль, `tbin` 2, `speed` 2 | этап previews: умолчания `bm_qc.make_dff_video`, `preview.tbin` 2, `preview.speed` 4 | только просмотр |
| Н38 | Видео QC | ячейка 28: `video_tbin` 2, `video_speed` 2, `gain_res` 3 | этап previews: `video_tbin = preview.tbin` | только просмотр |
| Н39 | Сжатие | ячейка 33: `remove_bad_components=True` | не этап конвейера | уходит с ноутбуком |
| Н40 | События | ячейки 48, 53: `sd_dict`, `wvt_param_dict` | не этап (события — не цель проекта) | уходит с ноутбуком |
| Н41 | Порт Bokeh | ячейка 31: `localhost:8888` | `inspection.port: 0` | заменено (А24) |

**Вывод для задачи 5, шаг 2.** Все значения ноутбука либо есть в `defaults.yaml`, либо заменены решением из этой таблицы. Н2 (`tsub`) решён автором 2026-11-09: оставить 1. Если решение «оставить 1», задача 5 не дописывает значения в `defaults.yaml`.

## 6. Импорты старых имён (шаг 5)

Старые имена — то, что задачи 3–5 переносят или удаляют. Поиск: `git grep` по `*.py` на `53dece2`, включая `paper/`, `experiments/`, `ml/`, `tests/` и корень.

| Имя | Кто импортирует | Задача |
|---|---|---|
| `bm_batch_routines` (`DoMotionCorrection`, `DoCNMF`, `MotionCorrect`) | `tests/bearmind/equivalence_runs.py:467, 525` (`import bm_batch_routines as old`; строка 471 подменяет `old.MotionCorrect`, 523 — `caiman.cluster.setup_cluster`) | 3 |
| файл `bm_batch_routines.py` в ключе эквивалентности | `tests/bearmind/test_equivalence_key.py:10, 23, 45`; сам ключ — `tests/bearmind/equivalence_runs.py:75` `code_files` (глоб `bearmind/**`, `*.py`, `ml/**`; `history/**` нет) | 3 (П-5) |
| `bm_batch_routines.extract_name_with_pattern` | `bm_examinator.py:55` | остаётся в `bm_batch_routines.py` |
| `from bm_batch_routines import *` | `startup.py:1` | 5 (удаляется) |
| `batch_export` | `batch_export_all.py:17`; `tests/bearmind/test_equivalence.py:200`; `tests/bearmind/test_stage_export.py:339, 385` | 3 |
| `batch_autoinspect` | никто | 3 |
| `batch_autoinspect_all`, `batch_export_all`, `chain_run_export_after_autoinspect`, `resplit_fof` | никто (запускаются как скрипты) | 3 |
| `startup` | только ячейка 1 ноутбука (копирует файл) | 5 |
| `test_export_e2e` | никто | 5 |
| `export_estimates_data.main` | никто | 4 (О5) |
| `ae_launch.get_fps_from_table` | `experiments/testing/test_fps_lookup.py`, `experiments/testing/test_robust_fps_integration.py` | 4: при сведении читалок FPS сохранить имя (П-8) |
| `bm_examinator.ExamineCells`, `ManualSeeds` | `tests/test_examinator_callbacks.py:103`, `tests/test_examinator_serve.py:67, 77`; `auto_inspector.py` (только упоминание) | не меняются |

В `paper/` (2 файла) и `ml/` импортов старых имён нет. В `experiments/` — только `auto_inspector`, `ae_launch`, `capcan_validation`: эти модули остаются.

## 7. Окружение (шаг 6)

| № | Что | Факт | Решение для задачи 7 |
|---|---|---|---|
| Е1 | Lock на `05ad179` против `53dece2` | `env/bearmind-lock.txt` не менялся (429 строк, без разницы) | — |
| Е2 | Живое окружение против lock | `conda list -n bearmind --export` совпадает с lock построчно | фазы 3–5 пакетов не ставили |
| Е3 | `PySide6` (окно, фазы 2–5) | lock `pyside6=6.11.2`; в файлах окружения нет; приносит caiman | закрепить в `environment.yml` |
| Е4 | `bokeh` (сервер, окно инспекции) | lock `3.10.0`; нет; приносят caiman и panel (`<3.11`) | закрепить |
| Е5 | `tornado` (`bearmind/inspect_server.py:48-49`) | lock `6.5.10`; нет; приносит bokeh | закрепить |
| Е6 | `threadpoolctl` (`bm_examinator.py:270`, `tests/test_environment.py`) | lock `3.7.0`; нет; приносит scikit-learn | закрепить |
| Е7 | `psutil` (`bearmind/config.py`, `fileio.py`, `inspect_lock.py`, окно) | lock `7.2.2`; нет; приносят caiman, driada | закрепить |
| Е8 | `pytorch` (`inspect_server.py:179`, `wavelet_backend.py:36`, оба по желанию) | lock `pytorch=2.14.1=cpu_openblas_py312he688b4c_0`; нет; приносит caiman | закрепить вместе со сборкой `cpu_openblas` (иначе решатель может взять MKL — см. §5.14) |
| Е9 | `pandas`, `matplotlib`, `tifffile`, `joblib`, `ipywidgets`, `ipython` | lock `3.1.0`, `3.11.2`, `2026.10.15`, `1.6.0`, `8.1.9`, `9.18.0`; нет; приходят с caiman и др. | закрепить |
| Е10 | `ssqueezepy` (автоинспекция через driada) | lock `0.6.6` (`pypi_0`); нет в `requirements-pip.txt`; приносит driada | закрепить в `requirements-pip.txt` |
| Е11 | `pytz` (`utils.py:2`; `utils` импортируют `auto_inspector` и `bm_examinator`) | lock `2026.6` (`pypi_0`); нет; приносит driada (pandas 3 больше не тянет pytz) | закрепить в `requirements-pip.txt` |
| Е12 | `imageio-ffmpeg` (`bm_qc.py`) | lock `0.7.0` (`pypi_0`); нет; приносит moviepy | закрепить в `requirements-pip.txt` |
| Е13 | `moviepy` | `requirements-pip.txt` `moviepy==1.0.3` = lock. После задачи 3 нужен только `history/legacy_pipeline.py` (тест эквивалентности) | оставить; поправить комментарий |
| Е14 | `interpret` | `requirements-pip.txt` `interpret==0.7.8`; lock: `interpret=0.7.8` и `interpret-core=0.7.8` (`pypi_0`) | совпадает |
| Е15 | `scipy` | `environment.yml` `scipy=1.18.1` (conda); lock `scipy=1.18.1=pypi_0` (pip заменил conda) | версия та же; после установки с нуля проверить происхождение `scipy` |
| Е16 | `cupy` (`wavelet_backend.py:35`) | нет в lock; импорт по желанию (GPU) | не добавлять |
| Е17 | `av`, `numpy`, `caiman`, `ffmpeg`, `opencv`, `natsort`, `pyyaml`, `tqdm`, `pywavelets`, `driada`, `sortedcontainers`, `pytest` | закреплены, версии равны lock | — |
| Е18 | Тест `test_env_files_match_lock` | в `environment.yml` есть `libblas=*=*openblas` (шаблон) и `pip` без версии; `numpy` и `pytest` закреплены и в pip, и в conda-lock | тест должен понимать шаблоны и pip-строки |
| Е19 | `OPENBLAS_CORETYPE`, шапка lock, `python=3.12.16`, установщики | О11–О14 | см. П-15, П-16 |

## 8. Предложения к задачам 2–7

Каждый пункт: что не предусмотрено, почему важно, что предлагаю. Решает контролёр (или автор, где сказано).

- **П-1 (задача 2).** `os.chdir(REPO_ROOT)` есть ещё в `bearmind/runner.py:1583` (`run`, возврат в 1608). Добавить `bearmind/runner.py` в файлы задачи 2. Тест `test_cli_has_no_chdir` распространить на `runner.py`. Комментарий `tests/bearmind/test_isolate.py:107` поправить.
- **П-2 (задача 2).** `auto_inspector.py:11` берёт `config.get_parallelization_backend` (`config.py:131-150`), а тот читает `CONFIG` из `config.json` текущей папки. Сейчас батч и сервер работают из корня репозитория, где лежит `config.json` автора (без ключа `PARALLELIZATION_BACKEND` → `loky`). Без `chdir` бэкенд станет зависеть от папки запуска. Предлагаю: `config.py` читает и пишет `config.json` в корне репозитория (`Path(__file__).parent`), а не в cwd. Поведение конвейера не меняется.
- **П-3 (задачи 2 и 5).** Список корневых модулей для `test_root_modules_import_cleanly`. Глобальные правки при импорте есть ещё в `bm_batch_routines.py:36` (`warnings.filterwarnings('ignore')`), `collect_nof_aligned_events.py:46, 62`, `collect_wavelet_event_stats.py:35`, `summarize_nof_events.py:14`. Предлагаю перенести их в `main()` или под `if __name__ == "__main__"`. `capcan_validation.py` при импорте только читает и печатает — оставить. Из теста исключить `conftest.py` и корневые `test_*.py`. `test_event_stats.py` и `test_wavelet_backend_integration.py` перенести `git mv` в `experiments/testing/` (задача 5). `test_env.py` — П-16.
- **П-4 (задача 3).** План оставляет в `bm_batch_routines.py` «функции, которые использует код вне history», но не говорит о 16 функциях ноутбука без вызывающих (О18). Схема фазы говорит: «only functions still used». Предлагаю: все функции старого конвейера и его GUI уходят в `history/legacy_pipeline.py`; мёртвый код (А41, А42) удаляется при переезде. В `bm_batch_routines.py` остаются `find_identifiers` и `extract_name_with_pattern`. Тогда цепочка импорта `bm_examinator` теряет `moviepy`, `ipywidgets`, `bokeh`, `caiman` из `bm_batch_routines` и `warnings.filterwarnings` (П-3).
- **П-5 (задача 3).** Ключ эквивалентности строит `tests/bearmind/equivalence_runs.py:75` (`code_files`); `history/**` в него не входит. Добавить глоб `history/**/*.py` там, а не только в список `tests/bearmind/test_equivalence_key.py`. Строку 471 (`old.MotionCorrect`) перенаправить на модуль в `history`.
- **П-6 (задача 3).** Скрипты MINI3 строят пути от своей папки: `ROOT = Path(__file__).parent` (`batch_autoinspect_all.py:56`, `batch_export_all.py:19`, `chain_run_export_after_autoinspect.py:21`). После `git mv` `ROOT` станет `history/mini3/`; `output/` и логи будут искаться там. Тест `test_mini3_scripts_kept_verbatim` правку запрещает. Записать это в `history/README.md`: как запускать (копия в корень или `output/` рядом со скриптом).
- **П-7 (задача 4).** `test_autoinspect_params_single_source` в текущем виде невыполним. Предлагаю: (1) область — корневые модули и `bearmind/`, без `history/`, `experiments/`, `ml/`, `paper/`, `tests/`; (2) умолчания `ae_launch.run_auto_inspection` для `brain`, `ml_model_path`, `ml_threshold`, `n_iter`, `detect_corner_artifacts` привести к `defaults.yaml` или читать оттуда — этап их передаёт, батч не меняется; (3) 14 скрытых параметров (Н22) не трогать. Либо явно внести их в раздел `autoinspect` `defaults.yaml`; это пересчитает autoinspect и export обработанных сессий — решение контролёра. (4) Внутренние умолчания `auto_inspector.py` (`n_iter=2`, `ml_threshold=0.5`, `cthr=0.3`) и вид `bm_examinator.py` — библиотечные; тест их не проверяет. `capcan_validation.py` — свой набор валидации; оставить.
- **П-8 (задача 4).** Интерфейс «чтение FPS для старых скриптов — `bearmind.timestamps.robust_fps` через прослойку» невыполним. `capcan_validation` (`data/raw_compressed`), `collect_nof_aligned_events` (Synchro npz) и `history/mini3/batch_export.py` работают с обработанными файлами корпуса статьи, без файлов таймстемпов. Предлагаю одну читалку `fps_data.csv` (ключ — `extract_session_id`). Округление остаётся явным там, где оно есть сейчас: автоинспекция и валидация округляют, экспорт — нет. Имя `ae_launch.get_fps_from_table` сохранить (его импортируют `experiments/`).
- **П-9 (задача 4).** Шаблон `test_no_absolute_user_paths` должен ловить и прямую косую (`D:/_Projects` в `collect_nof_aligned_events.py:31, 174`). `experiments/` и `ml/` в его области: 4 файла с путями (А20). Решить: править их или исключить эти папки.
- **П-10 (задача 4).** Замена `[:10]` в `capcan_validation.py:54, 80` на `extract_session_id` может изменить FPS валидации для сессий с ID не из 10 символов. Перед заменой сверить ключи корпуса валидации; если что-то меняется — записать.
- **П-11 (задача 4).** Широкие `except` в `auto_inspector.py` (А28) решают ветки каскада кинетики. Сужение типов изменит решения на краевых случаях (нарушит Global Constraints). В `auto_inspector.py` — только запись в лог, без сужения.
- **П-12 (задача 4, по желанию).** `ae_launch.py:26` импортирует `bm_examinator` ради `LoadEstimates` (А33). План пункт не назначил. После П-4 цена падает. Предлагаю оставить; если делать — вынести `LoadEstimates` и `get_timestamps` в лёгкий модуль.
- **П-13 (задача 5, шаг 3).** Проверка «метаданные под выгруженные трейсы» покрыта только тестом на сессии из `.gitignore` (на этой машине его нет); «нет нулевых строк ASP» не покрыта (О1). До удаления `test_export_e2e.py` добавить синтетический тест: сжатие → `approved_components` → `extract_data`, `build_metadata`, `export_filters_mat`. Число строк `C`, `asp`, `metrics_df` в метаданных и `A` в mat равны; нулевых строк ASP нет.
- **П-14 (задача 6).** О4: копия `startup.py` в профиле IPython у тех, кто запускал ячейку 1. Строка в инструкции: проверить и удалить файл, если он есть. Копию автора не трогать без его ответа.
- **П-15 (задача 7).** Добавить в `environment.yml` раздел `variables: {OPENBLAS_CORETYPE: Sandybridge}` или исправить комментарий. Иначе окружение с нуля работает без обхода Zen 4 вне `bearmind` и тестов. Закрепить пакеты Е3–Е12 версиями lock. Шапка lock: строки `pypi_0` ставит `requirements-pip.txt`.
- **П-16 (задача 7).** `install2_pip.bat:86` запускает `test_env.py`, который требует `tensorflow` → проверка установки печатает FAILED. Предлагаю: `test_env.py` удалить, `install2_pip.bat` запускать `python -m pytest -q tests/test_environment.py`. `bearmind2025.yml` (старое окружение `caiman`) удалить или перенести в `history/`. Инструкция (задача 6) выбирает один путь установки: `.bat` или команды.
- **П-17 (задача 6).** Корневой `README.md` описывает старую установку и ноутбук (О15). Заменить его коротким текстом: что это, ссылка на `docs/students/README.md` и `docs/students/series-workflow.md`. Тест `test_guide_paths_exist` можно распространить на него.
