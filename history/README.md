# history: старый конвейер и скрипты MINI3

Здесь лежит старый код. Пакет `bearmind` его заменил. Данные статьи посчитаны этим кодом или его более ранними версиями. Отличия описаны ниже. Пакет `bearmind` этот код не импортирует.

Не правьте файлы в этой папке: они — запись старого кода.

## Что здесь лежит

| Файл | Что делает |
|---|---|
| `legacy_pipeline.py` | Функции конвейера ноутбука `BEARMiND_full_pipeline.ipynb`. Раньше они лежали в `bm_batch_routines.py`. |
| `mini3/batch_autoinspect_all.py` | Автоинспекция наборов CC, RFC и FOF с параметрами ноутбука (`pxlthr_distance_boundary = 15`). |
| `mini3/batch_export.py` | Экспорт файлов `*_processed.pickle` в NPZ, JSON и MAT. |
| `mini3/batch_export_all.py` | Запускает `batch_export` по наборам CC, RFC и FOF. |
| `mini3/chain_run_export_after_autoinspect.py` | Ждёт конца автоинспекции по её логу, потом запускает экспорт и пишет отчёт. |
| `mini3/resplit_fof.py` | Делит сшитые сессии FOF на дни 1D, 2D и 3D. |
| `__init__.py`, `mini3/__init__.py` | Делают папки пакетами Python: тесты импортируют `history.legacy_pipeline` и `history.mini3.batch_export`. |

Главные функции `legacy_pipeline.py`:

- `DoCropAndRewrite` — кроп AVI и запись одного TIF;
- `DoMotionCorrection` — коррекция движения (MC);
- `DoCNMF` — CNMF-E;
- `ReDoCNMF` — повтор CNMF-E с seeds (известный баг, см. ниже);
- `DrawCropper`, `plot_gsig_range`, `plot_min_corr_and_pnr_range` — виджеты кропа и порогов для ноутбука.

`legacy_pipeline.py` — версия этой ветки, а не те же байты, что запускал ноутбук. Правки: работа на NumPy 2 и CaImAn 1.13.2 (коммит `e64d693`), необязательный QC (коммит `863b307`). Основной расчёт тот же.

Скрипты папки `mini3/` сделали на компьютере MINI3 автоинспекцию и экспорт наборов CC, RFC и FOF для статьи. Здесь они лежат байт в байт такими, какими были в корне репозитория до переезда. Тест `tests/test_history_moves.py` это проверяет. `batch_export.py` после прогонов на MINI3 менялся: см. раздел «Правило экспорта статьи».

Код этой папки использует только тест эквивалентности:

- `tests/bearmind/equivalence_runs.py` берёт старую сторону MC и CNMF-E из `history.legacy_pipeline`;
- `tests/bearmind/test_equivalence.py` и `tests/bearmind/test_stage_export.py` сверяют экспорт пакета с `history.mini3.batch_export`.

Ключ кэша теста эквивалентности учитывает все файлы `history/**/*.py`.

## Правило экспорта статьи

Экспорт статьи шёл по старому правилу: нейрон выгружался, если `ml_keep_probability >= ml_threshold`. У этого правила известные дефекты. Оно теряет все нейроны, которые слила автоинспекция: у них нет вероятности ML. Оно выгружает дубликаты, которые автоинспекция пометила на удаление по SNR.

После прогонов на MINI3 правило в `batch_export.py` поменяли (коммиты `0f51678` и `9434c23`). Теперь скрипт берёт нейроны из `approved_components` и предупреждает о слитых нейронах старых файлов (`warn_merge_products_left_out`). Самая ранняя версия `batch_export.py` в репозитории — коммит `cf92c83`.

Поэтому повторный запуск `history/mini3/batch_export.py` теперь даёт другие наборы нейронов, чем в экспорте статьи.

## Ноутбук больше не работает

Ноутбук `BEARMiND_full_pipeline.ipynb` брал эти функции через `startup.py`: `from bm_batch_routines import *`. В `bm_batch_routines.py` теперь остались только `find_identifiers` и `extract_name_with_pattern`. Поэтому ячейки ноутбука со старыми функциями падают с `NameError`.

Пакет `bearmind` заменил ноутбук и `startup.py`. Старые функции лежат в `bm_batch_routines.py` в родителе коммита, который создал `history/`. Этот коммит покажет команда `git log --diff-filter=A --format=%h -- history/README.md`.

## Как вызвать функцию из `legacy_pipeline.py`

Предупреждение: функции пишут файлы рядом с входным видео или в папку `ROOT` из `config.json`. Запускайте их только на копии данных.

1. Откройте Anaconda Prompt.
2. Перейдите в корень репозитория: `cd C:\путь\к\bearmind`.
3. Активируйте окружение: `conda activate bearmind`.
4. Запустите Python: `python`.
5. Импортируйте функцию: `from history.legacy_pipeline import DoCNMF`.

`DoCropAndRewrite` и `extract_and_copy_ts` читают `ROOT` и `DATA_PATHWAY` из `config.json` в корне репозитория. `SaveCrops` читает только `DATA_PATHWAY` (через `config.py`).

## Как запустить скрипты MINI3

Скрипты строят пути от своей папки: `ROOT = Path(__file__).parent`. Теперь `ROOT` — это `history\mini3`. Поэтому результаты лягут в `history\mini3\output\<набор>`, а логи и отчёт — в `history\mini3`. Git не видит папку `output\` (она в `.gitignore`).

Пути к данным в скриптах — пути компьютера MINI3 (`W:\DATASET\...`). Чтобы поменять пути, скопируйте папку `history\mini3` за пределы репозитория. Правьте копию, а не файлы в `history`.

Скрипты импортируют модули корня репозитория: `ae_launch`, `naming`, `export_estimates_data`. Поэтому корень репозитория должен быть в `PYTHONPATH`. Путь к ML-модели `production_models/ebm_v9_iter8.pkl` скрипт берёт от текущей папки. Поэтому запускайте скрипты из корня репозитория.

`batch_export.py` берёт частоту кадров из `fps_data.csv` в корне репозитория. Если сессии нет в таблице, экспорт этой сессии падает с ошибкой.

`batch_autoinspect_all.py` вызывает автоинспекцию без частоты кадров (`fps=None`). Тогда `ae_launch` берёт частоту из `fps_data.csv`. Если сессии нет в таблице, `ae_launch` выдаёт ошибку, и скрипт отмечает сессию как FAIL. В текущей таблице нет ни одной сессии CC и примерно половины сессий RFC и FOF. На MINI3 была своя таблица частот: в метаданных CC записано `autoinspection_config.fps` = 20. Для повторного прогона нужна таблица частот MINI3: добавьте её строки в `fps_data.csv`.

1. Откройте Anaconda Prompt.
2. Перейдите в корень репозитория: `cd C:\путь\к\bearmind`.
3. Активируйте окружение: `conda activate bearmind`.
4. Добавьте корень репозитория в `PYTHONPATH`: `set PYTHONPATH=%CD%`.

Скрипт `batch_autoinspect_all.py` сам лог не пишет. `chain_run_export_after_autoinspect.py` ждёт строку `ALL DONE` в файле `history\mini3\batch_autoinspect_all.log`. Поэтому в шаге 5 вывод идёт в этот файл.

5. Запустите автоинспекцию: `python history\mini3\batch_autoinspect_all.py --datasets CC,RFC,FOF > history\mini3\batch_autoinspect_all.log 2>&1`.
6. Откройте второе окно Anaconda Prompt.
7. Во втором окне выполните шаги 2–4.
8. Во втором окне запустите цепочку: `python history\mini3\chain_run_export_after_autoinspect.py`.

Цепочка ждёт конца автоинспекции не дольше 16 часов. Потом она запускает `batch_export_all.py` по всем трём наборам. Лог экспорта — `history\mini3\batch_export_all.log`, отчёт — `history\mini3\overnight_report.txt`.

Экспорт без цепочки: `python history\mini3\batch_export_all.py --datasets CC,RFC,FOF`.

Чтобы пропустить сессии, у которых уже есть `*_processed.pickle`, добавьте к шагу 5 ключ `--skip-existing`.

### `resplit_fof.py`

Предупреждение: скрипт удаляет файлы без подтверждения. Он очищает папки `traces` и `metadata` в `W:\DATASET\FOF\Synchro` и удаляет папку `synchronized`. Не запускайте его на оригиналах.

Пути в скрипте абсолютные. Путь `BM_OUT` — путь компьютера MINI3 (`C:\Users\1\pycharmprojects\bearmind\output\FOF`).

## Известный баг `ReDoCNMF`

`ReDoCNMF` строит параметры CNMF-E (`opts`) раньше, чем записывает в `params_dict` настройки seeds: `min_corr = 0`, `min_pnr = 0`, `seed_method`, `init_iter = 0`, `rf = None`. Поэтому CNMF-E идёт без seeds, с порогами и `rf` исходной сессии.

Настройки seeds попадают только в `cnm.estimates.cnmf_dict`. Файл `*_redo_cnmf_estimates.pickle` поэтому записывает настройки, с которыми его не считали.

Баг не исправлен: файл — запись старого кода. Пакет `bearmind` повтор CNMF-E с seeds не делает.
