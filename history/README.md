# history: старый конвейер и скрипты MINI3

Здесь лежит старый код. Пакет `bearmind` его заменил. Этим кодом посчитаны данные статьи. Пакет `bearmind` этот код не импортирует.

Не правьте файлы в этой папке. Они записывают, как шла обработка данных статьи.

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

Скрипты папки `mini3/` сделали автоинспекцию и экспорт наборов CC, RFC и FOF для статьи на компьютере MINI3. Они лежат здесь байт в байт, как в коммите `5527875`. Тест `tests/test_history_moves.py` это проверяет.

Код этой папки использует только тест эквивалентности:

- `tests/bearmind/equivalence_runs.py` берёт старую сторону MC и CNMF-E из `history.legacy_pipeline`;
- `tests/bearmind/test_equivalence.py` и `tests/bearmind/test_stage_export.py` сверяют экспорт пакета с `history.mini3.batch_export`.

Ключ кэша теста эквивалентности учитывает все файлы `history/**/*.py`.

## Ноутбук больше не работает

Ноутбук `BEARMiND_full_pipeline.ipynb` брал эти функции через `startup.py`: `from bm_batch_routines import *`. В `bm_batch_routines.py` теперь остались только `find_identifiers` и `extract_name_with_pattern`. Поэтому ячейки ноутбука со старыми функциями падают с `NameError`.

Пакет `bearmind` заменил ноутбук и `startup.py`. Последний коммит, где старые функции лежат в `bm_batch_routines.py`, — `5527875`.

## Как вызвать функцию из `legacy_pipeline.py`

Предупреждение: функции пишут файлы рядом с входным видео или в папку `ROOT` из `config.json`. Запускайте их только на копии данных.

1. Откройте Anaconda Prompt.
2. Перейдите в корень репозитория: `cd C:\путь\к\bearmind`.
3. Активируйте окружение: `conda activate bearmind`.
4. Запустите Python: `python`.
5. Импортируйте функцию: `from history.legacy_pipeline import DoCNMF`.

`DoCropAndRewrite`, `extract_and_copy_ts` и `SaveCrops` читают `ROOT` и `DATA_PATHWAY` из `config.json` в корне репозитория.

## Как запустить скрипты MINI3

Скрипты строят пути от своей папки: `ROOT = Path(__file__).parent`. Теперь `ROOT` — это `history\mini3`. Поэтому результаты лягут в `history\mini3\output\<набор>`, а логи и отчёт — в `history\mini3`. Git не видит папку `output\` (она в `.gitignore`).

Пути к данным в скриптах — пути компьютера MINI3 (`W:\DATASET\...`). Чтобы поменять пути, скопируйте папку `history\mini3` за пределы репозитория. Правьте копию, а не файлы в `history`.

Скрипты импортируют модули корня репозитория: `ae_launch`, `naming`, `export_estimates_data`. Поэтому корень репозитория должен быть в `PYTHONPATH`. Путь к ML-модели `production_models/ebm_v9_iter8.pkl` скрипт берёт от текущей папки. Поэтому запускайте скрипты из корня репозитория.

`batch_export.py` берёт частоту кадров из `fps_data.csv` в корне репозитория. Если сессии нет в таблице, экспорт этой сессии падает с ошибкой.

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
