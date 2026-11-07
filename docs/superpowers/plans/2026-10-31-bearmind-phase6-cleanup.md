# BEARMiND фаза 6: чистка остального и инструкция для студентов — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Убрать из репозитория то, что осталось из приложения А дизайна (побочные эффекты импорта, хардкоды, копии кода, мёртвый код, ноутбуки), и написать инструкцию для студентов по установке и работе с BEARMiND: по-русски, с названиями кнопок и этапов как в программе (по-английски).

**Architecture:** Сначала аудит: каждый пункт приложения А, разведки и отложенных пунктов волн сверяется с текущим кодом, итог — таблица статусов. Затем правки по группам, каждая со своими тестами. Старые функции конвейера, которые заменил пакет `bearmind`, переезжают в `history/` вместе со скриптами MINI3 и не переписываются. Ноутбуки удаляются (решение автора). В конце — инструкция по протоколу STE, файлы окружения с пакетами фаз 3–5 и проверка установки с нуля.

**Tech Stack:** Python 3.12, окружение `bearmind`; pytest; conda/mamba для проверки окружения.

**Spec:** `docs/superpowers/specs/2026-10-30-bearmind-batch-pipeline-design.md` — §2 (цель «убраны хардкоды, костыли, копии кода и найденные баги»), §5.14 (окружение, инструкция по STE), §5.16, приложение А. Подробности с номерами строк — `docs/superpowers/specs/2026-10-30-bearmind-recon.md`. **Схема фазы (одобрена автором):** `docs/superpowers/plans/2026-11-06-phase6-cleanup-explainer.html` (части 1–6, оглавление инструкции). Решения автора: память `author-decisions-2026-11-06` (инструкция по-русски с английскими названиями; пакеты в окружение `bearmind` с записью в `env/bearmind-lock.txt`, `environment.yml`, `requirements-pip.txt`; рекомендация 5 — ноутбук удаляется). Выполняется после фаз 2–5: инструкция описывает их вкладки.

## Changes on 2026-11-07

План написан 2026-10-31, до разбора автором. Что изменено.

По решениям автора:
- Ноутбук `BEARMiND_full_pipeline.ipynb` удаляется в этой фазе (рекомендация 5). Незакоммиченные правки автора в нём — только метаданные Jupyter; задача 5 проверяет это перед удалением и останавливается, если изменён код ячеек. `MoviePlay.ipynb` удаляется по одобренной схеме. Значения ноутбука сначала сверяются с `bearmind/defaults.yaml` (аудит).
- Инструкция для студентов — по-русски, по протоколу STE; названия вкладок, кнопок и этапов — как в программе, по-английски: Monitor, Scan, Thresholds, Inspection, Start, Stop after current stage, Rescan, Accept → next, Manual inspection, Not needed и т. д. Порядок вкладок — Monitor, Scan, Thresholds, Inspection. Прежние «Скан», «Пороги» убраны. Новый тест сверяет английские названия из инструкции с текстами окна.
- Пакеты: фазы 3–5 могли добавить пакеты в окружение `bearmind`. Они должны попасть в `environment.yml` или `requirements-pip.txt` с теми же версиями, что в `env/bearmind-lock.txt`. Новый тест сверяет файлы окружения с файлом блокировки; установка с нуля это ловит.
- Отложенные пункты волны A1: корневой скрипт `test_export_e2e.py` повторяет старое правило отбора `>=` (строка 133) — удаляется после проверки, что его проверки покрыты тестами в `tests/`; ячейки экспорта ноутбука без защиты от старых обработанных файлов — уходят вместе с ноутбуком.
- Батч не останавливается из-за одной сессии и окна по-английски — правила фаз 2–5; эта фаза их не меняет.

По реальному коду:
1. `export_estimates_data.py` удалить нельзя: его импортируют `bearmind/stages/export.py` (как `old`: правило `approved_components`, `extract_data`, `build_metadata`, `export`, `export_filters_mat`), `bearmind/runner.py` и `ae_launch.py` (`MERGE_GROUPING`), тесты `tests/test_export_*.py`. Схема говорила «удаляется». Файл остаётся библиотекой экспорта; аудит решает только судьбу его командной строки.
2. `batch_export.py` нужен скрипту MINI3 `batch_export_all.py` и тестам `tests/bearmind/test_equivalence.py`, `tests/bearmind/test_stage_export.py` (сверка трёх путей экспорта). Он переезжает в `history/mini3/` вместе с MINI3, тесты импортируют `history.mini3.batch_export`. `batch_autoinspect.py` ни у кого не в импортах — удаляется.
3. Старый конвейер в `bm_batch_routines.py`: `DoCropAndRewrite`, `DoMotionCorrection`, `DoCNMF`, `ReDoCNMF` и помощники. Их импортируют `tests/bearmind/equivalence_runs.py` (строки 467, 525) и ключ эквивалентности `tests/bearmind/test_equivalence_key.py` (список файлов содержит `bm_batch_routines.py`). `bm_examinator.py` берёт оттуда `extract_name_with_pattern`, `startup.py` — всё (`import *`).
4. `startup.py` создаёт окно Tk при импорте, и его импортирует только ноутбук. Он удаляется вместе с ноутбуком, а не чинится.
5. `config.py` (корень) создаёт `config.json` при импорте (строка 153) и бросает `UnboundLocalError` при неизвестном `DATA_PATHWAY`. `os.chdir(provenance.REPO_ROOT)` стоит в `bearmind/cli.py` в двух местах: `cmd_gui` и `main`. `launcher` запускает батч и сервер инспекции с `cwd` = корень репозитория — это остаётся (безвредно).
6. `sys.modules["torch"] = None` есть только в `collect_nof_aligned_events.py`; `np.Inf` в коде вне `experiments/`, `paper/`, `ml/` не найден. Остальное — по аудиту.
7. `output_notebook()` при импорте `bm_examinator.py` убрала фаза 5. Мёртвая блок-строка в конце `bm_examinator.py` (после `merge_estimates`, строки 1793–1850 на 2026-11-07) — эта фаза.
8. Отслеживаемый мусор: `.DS_Store`, `.ipynb_checkpoints/MoviePlay-checkpoint.ipynb`.
9. Файлы окружения: `environment.yml`, `requirements-pip.txt`, `env/bearmind-lock.txt` (экспорт `conda list`); тест окружения — `tests/test_environment.py`.
10. Заметки оператора `docs/superpowers/specs/2026-11-01-operator-notes.md` обещают: «Фаза 6 перенесёт их в руководство студента». Инструкция ссылается на них для тонких настроек (рамка кропа, правило экспорта, частота кадров, пул MC); сами заметки остаются.
11. Проверки на реальных данных: `series_short` и `series1` только читаются; свежая установка и свежий клон — во временных папках.

## Global Constraints

- Не меняются: алгоритмы CaImAn и логика автоинспекции (метрики, правила, ML-модель, группировка слияний `fbd_complete`), правило экспорта `ml_or_merged_not_deleted`, формат `npz` / `mat` / `json` (§2).
- Скрипты MINI3 (`batch_autoinspect_all.py`, `batch_export_all.py`, `chain_run_export_after_autoinspect.py`, `resplit_fof.py`) и их зависимость `batch_export.py` — запись того, как обработаны данные статьи: переезжают в `history/mini3/` через `git mv`, без переписывания (приложение А).
- Ноутбуки удаляются только после того, как аудит подтвердил: значения `cnmf_dict`, `mc_dict` и параметры автоинспекции из ноутбука есть в `bearmind/defaults.yaml` (§5.16). Исключение — `fr`: в новом коде `fr = FPS`, без деления.
- Порог `pxlthr_distance_boundary` = 15 px не трогать.
- Окна и их тексты (английские) не меняются. Инструкция — по-русски, по протоколу STE (§5.14): шаги нумерованы, одно действие в шаге, глагол в начале, предупреждение до шага, предложение в шаге до 20 слов; названия вкладок, кнопок и этапов — как в программе.
- Пакеты можно ставить в окружение `bearmind`, если нет конфликта с CaImAn; каждый пакет — в `env/bearmind-lock.txt`, `environment.yml` или `requirements-pip.txt`.
- Батч не останавливается из-за одной сессии; панель Monitor — ровно шесть видов (фаза их не трогает).
- Тяжёлых запусков MC и CNMF-E нет (правило автора); медленный тест эквивалентности — только если для него уже есть данные, иначе пропуск.
- Исходные данные только читаются (`G:\`, `H:\Dataset`, `H:\Projects\LT1`, включая `LT1\CalciumData\1_Raw`). `series_short`, `series1` только читаются; проверки пишут во временные папки или в `_bearmind` `H:\Projects\_bearmind_test\series_lt1`.
- Консоль — ASCII. Коммиты без строк об ИИ, только `git commit -- <пути>` (новые файлы — сначала `git add <эти пути>`; переезды — `git mv`); не `git add -A`. В репозитории могут работать другие агенты: коммитить только свои пути. Ничего не пушить.

## Review Focus

- Импорт модуля корня в пустой папке создаёт файлы или окна: после фазы этого нет ни у одного модуля корня. Тест: задача 2 `test_root_modules_import_cleanly`.
- Скрипт из `paper/`, `experiments/`, `ml/` или `tests/` импортирует функцию, которая уехала в `history/`: импорт падает. Тест: задача 3 `test_no_imports_of_moved_names`.
- Батч после переезда не нашёл библиотеку экспорта: этапы export и autoinspect и `runner` по-прежнему импортируют `export_estimates_data`. Тест: задача 3 `test_export_library_stays`.
- Студент запускает команду из инструкции, которой нет, или ищет кнопку, которая называется иначе. Тесты: задача 6 `test_guide_commands_exist`, `test_guide_ui_names_exist`.
- Пакет, добавленный в фазах 3–5, не попал в файлы окружения: окружение с нуля не соберётся. Тест: задача 7 `test_env_files_match_lock`; проверка: задача 7, шаг 2.
- Код всё ещё полагается на `config.json` в текущей папке: после фазы окна и CLI работают из любой папки. Тест: задача 2 `test_cli_from_other_cwd`.
- В ноутбуке были правки автора, кроме метаданных: ноутбук не удаляется без ответа автора. Проверка: задача 5, шаг 1.

## Решения плана (моё решение, автор может отменить)

- Старые `DoCropAndRewrite`, `DoMotionCorrection`, `DoCNMF`, `ReDoCNMF` и их помощники переезжают в `history/legacy_pipeline.py`. Их импортирует только проверка эквивалентности (`tests/bearmind/equivalence_runs.py`); ключ эквивалентности считает `history/legacy_pipeline.py` вместо `bm_batch_routines.py`. Баг `ReDoCNMF` (seeds ставятся после сборки `opts`) не чинится, а описывается в `history/README.md`.
- `export_estimates_data.py` остаётся в корне как библиотека экспорта пакета (см. «Changes» п. 1). Его командную строку аудит либо оставляет (она одна из трёх путей с правилом `ml_or_merged_not_deleted`), либо удаляет вместе с функциями, которые нужны только ей; решение и причина — в таблице аудита.
- `batch_export.py` уходит в `history/mini3/` (его импортирует `batch_export_all.py`); `batch_autoinspect.py` удаляется (его никто не импортирует; его работу делает этап autoinspect).
- `startup.py` удаляется вместе с ноутбуком: он нужен только ноутбуку и при импорте открывает окно Tk.
- `test_export_e2e.py` (корень, pytest его не собирает) удаляется, если аудит подтвердит, что его три проверки (`metrics_df` и `asp_cache` после сжатия, метаданные под выгруженные трейсы) покрыты тестами `tests/test_compression_*.py`, `tests/test_export_*.py`. Непокрытая проверка переносится в `tests/` на правило `approved_components`, а не на старое `>=`.
- Старый `config.py` остаётся для `bm_examinator.py` и других модулей корня. Он больше не создаёт `config.json` при импорте и не бросает `UnboundLocalError`.
- Инструкция ссылается на заметки оператора для тонких настроек и не копирует их.

---

### Task 1: Аудит приложения А и отложенных пунктов

**Files:**
- Create: `docs/superpowers/specs/<дата>-cleanup-audit.md` (`<дата>` — день выполнения, `YYYY-MM-DD`)

- [ ] **Step 1:** Пройти каждый пункт приложения А и каждую строку разведки `2026-10-30-bearmind-recon.md`. Для каждого найти текущее место в коде (`grep`, номер строки на текущем коммите).
- [ ] **Step 2:** Добавить отложенные пункты: `test_export_e2e.py:133` (A1, задача 3), ячейки экспорта ноутбука без защиты (A1, повторный разбор), `startup.py`, командная строка `export_estimates_data.py`, импорты `batch_export.py` и `batch_autoinspect.py`.
- [ ] **Step 3:** Записать таблицу: пункт · где сейчас · статус (`исправлено в <коммит>` / `осталось` / `уходит в history` / `уходит с ноутбуком` / `удаляется`) · задача этого плана.
- [ ] **Step 4:** Сверить значения ноутбука с `bearmind/defaults.yaml`: `cnmf_dict`, `mc_dict`, параметры автоинспекции. Каждое расхождение — строка таблицы с решением. `fr` — известное намеренное расхождение.
- [ ] **Step 5:** Найти все импорты старых имён из `paper/`, `experiments/`, `ml/`, `tests/` и корня; записать, какие затронут задачи 3–5.
- [ ] **Step 6:** Сравнить `env/bearmind-lock.txt` на 2026-11-07 (`git show 05ad179:env/bearmind-lock.txt`) с текущим: пакеты, добавленные фазами 3–5, — строки таблицы для задачи 7.
- [ ] **Step 7:** Commit: `git add docs/superpowers/specs/<дата>-cleanup-audit.md` и `git commit -m "docs: cleanup audit of appendix A against the current code" -- docs/superpowers/specs/<дата>-cleanup-audit.md`

### Task 2: Побочные эффекты импорта

**Files:**
- Modify: `config.py` (корень), `bearmind/cli.py` (убрать оба `os.chdir`), модули корня из таблицы аудита (например, `collect_nof_aligned_events.py` с `sys.modules["torch"] = None`)
- Test: `tests/test_import_side_effects.py`

**Interfaces:**
- Produces: `config.py` создаёт `config.json` только в `write_config` / `update_config`. `read_config` при отсутствии файла возвращает `DEFAULT_CONFIG`. Неизвестный `DATA_PATHWAY` → `ValueError("unknown DATA_PATHWAY: <value>; expected one of ...")`. `update_config` меняет словарь `CONFIG` на месте (`clear` + `update`), поэтому модули, импортировавшие `CONFIG`, видят новые значения.

- [ ] **Step 1: Тесты:**
```python
def test_root_modules_import_cleanly(tmp_path): ...   # for every root module that stays (list from the audit; startup.py is deleted in task 5): python -c "import <m>" with cwd=tmp_path -> exit 0, tmp_path empty, tkinter.Tk never created (sitecustomize replaces Tk with a failure)
def test_update_config_visible_to_importers(tmp_path): ...
def test_unknown_data_pathway(tmp_path): ...          # ValueError text names the allowed values
def test_cli_from_other_cwd(tmp_path): ...            # python -m bearmind status <series> with cwd=tmp_path: exit 0, no config.json anywhere new
def test_cli_has_no_chdir(): ...                      # ast of bearmind/cli.py: no os.chdir call
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS; быстрый набор PASS.
- [ ] **Step 5:** Commit: `git commit -m "fix: no files, windows or global patches on import of root modules" -- <пути из шага 3> tests/test_import_side_effects.py`

### Task 3: Старый конвейер и скрипты — в `history/`

**Files:**
- Create: `history/README.md`, `history/legacy_pipeline.py`, `history/mini3/` (через `git mv`: `batch_autoinspect_all.py`, `batch_export_all.py`, `chain_run_export_after_autoinspect.py`, `resplit_fof.py`, `batch_export.py`)
- Modify: `bm_batch_routines.py` (остаются функции, которые использует код вне `history/`: `extract_name_with_pattern` и другие по аудиту), `tests/bearmind/equivalence_runs.py` (импорт из `history.legacy_pipeline`), `tests/bearmind/test_equivalence_key.py` (`history/legacy_pipeline.py` в списке файлов), `tests/bearmind/test_equivalence.py` и `tests/bearmind/test_stage_export.py` (импорт `history.mini3.batch_export`)
- Delete: `batch_autoinspect.py`
- Test: `tests/test_history_moves.py`

- [ ] **Step 1: Тесты:**
```python
def test_no_imports_of_moved_names(): ...      # ast walk of every .py outside history/: no import of the moved names from bm_batch_routines, no "import batch_export" / "batch_autoinspect"
def test_legacy_pipeline_importable(): ...     # import history.legacy_pipeline; DoMotionCorrection, DoCNMF, ReDoCNMF present
def test_mini3_scripts_kept_verbatim(): ...    # the moved scripts equal their last versions before the move (git show <base>:<path>)
def test_export_library_stays(): ...           # export_estimates_data importable; bearmind.stages.export, bearmind.runner, ae_launch import it
def test_deleted_scripts_gone(): ...           # batch_autoinspect.py absent
```
- [ ] **Step 2:** FAIL. **Step 3:** Перенести (`git mv`, чтобы сохранить историю). Удалить мёртвый код: блок-строку в конце `bm_examinator.py`, код после `return` в `bm_batch_routines.py`, скрытые кнопки кропа. В `history/README.md` записать: что здесь лежит, зачем (данные статьи), как запускать скрипты MINI3 из `history/mini3/` (с `PYTHONPATH` = корень репозитория), известный баг `ReDoCNMF`. **Step 4:** PASS; быстрый набор PASS; медленный тест эквивалентности — если для него есть данные.
- [ ] **Step 5:** Commit: `git add history tests/test_history_moves.py` и `git commit -m "refactor: move the replaced pipeline and MINI3 scripts to history/" -- history bm_batch_routines.py bm_examinator.py batch_autoinspect.py batch_autoinspect_all.py batch_export_all.py chain_run_export_after_autoinspect.py resplit_fof.py batch_export.py tests/bearmind/equivalence_runs.py tests/bearmind/test_equivalence_key.py tests/bearmind/test_equivalence.py tests/bearmind/test_stage_export.py tests/test_history_moves.py`

### Task 4: Хардкоды и копии кода

**Files:**
- Modify: места из таблицы аудита (абсолютные пути, путь к модели относительно текущей папки, порт Bokeh, срезы имён файлов, списки наборов CC/RFC/FOF), `auto_inspector.py` и модули с копиями чтения таблицы FPS и PNR → imax
- Test: `tests/test_no_hardcodes.py`

**Interfaces:**
- Produces: один источник для каждой копии из приложения А. Параметры автоинспекции — только `bearmind/defaults.yaml` (раздел `autoinspect`). PNR → imax — одна функция в `bm_qc` (её же использует `bearmind/stages/cnmf._imax`). Чтение FPS для старых скриптов — `bearmind.timestamps.robust_fps` через прослойку.

- [ ] **Step 1: Тесты:**
```python
def test_no_absolute_user_paths(): ...         # regex over .py outside history/ and paper/: no "C:\\Users\\", "W:\\DATASET", "D:\\_Projects"
def test_no_name_slicing(): ...                # no name slices from the recon list (patterns like name[-20:-10], split('/')[6:])
def test_autoinspect_params_single_source(): ... # literals of ml_threshold / n_iter / pxlthr_distance_boundary outside defaults.yaml and history/ absent
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать по таблице аудита. Широкие `except` с молчаливым продолжением заменить узкими или записью в лог. **Step 4:** PASS; быстрый набор PASS.
- [ ] **Step 5:** Commit: `git commit -m "refactor: remove hardcoded paths and copies listed in appendix A" -- <пути из шага 3> tests/test_no_hardcodes.py`

### Task 5: Ноутбуки и мусор в репозитории

**Files:**
- Delete: `BEARMiND_full_pipeline.ipynb`, `MoviePlay.ipynb`, `startup.py`, `test_export_e2e.py`, `.DS_Store`, `.ipynb_checkpoints/MoviePlay-checkpoint.ipynb`
- Modify: `.gitignore` (`.DS_Store`, `.ipynb_checkpoints/`, `temp/`, `__pycache__/`)

- [ ] **Step 1: Правки автора в ноутбуке.** Сравнить рабочую копию `BEARMiND_full_pipeline.ipynb` с `HEAD` по ячейкам (скриптом: `source` и `cell_type` каждой ячейки). Совпадают — правки только в метаданных, автор разрешил их потерять (решение 5). Не совпадают — остановиться и спросить автора; ноутбук не трогать.
- [ ] **Step 2:** Проверить по таблице аудита (задача 1, шаг 4), что все значения ноутбука есть в `defaults.yaml`. Если нет — остановиться и дописать значения отдельным коммитом.
- [ ] **Step 3:** Проверить по аудиту, что проверки `test_export_e2e.py` покрыты тестами в `tests/` (иначе перенести непокрытую на правило `approved_components` отдельным коммитом) и что `startup.py` никто, кроме ноутбука, не импортирует.
- [ ] **Step 4:** Удалить файлы (`git rm -f` для ноутбука с правками метаданных, `git rm` для остальных). Обновить `.gitignore`.
- [ ] **Step 5:** `git ls-files | grep -E "\.DS_Store|ipynb_checkpoints|\.ipynb$|^startup\.py$|^test_export_e2e\.py$"` — пусто.
- [ ] **Step 6:** Commit: `git commit -m "chore: remove notebooks, notebook helpers and tracked junk; defaults live in bearmind/defaults.yaml" -- BEARMiND_full_pipeline.ipynb MoviePlay.ipynb startup.py test_export_e2e.py .DS_Store .ipynb_checkpoints .gitignore`

### Task 6: Инструкция для студентов

**Files:**
- Create: `docs/students/README.md` (установка и запуск), `docs/students/series-workflow.md` (работа с серией), картинки окон из отчётов проверок фаз 2–5 (`docs/students/img/`, каждая < 300 КБ)
- Test: `tests/test_student_guide.py`

Содержание (протокол STE, по-русски; названия вкладок, кнопок и этапов по-английски, как в программе; оглавление — как в одобренной схеме):
- `README.md`: что нужно на компьютере; как создать окружение `bearmind` по `environment.yml` и `requirements-pip.txt`; как проверить установку (`run_bearmind.bat --help`, `bearmind_gui.bat`); что делать, если окружение не найдено.
- `series-workflow.md`:
  - Папка серии: `1_Raw`, `0_TimeStamps` и другие (§5.2); пайплайн никогда не пишет в `1_Raw`.
  - Вкладка Scan: найти сессии (Scan, Rescan), разобрать аномалии (Exclude, Include, Accept with trimming to common length, Confirm stitch).
  - Запуск батча на вкладке Monitor (Start, Stop after current stage).
  - Вкладка Thresholds (Accept → next, Accept for all sessions of this mouse).
  - Вкладка Inspection: Manual inspection в браузере (Save results, Next) или Not needed.
  - Сводка: исключить сессию, пересчитать (меню Actions: Exclude, Rerun from stage…, Retry failed).
  - Цвета клеток и три самых частых предупреждения из панели Warnings.
  - Если что-то пошло не так: батч упал или машину перезагрузили (нажать Start — продолжит); `summary.csv` открыт в Excel; ошибка этапа (Retry); где логи (`_bearmind/logs/`).
  - Тонкие настройки — ссылка на `docs/superpowers/specs/2026-11-01-operator-notes.md`.

- [ ] **Step 1: Тесты:**
```python
def test_guide_commands_exist(): ...     # every "run_bearmind.bat <cmd>" in the guide is a command of python -m bearmind --help
def test_guide_paths_exist(): ...        # repository files and folders named in the guide exist
def test_guide_sentence_length(): ...    # sentences in numbered steps have at most 20 words (STE)
def test_guide_ui_names_exist(): ...     # every English UI name in **bold** in the guide occurs as a string in bearmind/gui/*.py or bm_examinator.py
def test_guide_tab_order(): ...          # the guide names the tabs in the order Monitor, Scan, Thresholds, Inspection; no "Скан", "Пороги"
```
- [ ] **Step 2:** FAIL. **Step 3:** Написать инструкцию. **Step 4:** PASS.
- [ ] **Step 5:** Commit: `git add docs/students tests/test_student_guide.py` и `git commit -m "docs: student guide - install, series workflow, troubleshooting" -- docs/students tests/test_student_guide.py`

### Task 7: Файлы окружения, установка с нуля и финальный прогон

**Files:**
- Modify: `environment.yml`, `requirements-pip.txt` (пакеты фаз 3–5 из таблицы аудита, версии из `env/bearmind-lock.txt`), `tests/test_environment.py`
- Create: `docs/superpowers/specs/<дата>-phase6-check.md`

- [ ] **Step 1: Тест файлов окружения:**
```python
def test_env_files_match_lock(): ...     # every package pinned in environment.yml and requirements-pip.txt is in env/bearmind-lock.txt with the same version
```
FAIL, если фазы 3–5 добавили пакет без записи в файлы окружения; дописать пакеты; PASS.
- [ ] **Step 2:** Собрать окружение с нуля во временный префикс: `mamba env create -p <temp>\bm_fresh -f environment.yml`, затем `pip install -r requirements-pip.txt`. Запустить в нём быстрый набор тестов. Удалить временный префикс.
- [ ] **Step 3:** Свежий клон репозитория во временную папку. Запустить `run_bearmind.bat --help` и `python -m bearmind status` на копии `series_short` (копия во временной папке; источник только читается).
- [ ] **Step 4:** Полный набор тестов, включая `slow`; `realdata` — если есть тестовые серии.
- [ ] **Step 5:** Отчёт: что собрано, время установки, результаты тестов, что не проверено (машина без CUDA, другая модель процессора: нужен ли там обход OpenBLAS `OPENBLAS_CORETYPE`).
- [ ] **Step 6:** Commit: `git add docs/superpowers/specs/<дата>-phase6-check.md` и `git commit -m "docs: phase 6 check - env files, fresh install and full test run" -- environment.yml requirements-pip.txt tests/test_environment.py docs/superpowers/specs/<дата>-phase6-check.md`
