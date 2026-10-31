# BEARMiND фаза 6: чистка остального и инструкция для студентов — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Убрать из репозитория то, что осталось из приложения А дизайна (побочные эффекты импорта, хардкоды, копии кода, мёртвый код, ноутбуки), и написать инструкцию для студентов по установке и работе с BEARMiND.

**Architecture:** Сначала аудит: каждый пункт приложения А и разведки сверяется с текущим кодом, итог — таблица статусов. Затем правки по группам, каждая со своими тестами. Старые функции конвейера, которые заменил пакет `bearmind`, переезжают в `history/` вместе со скриптами MINI3 и не переписываются. В конце — инструкция по протоколу STE и проверка установки с нуля.

**Tech Stack:** Python 3.12, окружение `bearmind`; pytest; conda/mamba для проверки окружения.

**Spec:** `docs/superpowers/specs/2026-10-30-bearmind-batch-pipeline-design.md` — §2 (цель «убраны хардкоды, костыли, копии кода и найденные баги»), §5.14 (окружение, инструкция по STE), §5.16, приложение А. Подробности с номерами строк — `docs/superpowers/specs/2026-10-30-bearmind-recon.md`. Выполняется после фаз 2–5: инструкция описывает их окна.

## Global Constraints

- Не меняются: алгоритмы CaImAn и логика автоинспекции (метрики, правила, ML-модель), формат `npz` / `mat` / `json` (§2).
- Скрипты MINI3 (`batch_autoinspect_all.py`, `batch_export_all.py`, `chain_run_export_after_autoinspect.py`, `resplit_fof.py`) — запись того, как обработаны данные статьи: переносятся в `history/mini3/` без переписывания (приложение А).
- Ноутбуки `BEARMiND_full_pipeline.ipynb` и `MoviePlay.ipynb` удаляются только после того, как аудит подтвердил: значения `cnmf_dict`, `mc_dict` и параметры автоинспекции из ноутбука есть в `bearmind/defaults.yaml` (§5.16). Исключение — `fr`: в новом коде `fr = FPS`, без деления (раздел 3).
- Порог `pxlthr_distance_boundary` = 15 px не трогать (раздел 3).
- Инструкция для студентов — по-русски, по протоколу STE (§5.14): шаги нумерованы, одно действие в шаге, глагол в начале, предупреждение до шага.
- Коммиты без строк об ИИ, только `git commit -- <пути>`. Ничего не пушить.

## Review Focus

- Импорт старого модуля в новой папке создаёт файлы или окна: после фазы этого нет ни у одного модуля корня. Тест: задача 2 `test_root_modules_import_cleanly`.
- Скрипт из `paper/` или `experiments/` импортирует функцию, которая уехала в `history/`: импорт падает. Тест: задача 3 `test_no_imports_of_moved_names`.
- Студент запускает команду из инструкции, которой нет или которая называется иначе. Тест: задача 6 `test_guide_commands_exist`.
- Окружение с нуля по `environment.yml` и `requirements-pip.txt` не собирается на чистой машине. Проверка: задача 7, шаг 1.
- Код всё ещё полагается на `config.json` в текущей папке (старый `config.py`): после фазы окна и CLI работают из любой папки. Тест: задача 2 `test_cli_from_other_cwd`.

## Решения плана (моё решение, автор может отменить)

- Старые `DoCropAndRewrite`, `DoMotionCorrection`, `DoCNMF`, `ReDoCNMF` и их помощники переезжают в `history/legacy_pipeline.py`. Их импортирует только проверка эквивалентности (задача 18 ядра). Баг `ReDoCNMF` (seeds ставятся после сборки `opts`) не чинится, а описывается в `history/README.md`.
- `export_estimates_data.py`, `batch_export.py`, `batch_autoinspect.py` удаляются. Их работу делают этапы `bearmind`. Перед удалением сверить ключи `npz` / `mat` с экспортом `bearmind`: это делает задача 18 ядра (шаг 4), здесь достаточно сослаться на её результат.
- Старый `config.py` остаётся для `bm_examinator.py` и других модулей корня. Он больше не создаёт `config.json` при импорте и не бросает `UnboundLocalError`.

---

### Task 1: Аудит приложения А

**Files:**
- Create: `docs/superpowers/specs/<дата>-cleanup-audit.md`

- [ ] **Step 1:** Пройти каждый пункт приложения А и каждую строку разведки `2026-10-30-bearmind-recon.md`. Для каждого найти текущее место в коде (`grep`, номер строки на текущем коммите).
- [ ] **Step 2:** Записать таблицу: пункт · где сейчас · статус (`исправлено в <коммит>` / `осталось` / `уходит в history` / `уходит с ноутбуком`) · задача этого плана.
- [ ] **Step 3:** Сверить значения ноутбука с `bearmind/defaults.yaml`: `cnmf_dict`, `mc_dict`, параметры автоинспекции (раздел 3 дизайна). Каждое расхождение — строка таблицы с решением. `fr` — известное намеренное расхождение.
- [ ] **Step 4:** Найти все импорты старых имён из `paper/`, `experiments/`, `ml/`, `tests/` и записать, какие из них затронут задачи 3–5.
- [ ] **Step 5:** Commit: `docs: cleanup audit of appendix A against the current code`.

### Task 2: Побочные эффекты импорта

**Files:**
- Modify: `config.py` (корень), `startup.py`, модули корня с `np.Inf`, `sys.modules['torch'] = None`, `SSQ_GPU` после импорта driada (точные места — из аудита)
- Test: `tests/test_import_side_effects.py`

**Interfaces:**
- Produces: `config.py` создаёт `config.json` только в `write_config` / `update_config`. `read_config` при отсутствии файла возвращает `DEFAULT_CONFIG`. Неизвестный `DATA_PATHWAY` → `ValueError("unknown DATA_PATHWAY: <значение>; expected one of ...")`. `update_config` меняет словарь `CONFIG` на месте (`clear` + `update`), поэтому модули, импортировавшие `CONFIG`, видят новые значения.

- [ ] **Step 1: Тесты:**
```python
def test_root_modules_import_cleanly(tmp_path): ...   # для каждого модуля корня: python -c "import <m>" с cwd=tmp_path -> код 0, tmp_path пуст, tkinter.Tk не создавался (sitecustomize подменяет Tk на падение)
def test_update_config_visible_to_importers(tmp_path): ...
def test_unknown_data_pathway(tmp_path): ...
def test_cli_from_other_cwd(tmp_path): ...            # python -m bearmind status <серия> с cwd=tmp_path: config.json нигде не создан
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. Убрать `os.chdir` к корню репозитория из `bearmind/cli.py`: после этой задачи он не нужен. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `fix: no files, windows or global patches on import of root modules`.

### Task 3: Старый конвейер и скрипты — в `history/`

**Files:**
- Create: `history/README.md`, `history/legacy_pipeline.py`, `history/mini3/` (перенос скриптов MINI3 через `git mv`)
- Modify: `bm_batch_routines.py` (остаются только функции, которые использует код вне `history/`), тест эквивалентности задачи 18 ядра (импорт из `history.legacy_pipeline`)
- Delete: `batch_autoinspect.py`, `batch_export.py`, `export_estimates_data.py`
- Test: `tests/test_history_moves.py`

- [ ] **Step 1: Тесты:**
```python
def test_no_imports_of_moved_names(): ...      # ast-обход всех .py вне history/: нет импортов перенесённых имён из bm_batch_routines
def test_legacy_pipeline_importable(): ...     # import history.legacy_pipeline; DoMotionCorrection, DoCNMF доступны
def test_deleted_scripts_gone(): ...
```
- [ ] **Step 2:** FAIL. **Step 3:** Перенести (`git mv`, чтобы сохранить историю). Мёртвый код удалить: блок-строку в конце `bm_examinator.py`, код после `return` в `bm_batch_routines.py`, скрытые кнопки кропа. В `history/README.md` записать: что здесь лежит, зачем (данные статьи), известный баг `ReDoCNMF`. **Step 4:** PASS + быстрый набор + медленный тест эквивалентности, если есть данные.
- [ ] **Step 5:** Commit: `refactor: move the replaced pipeline and MINI3 scripts to history/`.

### Task 4: Хардкоды и копии кода

**Files:**
- Modify: места из таблицы аудита (абсолютные пути, `n_processes=14`, `max_mem_gb=16`, путь к модели относительно текущей папки, порт Bokeh, срезы имён файлов, списки наборов CC/RFC/FOF), `auto_inspector.py` и модули с копиями чтения таблицы FPS и PNR → imax
- Test: `tests/test_no_hardcodes.py`

**Interfaces:**
- Produces: один источник для каждой копии из приложения А. Параметры автоинспекции — только `bearmind/defaults.yaml` (раздел `autoinspect`). PNR → imax — `bm_qc` (одна функция). Чтение FPS для старых скриптов — `bearmind.timestamps.robust_fps` через прослойку.

- [ ] **Step 1: Тесты:**
```python
def test_no_absolute_user_paths(): ...         # regex по .py вне history/ и paper/: нет "C:\\Users\\", "W:\\DATASET", "D:\\_Projects"
def test_no_name_slicing(): ...                # нет срезов имён из списка разведки (шаблоны вида name[-20:-10], split('/')[6:])
def test_autoinspect_params_single_source(): ...   # литералы ml_threshold / n_iter / pxlthr_distance_boundary вне defaults.yaml и history/ не встречаются
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать по таблице аудита. Широкие `except` с молчаливым продолжением заменить узкими или записью в лог. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `refactor: remove hardcoded paths and copies listed in appendix A`.

### Task 5: Ноутбуки и мусор в репозитории

**Files:**
- Delete: `BEARMiND_full_pipeline.ipynb`, `MoviePlay.ipynb`, отслеживаемые `.DS_Store`, `.ipynb_checkpoints/`
- Modify: `.gitignore` (`.DS_Store`, `.ipynb_checkpoints/`, `temp/`, `__pycache__/`)

- [ ] **Step 1:** Проверить по таблице аудита (задача 1, шаг 3), что все значения ноутбука есть в `defaults.yaml`. Если нет — остановиться и дописать значения отдельным коммитом.
- [ ] **Step 2:** Удалить файлы (`git rm`). Обновить `.gitignore`.
- [ ] **Step 3:** `git ls-files | grep -E "\.DS_Store|ipynb_checkpoints|\.ipynb$"` — пусто.
- [ ] **Step 4:** Commit: `chore: remove notebooks and tracked junk; defaults live in bearmind/defaults.yaml`.

### Task 6: Инструкция для студентов

**Files:**
- Create: `docs/students/README.md` (установка и запуск), `docs/students/series-workflow.md` (работа с серией), картинки окон из отчётов проверок фаз 2–5
- Test: `tests/test_student_guide.py`

Содержание (протокол STE, по-русски):
- **Установка:** что нужно на машине; как создать окружение `bearmind` по `environment.yml` и `requirements-pip.txt`; как проверить установку (`run_bearmind.bat --help`, `bearmind_gui.bat`); что делать, если окружение не найдено.
- **Серия:** раскладка папки (§5.2); куда класть сырые видео; что пайплайн никогда не пишет в `1_Raw/`.
- **Работа:** скан и разбор аномалий (вкладка «Скан»); запуск батча; пороги (вкладка «Пороги»); ручная инспекция или «не нужна»; сводка, исключение и переанализ.
- **Цвета и сообщения:** что значит каждый цвет клетки; три самых частых предупреждения и что с ними делать.
- **Если что-то пошло не так:** батч упал или машину перезагрузили (запустить снова — продолжит); `summary.csv` открыт в Excel; ошибка на этапе (`retry`); где логи.

- [ ] **Step 1: Тесты:**
```python
def test_guide_commands_exist(): ...     # каждая команда run_bearmind.bat <cmd> из инструкции есть в python -m bearmind --help
def test_guide_paths_exist(): ...        # файлы и папки репозитория, упомянутые в инструкции, существуют
def test_guide_sentence_length(): ...    # в нумерованных шагах предложения не длиннее 20 слов (правило STE)
```
- [ ] **Step 2:** FAIL. **Step 3:** Написать инструкцию. **Step 4:** PASS.
- [ ] **Step 5:** Commit: `docs: student guide - install, series workflow, troubleshooting`.

### Task 7: Проверка установки с нуля и финальный прогон

**Files:**
- Create: `docs/superpowers/specs/<дата>-phase6-check.md`

- [ ] **Step 1:** Собрать окружение с нуля во временный префикс: `mamba env create -p <temp>\bm_fresh -f environment.yml`, затем `pip install -r requirements-pip.txt`. Запустить быстрый набор тестов в нём. Удалить временный префикс.
- [ ] **Step 2:** Свежий клон репозитория во временную папку. Запустить `run_bearmind.bat --help` и `python -m bearmind status` на копии series1 из этой папки.
- [ ] **Step 3:** Полный набор тестов, включая `slow`; `realdata` — если есть тестовые серии.
- [ ] **Step 4:** Отчёт: что собрано, время установки, результаты тестов, что не проверено (машина без CUDA, другая модель процессора: обход OpenBLAS `OPENBLAS_CORETYPE` нужен ли там).
- [ ] **Step 5:** Commit: `docs: phase 6 check - fresh install and full test run`.
