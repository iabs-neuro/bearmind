# BEARMiND фаза 2: мониторинг и сводка — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Окно BEARMiND с вкладкой «Мониторинг / сводка»: статус фонового батча, таблица сессий по этапам с цветом и ключевыми числами, логи, запуск и остановка батча и действия сводки без командной строки.

**Architecture:** Модель `bearmind/monitor.py` без Qt читает реестр, пульс и конфиг и строит `SeriesView`. Все записи идут через `bearmind/actions.py`, общий с командной строкой. Окно на PySide6 в `bearmind/gui/` только показывает модель и вызывает действия. Батч запускается отдельным отсоединённым процессом через `bearmind/launcher.py` и переживает закрытие окна.

**Tech Stack:** Python 3.12, PySide6 6.11.2 (уже в окружении `bearmind`), psutil, pandas; тесты pytest с `QT_QPA_PLATFORM=offscreen`, без pytest-qt.

**Spec:** `docs/superpowers/specs/2026-10-30-bearmind-batch-pipeline-design.md` — §4 (путь серии, батч отдельным процессом), §5.9 (массовое «не нужна»), §5.10, §5.11, §5.13 (мониторинг). Опирается на ядро фаз 0–1: `docs/superpowers/plans/2026-10-31-bearmind-batch-core.md` (runner, пульс, реестр, CLI).

## Global Constraints

- Новых пакетов не ставить: окружение `bearmind` закреплено (`env/bearmind-lock.txt`).
- Текст окна — по-русски. Вывод в консоль — только ASCII (`[OK]` / `[FAILED]` / `[WARN]`).
- Батч — отдельный процесс: закрытие окна его не останавливает (§4).
- Окно пишет в реестр и конфиги только через `bearmind/actions.py`, то есть через `registry.update_record` и `config.update_yaml`.
- Одно обновление вкладки на серии из 300 сессий — не дольше 200 мс.
- Окно не добавляет батчу новых ожиданий человека (анти-цель §2).
- Вне этого плана: вкладка «Скан» (фаза 3), окно порогов (фаза 4), кнопка ручной инспекции Bokeh (фаза 5).
- Коммиты без строк об ИИ (`Co-Authored-By` и т. п.), только `git commit -- <пути>`.

## Review Focus

- Пульс от убитого батча (процесс снят диспетчером задач): окно пишет «батч не отвечает», кнопка «Запустить» доступна. Тесты: задача 1 `test_runner_state_dead_when_pid_gone`, задача 6 `test_start_enabled_when_runner_dead`.
- Файл реестра повреждён или читается в момент записи: строка «не читается», окно не падает. Тест: задача 1 `test_unreadable_registry_file_row`.
- Окно закрыли или оно упало во время батча: батч продолжает. Тест: задача 3 `test_start_detached_survives_parent`.
- «Запустить» нажали дважды, или батч уже запущен из командной строки: второго процесса нет, сообщение понятное. Тесты: задача 3 `test_start_detached_refuses_when_alive`, задача 6 `test_start_disabled_while_alive`.
- Путь серии с кириллицей и пробелами доходит до батча целым. Тест: задача 3 `test_cyrillic_and_space_path_passed_intact`.

## Решения плана (моё решение, автор может отменить)

- Счётчики предупреждений и ошибок считают **сессии** с худшим уровнем «предупреждение» или «ошибка», а не отдельные проверки.
- Оценка оставшегося времени не учитывает ожидание человека: этапы, которые ждут порогов или ручной инспекции, в неё не входят. Окно пишет это рядом с оценкой.
- Проверки хранятся в реестре по-английски (поле `message`). Окно строит русский текст подсказки из имени и значения проверки (`bearmind/texts.py`), а при незнакомом имени показывает `message`.
- Цвета клеток: норма `#c8e6c9`, предупреждение `#fff3b0`, ошибка `#ffcdd2`, идёт `#bbdefb`, ждёт человека `#e1bee7`, не начато — без цвета, исключённая сессия — серый текст.

---

### Task 1: Модель мониторинга без Qt

**Files:**
- Create: `bearmind/monitor.py`, `bearmind/texts.py`
- Modify: `bearmind/runner.py` (переименовать `_KEYS` в публичный `STAGE_KEYS`, старое имя оставить синонимом), `bearmind/defaults.yaml` (раздел `monitor`)
- Test: `tests/bearmind/test_monitor.py`

**Interfaces:**
- Consumes: `registry.STAGES`, `State`, `StageRecord`, `load_record`; `runner.STAGE_KEYS`, `runner.COUNT_KEYS`; `config.resolve`; `SeriesPaths.registry_file / heartbeat / lock`. В `run.lock` лежит JSON `{"pid", "create_time", "time"}`, в пульсе — `{"pid", "time", "current", "counts", "by_stage", "status"}`.
- Produces:
```python
@dataclass
class RunnerState:
    alive: bool
    status: str            # "running" | "waiting" | "idle" | "stopped" | "dead" | "none"
    pid: int | None
    age_s: float | None    # возраст пульса
    current: tuple[str, str] | None   # (sid, stage)
    stage_elapsed_s: float | None

@dataclass
class Cell:
    state: str             # значение State или "unreadable"
    level: str             # "ok" | "warning" | "error" | "running" | "waiting" | "none" | "excluded"
    text: str
    tooltip: str

@dataclass
class Row:
    sid: str
    mouse: str
    excluded: bool
    cells: dict[str, Cell]          # ключи = STAGES
    gsig: int | None
    min_corr: float | None
    min_pnr: float | None
    median_snr: float | None
    waiting_for: str | None         # "thresholds" | "manual_inspection" | "crop_finalize" | None
    unreadable: bool = False

@dataclass
class SeriesView:
    series: Path
    scanned: bool
    runner: RunnerState
    rows: list[Row]                 # по mouse, затем по sid
    progress: tuple[int, int]       # (готовых пар сессия-этап, всего пар) по неисключённым сессиям
    eta_s: float | None
    n_warning_sessions: int
    n_error_sessions: int
    waiting: dict[str, int]         # {"thresholds", "manual_inspection", "crop_finalize"}

class RecordCache:                  # перечитывает только JSON с новым mtime_ns или размером
    def records(self, paths) -> list[tuple[str, "SessionRecord | Exception"]]

def runner_state(paths, now: float | None = None) -> RunnerState
def series_view(paths, now: float | None = None, cache: RecordCache | None = None) -> SeriesView
```
- `texts.py`: `STAGE_TITLE` (crop «Кроп», mc «MC», thresholds «Карты порогов», cnmf «CNMF-E», gate «Ручная инспекция», autoinspect «Автоинспекция», export «Экспорт»), `WAITING_TEXT` (thresholds «ждёт порогов», manual_inspection «ждёт ручной инспекции», crop_finalize «ждёт кропа мыши»), `check_text(check: Check) -> str`, `anomaly_text(a: Anomaly) -> str`, `failure_text(code: str, detail: str) -> str` (коды из задачи 2).

Правила:
- `alive`: в `run.lock` PID жив и его `create_time` совпадает с записанным (psutil), **и** пульс моложе `monitor.heartbeat_dead_s` (по умолчанию 120). Тогда `status` берётся из пульса.
- Батч не жив: пульс со статусом `idle` или `stopped` (нормальный выход, `run.lock` уже удалён) → этот статус, `alive=False`. Пульс со статусом `running` или `waiting` → `status="dead"`. Пульса нет → `status="none"`.
- Уровень клетки: DONE → худший уровень проверок этапа (нет проверок → `ok`); FAILED → `error`; WAITING → `waiting`; RUNNING → `running`; PENDING и SKIPPED → `none`; у исключённой сессии все клетки `excluded`.
- Текст клетки: DONE → ключевое число из `STAGE_KEYS`. Для autoinspect это «принято (удалено N %)». WAITING → `WAITING_TEXT`. FAILED → «упал». RUNNING → «идёт ММ:СС».
- Подсказка: по строке `check_text` на каждую проверку; у FAILED — последняя строка ошибки; аномалии с меткой этого этапа. Пример `check_text` для `cnmf_n_components_vs_median` со значением 4.8: «2950 компонент, в 4.8 раза больше медианы серии».
- Оценка времени: для каждой неисключённой сессии и каждого этапа не в DONE и не ждущего человека берётся `n_frames × s_per_frame[stage]`. `s_per_frame` — медиана `(finished − started) / n_frames` по DONE этапам серии, если таких записей не меньше двух. Иначе — `monitor.eta_s_per_frame[stage]` из `defaults.yaml`; значения взять из таблицы времени по этапам в `docs/superpowers/specs/2026-10-31-e2e-report.md` (задача 17 ядра). У идущего этапа вычесть прошедшее время, не меньше нуля.
- Нечитаемый JSON реестра → `Row(sid=<имя файла>, unreadable=True)`, все клетки `level="error"`, `text="?"`, подсказка «файл реестра не читается: <путь>».
- Нет папки реестра или в ней нет записей → `scanned=False`, `rows=[]`.

- [ ] **Step 1: Тесты** (записи реестра собираются в `tmp_path` через `registry.save_record`, время подставляется параметром `now`):
```python
def test_runner_state_none_without_heartbeat(tmp_series): ...   # status == "none", alive is False
def test_runner_state_alive(tmp_series): ...                    # lock с os.getpid() и его create_time, свежий пульс -> alive, current == ("S1", "mc")
def test_runner_state_dead_when_pid_gone(tmp_series): ...       # lock с PID завершившегося subprocess, свежий пульс -> status == "dead"
def test_runner_state_stale_heartbeat(tmp_series): ...          # PID жив, пульсу 300 с -> alive is False, status == "dead"
def test_runner_state_after_normal_exit(tmp_series): ...        # нет run.lock, пульс "idle" -> alive is False, status == "idle"
def test_cell_levels_and_texts(tmp_series): ...                 # DONE + warning-проверка -> level "warning", text = ключевое число; WAITING thresholds -> "ждёт порогов"; FAILED -> "error", ошибка в tooltip
def test_tooltip_russian_for_known_check(tmp_series): ...       # cnmf_n_components_vs_median 4.8 -> "в 4.8 раза" in tooltip
def test_unknown_check_falls_back_to_message(tmp_series): ...
def test_excluded_row_cells(tmp_series): ...                    # все клетки "excluded"; сессия не входит в progress и eta
def test_progress_counts_pairs(tmp_series): ...                 # 2 сессии, у одной 3 этапа DONE -> progress == (3, 14)
def test_eta_from_history_and_defaults(tmp_series): ...         # 2 DONE mc с известной длительностью -> медиана; без истории -> defaults; ждущие человека этапы не входят
def test_unreadable_registry_file_row(tmp_series): ...          # мусор в <sid>.json -> Row.unreadable, остальные строки на месте
def test_cache_rereads_only_changed(tmp_series, monkeypatch): ...  # второй series_view без изменений: 0 чтений JSON; изменили одну запись: 1 чтение
def test_rows_sorted_by_mouse_then_sid(tmp_series): ...
def test_not_scanned_series(tmp_path): ...                      # scanned is False, rows == []
```
- [ ] **Step 2:** Запустить `pytest tests/bearmind/test_monitor.py -p no:cacheprovider`; ожидается FAIL (`bearmind.monitor` нет).
- [ ] **Step 3:** Реализовать `monitor.py` и `texts.py` по правилам выше; в `runner.py` переименовать `_KEYS` в `STAGE_KEYS`, оставить `_KEYS = STAGE_KEYS`.
- [ ] **Step 4:** Тест-файл PASS; весь быстрый набор PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): monitor model - runner state, per-stage cells, progress, ETA`.

### Task 2: Общий модуль действий для окна и командной строки

**Files:**
- Create: `bearmind/actions.py`
- Modify: `bearmind/cli.py` (команды thresholds, gate, decide, rerun, retry вызывают `actions`; тексты и коды возврата CLI не меняются)
- Test: `tests/bearmind/test_actions.py`; `tests/bearmind/test_cli.py` проходит без правок

**Interfaces:**
- Consumes: `registry.update_record`, `config.update_yaml`, `runner.reset_from`, `stages.gate.decide`, `scan.evaluate_excluded`, правила проверки порогов из `cli.py` (переносятся сюда).
- Produces:
```python
@dataclass
class ActionResult:
    ok: list[str]                          # sid, к которым действие применено
    failed: dict[str, tuple[str, str]]     # sid -> (code, detail)
    warnings: list[tuple[str, str]]        # (sid, code)

def current_user() -> str                  # имя пользователя Windows (getpass.getuser)
def set_excluded(paths, sids: list[str], excluded: bool, who: str) -> ActionResult
def accept_mismatch(paths, sids: list[str], who: str) -> ActionResult
def gate_not_needed(paths, sids: list[str] | None, who: str) -> ActionResult   # None = все, кто ждёт ручной инспекции
def gate_cleaned(paths, sid: str, who: str) -> ActionResult
def rerun_from(paths, sids: list[str], stage: str, who: str) -> ActionResult
def retry_failed(paths, sids: list[str] | None, who: str) -> ActionResult     # None = все с FAILED-этапом
def set_thresholds(paths, sid: str, gsig: int, min_corr: float, min_pnr: float) -> ActionResult
def accept_suggested(paths, sids: list[str] | None) -> ActionResult           # None = все, кто ждёт порогов
def export_summary(paths, dest: Path) -> Path
```
- Коды `failed`: `no_suggestion`, `invalid_suggestion`, `manual_result_missing`, `no_failed_stage`. Коды `warnings`: `excluded_session`, `no_mismatch_error`. Неизвестный sid или этап → `ValueError` **до любой записи**. Неверный порог в `set_thresholds` → `ValueError` с текстом CLI («--min-corr must be a number above 0 and at most 1, got 92.0»).

- [ ] **Step 1: Тесты** на крошечной серии (этапы-заглушки как в `test_cli.py`):
```python
def test_set_excluded_and_back(series): ...         # excluded True -> False, Decision("exclude", "true"/"false", who)
def test_unknown_sid_writes_nothing(series): ...    # ["S1", "NOPE"] -> ValueError; S1 не изменилась
def test_gate_not_needed_all_waiting(series): ...   # None -> только сессии, у которых gate WAITING
def test_gate_cleaned_without_file(series): ...     # failed == {"S1": ("manual_result_missing", <путь>)}, ничего не записано
def test_rerun_from(series): ...
def test_retry_failed(series): ...                  # сессия без FAILED -> failed code "no_failed_stage"
def test_set_thresholds_invalid(series): ...        # 92 / nan / 0 -> ValueError, конфиг не создан
def test_accept_suggested_mixed(series): ...        # одна с предложением, одна без -> ok и failed "no_suggestion"
def test_accept_mismatch_warns_without_error(series): ...
def test_export_summary(series, tmp_path): ...      # CSV с колонкой sid
```
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Перенести логику записи из `cli.py` в `actions.py`; CLI печатает свои прежние английские тексты по кодам `ActionResult`.
- [ ] **Step 4:** `test_actions.py` и `test_cli.py` PASS без правок `test_cli.py`; быстрый набор PASS.
- [ ] **Step 5:** Commit: `refactor(bearmind): actions module shared by CLI and GUI`.

### Task 3: Запуск батча в фоне и остановка после текущего этапа

**Files:**
- Create: `bearmind/launcher.py`
- Modify: `bearmind/paths.py` (`stop_request` → `_bearmind/stop.request`), `bearmind/runner.py`, `bearmind/cli.py` (команда `stop SERIES`)
- Test: `tests/bearmind/test_launcher.py`, дополнить `tests/bearmind/test_runner.py`

**Interfaces:**
- Consumes: `monitor.runner_state`, `runner.run`.
- Produces:
```python
def runner_command(series_root: Path) -> list[str]     # [<python.exe окружения>, "-m", "bearmind", "run", str(series_root)]
def start_detached(series_root: Path, cmd: list[str] | None = None) -> int   # pid; RuntimeError("батч уже запущен: PID N"), если runner_state(...).alive
def request_stop(paths) -> None
```
- `run()`: в начале удаляет старый `stop.request`. Перед каждым этапом и в цикле ожидания проверяет файл. Если файл есть, `run()` возвращается, не начиная новый этап: `result["stopped"] = True`, пульс `status="stopped"`, замок снят.
- `runner_command`: если `sys.executable` — `pythonw.exe`, взять `python.exe` из той же папки.
- `start_detached`: `subprocess.Popen(cmd, cwd=<корень репо>, stdin=DEVNULL, stdout/stderr -> добавлять в _bearmind/logs/runner_console.log, close_fds=True)`. Окружение: `PYTHONPATH=<корень репо>`, `OPENBLAS_CORETYPE` по умолчанию `Sandybridge`. Флаги Windows: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB`; при `OSError` (job не разрешает breakaway) — повторить без `CREATE_BREAKAWAY_FROM_JOB`.
- CLI `stop SERIES` печатает `[OK] stop requested; the runner stops before its next stage`, код 0.

- [ ] **Step 1: Тесты:**
```python
def test_stop_request_stops_between_stages(series): ...    # заглушка этапа crop создаёт stop.request -> run(once=True) вернулся, mc и дальше PENDING, result["stopped"], пульс "stopped"
def test_stale_stop_request_removed_at_start(series): ...
def test_runner_command_uses_python_not_pythonw(monkeypatch): ...
def test_start_detached_survives_parent(tmp_path): ...      # вспомогательный процесс вызывает start_detached(cmd=[python, -c, "sleep 2; write marker"]) и сразу выходит; marker появляется
def test_start_detached_refuses_when_alive(tmp_path, monkeypatch): ...   # runner_state alive -> RuntimeError, Popen не вызван
def test_cyrillic_and_space_path_passed_intact(tmp_path): ...            # серия "серия тест & co"; cmd пишет sys.argv в файл; путь совпал
def test_cli_stop(series): ...
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): start the runner detached; stop it before the next stage`.

### Task 4: Каркас окна

**Files:**
- Create: `bearmind/gui/__init__.py`, `bearmind/gui/app.py`, `bearmind_gui.bat`
- Modify: `bearmind/cli.py` (команда `gui [SERIES]`, ленивый импорт `bearmind.gui.app`)
- Test: `tests/bearmind/gui/conftest.py` (одна `QApplication` на сессию тестов; `QT_QPA_PLATFORM=offscreen` ставится до импорта PySide6), `tests/bearmind/gui/test_app.py`

**Interfaces:**
- Consumes: `config.machine_config`, файл машины `<BEARMIND_HOME>/machine.yaml` через `config.update_yaml`.
- Produces:
```python
class MainWindow(QMainWindow):
    tabs: QTabWidget                     # сейчас одна вкладка «Мониторинг / сводка»; фазы 3–4 добавят «Скан» и «Пороги»
    def __init__(self, series: Path | None = None, message=None): ...   # message(title, text) — для тестов, по умолчанию QMessageBox
    def open_series(self, series: Path) -> bool
def main(argv: list[str] | None = None) -> int
```
- Меню «Серия»: «Открыть серию…» (`QFileDialog.getExistingDirectory`) и «Недавние» (`gui.recent_series` в `machine.yaml`, до 10 путей, новые сверху).
- Заголовок окна: `BEARMiND — <имя папки серии>`.
- Несуществующая папка → сообщение «Папка серии не найдена: <путь>»; окно остаётся рабочим.
- Необработанное исключение в слоте → `sys.excepthook` пишет трейсбек в `<BEARMIND_HOME>/gui.log` и показывает первую строку; окно не закрывается.
- `bearmind_gui.bat` устроен как `run_bearmind.bat` (тот же поиск окружения), но запускает `start "" "<env>\pythonw.exe" -m bearmind gui %*`, без консольного окна.

- [ ] **Step 1: Тесты:**
```python
def test_window_opens_with_series(qapp, tmp_series): ...       # заголовок содержит имя папки; вкладка "Мониторинг / сводка"
def test_recent_series_saved(qapp, tmp_series): ...            # machine.yaml gui.recent_series[0] == путь; повтор не дублирует
def test_missing_folder_message(qapp, tmp_path): ...           # message вызван с "Папка серии не найдена"; open_series -> False
def test_excepthook_logs_and_keeps_window(qapp, tmp_series): ...
def test_cli_gui_help(): ...                                   # subprocess "python -m bearmind gui --help" -> код 0
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): GUI window skeleton with series chooser`.

### Task 5: Вкладка мониторинга: шапка и таблица

**Files:**
- Create: `bearmind/gui/monitor_tab.py`
- Modify: `bearmind/gui/app.py` (вкладка подключается при открытии серии), `bearmind/defaults.yaml` (`monitor.refresh_ms: 2000`)
- Test: `tests/bearmind/gui/test_monitor_tab.py`

**Interfaces:**
- Consumes: `monitor.series_view`, `monitor.RecordCache`, `texts.STAGE_TITLE`.
- Produces:
```python
class MonitorTab(QWidget):
    header: QLabel
    table: QTreeView
    model: QStandardItemModel
    def __init__(self, series: Path, parent=None): ...
    def refresh(self) -> None
    def selected_sids(self) -> list[str]
```
- Шапка, если батч жив: «Работает, обновлено 5 с назад · NOF_H03_3D / MC (идёт 03:12) · готово 14 из 42 этапов · осталось ≈ 1 ч 20 мин (без ожидания человека) · сессий с предупреждениями 2 · с ошибками 0 · ждут порогов 3 · ждут ручной инспекции 1». Без пульса, а также при `idle` или `stopped`: «Батч не запущен» (при `stopped` — «Батч остановлен»). Если `status == "dead"`: «Батч не отвечает (последний пульс 12 мин назад)», красным.
- Таблица: верхний уровень — мыши («мышь H03 · 2 сессии»), под ними сессии. Столбцы: «Сессия», семь этапов по `STAGE_TITLE`, «gSig», «min_corr», «min_pnr», «Медиана SNR», «Ждёт». В клетке — `Cell.text`, фон по уровню (цвета в «Решениях плана»), подсказка `Cell.tooltip`.
- `QTimer` с интервалом `monitor.refresh_ms`. При обновлении раскрытие, выделение и прокрутка сохраняются. Дерево перестраивается, только если изменился набор sid; иначе меняются тексты и цвета существующих элементов.
- Серия не сканирована → вместо таблицы текст «Серия не сканирована. Запустите скан: run_bearmind.bat scan <путь>».
- Обновление дольше 1 с → одно предупреждение в лог `bearmind`.

- [ ] **Step 1: Тесты** (серия: 2 мыши, 3 сессии, пульс в `tmp_path`):
```python
def test_tree_groups_by_mouse(qapp, series3): ...         # 2 элемента верхнего уровня; у первого 2 дочерних
def test_cell_color_text_tooltip(qapp, series3): ...      # совпадают с monitor.series_view для той же клетки
def test_header_states(qapp, series3): ...                # без пульса "Батч не запущен"; мёртвый PID "не отвечает"; живой "Работает"
def test_refresh_keeps_selection(qapp, series3): ...      # выделили S2, изменили запись S1, refresh -> выделена S2
def test_not_scanned_message(qapp, tmp_path): ...
def test_refresh_fast_on_300_sessions(qapp, tmp_path): ...  # 300 синтетических записей: второй refresh < 200 мс
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): monitoring tab - status header and per-stage table`.

### Task 6: Кнопки сводки, запуск и остановка, логи

**Files:**
- Create: `bearmind/gui/logs.py`
- Modify: `bearmind/gui/monitor_tab.py`
- Test: `tests/bearmind/gui/test_actions_ui.py`

**Interfaces:**
- Consumes: `actions.*`, `launcher.start_detached`, `launcher.request_stop`, `monitor.runner_state`, `texts.failure_text`.
- Produces:
```python
class MonitorTab(QWidget):   # дополнение задачи 5
    def __init__(self, series: Path, parent=None, confirm=None, choose_stage=None, choose_file=None): ...
    # confirm(text) -> bool; choose_stage() -> str | None; choose_file(default: Path) -> Path | None — подменяются в тестах
class LogView(QWidget):
    def show_log(self, path: Path) -> None
    def poll(self) -> None                 # дочитывает только новые байты; файл уменьшился -> читать с начала
```
- Панель кнопок над таблицей: «Запустить батч» (доступна, если батч не жив), «Остановить после текущего этапа» (доступна, если жив), «Исключить», «Вернуть», «Принять несовпадение кадров», «Переанализировать с этапа…», «Повторить упавшие», «Ручная инспекция не нужна», «Не нужна для всех ожидающих», «Выгрузить сводку CSV…».
- Каждое пишущее действие сначала вызывает `confirm` с текстом: какие сессии и что изменится. Пример: «Сессии NOF_H01_1D, NOF_H03_3D будут пересчитаны с этапа CNMF-E. Старые результаты останутся, пока новые не готовы.» Отказ → ничего не пишется.
- Результат: строка в статусной строке окна; `failed` и `warnings` — окном сообщения с текстами `texts.failure_text`. `who = actions.current_user()`.
- Логи снизу (`QSplitter`): последние 500 строк `series.log`; выбор сессии в таблице переключает на её лог; опрос по таймеру таблицы; кнопка «Открыть папку логов» (`QDesktopServices.openUrl`).
- CSV по умолчанию: `<папка серии>/<имя серии>_summary.csv`.

- [ ] **Step 1: Тесты:**
```python
def test_exclude_selected(qapp, series3): ...              # confirm -> True; S1 excluded в реестре; таблица обновилась
def test_confirm_no_writes_nothing(qapp, series3): ...
def test_rerun_with_stage_choice(qapp, series3): ...       # choose_stage -> "cnmf"; cnmf..export PENDING
def test_not_needed_all_waiting(qapp, series3): ...
def test_failures_shown_in_russian(qapp, series3): ...     # retry без FAILED -> сообщение с текстом failure_text("no_failed_stage", ...)
def test_start_disabled_while_alive(qapp, series3, monkeypatch): ...
def test_start_enabled_when_runner_dead(qapp, series3): ...
def test_start_calls_launcher(qapp, series3, monkeypatch): ...   # start_detached вызван с путём серии
def test_stop_writes_request(qapp, series3): ...
def test_log_tail_and_switch(qapp, series3): ...           # 600 строк -> видно последние 500; выбор S2 -> лог S2
def test_log_truncated_rereads(qapp, series3): ...
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): summary actions, run/stop buttons and log view`.

### Task 7: Проверка на реальных сериях

**Files:**
- Create: `docs/superpowers/specs/<дата>-phase2-check.md` и папка `docs/superpowers/specs/phase2-check/` (снимки окна, каждый < 300 КБ)

- [ ] **Step 1:** Открыть окно offscreen на `H:\Projects\_bearmind_test\series1` (результат задачи 17 ядра) и на `H:\Projects\_bearmind_test\scan_nof` (112 сессий после скана). Сохранить снимок вкладки (`QWidget.grab().save`).
- [ ] **Step 2:** Замерить медиану 20 обновлений на `scan_nof`. Ожидается < 200 мс.
- [ ] **Step 3:** На `series1` нажать «Запустить батч» через `MonitorTab`. Батч должен стать «живым», а после выхода — «не запущен». Закрыть окно во время работы батча: процесс батча жив (`psutil`). Затем «Остановить после текущего этапа».
- [ ] **Step 4:** Записать отчёт: снимки, время обновления, что проверено руками и что нет.
- [ ] **Step 5:** Commit: `docs: phase 2 check on real series`.
