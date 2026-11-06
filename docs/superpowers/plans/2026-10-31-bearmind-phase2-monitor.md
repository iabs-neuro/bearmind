# BEARMiND фаза 2: мониторинг и сводка — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Окно BEARMiND с вкладкой Monitor из шести частей (макет, одобренный автором 2026-11-06): шапка батча, прогресс по этапам, «ждут вас», таблица сессий, панель предупреждений, лог. Запуск и остановка батча и действия сводки без командной строки. Весь текст окна на английском.

**Architecture:** Модель `bearmind/monitor.py` без Qt читает реестр, пульс и конфиг и строит `SeriesView`. Все записи идут через `bearmind/actions.py`, общий с командной строкой. Окно на PySide6 в `bearmind/gui/` только показывает модель и вызывает действия. Батч запускается отдельным отсоединённым процессом через `bearmind/launcher.py` и переживает закрытие окна.

**Tech Stack:** Python 3.12, PySide6 6.11.2 (уже в окружении `bearmind`), psutil, pandas; тесты pytest с `QT_QPA_PLATFORM=offscreen`, без pytest-qt.

**Spec:** `docs/superpowers/specs/2026-10-30-bearmind-batch-pipeline-design.md` — §4 (путь серии, батч отдельным процессом), §5.9 (массовое «не нужна»), §5.10, §5.11, §5.13 (мониторинг). **Макет (источник истины для содержимого окна):** `docs/superpowers/plans/2026-11-06-phase2-monitor-explainer.html`. Опирается на ядро фаз 0–1: `docs/superpowers/plans/2026-10-31-bearmind-batch-core.md` (runner, пульс, реестр, CLI) в его финальном виде (`bearmind/runner.py`, `registry.py`, `cli.py`, `checks.py`).

## Changes on 2026-11-06

План написан 2026-10-31, до разбора автором и до последних исправлений ядра. Что изменено.

По решениям автора:
- Весь текст окна по-английски. Слой русских текстов `bearmind/texts.py` убран; названия этапов (`STAGE_TITLE`) и маленькая таблица английских подсказок (`HINTS`) живут в `monitor.py`, тексты отказов действий (`failure_text`) — в `actions.py`. Консоль по-прежнему ASCII. Сам документ плана остаётся по-русски.
- Ограничение «новых пакетов не ставить» снято: можно ставить в окружение `bearmind`, если нет конфликта с CaImAn; каждый добавленный пакет записать в `env/bearmind-lock.txt`.
- Вкладка Monitor ровно из шести частей макета. Убраны столбцы gSig / min_corr / min_pnr / «Медиана SNR» / «Ждёт», группировка по мышам деревом и длинная панель кнопок. Таблица плоская (сессия x семь этапов), отсортирована по мыши и sid.
- Добавлены прогресс-полосы по этапам, кнопки «ждут вас» и панель предупреждений ровно шести видов с кнопкой Retry. Счётчик «сессий с предупреждениями» в шапке заменён заголовком панели «Warnings — N sessions».
- Остальные действия сводки (Exclude, Include, Accept frame mismatch, Rerun from, Retry all failed, Inspection not needed, Inspection not needed for all waiting, Export summary CSV) сохранены, но вынесены из панели кнопок в меню Actions и контекстное меню таблицы: на макете для них нет места, а отказываться от них автор не просил.
- Вкладки Scan, Thresholds, Inspection в этом плане — заглушки (фазы 3, 4, 5). Названия вкладок на английском: Monitor, Scan, Thresholds, Inspection. Планы фаз 3–6 всё ещё говорят «Скан», «Пороги» и русский текст окна; их нужно привести к этим названиям отдельно.
- Проверка на реальных данных (задача 7) переведена на эксперимент LT1 и не запускает MC и CNMF-E.

Исправления по реальному коду ядра:
1. Пульс (`runner._Heartbeat`): поля `pid, time, current, counts, by_stage, status, unschedulable`. `time` — локальная ISO-строка без пояса, не число. `current` — словарь `{"sid", "stage"}` или `null`, не пара; времени старта этапа в пульсе нет, поэтому длительность идущего этапа берётся из `StageRecord.started` реестра. Статусы пульса только `running`, `waiting`, `idle`: статуса `stopped` в ядре нет, а `RunnerState.status` теперь `running | waiting | stopped | dead` (`stopped` — пульс `idle` или пульса нет; прежнее `none` слито со `stopped`).
2. Остановка по запросу в ядре не существует. Её добавляет задача 3. Нормальный выход (в том числе по запросу) оставляет пульс `idle` через `_Run.close()`; прежнее «пульс `stopped`» убрано.
3. `run.lock` пишет `{"pid", "create_time", "time"}` (`runner._acquire_lock`), на нормальном выходе файл удаляется; у убитого процесса файл остаётся, а ОС-блокировка снимается. Тест «мёртвый PID» опирается на это.
4. Порог устаревания пульса `HEARTBEAT_MAX_AGE_S = 120` уже есть в `cli.py`; `monitor.heartbeat_dead_s` по умолчанию тоже 120 (значение из `defaults.yaml`, `cli.py` оставить как есть).
5. Реестр: читать каталог надо через `registry.read_all(paths) -> (records, broken)`, а не собственным обходом. Битый файл приходит в `broken` как `(Path, причина)`. Для кэша по mtime добавляется публичная `registry.read_file(path)` (синоним закрытой `registry._read`).
6. Ключевые числа этапов уже собраны в `runner._KEYS` (в плане было так же). Добавляется публичное имя `STAGE_KEYS`; `_crop_box(rec)` даёт строку `WxH@x,y` или `uncropped`, окно берёт часть до `@`. Метрика ворот `gate` принимает значения `cleaned`, `not_needed`, `disabled`.
7. Ожидание человека в реестре: порогов — `cnmf` в WAITING с причиной `thresholds`; ручной инспекции — `gate` в WAITING с причиной `manual_inspection`; `crop_finalize` — внутреннее ожидание ядра, это не «ждёт вас». Причина берётся через `runner._what(st)`; добавляется публичное имя `runner.wait_reason`. Случай «пороги предложены, а CNMF-E ещё PENDING после `run --until thresholds`» ядро обрабатывает в `cli._suggested_unset`; логика переносится в `actions` и в модель.
8. Имена команд CLI: `scan, run, status, thresholds (accept-suggested | set), gate, decide (exclude | include | accept-mismatch), rerun, retry`. Команды `stop` и `gui` добавляются этим планом. Эквивалент кнопки Retry — `retry SERIES SID...` (`cli.cmd_retry`: `runner.reset_from` с первого FAILED-этапа). Массовое «не нужна» — `gate SERIES --all-waiting not-needed`, у него есть `--force` для уже пройденных ворот (в прежнем плане этого не было: добавлен код отказа `already_passed`).
9. Коды `Anomaly`, которые показывает панель: `stage_crashed` (с `data.cause` = `no_progress` для зависания по CPU, `pool_worker_died`, иначе None), `frames_ts_mismatch`, `fps_outlier`. Имена проверок в `checks.py`: `mc_max_shift_px`, `cnmf_n_components_vs_median`, `auto_deleted_frac`, `export_n_neurons`. Пороги в `defaults.yaml` (раздел `checks`): 10 / 30, 2.0 / 3.0, 0.8 / 0.95, 30 / 5 (меньше — хуже). Их пока не трогаем, калибровка на LT1.
10. Сообщения проверок (`Check.message`) уже английские и содержат пороги, например `max shift 18.0 px (warn > 10, error > 30)`. Окно показывает их как есть.
11. Логи: `series.log` и `<sid>.log` в `_bearmind/logs/`; формат строки `%(asctime)s %(levelname)s %(message)s`, окно показывает строки как записаны.
12. Валидаторы порогов (`_gsig_error`, `_min_corr_error`, `_min_pnr_error`), `_gate_waits`, `_suggested_unset`, `_write_thresholds`, `_warn_excluded` сейчас закрытые функции в `cli.py`; задача 2 переносит их в `actions`.
13. `cli.main` делает `Path(args.series).resolve()` и берёт `args.series` у каждой команды: у `gui` серия необязательна, поэтому `gui` разбирается до этой строки.
14. Таблица времени этапов для оценки: файл называется `docs/superpowers/specs/2026-11-01-e2e-report.md` (раньше стояло «e2e-report.md» без даты), задача 17 ядра.
15. В окружении `bearmind` PySide6 6.11.2 (`env/bearmind-lock.txt`).

## Global Constraints

- Пакеты можно ставить в окружение `bearmind`, если они не конфликтуют с CaImAn (после установки проверить `python -c "import caiman"` и `pip check`). Каждый добавленный пакет записать в `env/bearmind-lock.txt`.
- Текст окна — по-английски. Вывод в консоль — только ASCII (`[OK]` / `[FAILED]` / `[WARN]`).
- Батч — отдельный процесс: закрытие окна его не останавливает (§4). Сбой одной сессии батч не останавливает (изоляция этапов и один автоповтор уже в ядре); окно только показывает его.
- Окно пишет в реестр и конфиги только через `bearmind/actions.py`, то есть через `registry.update_record` и `config.update_yaml`.
- Одно обновление вкладки на серии из 300 сессий — не дольше 200 мс.
- Окно не добавляет батчу новых ожиданий человека (анти-цель §2).
- Вне этого плана: вкладка Scan (фаза 3), окно Thresholds (фаза 4), кнопка ручной инспекции Bokeh (фаза 5): здесь заглушки.
- Данные `H:\Projects\LT1\CalciumData\1_Raw` только для чтения: ни этот план, ни его проверки туда не пишут. Тяжёлых запусков MC и CNMF-E нет (правило автора).
- Коммиты без строк об ИИ (`Co-Authored-By` и т. п.), только `git commit -- <пути>`.

## Review Focus

- Пульс от убитого батча (процесс снят диспетчером задач): окно пишет «Not responding», кнопка Start доступна. Тесты: задача 1 `test_runner_state_dead_when_pid_gone`, задача 5 `test_start_enabled_when_runner_dead`.
- Файл реестра повреждён или читается в момент записи: строка с `?` и красным фоном, окно не падает. Тест: задача 1 `test_unreadable_registry_file_row`.
- Окно закрыли или оно упало во время батча: батч продолжает. Тест: задача 3 `test_start_detached_survives_parent`.
- Start нажали дважды, или батч уже запущен из командной строки: второго процесса нет, сообщение понятное. Тесты: задача 3 `test_start_detached_refuses_when_alive`, задача 5 `test_start_disabled_while_alive`.
- Путь серии с кириллицей и пробелами доходит до батча целым. Тест: задача 3 `test_cyrillic_and_space_path_passed_intact`.
- Одна упавшая сессия не меняет батч: окно показывает её красной, в панели одна ошибка с Retry, остальные сессии идут. Тесты: задача 1 `test_failed_session_does_not_hide_others`, задача 6 `test_retry_button_resets_failed_stage`.
- В панели ровно шесть видов; другие проверки красят клетку, но в панель не попадают. Тест: задача 1 `test_warning_panel_has_only_six_kinds`.
- Оценка времени не включает ожидание человека. Тест: задача 1 `test_eta_excludes_human_waits`.

## Решения плана (моё решение, автор может отменить)

- «Warnings — N sessions» считает **сессии** с хотя бы одной записью в панели.
- Оценка оставшегося времени не включает этапы, которые ждут человека, и все более поздние этапы тех же сессий (они не могут пойти, пока человек не решит). Рядом с оценкой окно пишет «time waiting for you not counted», как на макете.
- Запись панели одна на пару сессия-этап: сообщения проверок одного этапа склеиваются через « ; ». Исправленные автоматически аномалии (`auto_fixed`, например принятое несовпадение кадров) в панель не попадают.
- Аномалии скана (`frames_ts_mismatch`, `fps_outlier`) показываются и для исключённых сессий: именно они объясняют исключение. Остальные виды для исключённых сессий не показываются.
- Ждущие человека: порогов — `cnmf` WAITING (причина `thresholds`) или «пороги предложены, но не заданы»; ручной инспекции — `gate` WAITING. Ожидание `crop_finalize` и сессии, которые ядро не может поставить в расписание (`heartbeat.unschedulable`), красят клетку в её цвет, но не «ждут вас».
- Нечитаемый файл реестра: красная строка в таблице с подсказкой; в панель не попадает (седьмого вида нет). Так же для «не может быть запланирована».
- Retry только сбрасывает этап (`runner.reset_from`). Если батч не запущен, окно пишет в статусной строке «Start the batch to run it» и сам его не запускает.
- Цвета клеток: норма `#c8e6c9`, предупреждение `#fff3b0`, ошибка и упал `#ffcdd2`, идёт `#bbdefb`, ждёт вас `#e1bee7`, не начато и пропущено — без цвета, исключённая сессия — серый текст. Цвета полос прогресса те же, что у клеток (макет: зелёный, синий, фиолетовый, красный).
- Тексты клеток по макету: Crop `512×480`, MC `3.2 px`, Thresholds `gSig 4`, CNMF-E `231 cells`, Inspection `not needed` / `cleaned`, Autoinspect `128 kept`, Export `121 neurons`; идёт — `running`, ждёт — `waits`, упал — `failed`, не начато и пропущено — `–`. Пороги берутся из разрешённой конфигурации сессии (`config.resolve`), с кэшем по mtime трёх yaml-файлов.

---

### Task 1: Модель мониторинга без Qt

**Files:**
- Create: `bearmind/monitor.py`
- Modify: `bearmind/runner.py` (публичные `STAGE_KEYS` и `wait_reason`, старые имена `_KEYS`, `_what` остаются синонимами), `bearmind/registry.py` (`read_file`), `bearmind/defaults.yaml` (раздел `monitor`)
- Test: `tests/bearmind/test_monitor.py`

**Interfaces:**
- Consumes: `registry.STAGES`, `State`, `StageRecord`, `Check`, `Anomaly`, `read_all`, `read_file`; `runner.STAGE_KEYS`, `runner.COUNT_KEYS`, `runner.wait_reason`; `config.resolve`; `SeriesPaths.registry_file / heartbeat / lock / session_config / mouse_config / series_config`. В `run.lock` лежит JSON `{"pid", "create_time", "time"}`, в пульсе — `{"pid", "time", "current", "counts", "by_stage", "status", "unschedulable"}`.
- Produces:
```python
STAGE_TITLE: dict[str, str]   # crop "Crop", mc "MC", thresholds "Thresholds", cnmf "CNMF-E",
                              # gate "Inspection", autoinspect "Autoinspect", export "Export"
HINTS: dict[str, str]         # kind or cause -> one English sentence appended to a panel entry

@dataclass
class RunnerState:
    alive: bool
    status: str            # "running" | "waiting" | "stopped" | "dead"
    pid: int | None
    age_s: float | None    # возраст пульса
    current: tuple[str, str] | None   # (sid, stage)
    stage_elapsed_s: float | None     # now - StageRecord.started текущего этапа

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
    unreadable: bool = False

@dataclass
class StageBar:
    done: int
    running: int
    waiting: int           # ждут человека (только thresholds и gate)
    failed: int
    pending: int           # не начато, пропущено, ждёт ядро (crop_finalize)
    total: int             # неисключённые сессии

@dataclass
class WarningItem:
    level: str             # "error" | "warning"
    kind: str              # "stage_failed" | "scan_data" | "mc_shift" | "cnmf_count" | "autoinspect_deleted" | "export_count"
    sid: str
    stage: str             # ключ из STAGES; для scan_data "crop"-колонка не используется: stage = ""
    text: str              # английское сообщение + подсказка из HINTS
    retry: bool            # True только у kind == "stage_failed"

@dataclass
class SeriesView:
    series: Path
    scanned: bool
    runner: RunnerState
    rows: list[Row]                 # по mouse, затем по sid
    bars: dict[str, StageBar]       # ключи = STAGES
    eta_s: float | None             # None: нечего оценивать или нет данных
    waiting: dict[str, int]         # {"thresholds": N, "manual_inspection": M}
    warnings: list[WarningItem]     # сначала ошибки, затем предупреждения; внутри по sid и порядку этапов
    n_warning_sessions: int
    unreadable_files: list[Path]

class RecordCache:                  # перечитывает только JSON с новым mtime_ns или размером; держит и разрешённые пороги
    def records(self, paths) -> tuple[list[SessionRecord], list[tuple[Path, str]]]
    def gsig(self, paths, rec) -> int | None

def runner_state(paths, now: datetime | None = None) -> RunnerState
def series_view(paths, now: datetime | None = None, cache: RecordCache | None = None) -> SeriesView
```

Правила:
- `alive`: в `run.lock` PID жив и его `create_time` совпадает с записанным (psutil, допуск 1 с), **и** пульс моложе `monitor.heartbeat_dead_s` (по умолчанию 120). Тогда `status` берётся из пульса (`running` или `waiting`).
- Батч не жив: пульс со статусом `idle` или пульса нет → `status="stopped"`. Пульс со статусом `running` или `waiting` → `status="dead"` (окно пишет «Not responding»).
- Уровень клетки: DONE → худший уровень проверок этапа (нет проверок → `ok`); FAILED → `error`; WAITING → `waiting`; RUNNING → `running`; PENDING и SKIPPED → `none`; у исключённой сессии все клетки `excluded`.
- Текст клетки по «Решениям плана». Для DONE берётся ключевое число из `STAGE_KEYS`; `gate` с метрикой `disabled` показывает `off`.
- Подсказка клетки: по строке `Check.message` на каждую проверку этапа; у FAILED — последняя строка ошибки; аномалии с меткой этого этапа.
- Панель предупреждений, ровно шесть видов (остальные проверки и аномалии только красят клетку):
  1. `stage_failed`: этап FAILED (текст: последняя строка `error` + подсказка; для аномалии `stage_crashed` с `data.cause == "no_progress"` — «stalled: no CPU progress», с `pool_worker_died` — «a CaImAn worker died»). Уровень `error`, `retry=True`.
  2. `scan_data`: аномалии `frames_ts_mismatch` и `fps_outlier` (их `level` и `message`), кроме `auto_fixed`.
  3. `mc_shift`: проверка `mc_max_shift_px` не `ok`.
  4. `cnmf_count`: проверка `cnmf_n_components_vs_median` не `ok`.
  5. `autoinspect_deleted`: проверка `auto_deleted_frac` не `ok`.
  6. `export_count`: проверка `export_n_neurons` не `ok`.
  Подсказки `HINTS`: `mc_shift` — «Check the MC preview.», `stage_failed` — «Other sessions continue.».
- Оценка времени: для каждой неисключённой сессии и каждого этапа в PENDING или RUNNING, кроме хвоста за ожиданием человека, берётся `n_frames x s_per_frame[stage]`. `s_per_frame` — медиана `(finished - started) / n_frames` по DONE этапам серии, если таких записей не меньше двух, иначе `monitor.eta_s_per_frame[stage]` из `defaults.yaml` (числа из таблиц времени `docs/superpowers/specs/2026-11-01-e2e-report.md`; если для этапа там нет числа, взять медиану по `series_short`, а если и её нет, этап не оценивается и в `eta_s` не входит). У идущего этапа вычесть прошедшее время, не меньше нуля. FAILED и SKIPPED этапы не оцениваются.
- Нечитаемый JSON реестра (`broken` из `read_all`) → `Row(sid=<stem файла>, unreadable=True)`, все клетки `level="error"`, `text="?"`, подсказка «registry file cannot be read: <путь>: <причина>».
- Нет папки реестра или в ней нет записей → `scanned=False`, `rows=[]`.

- [ ] **Step 1: Тесты** (записи реестра собираются в `tmp_path` через `registry.save_record`; живой раннер имитируется вспомогательным процессом `tests/bearmind/fake_runner.py`, который берёт настоящий `runner._acquire_lock` и пишет пульс через `runner._Heartbeat`; время подставляется параметром `now`):
```python
def test_runner_state_stopped_without_heartbeat(tmp_series): ...   # status == "stopped", alive is False
def test_runner_state_alive(tmp_series): ...                        # fake_runner держит lock, пульс свежий, current {"sid":"S1","stage":"mc"} -> alive, current == ("S1", "mc"), stage_elapsed_s из StageRecord.started
def test_runner_state_dead_when_pid_gone(tmp_series): ...           # lock с PID завершившегося subprocess, пульс "running" -> status == "dead", alive is False
def test_runner_state_stale_heartbeat(tmp_series): ...              # PID жив, пульсу 300 с -> alive is False, status == "dead"
def test_runner_state_after_normal_exit(tmp_series): ...            # нет run.lock, пульс "idle" -> alive is False, status == "stopped"
def test_cell_levels_and_texts(tmp_series): ...                     # DONE + warning-проверка -> level "warning", text = ключевое число; cnmf WAITING thresholds -> "waits"; FAILED -> "error", "failed", ошибка в tooltip
def test_cell_texts_match_mockup(tmp_series): ...                   # crop "512×480", mc "3.2 px", thresholds "gSig 4", cnmf "231 cells", gate "not needed", autoinspect "128 kept", export "121 neurons"
def test_tooltip_uses_check_message(tmp_series): ...                # tooltip содержит Check.message целиком, включая "(warn > 2, error > 3)"
def test_excluded_row_cells(tmp_series): ...                        # все клетки "excluded"; сессия не входит в bars и eta
def test_bars_count_sessions_per_stage(tmp_series): ...             # 4 сессии: crop done x4; mc done x2, running x1, pending x1; thresholds: 1 сессия с cnmf WAITING(thresholds) -> bars["thresholds"].waiting == 1; gate WAITING -> bars["gate"].waiting == 1; failed cnmf -> bars["cnmf"].failed == 1
def test_waiting_counts(tmp_series): ...                            # waiting == {"thresholds": 2, "manual_inspection": 1}; crop_finalize не считается; "пороги предложены, не заданы" считается как thresholds
def test_eta_from_history_and_defaults(tmp_series): ...             # 2 DONE mc с известной длительностью -> медиана; без истории -> defaults; FAILED и SKIPPED не входят
def test_eta_excludes_human_waits(tmp_series): ...                  # сессия с cnmf WAITING(thresholds): ни cnmf, ни gate, ни autoinspect, ни export этой сессии в eta; сессия без ожидания входит
def test_warning_panel_has_only_six_kinds(tmp_series): ...          # проверки mc_template_corr, cnmf_missed_cells, export_median_snr и аномалии no_metadata, broken_last_frame в warnings отсутствуют; клетки при этом окрашены
def test_warning_panel_six_kinds_present(tmp_series): ...           # по одной записи каждого вида; kind-ы из списка; retry True только у stage_failed
def test_warning_panel_errors_first(tmp_series): ...                # error раньше warning независимо от sid
def test_warning_panel_stall_and_crash_text(tmp_series): ...        # stage_crashed cause "no_progress" -> "stalled" in text
def test_warning_auto_fixed_mismatch_hidden(tmp_series): ...        # frames_ts_mismatch с auto_fixed=True не в панели; без auto_fixed в панели, в том числе у исключённой сессии
def test_failed_session_does_not_hide_others(tmp_series): ...       # S2 FAILED на cnmf, S1 и S3 идут: у S1 и S3 клетки running/done, в warnings одна запись S2, n_warning_sessions == 1
def test_unreadable_registry_file_row(tmp_series): ...              # мусор в <sid>.json -> Row.unreadable, остальные строки на месте, файл в unreadable_files
def test_cache_rereads_only_changed(tmp_series, monkeypatch): ...  # второй series_view без изменений: 0 чтений JSON и 0 вызовов config.resolve; изменили одну запись: 1 чтение
def test_rows_sorted_by_mouse_then_sid(tmp_series): ...
def test_not_scanned_series(tmp_path): ...                          # scanned is False, rows == []
```
- [ ] **Step 2:** Запустить `pytest tests/bearmind/test_monitor.py -p no:cacheprovider`; ожидается FAIL (`bearmind.monitor` нет).
- [ ] **Step 3:** Реализовать `monitor.py` по правилам выше; в `runner.py` добавить `STAGE_KEYS = _KEYS` и `wait_reason = _what`, в `registry.py` — `read_file = _read`.
- [ ] **Step 4:** Тест-файл PASS; весь быстрый набор PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): monitor model - runner state, stage cells, progress bars, ETA, warning panel`.

### Task 2: Общий модуль действий для окна и командной строки

**Files:**
- Create: `bearmind/actions.py`
- Modify: `bearmind/cli.py` (команды thresholds, gate, decide, rerun, retry вызывают `actions`; тексты и коды возврата CLI не меняются)
- Test: `tests/bearmind/test_actions.py`; `tests/bearmind/test_cli.py` проходит без правок

**Interfaces:**
- Consumes: `registry.update_record`, `config.update_yaml`, `runner.reset_from`, `stages.gate.decide / gate.run`, `scan.evaluate_excluded`, `runner.summary`, валидаторы порогов и функции `_gate_waits`, `_suggested_unset`, `_write_thresholds` из `cli.py` (переносятся сюда).
- Produces:
```python
@dataclass
class ActionResult:
    ok: list[str]                          # sid, к которым действие применено
    failed: dict[str, tuple[str, str]]     # sid -> (code, detail)
    warnings: list[tuple[str, str]]        # (sid, code)

class UnknownSession(ValueError): ...      # .unknown: list[str]; бросается до любой записи

def current_user() -> str                  # имя пользователя Windows (getpass.getuser)
def failure_text(code: str, detail: str) -> str     # английский текст для окна
def set_excluded(paths, sids: list[str], excluded: bool, who: str) -> ActionResult
def accept_mismatch(paths, sids: list[str], who: str) -> ActionResult
def gate_not_needed(paths, sids: list[str] | None, who: str, force: bool = False) -> ActionResult   # None = все, кто ждёт ручной инспекции
def gate_cleaned(paths, sid: str, who: str) -> ActionResult
def rerun_from(paths, sids: list[str], stage: str, who: str) -> ActionResult
def retry_failed(paths, sids: list[str] | None, who: str) -> ActionResult     # None = все неисключённые с FAILED-этапом; эквивалент CLI retry
def set_thresholds(paths, sid: str, gsig: int, min_corr: float, min_pnr: float) -> ActionResult
def accept_suggested(paths, sids: list[str] | None) -> ActionResult           # None = все, кто ждёт порогов
def export_summary(paths, dest: Path) -> Path                                  # runner.summary(paths).to_csv(dest)
```
- Коды `failed`: `no_suggestion`, `invalid_suggestion`, `manual_result_missing`, `no_failed_stage`, `already_passed`. Коды `warnings`: `excluded_session`, `no_mismatch_error`, `no_fps_stays_excluded`. Неизвестный sid или этап → `UnknownSession` / `ValueError` **до любой записи**; CLI превращает `UnknownSession` в прежнее `CliError("unknown session ...; known sessions: ...")`. Неверный порог в `set_thresholds` → `ValueError` с текстом CLI («--min-corr must be a number above 0 and at most 1, got 92.0»).
- `retry_failed` повторяет логику `cli.cmd_retry`: `runner.reset_from` с первого FAILED-этапа сессии; сессия без FAILED-этапа → `failed[sid] = ("no_failed_stage", ...)`.

- [ ] **Step 1: Тесты** на крошечной серии (этапы-заглушки как в `test_cli.py`):
```python
def test_set_excluded_and_back(series): ...         # excluded True -> False, Decision("exclude", "true"/"false", who)
def test_unknown_sid_writes_nothing(series): ...    # ["S1", "NOPE"] -> UnknownSession; S1 не изменилась
def test_gate_not_needed_all_waiting(series): ...   # None -> только сессии, у которых ворота ждут человека
def test_gate_not_needed_already_passed(series): ...# force=False -> failed code "already_passed", ничего не записано; force=True -> записано
def test_gate_cleaned_without_file(series): ...     # failed == {"S1": ("manual_result_missing", <путь>)}, ничего не записано
def test_rerun_from(series): ...
def test_retry_failed(series): ...                  # сессия без FAILED -> failed code "no_failed_stage"; с FAILED cnmf -> cnmf..export PENDING
def test_retry_failed_none_means_all(series): ...   # None -> только неисключённые с FAILED-этапом
def test_set_thresholds_invalid(series): ...        # 92 / nan / 0 -> ValueError, конфиг не создан
def test_accept_suggested_mixed(series): ...        # одна с предложением, одна без -> ok и failed "no_suggestion"
def test_accept_mismatch_warns_without_error(series): ...
def test_failure_text_english_ascii_free_of_cyrillic(): ...   # failure_text для каждого кода не пуст и не содержит кириллицы
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
- Test: `tests/bearmind/test_launcher.py`, дополнить `tests/bearmind/test_runner.py`, `tests/bearmind/test_cli.py` (только новый тест)

**Interfaces:**
- Consumes: `monitor.runner_state`, `runner.run`, `cli.cmd_run` (команда `python -m bearmind run SERIES`; код возврата 3 = серия занята).
- Produces:
```python
def runner_command(series_root: Path) -> list[str]     # [<python.exe окружения>, "-m", "bearmind", "run", str(series_root)]
def start_detached(series_root: Path, cmd: list[str] | None = None) -> int   # pid; RuntimeError("batch already running: PID N"), если runner_state(...).alive
def request_stop(paths) -> None                        # пишет _bearmind/stop.request
```
- `run()`: в начале (после захвата замка) удаляет старый `stop.request`. Файл проверяется в начале каждого витка `_loop` (до выбора следующей работы, поэтому идущий этап дорабатывает) и при ожидании человека: `time.sleep(poll_s)` заменить ожиданием кусками по 1 с с проверкой файла, иначе остановка из ожидания занимает до 30 с. Если файл есть, `run()` возвращается: `result["stopped"] = True`, пульс `idle` (`_Run.close()`), `run.lock` снят. Файл `stop.request` после остановки удаляется.
- `runner_command`: если `sys.executable` — `pythonw.exe`, взять `python.exe` из той же папки.
- `start_detached`: `subprocess.Popen(cmd, cwd=<корень репо>, stdin=DEVNULL, stdout/stderr -> добавлять в _bearmind/logs/runner_console.log, close_fds=True)`. Окружение: `PYTHONPATH=<корень репо>`, `OPENBLAS_CORETYPE` по умолчанию `Sandybridge`. Флаги Windows: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB`; при `OSError` (job не разрешает breakaway) — повторить без `CREATE_BREAKAWAY_FROM_JOB`. Батч запускается без `--once`: он ждёт человека и подхватывает решения сам.
- CLI `stop SERIES` печатает `[OK] stop requested; the runner stops before its next stage`, код 0.

- [ ] **Step 1: Тесты:**
```python
def test_stop_request_stops_between_stages(series): ...    # заглушка этапа crop создаёт stop.request -> run(once=True) вернулся, mc и дальше PENDING, result["stopped"] is True, пульс "idle", run.lock нет, stop.request удалён
def test_stop_request_stops_waiting_runner(series): ...     # раннер ждёт человека (poll_s=30), request_stop -> run вернулся за < 5 с
def test_stale_stop_request_removed_at_start(series): ...   # stop.request от прошлого запуска не останавливает новый
def test_runner_command_uses_python_not_pythonw(monkeypatch): ...
def test_start_detached_survives_parent(tmp_path): ...      # вспомогательный процесс вызывает start_detached(cmd=[python, -c, "sleep 2; write marker"]) и сразу выходит; marker появляется
def test_start_detached_refuses_when_alive(tmp_path, monkeypatch): ...   # runner_state alive -> RuntimeError, Popen не вызван
def test_cyrillic_and_space_path_passed_intact(tmp_path): ...            # серия "серия тест & co"; cmd пишет sys.argv в файл; путь совпал
def test_cli_stop(series): ...
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): start the runner detached; stop it before the next stage`.

### Task 4: Каркас окна и вкладки-заглушки

**Files:**
- Create: `bearmind/gui/__init__.py`, `bearmind/gui/app.py`, `bearmind_gui.bat`
- Modify: `bearmind/cli.py` (команда `gui [SERIES]`, ленивый импорт `bearmind.gui.app`; разбирается до `Path(args.series).resolve()` в `main`, серия необязательна)
- Test: `tests/bearmind/gui/conftest.py` (одна `QApplication` на сессию тестов; `QT_QPA_PLATFORM=offscreen` ставится до импорта PySide6), `tests/bearmind/gui/test_app.py`

**Interfaces:**
- Consumes: `config.machine_config`, файл машины `<BEARMIND_HOME>/machine.yaml` через `config.update_yaml`.
- Produces:
```python
class MainWindow(QMainWindow):
    tabs: QTabWidget                     # вкладки Monitor, Scan, Thresholds, Inspection; Scan, Thresholds, Inspection — заглушки
    def __init__(self, series: Path | None = None, message=None): ...   # message(title, text) — для тестов, по умолчанию QMessageBox
    def open_series(self, series: Path) -> bool
    def show_tab(self, name: str) -> None       # переключает на вкладку по английскому имени
def main(argv: list[str] | None = None) -> int
```
- Заглушка вкладки: одна строка «This tab is coming in a later version of BEARMiND.» и название фазы (Scan: phase 3, Thresholds: phase 4, Inspection: phase 5).
- Меню Series: «Open series…» (`QFileDialog.getExistingDirectory`) и «Recent» (`gui.recent_series` в `machine.yaml`, до 10 путей, новые сверху). Меню Actions заполняет задача 6.
- Заголовок окна: `BEARMiND — <полный путь серии>` (как на макете).
- Несуществующая папка → сообщение «Series folder not found: <путь>»; окно остаётся рабочим.
- Необработанное исключение в слоте → `sys.excepthook` пишет трейсбек в `<BEARMIND_HOME>/gui.log` и показывает первую строку; окно не закрывается.
- `bearmind_gui.bat` устроен как `run_bearmind.bat` (тот же поиск окружения), но запускает `start "" "<env>\pythonw.exe" -m bearmind gui %*`, без консольного окна.

- [ ] **Step 1: Тесты:**
```python
def test_window_opens_with_series(qapp, tmp_series): ...       # заголовок содержит путь серии; имена вкладок == ["Monitor", "Scan", "Thresholds", "Inspection"]
def test_placeholder_tabs_text(qapp, tmp_series): ...          # у Scan, Thresholds, Inspection есть текст-заглушка на английском (без кириллицы)
def test_show_tab(qapp, tmp_series): ...                       # show_tab("Thresholds") -> tabs.currentIndex() == 2
def test_recent_series_saved(qapp, tmp_series): ...            # machine.yaml gui.recent_series[0] == путь; повтор не дублирует
def test_missing_folder_message(qapp, tmp_path): ...           # message вызван с "Series folder not found"; open_series -> False
def test_excepthook_logs_and_keeps_window(qapp, tmp_series): ...
def test_cli_gui_help(): ...                                   # subprocess "python -m bearmind gui --help" -> код 0
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): GUI window skeleton with series chooser and placeholder tabs`.

### Task 5: Вкладка Monitor, части 1–4: шапка, прогресс, «ждут вас», таблица

**Files:**
- Create: `bearmind/gui/monitor_tab.py`
- Modify: `bearmind/gui/app.py` (вкладка подключается при открытии серии), `bearmind/defaults.yaml` (`monitor.refresh_ms: 2000`)
- Test: `tests/bearmind/gui/test_monitor_tab.py`

**Interfaces:**
- Consumes: `monitor.series_view`, `monitor.RecordCache`, `monitor.STAGE_TITLE`, `launcher.start_detached / request_stop`, `monitor.runner_state`.
- Produces:
```python
class MonitorTab(QWidget):
    open_tab = Signal(str)               # "Thresholds" | "Inspection": окно переключает вкладку (show_tab)
    status_message = Signal(str)         # в статусную строку окна
    batch_label: QLabel                  # часть 1: "Running" | "Not responding" | "Stopped"
    batch_detail: QLabel                 # "LT1_J05_2D · MC · 6 min" и "heartbeat 4 s ago"
    start_button: QPushButton            # "Start"
    stop_button: QPushButton             # "Stop after current stage"
    folder_button: QPushButton           # "Open folder"
    bars: dict[str, QWidget]             # часть 2: по одной полосе на этап
    eta_label: QLabel
    thresholds_button: QPushButton       # часть 3: "Thresholds: N sessions →"
    inspection_button: QPushButton       # "Manual inspection: M sessions →"
    table: QTableView                    # часть 4
    model: QStandardItemModel
    def __init__(self, series: Path, parent=None, start=None, stop=None): ...   # start(series) -> pid, stop(paths): подменяются в тестах; по умолчанию launcher
    def refresh(self) -> None
    def selected_sids(self) -> list[str]
```
- Часть 1, шапка: статус цветом (Running зелёный, Not responding красный, Stopped серый); при Running — «`<sid> · <этап> · <длительность>`» из `RunnerState.current` и `stage_elapsed_s`, а если `current` нет и статус `waiting` — «waiting for you»; «heartbeat N s ago» из `age_s`; при Not responding — «last heartbeat N min ago». Кнопки: Start доступна, если батч не жив; Stop after current stage — если жив; Open folder открывает корень серии (`QDesktopServices.openUrl`). Start при живом батче (гонка) показывает сообщение из `RuntimeError`.
- Часть 2, прогресс: заголовок «Progress — N sessions» (неисключённые); одна полоса на этап (названия из `STAGE_TITLE`), сегменты done / running / waits for you / failed с цветами из «Решений плана» и подпись счётчиков вида «14 done · 1 running», «8 done · 4 wait», «6 done · 1 failed»; легенда; строка «Estimated time left: ~3 h 10 min (time waiting for you not counted)», при `eta_s is None` — «Estimated time left: unknown».
- Часть 3, «Waiting for you»: две кнопки с числами; нажатие эмитит `open_tab("Thresholds")` и `open_tab("Inspection")`; при нуле кнопка отключена. Сами вкладки — заглушки до фаз 4 и 5.
- Часть 4, таблица: столбцы «Session» + семь этапов по `STAGE_TITLE`; строка — сессия, плоско, по мыши и sid; в клетке `Cell.text`, фон по уровню, подсказка `Cell.tooltip`; выбор строки выделяет сессию для лога (`selected_sids`). Нечитаемая строка красная с `?`.
- `QTimer` с интервалом `monitor.refresh_ms`. При обновлении выделение и прокрутка сохраняются. Таблица перестраивается, только если изменился набор sid; иначе меняются тексты и цвета существующих элементов.
- Серия не сканирована → вместо таблицы текст «Series not scanned. Run: run_bearmind.bat scan <path>».
- Обновление дольше 1 с → одно предупреждение в лог `bearmind`.

- [ ] **Step 1: Тесты** (серия: 2 мыши, 4 сессии, пульс в `tmp_path`):
```python
def test_batch_header_states(qapp, series4): ...            # без пульса "Stopped"; мёртвый PID "Not responding"; fake_runner "Running" и текст содержит sid и "MC"
def test_header_heartbeat_age(qapp, series4): ...           # "heartbeat 4 s ago" для пульса 4 с назад (now подставлен)
def test_start_disabled_while_alive(qapp, series4): ...
def test_start_enabled_when_runner_dead(qapp, series4): ...
def test_start_calls_launcher(qapp, series4): ...           # start вызван с путём серии; кнопка нажата один раз
def test_start_refused_shows_message(qapp, series4): ...    # start бросает RuntimeError -> status_message с "already running"
def test_stop_writes_request(qapp, series4): ...            # кнопка Stop при живом батче -> stop.request существует
def test_progress_bars_counts_text(qapp, series4): ...      # у bars["mc"] подпись "2 done · 1 running"; у bars["thresholds"] есть "wait"
def test_eta_label_mentions_waiting_not_counted(qapp, series4): ...
def test_waiting_buttons_counts_and_signal(qapp, series4): ...   # тексты "Thresholds: 2 sessions", нажатие -> open_tab("Thresholds"); при 0 кнопка отключена
def test_table_flat_sorted(qapp, series4): ...              # 4 строки, по мыши и sid; столбцы == ["Session", "Crop", "MC", "Thresholds", "CNMF-E", "Inspection", "Autoinspect", "Export"]
def test_cell_color_text_tooltip(qapp, series4): ...        # совпадают с monitor.series_view для той же клетки
def test_refresh_keeps_selection(qapp, series4): ...        # выделили S2, изменили запись S1, refresh -> выделена S2
def test_not_scanned_message(qapp, tmp_path): ...
def test_refresh_fast_on_300_sessions(qapp, tmp_path): ...  # 300 синтетических записей: второй refresh < 200 мс
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): monitor tab - batch header, stage progress, waiting buttons, session table`.

### Task 6: Вкладка Monitor, части 5–6: предупреждения, лог; меню Actions

**Files:**
- Create: `bearmind/gui/logs.py`
- Modify: `bearmind/gui/monitor_tab.py`, `bearmind/gui/app.py` (меню Actions)
- Test: `tests/bearmind/gui/test_actions_ui.py`

**Interfaces:**
- Consumes: `actions.*`, `monitor.WarningItem`, `SeriesPaths.session_log / series_log`.
- Produces:
```python
class MonitorTab(QWidget):   # дополнение задачи 5
    warnings_list: QWidget               # часть 5
    log_view: "LogView"                  # часть 6
    def __init__(self, series: Path, parent=None, start=None, stop=None, confirm=None, choose_stage=None, choose_file=None): ...
    # confirm(text) -> bool; choose_stage() -> str | None; choose_file(default: Path) -> Path | None — подменяются в тестах
    def run_action(self, name: str) -> None      # name из ACTIONS; вызывает confirm, затем actions.*
ACTIONS: tuple[str, ...]   # "exclude", "include", "accept_mismatch", "rerun_from", "retry_failed", "inspection_not_needed", "inspection_not_needed_all", "export_csv"
class LogView(QWidget):
    def show_log(self, path: Path) -> None
    def poll(self) -> None                 # дочитывает только новые байты; файл уменьшился -> читать с начала
```
- Часть 5, предупреждения: заголовок «Warnings — N sessions»; записи в порядке модели (ошибки первыми), у записи метка ERROR / WARN, «`<sid> · <название этапа>`», текст `WarningItem.text`; у `retry=True` кнопка «Retry». Retry вызывает `actions.retry_failed(paths, [sid], who)` (эквивалент `run_bearmind retry SERIES SID`) и показывает в статусной строке «`<sid>`: reset from `<этап>`»; если батч не жив, добавляет «Start the batch to run it». Подтверждения не спрашивает (действие обратимо, старые результаты остаются). Нет записей — «No warnings».
- Часть 6, лог: заголовок «Log — `<sid>`»; последние 500 строк `<sid>.log`, а при отсутствии выбранной сессии — `series.log`; строки как записаны; выбор строки таблицы переключает лог; опрос по таймеру вкладки; файл не прочитывается целиком (чтение с конца).
- Меню Actions и контекстное меню таблицы (одни и те же пункты): «Exclude», «Include», «Accept frame mismatch», «Rerun from stage…», «Retry failed (selected or all)», «Inspection not needed (selected)», «Inspection not needed for all waiting», «Export summary CSV…». Это действия §5.9–5.10 и 5.13; на макете для них нет отдельной части, поэтому они в меню.
- Каждое пишущее действие, кроме Retry в панели, сначала вызывает `confirm` с английским текстом: какие сессии и что изменится. Пример: «Sessions NOF_H01_1D, NOF_H03_3D will be recomputed from CNMF-E. Old results stay until the new ones are ready.» Отказ → ничего не пишется. Результат: строка в статусной строке окна; `failed` и `warnings` — окном сообщения с `actions.failure_text`. `who = actions.current_user()`.
- CSV по умолчанию: `<папка серии>/<имя серии>_summary.csv`.

- [ ] **Step 1: Тесты:**
```python
def test_panel_lists_errors_first_with_retry(qapp, series4): ...      # запись FAILED первой и с кнопкой Retry; у предупреждений кнопки нет
def test_panel_header_counts_sessions(qapp, series4): ...             # "Warnings — 2 sessions"
def test_panel_empty(qapp, series4_clean): ...                        # "No warnings"
def test_retry_button_resets_failed_stage(qapp, series4): ...         # Retry у S2 -> cnmf..export PENDING; запись из панели исчезла после refresh; статусная строка содержит "Start the batch" при мёртвом батче
def test_retry_does_not_ask_confirmation(qapp, series4): ...
def test_panel_texts_are_english(qapp, series4): ...                  # ни одного кириллического символа во всех QLabel и кнопках вкладки
def test_exclude_selected(qapp, series4): ...                         # confirm -> True; S1 excluded в реестре; таблица обновилась
def test_confirm_no_writes_nothing(qapp, series4): ...
def test_rerun_with_stage_choice(qapp, series4): ...                  # choose_stage -> "cnmf"; cnmf..export PENDING
def test_inspection_not_needed_all_waiting(qapp, series4): ...
def test_failures_shown_in_english(qapp, series4): ...                # retry_failed без FAILED -> сообщение с failure_text("no_failed_stage", ...)
def test_export_csv(qapp, series4, tmp_path): ...
def test_log_tail_and_switch(qapp, series4): ...                      # 600 строк -> видно последние 500; выбор S2 -> лог S2; без выбора series.log
def test_log_truncated_rereads(qapp, series4): ...
def test_log_does_not_read_whole_file(qapp, series4, monkeypatch): ... # файл 50 МБ: show_log читает не больше 1 МБ
def test_closing_window_does_not_touch_runner(qapp, series4): ...     # fake_runner жив; закрыли окно -> процесс жив (psutil)
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): monitor tab - warning panel with Retry, log tail, summary actions menu`.

### Task 7: Проверка на реальных данных (LT1), без тяжёлых запусков

**Files:**
- Create: `docs/superpowers/specs/<дата>-phase2-check.md` и папка `docs/superpowers/specs/phase2-check/` (снимки окна, каждый < 300 КБ)

Правила: сырые данные `H:\Projects\LT1\CalciumData\1_Raw` (24 сессии: мыши J01 J03 J05 J06 J08 J10 x 4 дня) только читаются. Корень серии для проверки создаётся в другом месте. MC, CNMF-E и реальный запуск батча не выполняются (правило автора: без тяжёлых запусков). Окну нужны скан и состояния реестра.

- [ ] **Step 1:** Создать корень серии `H:\Projects\_bearmind_test\series_lt1` с junction `1_Raw` на `H:\Projects\LT1\CalciumData\1_Raw` (так же устроен `H:\Projects\_bearmind_test\scan_lt1`; проверить, куда указывает его junction, и взять рабочий путь). Выполнить `run_bearmind.bat scan H:\Projects\_bearmind_test\series_lt1`: скан читает только заголовки видео и таймстемпы и пишет только в `series_lt1\_bearmind`. Убедиться, что в `LT1\CalciumData` после скана ничего не изменилось (сравнить список файлов и mtime до и после).
- [ ] **Step 2:** Состояния для снимков: записать в реестр `series_lt1` (только в этой копии, через `registry.update_record`) синтетические состояния по образцу макета: часть сессий DONE по этапам с метриками, одна RUNNING (с `started`), одна FAILED на cnmf с ошибкой и аномалией `stage_crashed`, две с cnmf WAITING (thresholds), две с gate WAITING, одна с MC `max_shift_px` 18 и проверкой, одна с `export` 19 нейронов. Пульс на время снимка даёт `tests/bearmind/fake_runner.py`. Отметить в отчёте, что состояния синтетические. Файлы выходов не создавать: окно их не проверяет.
- [ ] **Step 3:** Открыть окно offscreen на `series_lt1` и сохранить снимок вкладки Monitor (`QWidget.grab().save`): все шесть частей видны, текст английский, сверить с макетом по номерам частей. Второй снимок: пульс остановлен без нормального выхода (имитация убитого батча) — «Not responding».
- [ ] **Step 4:** Открыть окно на готовой серии `H:\Projects\_bearmind_test\series_short` только для чтения (без Start, без действий) и сохранить снимок; убедиться, что файлы серии не изменились (хэш `_bearmind\registry\*.json` до и после).
- [ ] **Step 5:** Замерить медиану 20 обновлений на `series_lt1` и на `scan_nof` (112 сессий). Ожидается < 200 мс.
- [ ] **Step 6:** Записать отчёт: снимки, время обновления, что проверено и что нет. Заранее известно, что не проверено: реальный запуск батча кнопкой Start на реальных этапах (покрыт тестами задач 3 и 5 с процессом-заглушкой; первый настоящий запуск автор делает из окна сам) и поведение при настоящей упавшей сессии.
- [ ] **Step 7:** Commit: `docs: phase 2 check on LT1 and series_short`.
