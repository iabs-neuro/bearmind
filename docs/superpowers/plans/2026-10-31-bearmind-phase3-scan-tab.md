# BEARMiND фаза 3: вкладка Scan — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Вкладка Scan в окне BEARMiND (на своём месте: Monitor, Scan, Thresholds, Inspection): скан серии кнопкой, таблица найденных сессий со всеми аномалиями скана, подробности сессии и решения по ним (Exclude, Include, Accept with trimming to common length, Confirm stitch, Rescan) без командной строки. Весь текст окна на английском.

**Architecture:** Модель `bearmind/scan_view.py` без Qt собирает из реестра (`registry.read_all`) и списка нераспознанных папок (`scan.read_unrecognized`) всё, что показывает вкладка. Скан идёт в рабочем потоке Qt и сообщает прогресс через новый параметр `progress` у `scan.scan_series`. Решения пишутся через `bearmind/actions.py` (уже есть `set_excluded`, `accept_mismatch`; добавляются `confirm_stitch`, `rescan`). Вкладка `bearmind/gui/scan_tab.py` заменяет заглушку Scan в `MainWindow` фазы 2 (`bearmind/gui/app.py`).

**Tech Stack:** Python 3.12, PySide6 6.11.2, pytest с `QT_QPA_PLATFORM=offscreen` (фикстура `qapp` из `tests/bearmind/gui/conftest.py`, фабрика серий `make_series` из `tests/bearmind/conftest.py`).

**Spec:** `docs/superpowers/specs/2026-10-30-bearmind-batch-pipeline-design.md` — §4 пп. 1–2, §5.4, §5.5 (режимы, FPS, аномалии, «из окна скана: открыть папку, исключить, принять, склеить, пересканировать»), §5.13. **Макет (источник истины для содержимого вкладки, одобрен автором):** `docs/superpowers/plans/2026-11-06-phase3-scan-explainer.html`, части 1–7. Решения автора: память `author-decisions-2026-11-06` (английский текст окна, пакеты, мало главных предупреждений, рекомендация 1 от 2026-11-07). Опирается на сделанную фазу 2: код `bearmind/gui/app.py`, `bearmind/gui/monitor_tab.py`, `bearmind/monitor.py`, `bearmind/actions.py` и план `docs/superpowers/plans/2026-10-31-bearmind-phase2-monitor.md`. Выполняется после волны A2 (`docs/superpowers/plans/2026-11-06-bearmind-long-sessions-previews.md`): она сейчас меняет `runner.py`, этапы, `video.py` и вкладку Monitor.

## Changes on 2026-11-07

План написан 2026-10-31, до разбора автором и до фазы 2. Что изменено.

По решениям автора:
- Весь текст окна по-английски: кнопки Scan / Rescan, Open folder, Exclude, Include, Accept with trimming to common length, Confirm stitch, заголовки столбцов, сводка, вопросы и сообщения. Слоя `bearmind/texts.py` нет (фаза 2 его убрала). Тексты аномалий — `Anomaly.message`, которые скан уже пишет по-английски с числами; короткие названия кодов для столбца Anomalies — таблица `ANOMALY_TITLE` в `scan_view.py`. Тесты, которые раньше требовали русский текст, теперь проверяют английский текст и отсутствие кириллицы. Сам документ плана по-русски.
- Вкладка стоит на своём месте рядом с Monitor: Monitor, Scan, Thresholds, Inspection (как в фазе 2). Убраны «Scan первой» и открытие окна на Scan (тест `test_window_opens_on_scan_tab_when_not_scanned`). Вместо этого у несканированной серии Monitor показывает кнопку Open Scan tab.
- Рекомендация 1 принята: вкладка Scan показывает все аномалии скана сессии (все коды, в том числе исправленные автоматически, с пометкой «fixed automatically»). Monitor по-прежнему показывает один вид `scan_data` (`frames_ts_mismatch`, `fps_outlier`). Панель предупреждений остаётся из шести видов; фаза не трогает `monitor.py`.
- Пакеты можно ставить в окружение `bearmind`, если нет конфликта с CaImAn (запись в `env/bearmind-lock.txt`). Этой фазе новые пакеты не нужны.
- Батч не останавливается из-за одной сессии: вкладка ничего не ждёт и ничего не блокирует; нечитаемая папка уже даёт аномалию `unreadable_video` только своей сессии (ядро).

По реальному коду:
1. `texts.anomaly_text` и правка `bearmind/texts.py` убраны: файла нет.
2. Реестр читается через `registry.read_all(paths) -> (records, broken)`. Битые файлы реестра показываются списком под таблицей; вкладка не падает.
3. Нераспознанные папки уже читает `scan.read_unrecognized(paths) -> list[(path, reason)]` (файл `SeriesPaths.scan_unrecognized_csv`). Своё чтение CSV не нужно.
4. `scan.scan_series(series_root)` идёт так: обход `1_Raw` (`_walk`), записи по сессиям (`_build_record`), проверка FPS по мыши, запись реестра. `progress` вызывается после обхода — `(0, N, "")` — и после записи каждой сессии.
5. Аномалии скана и их поля (`bearmind/scan.py`): `frames_ts_mismatch` (`data.diff`, `data.where` = `end` или `part k end`; малая разница — warning с `auto_fixed`, большая — error), `split_recording` (warning «stitched by start time» или error «cannot order or stitch»; `data.gaps_s`), `no_timestamps`, `fps_unavailable`, `dropped_frames` (`data.n_gaps`, `data.max_gap_ms`), `fps_outlier` (`data.fps` и `data.median` своей мыши или `data.nominal`), `broken_last_frame` и `chunk_tail_frames` (`auto_fixed`), `no_metadata`, `unreadable_video`, `registry_recreated`. С 2026-11-07 (`0ddb65e`) `fps_outlier` сравнивает сессию с медианой её мыши, а не серии. Аномалии с `data.stage` принадлежат этапам; во вкладку Scan они не идут, их показывает Monitor.
6. Исключение решает `scan.evaluate_excluded(rec)`: последнее решение `exclude` человека, иначе любая ошибка скана. Сессия без FPS исключена всегда; Include для неё даёт предупреждение `no_fps_stays_excluded` (уже есть в `actions.set_excluded`).
7. `actions.set_excluded(paths, sids, excluded, who)`, `actions.accept_mismatch(paths, sids, who)`, `actions.current_user()`, `actions.failure_text(code, detail)`, исключение `actions.UnknownSession` уже есть. Добавляется только текст кода `no_split` в `_FAILURE_TEXT`.
8. `MainWindow.open_series` строит только `MonitorTab` (`_set_monitor`); вкладки 1–3 — заглушки `_placeholder` по словарю `_PHASE`. Фаза обобщает замену вкладки и убирает Scan из `_PHASE`. Тест `test_placeholder_tabs_text` проверяет теперь только Thresholds и Inspection.
9. Батч жив — `monitor.runner_state(paths).alive`.
10. Цвета уровней — `CELL_BG`, `CELL_INK`, `EXCLUDED_INK` из `bearmind/gui/monitor_tab.py`; тексты подтверждений Exclude / Include / Accept — `MonitorTab._ask_text`, чтобы вкладки спрашивали одинаково.
11. Monitor при несканированной серии пишет `Series not scanned. Run: run_bearmind.bat scan "<path>"` (тест `test_not_scanned_message`). Текст меняется на кнопку перехода во вкладку Scan.
12. Реальные данные: junction `scan_lt1\1_Raw` указывает на `H:\Projects\LT1\1_Raw`, которого больше нет. LT1 лежит в `H:\Projects\LT1\CalciumData\1_Raw`: там 84 сессии (21 мышь), а не 24. Проверка идёт на `H:\Projects\_bearmind_test\series_lt1` (пишет только в его `_bearmind`). NOF и RFC сканируются в новых корнях проверки; `scan_nof` и `scan_rfc` только читаются. `LT1_K02_1D` в `series_lt1` нет; вместо него берутся сессии `series_lt1` с `frames_ts_mismatch` (отчёт фазы 2 нашёл 7). `RFC_F10_1D` (+470 кадров) есть в RFC.

## Global Constraints

- Текст окна — по-английски. Вывод в консоль — только ASCII (`[OK]` / `[FAILED]` / `[WARN]`).
- Пакеты можно ставить в окружение `bearmind`, если они не конфликтуют с CaImAn (после установки `python -c "import caiman"` и `pip check`). Каждый пакет записать в `env/bearmind-lock.txt` (`conda list -n bearmind --export`). Этой фазе пакеты не нужны.
- Окно пишет в реестр только через `bearmind/actions.py` (внутри — `registry.update_record` и `scan.scan_series`).
- Пайплайн и вкладка не пишут в `1_Raw/` и `0_TimeStamps/` (§2, анти-цель). Вкладка открывает папки только в проводнике и файлы не трогает.
- Батч не ждёт человека и не останавливается из-за одной сессии. Склейку разорванной записи скан делает сам (`split_recording`); Confirm stitch только фиксирует проверку.
- Панель предупреждений Monitor — ровно шесть видов. Фаза не добавляет видов и не меняет `monitor.py`.
- Скан не блокирует окно: он идёт в рабочем потоке.
- Тяжёлых запусков MC и CNMF-E нет (правило автора). Проверки этой фазы — только скан.
- Исходные данные только читаются: `G:\`, `H:\Dataset`, `H:\Projects\LT1` (включая `H:\Projects\LT1\CalciumData\1_Raw`). Проверка на реальных данных пишет только в `_bearmind` тестовых корней: `H:\Projects\_bearmind_test\series_lt1` (junction `1_Raw` на LT1) и новые корни проверки в `H:\Projects\_bearmind_test` с junction `1_Raw` на сырые данные (так устроен `series_lt1`). `scan_nof`, `scan_rfc`, `series1`, `series_short` только читаются.
- Коммиты без строк об ИИ (`Co-Authored-By` и т. п.), только `git commit -- <пути>` (новые файлы — сначала `git add <эти пути>`); не `git add -A`. В репозитории параллельно работают другие агенты: коммитить только свои пути.

## Review Focus

- Rescan, пока батч работает: сессии с изменёнными файлами пересчитаются. Окно спрашивает «Batch is running. If session files changed, their stages will be recomputed. Rescan?». Тест: задача 3 `test_rescan_while_runner_alive_asks`.
- Ошибка скана посреди работы (папка недоступна, нет прав): окно показывает «Scan failed» с первой строкой ошибки, прежняя таблица остаётся. Тест: задача 3 `test_scan_error_keeps_table`.
- Серия без единой распознанной сессии, но с нераспознанными папками: видно, какие папки и почему. Тест: задача 1 `test_only_unrecognized`.
- Человек исправил файлы в проводнике и нажал Rescan: исключение по аномалии снимается само, ручное Exclude остаётся (правило I9 ядра). Тесты: задача 3 `test_rescan_after_fix_reincludes`, `test_manual_exclude_survives_rescan`.
- Очень длинный список аномалий у одной сессии: в таблице три коротких названия и «and N more», подробности — в панели. Тест: задача 1 `test_many_anomalies_summarized`.
- Вкладка показывает все аномалии скана, а Monitor — по-прежнему одну запись `scan_data`. Тест: задача 1 `test_scan_tab_shows_all_monitor_keeps_one_kind`.
- Окно отвечает во время скана. Тест: задача 3 `test_window_responds_during_scan`.

## Решения плана (моё решение, автор может отменить)

- Confirm stitch пишет `Decision("stitch_confirmed", "true", who)`. Аномалия остаётся в реестре; во вкладке она помечена «checked by you», и сессия больше не входит в счётчик «with problems». Кнопка работает только для склеенной записи (`split_recording` уровня warning). Запись уровня error («cannot order or stitch») склеить нельзя, такая сессия исключена сканом; для неё код отказа `no_split`. «Не склеивать» отдельно не делается: такую сессию исключают кнопкой Exclude.
- Удалить или добавить файлы человек может только в проводнике (Open folder), затем жмёт Rescan. Вкладка файлы не трогает.
- Подробности показывают `Anomaly.message` как его записал скан, с меткой ERROR / WARN / FIXED. Это тот же текст, что в панели Monitor и в выводе CLI `scan`: один текст на аномалию везде. Переформулированные фразы макета («174 more video frames than timestamps (at the end).») не повторяются.
- Столбец Anomalies — короткие названия с числами из `data`: «frames ≠ timestamps: +174 at the end», «split recording: 2 parts, pause 38 s», «no timestamps found», «FPS outlier: 24.9 (mouse median 30.0)». На макете стоит «series median»: скан с 2026-11-07 сравнивает с медианой мыши, вкладка показывает то, что считал скан.
- «Все аномалии сессии» — все аномалии скана (без `data.stage`). Аномалии этапов (сбой этапа, временная папка не удалена и т. п.) остаются в Monitor.
- Счётчик «with problems» в сводке — сессии с уровнем строки не `ok`, то есть с аномалией скана, которая не исправлена автоматически и не помечена «checked by you».
- Окно само на Scan не переключается. У несканированной серии Monitor вместо таблицы пишет «Series not scanned. Open the Scan tab and press Scan.» и показывает кнопку Open Scan tab.
- После успешного скана вкладка Monitor обновляется сразу (сигнал `scan_finished`), не ждёт своего таймера.
- Exclude, Include и Accept with trimming спрашивают подтверждение теми же английскими текстами, что меню Actions вкладки Monitor. Confirm stitch спрашивает «<Sessions> will be marked as checked: the stitch by start time stays as it is.».

---

### Task 1: Модель вкладки скана без Qt и прогресс скана

**Files:**
- Create: `bearmind/scan_view.py`
- Modify: `bearmind/scan.py` (`scan_series(series_root, progress=None)`)
- Test: `tests/bearmind/test_scan_view.py`; дополнить `tests/bearmind/test_scan.py`

**Interfaces:**
- Consumes: `registry.read_all`; `SessionRecord.sid / mouse / mode / parts / n_frames / fps / fps_info / anomalies / decisions / excluded`; `RawPart.folder / chunks / chunk_frames`; `fps_info` с полями `FpsEstimate` (`fps`, `n_outliers`, `n_gaps`, `max_gap_ms`, `median_interval_ms`) и `source`; `scan.read_unrecognized(paths)`; `SeriesPaths.scan_unrecognized_csv`. Модель ничего не пишет.
- Produces:
```python
ProgressFn = Callable[[int, int, str], None]      # (done, total, sid just written; "" after the folder walk)
def scan_series(series_root: Path, progress: ProgressFn | None = None) -> list[SessionRecord]

ANOMALY_TITLE: dict[str, str]   # code -> short English title: frames_ts_mismatch "frames ≠ timestamps",
                                # split_recording "split recording", no_timestamps "no timestamps found",
                                # fps_outlier "FPS outlier", dropped_frames "dropped frames",
                                # broken_last_frame "broken last frame", chunk_tail_frames "extra chunk frames",
                                # no_metadata "no recording start time", unreadable_video "unreadable video",
                                # fps_unavailable "no frame rate", registry_recreated "registry file recreated"

@dataclass
class AnomalyLine:
    code: str
    level: str          # "warning" | "error"
    text: str           # Anomaly.message as the scan wrote it
    short: str          # ANOMALY_TITLE + numbers from data, e.g. "frames ≠ timestamps: +174 at the end"
    auto_fixed: bool
    reviewed: bool      # split_recording with Decision("stitch_confirmed", "true")

@dataclass
class ScanRow:
    sid: str
    mouse: str
    mode: str                       # "bonsai" | "legacy"
    n_parts: int
    n_chunks: int
    n_video_frames: int             # sum of RawPart.chunk_frames
    n_frames: int                   # frames to process (SessionRecord.n_frames)
    fps: float | None
    duration_min: float | None      # n_frames / fps / 60
    level: str                      # worst line that is neither auto_fixed nor reviewed: "ok" | "warning" | "error"
    excluded: bool
    excluded_by: str                # "you" | "anomaly" | ""
    summary: str                    # up to 3 `short` texts, then "and N more"; "–" without anomalies
    anomalies: list[AnomalyLine]    # scan anomalies only (no data["stage"])
    folders: list[str]              # part folders relative to the series
    fps_text: str                   # "30.0 from timestamps · 0 intervals dropped · longest pause 0.07 s"
    decisions: list[str]            # "2026-11-07 14:02 vplyusnin: exclude = true"

@dataclass
class ScanView:
    series: Path
    scanned: bool                   # registry has records or scan_unrecognized.csv exists
    rows: list[ScanRow]             # by mouse, then sid
    unrecognized: list[tuple[str, str]]
    broken: list[tuple[Path, str]]  # registry files that cannot be read
    n_mice: int
    n_excluded: int
    n_problems: int                 # rows with level != "ok"
    modes: dict[str, int]           # {"bonsai": n, "legacy": m}
    def summary_text(self) -> str   # "Found 24 sessions of 6 mice · bonsai 24 · legacy 0 · excluded 3 ·
                                    #  with problems 4 · unrecognized folders 1"

def scan_view(paths) -> ScanView
```
- `progress` вызывается после обхода и после записи каждой сессии; без `progress` скан работает как раньше (все тесты `test_scan.py` проходят без правок).

- [ ] **Step 1: Тесты** (серии — через `make_series`; записи с нужными аномалиями и решениями — через `registry.save_record` / `registry.update_record`):
```python
# tests/bearmind/test_scan.py
def test_progress_called_per_session(make_series, tmp_path): ...  # 3 bonsai sessions -> calls == [(0, 3, ""), (1, 3, sid1), (2, 3, sid2), (3, 3, sid3)], sids in sorted order

# tests/bearmind/test_scan_view.py
def test_row_fields(make_series, tmp_path): ...                  # n_video_frames == sum(chunk_frames); n_chunks == sum(len(p.chunks)); duration_min == n_frames / fps / 60
def test_mismatch_line(make_series, tmp_path): ...               # large mismatch: line.level "error", line.text == anomaly.message, line.short == "frames ≠ timestamps: +<diff> at the end"; row.level "error", excluded_by "anomaly"
def test_auto_fixed_marked_not_counted(make_series, tmp_path): ... # chunk_tail_frames -> line.auto_fixed True; row.level "ok"; view.n_problems == 0
def test_split_recording_reviewed(make_series, tmp_path): ...    # legacy_split: row.level "warning", short startswith "split recording: 2 parts, pause"; + Decision stitch_confirmed -> reviewed True, row.level "ok"
def test_fps_outlier_short_names_mouse_median(tmp_path): ...     # data {"fps": 24.9, "median": 30.0} -> short == "FPS outlier: 24.9 (mouse median 30.0)"
def test_excluded_by(make_series, tmp_path): ...                 # no_timestamps -> "anomaly"; Decision exclude "true" -> "you"; exclude then include -> excluded False, ""
def test_stage_anomalies_not_shown(tmp_path): ...                # anomaly with data["stage"] == "cnmf" -> not in row.anomalies, row.level unchanged
def test_only_unrecognized(tmp_path): ...                        # no records, 2 rows in scan_unrecognized.csv -> scanned True, rows == [], len(unrecognized) == 2
def test_many_anomalies_summarized(tmp_path): ...                # 300 dropped_frames lines -> summary has 3 short texts and endswith "and 297 more"; len(row.anomalies) == 300
def test_scan_tab_shows_all_monitor_keeps_one_kind(tmp_path): ... # frames_ts_mismatch (error), fps_outlier, dropped_frames, no_metadata -> 4 lines in the row; monitor.series_view(...).warnings for this sid: exactly 1 item, kind "scan_data"
def test_summary_text(make_series, tmp_path): ...                # "Found 3 sessions of 2 mice · bonsai 3 · legacy 0 · excluded 1 · with problems 1 · unrecognized folders 0"
def test_texts_are_english(make_series, tmp_path): ...           # ANOMALY_TITLE values, every short / text / fps_text / decision string, summary_text: no Cyrillic
def test_broken_registry_file_listed(make_series, tmp_path): ... # garbage <sid>.json -> in view.broken with a reason; the other rows present
def test_not_scanned(tmp_path): ...                              # no _bearmind -> scanned False, rows == []
```
- [ ] **Step 2:** `pytest tests/bearmind/test_scan_view.py tests/bearmind/test_scan.py -p no:cacheprovider` — FAIL (`bearmind.scan_view` нет, `progress` не принимается).
- [ ] **Step 3:** Реализовать.
- [ ] **Step 4:** PASS; быстрый набор PASS.
- [ ] **Step 5:** Commit: `git commit -m "feat(bearmind): scan view model and scan progress callback" -- bearmind/scan_view.py bearmind/scan.py tests/bearmind/test_scan_view.py tests/bearmind/test_scan.py`

### Task 2: Действия скана

**Files:**
- Modify: `bearmind/actions.py`
- Test: дополнить `tests/bearmind/test_actions.py`

**Interfaces:**
- Consumes: `registry.update_record`, `scan.scan_series`, `scan_view.scan_view`, `actions.check_sids`, `actions._records`.
- Produces:
```python
def confirm_stitch(paths, sids: list[str], who: str) -> ActionResult
# Decision("stitch_confirmed", "true", who, now); failed[sid] = ("no_split", "") when the session has no
# split_recording of level warning; UnknownSession before any write
def rescan(paths, progress: ProgressFn | None = None) -> ScanView      # scan.scan_series(paths.root, progress), then scan_view(paths)
```
- `_FAILURE_TEXT["no_split"]`: «The session is not a stitched split recording, so there is nothing to confirm.»

- [ ] **Step 1: Тесты:**
```python
def test_confirm_stitch(make_series, tmp_path): ...             # legacy_split -> ok == [sid]; latest decision ("stitch_confirmed", "true", who); excluded unchanged
def test_confirm_stitch_no_split(series): ...                   # failed == {S1: ("no_split", "")}; decisions of S1 unchanged
def test_confirm_stitch_unknown_writes_nothing(series): ...     # [S1, "NOPE"] -> UnknownSession; S1 unchanged
def test_rescan_returns_view(make_series, tmp_path): ...        # view.scanned True, len(view.rows) == number of sessions
def test_rescan_keeps_stitch_confirmation(make_series, tmp_path): ... # confirm_stitch, rescan -> line.reviewed True
def test_failure_text_no_split_english(): ...                   # non-empty, no Cyrillic
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS; `test_actions.py` и `test_cli.py` PASS без правок старых тестов.
- [ ] **Step 5:** Commit: `git commit -m "feat(bearmind): scan actions - confirm stitch, rescan" -- bearmind/actions.py tests/bearmind/test_actions.py`

### Task 3: Вкладка Scan

**Files:**
- Create: `bearmind/gui/scan_tab.py`
- Modify: `bearmind/gui/app.py` (замена вкладки по имени вместо `_set_monitor`; Scan убран из `_PHASE`; `ScanTab.scan_finished` → `MonitorTab.refresh`; `ScanTab.status_message` → статусная строка), `bearmind/gui/monitor_tab.py` (несканированная серия: текст и кнопка Open Scan tab, сигнал `open_tab("Scan")`)
- Test: `tests/bearmind/gui/test_scan_tab.py`; поправить `tests/bearmind/gui/test_app.py` (`test_placeholder_tabs_text` — только Thresholds и Inspection) и `tests/bearmind/gui/test_monitor_tab.py` (`test_not_scanned_message` — новый текст)

**Interfaces:**
- Consumes: `scan_view.scan_view`, `actions.rescan / set_excluded / accept_mismatch / confirm_stitch / current_user / failure_text / UnknownSession`, `monitor.runner_state`, `MonitorTab._ask_text`, `CELL_BG`, `CELL_INK`, `EXCLUDED_INK`.
- Produces:
```python
class ScanTab(QWidget):
    scan_finished = Signal(bool)          # True: the scan finished and wrote the registry
    status_message = Signal(str)
    scan_button: QPushButton              # "Scan" before the first scan, "Rescan" after
    progress_bar: QProgressBar
    progress_label: QLabel                # "Scanning LT1_J10_3D … 21 / 24"
    summary_label: QLabel                 # ScanView.summary_text(); "Not scanned yet. Press Scan."
    table: QTableView
    details: QTextBrowser
    open_folder_button: QPushButton       # "Open folder"
    exclude_button: QPushButton           # "Exclude"
    include_button: QPushButton           # "Include"
    accept_button: QPushButton            # "Accept with trimming to common length"
    stitch_button: QPushButton            # "Confirm stitch"
    unrecognized_list: QListWidget        # "<path> — <reason>"; registry files that cannot be read below them
    def __init__(self, series: Path, parent=None, confirm=None, message=None, open_folder=None): ...
    # confirm(text) -> bool; message(title, text); open_folder(path): replaced in tests;
    # defaults: QMessageBox.question / QMessageBox.warning / os.startfile
    def start_scan(self) -> None          # worker thread; ignored while a scan runs
    def is_scanning(self) -> bool
    def refresh(self) -> None
    def selected_sids(self) -> list[str]

class MonitorTab(QWidget):                # addition to phase 2
    scan_tab_button: QPushButton          # "Open Scan tab", visible only for a series that is not scanned
```
- Часть 1 макета: кнопка Scan (после первого скана Rescan), полоса прогресса и подпись «Scanning <sid> … i / N».
- Часть 2: сводка `ScanView.summary_text()`.
- Часть 3: таблица, плоская, по мыши и sid. Столбцы: «Session», «Mode», «Parts», «Chunks», «Video frames», «Frames to process», «FPS», «Min», «Anomalies», «Excluded». Фон клетки Session — по `ScanRow.level` (`CELL_BG`); значения Excluded — «by anomaly» / «by you» / пусто.
- Часть 4: исключённая сессия — серый текст строки (`EXCLUDED_INK`).
- Часть 5: «Details — <sid>»: строки аномалий (`ERROR` / `WARN` + `text`; исправленные — `FIXED ... (fixed automatically)`; проверенная склейка — `... (checked by you)`), «Part folders: ...», «FPS: <fps_text>», «Decisions: none yet» или список `decisions`.
- Часть 6: кнопки. Open folder — выделена одна сессия, открывает первую папку части. Exclude — среди выделенных есть неисключённые. Include — среди выделенных есть исключённые. Accept with trimming to common length — у всех выделенных есть `frames_ts_mismatch` уровня error. Confirm stitch — у всех выделенных есть склеенная запись без отметки. Неподходящие кнопки серые. Во время скана решения недоступны.
- Часть 7: «Unrecognized folders»: строки «<path> — <reason>», двойной щелчок открывает папку; ниже — нечитаемые файлы реестра с причиной.
- Rescan при живом батче: `confirm("Batch is running. If session files changed, their stages will be recomputed. Rescan?")`; отказ — скан не запускается.
- Ошибка в рабочем потоке: `message("Scan failed", "<Type>: <first line>")`, трейсбек в лог `bearmind`, прежняя таблица остаётся, `scan_finished(False)`.
- Результат действий: строка в статусной строке; `failed` и `warnings` — окном сообщения с `actions.failure_text` (как в Monitor).
- `MainWindow`: вкладки строятся при открытии серии и заменяют заглушки на своих местах; порядок `TAB_NAMES` не меняется; окно открывается на Monitor.

- [ ] **Step 1: Тесты** (offscreen; конец скана ждут через `QEventLoop` с таймаутом по сигналу `scan_finished`):
```python
def test_scan_button_fills_table(qapp, make_series, tmp_path): ...       # 3 sessions: rowCount == 3, summary_label startswith "Found 3 sessions", scan_button text "Rescan"
def test_progress_bar_reaches_total(qapp, make_series, tmp_path): ...    # after scan progress_bar.value() == maximum() == 3; progress_label had "Scanning"
def test_window_responds_during_scan(qapp, make_series, tmp_path, monkeypatch): ... # scan sleeps 0.3 s per session in the thread; a 50 ms QTimer ticks >= 5 times before scan_finished
def test_columns_and_excluded_text(qapp, make_series, tmp_path): ...     # headers == ["Session", "Mode", "Parts", "Chunks", "Video frames", "Frames to process", "FPS", "Min", "Anomalies", "Excluded"]; error -> "by anomaly"; after Exclude -> "by you", grey ink
def test_details_show_all_anomalies(qapp, make_series, tmp_path): ...    # mismatch + dropped + broken_last_frame: details contain each Anomaly.message; "fixed automatically" next to broken_last_frame
def test_exclude_and_include(qapp, make_series, tmp_path): ...           # confirm -> True: excluded True, then False; Decision("exclude", ..., who)
def test_confirm_no_writes_nothing(qapp, make_series, tmp_path): ...     # confirm -> False: registry unchanged
def test_accept_enabled_only_on_mismatch_error(qapp, make_series, tmp_path): ...
def test_confirm_stitch_marks_reviewed(qapp, make_series, tmp_path): ... # legacy_split: stitch_button enabled; click -> details "(checked by you)", summary "with problems 0"
def test_open_folder_called_with_part_folder(qapp, make_series, tmp_path): ... # open_folder(series / rec.parts[0].folder)
def test_unrecognized_double_click_opens(qapp, make_series, tmp_path): ...
def test_rescan_while_runner_alive_asks(qapp, make_series, tmp_path, monkeypatch): ... # runner alive; confirm text == "Batch is running. If session files changed, their stages will be recomputed. Rescan?"; confirm -> False: scan_series not called
def test_scan_error_keeps_table(qapp, make_series, tmp_path, monkeypatch): ... # scan_series raises PermissionError("access denied") -> message("Scan failed", "PermissionError: access denied"); rows unchanged; scan_finished(False)
def test_rescan_after_fix_reincludes(qapp, make_series, tmp_path): ...   # no timestamps -> excluded "by anomaly"; write timestamps; Rescan -> excluded False
def test_manual_exclude_survives_rescan(qapp, make_series, tmp_path): ...
def test_scan_tab_texts_are_english(qapp, make_series, tmp_path): ...    # labels, buttons, headers, details, list items: no Cyrillic
def test_window_scan_tab_replaces_placeholder(qapp, tmp_series): ...     # MainWindow(series).tabs.widget(1) is ScanTab; tab texts == ["Monitor", "Scan", "Thresholds", "Inspection"]; currentIndex() == 0
def test_scan_finished_refreshes_monitor(qapp, make_series, tmp_path): ... # after the scan the Monitor table has 3 rows without its timer
def test_monitor_not_scanned_opens_scan_tab(qapp, tmp_path): ...         # not_scanned_label text == "Series not scanned. Open the Scan tab and press Scan."; scan_tab_button click -> open_tab("Scan")
# test_app.py (changed)
def test_placeholder_tabs_text(qapp, tmp_series): ...                    # only Thresholds (phase 4) and Inspection (phase 5) are placeholders; no Cyrillic
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS; весь быстрый набор, включая `tests/bearmind/gui`, PASS.
- [ ] **Step 5:** Commit: `git commit -m "feat(bearmind): scan tab with all scan anomalies and decisions" -- bearmind/gui/scan_tab.py bearmind/gui/app.py bearmind/gui/monitor_tab.py tests/bearmind/gui/test_scan_tab.py tests/bearmind/gui/test_app.py tests/bearmind/gui/test_monitor_tab.py`

### Task 4: Проверка на реальных сериях

**Files:**
- Create: `docs/superpowers/specs/<дата>-phase3-check.md` (`<дата>` — день выполнения, `YYYY-MM-DD`), снимки в `docs/superpowers/specs/phase3-check/` (каждый < 300 КБ)

Правила: сырые данные только читаются; скан читает заголовки видео и таймстемпы и пишет только в `_bearmind` своего корня. MC, CNMF-E и батч не запускаются.

- [ ] **Step 1: LT1.** Записать список файлов `H:\Projects\LT1\CalciumData` с размером и mtime. Открыть окно offscreen на `H:\Projects\_bearmind_test\series_lt1`, вкладка Scan, нажать Rescan. Снять таблицу, сводку и подробности одной сессии с `frames_ts_mismatch` и одной с `fps_outlier`, если правило мыши такие оставило. Снова записать список файлов LT1 и сравнить побайтно.
- [ ] **Step 2: RFC.** Создать корень `H:\Projects\_bearmind_test\phase3_rfc` с junction `1_Raw` → `H:\Dataset\CalciumData\RFC\1_Raw` (`cmd /c mklink /J`). Открыть вкладку, нажать Scan. Снять подробности `RFC_F10_1D` (+470 кадров).
- [ ] **Step 3: Сверка с CLI.** На тех же корнях выполнить `python -m bearmind scan <корень>` (повторный скан ничего не меняет). Сверить со сводкой вкладки: число сессий, исключённых, аномалий по кодам (по реестру).
- [ ] **Step 4: Время и отклик.** Создать корень `H:\Projects\_bearmind_test\phase3_nof` с junction `1_Raw` → `H:\Dataset\CalciumData\NOF\1_Raw`. Замерить время скана 112 сессий во вкладке. Проверить, что таймер окна срабатывал во время скана.
- [ ] **Step 5: Отчёт.** Снимки, совпадение с CLI, время, отклик, новые корни проверки (они остаются для следующих фаз), доказательство неизменности LT1. Что не проверено: настоящая разорванная запись, если в данных её нет.
- [ ] **Step 6:** Commit: `git add docs/superpowers/specs/<дата>-phase3-check.md docs/superpowers/specs/phase3-check` и `git commit -m "docs: phase 3 check on real series" -- docs/superpowers/specs/<дата>-phase3-check.md docs/superpowers/specs/phase3-check`
