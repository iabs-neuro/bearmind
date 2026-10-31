# BEARMiND фаза 3: вкладка «Скан» — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Вкладка «Скан» в окне BEARMiND: скан серии кнопкой, таблица найденных сессий с аномалиями по-русски, подробности сессии и решения по аномалиям (исключить, вернуть, принять с обрезкой, подтвердить склейку, пересканировать) без командной строки.

**Architecture:** Модель `bearmind/scan_view.py` без Qt собирает из реестра и `scan_unrecognized.csv` всё, что показывает вкладка. Скан идёт в рабочем потоке Qt и сообщает прогресс через новый параметр `progress` у `scan.scan_series`. Решения пишутся через `bearmind/actions.py` (фаза 2). Вкладка встраивается в `MainWindow` фазы 2.

**Tech Stack:** Python 3.12, PySide6 6.11.2, pytest с `QT_QPA_PLATFORM=offscreen`.

**Spec:** `docs/superpowers/specs/2026-10-30-bearmind-batch-pipeline-design.md` — §4 пп. 1–2, §5.4, §5.5 (режимы, FPS, аномалии, «из окна скана: открыть папку, исключить, принять, склеить, пересканировать»), §5.13. Опирается на план фазы 2 `docs/superpowers/plans/2026-10-31-bearmind-phase2-monitor.md` (`texts.py`, `actions.py`, `gui/app.py`, тестовый `qapp`).

## Global Constraints

- Все ограничения плана фазы 2 действуют и здесь: новых пакетов нет, окно по-русски, консоль ASCII, записи только через `actions`, коммиты без строк об ИИ и только `git commit -- <пути>`.
- Пайплайн не пишет в `1_Raw/` и `0_TimeStamps/` (§2, анти-цель). Вкладка открывает папки сессий только для просмотра в проводнике.
- Батч не ждёт человека: склейка разорванной записи уже делается сканом автоматически по времени начала (предупреждение `split_recording`). Кнопка «Подтвердить склейку» только фиксирует, что человек её проверил.
- Скан не блокирует окно: он идёт в рабочем потоке, окно отвечает.

## Review Focus

- Пересканирование, пока батч работает: сессии с изменёнными файлами пересчитаются. Окно предупреждает и спрашивает. Тест: задача 3 `test_rescan_while_runner_alive_asks`.
- Ошибка скана посреди работы (папка недоступна, нет прав): окно показывает ошибку, прежняя таблица остаётся. Тест: задача 3 `test_scan_error_keeps_table`.
- Серия без единой распознанной сессии, но с нераспознанными папками: видно, какие папки и почему. Тест: задача 1 `test_only_unrecognized`.
- Человек исправил файлы в проводнике и пересканировал: исключение по аномалии снимается само, ручное «исключить» остаётся (правило I9 ядра). Тест: задача 3 `test_rescan_after_fix_reincludes`.
- Очень длинный список аномалий у одной сессии (сотни пропусков кадров): таблица показывает сводку, подробности — в панели, окно не тормозит. Тест: задача 1 `test_many_anomalies_summarized`.

## Решения плана (моё решение, автор может отменить)

- «Подтвердить склейку» пишет `Decision("stitch_confirmed", "true")`. Аномалия остаётся в реестре, во вкладке она показывается как проверенная, и сессия больше не считается в счётчике непросмотренных предупреждений. «Не склеивать» отдельно не делается: такая сессия исключается кнопкой «Исключить».
- Удалить или добавить файлы человек может только в проводнике («Открыть папку»), затем жмёт «Пересканировать». Вкладка файлы не трогает.

---

### Task 1: Модель вкладки скана без Qt и прогресс скана

**Files:**
- Create: `bearmind/scan_view.py`
- Modify: `bearmind/scan.py` (`scan_series(series_root, progress=None)`), `bearmind/texts.py` (тексты всех кодов аномалий скана)
- Test: `tests/bearmind/test_scan_view.py`, дополнить `tests/bearmind/test_scan.py`

**Interfaces:**
- Consumes: `scan.scan_series`, `registry.all_records`, `SessionRecord.parts / anomalies / decisions / fps / fps_info / n_frames / excluded`, файл `_bearmind/scan_unrecognized.csv` (путь, причина), `texts.anomaly_text`.
- Produces:
```python
ProgressFn = Callable[[int, int, str], None]      # (сделано, всего, текущая папка)
def scan_series(series_root: Path, progress: ProgressFn | None = None) -> list[SessionRecord]

@dataclass
class AnomalyLine:
    code: str
    level: str          # "warning" | "error"
    text: str           # по-русски, с числами из data (разница кадров и где; паузы в секундах; число пропусков)
    reviewed: bool      # для split_recording: есть Decision("stitch_confirmed", "true")

@dataclass
class ScanRow:
    sid: str
    mouse: str
    mode: str                       # "bonsai" | "legacy"
    n_parts: int
    n_chunks: int
    n_video_frames: int             # сумма кадров кусков (RawPart.chunk_frames)
    n_frames: int                   # кадров к обработке после обрезки до таймстемпов
    fps: float | None
    duration_min: float | None
    level: str                      # худший уровень непросмотренных аномалий: "ok" | "warning" | "error"
    excluded: bool
    excluded_by: str                # "decision" | "anomaly" | ""
    summary: str                    # до 3 кратких названий аномалий и «ещё N»
    anomalies: list[AnomalyLine]
    folders: list[str]              # папки частей относительно серии
    decisions: list[str]            # «кто, когда, что» по-русски

@dataclass
class ScanView:
    series: Path
    scanned: bool
    rows: list[ScanRow]             # по mouse, затем по sid
    unrecognized: list[tuple[str, str]]
    n_mice: int
    n_excluded: int
    n_warning: int
    modes: dict[str, int]

def scan_view(paths) -> ScanView
```
- `progress` вызывается после каждой папки сессии; без `progress` скан работает как раньше.

- [ ] **Step 1: Тесты** (записи реестра — через `registry.save_record`; для `progress` — крошечная серия из `test_scan.py`):
```python
def test_progress_called_per_session(tiny_series): ...    # вызовы (1, 3, ...), (2, 3, ...), (3, 3, ...)
def test_row_fields(tmp_series): ...                      # n_video_frames = сумма chunk_frames; duration_min = n_frames / fps / 60
def test_mismatch_text(tmp_series): ...                   # frames_ts_mismatch diff 174 where end -> "кадров на 174 больше, чем таймстемпов (в конце)"
def test_split_recording_reviewed(tmp_series): ...        # Decision stitch_confirmed -> reviewed True, уровень строки "ok"
def test_excluded_by(tmp_series): ...                     # ошибка no_timestamps -> "anomaly"; Decision exclude -> "decision"
def test_only_unrecognized(tmp_path): ...                 # нет записей, 2 строки в csv -> rows == [], unrecognized из 2
def test_many_anomalies_summarized(tmp_series): ...       # 300 аномалий -> summary из 3 названий и "ещё 297"
def test_not_scanned(tmp_path): ...
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): scan view model and scan progress callback`.

### Task 2: Действия скана

**Files:**
- Modify: `bearmind/actions.py`
- Test: дополнить `tests/bearmind/test_actions.py`

**Interfaces:**
- Consumes: `registry.update_record`, `scan.scan_series`.
- Produces:
```python
def confirm_stitch(paths, sids: list[str], who: str) -> ActionResult   # failed code "no_split" у сессий без split_recording
def rescan(paths, progress: ProgressFn | None = None) -> ScanView      # scan_series + scan_view
```

- [ ] **Step 1: Тесты:**
```python
def test_confirm_stitch(series): ...         # Decision("stitch_confirmed", "true", who) записан
def test_confirm_stitch_no_split(series): ... # failed == {"S1": ("no_split", "")}
def test_rescan_returns_view(tiny_series): ...
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): scan actions - confirm stitch, rescan`.

### Task 3: Вкладка «Скан»

**Files:**
- Create: `bearmind/gui/scan_tab.py`
- Modify: `bearmind/gui/app.py` (вкладка «Скан» первой; если серия не сканирована, окно открывается на ней)
- Test: `tests/bearmind/gui/test_scan_tab.py`

**Interfaces:**
- Consumes: `scan_view.scan_view`, `actions.rescan / set_excluded / accept_mismatch / confirm_stitch / current_user`, `monitor.runner_state`.
- Produces:
```python
class ScanTab(QWidget):
    table: QTreeView
    details: QTextBrowser
    def __init__(self, series: Path, parent=None, confirm=None, message=None, open_folder=None): ...
    # confirm(text) -> bool; message(title, text); open_folder(path) — подменяются в тестах; по умолчанию QMessageBox и os.startfile
    def start_scan(self) -> None        # рабочий поток; по завершении сигнал scan_finished
    def refresh(self) -> None
    def selected_sids(self) -> list[str]
```
- Сверху: кнопка «Сканировать» (после первого скана — «Пересканировать»), полоса прогресса с текущей папкой, сводка: «Найдено 64 сессии у 32 мышей · bonsai 0 · legacy 64 · исключено 2 · с предупреждениями 4 · нераспознанных папок 0».
- Таблица по мышам. Столбцы: «Сессия», «Режим», «Частей», «Кусков», «Кадров в видео», «Кадров к обработке», «FPS», «Мин», «Аномалии», «Исключена». Фон строки — по `level` (цвета фазы 2); исключённая — серый текст.
- Панель подробностей выбранной сессии: аномалии (`AnomalyLine.text`, проверенные помечены), папки частей, FPS (`fps_info`: отброшено интервалов, самая большая пауза), история решений.
- Кнопки: «Открыть папку» (первая папка части), «Исключить», «Вернуть», «Принять с обрезкой до общей длины» (доступна при ошибке `frames_ts_mismatch`), «Подтвердить склейку» (доступна при `split_recording`), «Пересканировать».
- Список нераспознанных папок с причинами под таблицей. Двойной щелчок открывает папку.
- Пересканирование при живом батче: `confirm("Батч сейчас работает. Если файлы сессии изменились, её этапы пересчитаются. Пересканировать?")`.
- Ошибка в рабочем потоке: `message("Скан не удался", <первая строка>)`, трейсбек в лог `bearmind`, прежняя таблица остаётся.

- [ ] **Step 1: Тесты** (offscreen; скан ждут через `QEventLoop` с таймаутом по сигналу `scan_finished`):
```python
def test_scan_button_fills_table(qapp, tiny_series): ...         # после скана 3 строки, сводка "Найдено 3"
def test_progress_bar_reaches_total(qapp, tiny_series): ...
def test_details_show_anomaly_text(qapp, series_with_mismatch): ...
def test_exclude_and_include(qapp, tiny_series): ...
def test_accept_mismatch_enabled_only_on_error(qapp, series_with_mismatch): ...
def test_confirm_stitch_marks_reviewed(qapp, series_with_split): ...
def test_open_folder_called_with_part_folder(qapp, tiny_series): ...
def test_rescan_while_runner_alive_asks(qapp, tiny_series, monkeypatch): ...   # confirm -> False: скан не запускается
def test_scan_error_keeps_table(qapp, tiny_series, monkeypatch): ...           # scan_series бросает PermissionError -> message, строки на месте
def test_rescan_after_fix_reincludes(qapp, series_missing_ts): ...             # дописали таймстемпы, пересканировали -> не исключена
def test_window_opens_on_scan_tab_when_not_scanned(qapp, tmp_path): ...
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): scan tab with anomaly details and decisions`.

### Task 4: Проверка на реальных сериях

**Files:**
- Create: `docs/superpowers/specs/<дата>-phase3-check.md`, снимки в `docs/superpowers/specs/phase3-check/` (каждый < 300 КБ)

- [ ] **Step 1:** Открыть вкладку offscreen на `H:\Projects\_bearmind_test\scan_rfc`, `scan_nof`, `scan_lt1` (корни со ссылками только для чтения на сырые данные). Нажать «Пересканировать». Сохранить снимки таблицы и подробностей `RFC_F10_1D` (+470 кадров), `LT1_K02_1D` и одной сессии NOF с `fps_outlier`.
- [ ] **Step 2:** Сверить сводку вкладки с результатом `python -m bearmind scan` на тех же корнях: число сессий, исключённых, аномалий по кодам.
- [ ] **Step 3:** Замерить время скана 112 сессий NOF во вкладке и проверить, что окно отвечает во время скана (таймер окна срабатывает).
- [ ] **Step 4:** Отчёт: снимки, совпадение с CLI, время, что не проверено (настоящей разорванной записи в данных нет).
- [ ] **Step 5:** Commit: `docs: phase 3 check on real series`.
