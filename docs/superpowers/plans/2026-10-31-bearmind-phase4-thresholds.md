# BEARMiND фаза 4: вкладка Thresholds — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Вкладка Thresholds в окне BEARMiND: человек проходит сессии и для каждой подтверждает или правит gSig, min_corr и min_pnr по готовым картам. Выбор пишется в конфиг сессии рядом с тем, что предлагал автоподбор, и ждущий батч сразу его подхватывает. Весь текст окна на английском.

**Architecture:** Модель `bearmind/threshold_model.py` без Qt читает карты порогов (`SeriesPaths.threshold_maps(sid)`, пишет этап `bearmind/stages/thresholds.py`), считает маску «оба порога пройдены» и счётчики. Виджет `bearmind/gui/video_view.py` показывает MC-видео с ползунком, кадры декодирует рабочий поток. Вкладка `bearmind/gui/thresholds_tab.py` собирает всё вместе, заменяет заглушку Thresholds в `MainWindow` фазы 2 и пишет выбор через `actions.set_thresholds` (уже есть, дополняется журналом выбора).

**Tech Stack:** Python 3.12, PySide6 6.11.2, NumPy, SciPy (`ndimage.label`), PyAV через `bearmind.video.VideoArray`; pytest с `QT_QPA_PLATFORM=offscreen` (фикстуры `qapp`, `make_avi`).

**Spec:** `docs/superpowers/specs/2026-10-30-bearmind-batch-pipeline-design.md` — §4 п. 4, §5.6 (картинки порогов), §5.7 (окно порогов), §5.13. **Макет (источник истины для содержимого вкладки, одобрен автором):** `docs/superpowers/plans/2026-11-06-phase4-thresholds-explainer.html`, части 1–8. Решения автора: память `author-decisions-2026-11-06` (английский текст окна, пакеты). Опирается на сделанную фазу 2 (`bearmind/actions.py`, `bearmind/gui/app.py`, `bearmind/gui/monitor_tab.py`, `bearmind/monitor.py`; план `docs/superpowers/plans/2026-10-31-bearmind-phase2-monitor.md`) и на фазу 3 (замена заглушки вкладки по имени в `MainWindow`). Выполняется после волны A2 (`docs/superpowers/plans/2026-11-06-bearmind-long-sessions-previews.md`: параллельные карты порогов, превью по флажку).

## Changes on 2026-11-07

План написан 2026-10-31, до разбора автором и до фазы 2. Что изменено.

По решениям автора:
- Весь текст окна по-английски, как на макете: состояния `waiting`, `chosen`, `CNMF-E done`, `no maps: <reason>`, `excluded`; кнопки «← Previous», «Next →», «Reset to suggestion», «Accept → next», «Accept for all sessions of this mouse»; источник стартовых значений «Autotune suggestion (robust_bg): gSig 4, min_corr 0.85, min_pnr 8.0» / «Your earlier choice» / «No suggestion — start values from the config»; счётчики «blobs 212 · single pixels 18 % · blobs on mirrored maps 9»; подпись превью «gSig 4 · 212»; вопрос «CNMF-E of <SID> is already done. With the new thresholds it will be recomputed, and manual inspection will be needed again. Write?» с кнопками Write и Cancel; видео «frame 1234 / 18000 · 00:41.1» и «MC video not found: <path>». Тесты проверяют английский текст и отсутствие кириллицы. Документ плана по-русски.
- Вкладка Thresholds стоит на своём месте (индекс 2: Monitor, Scan, Thresholds, Inspection). Кнопка Monitor «Thresholds: N sessions →» уже переключает на неё (`open_tab("Thresholds")`). Прежнее «между Скан и Мониторинг» убрано.
- Пакеты можно ставить в окружение `bearmind` без конфликта с CaImAn (запись в `env/bearmind-lock.txt`). Фазе хватает NumPy, SciPy и PyAV.
- Батч не останавливается из-за одной сессии: сессия без карт видна с причиной, остальные идут; окно ничего не ждёт.
- Новых видов предупреждений нет; панель Monitor остаётся из шести видов.

По реальному коду:
1. `actions.set_thresholds(paths, sid, gsig, min_corr, min_pnr)` уже есть (фаза 2): пишет только конфиг сессии, без `who`, без решения в реестре и без журнала. Фаза добавляет параметры только по имени (`who`, `source`, `counters`); вызовы CLI с пятью позиционными аргументами работают как раньше.
2. `actions.accept_suggested(paths, sids)` пишет через `_write_thresholds`; теперь — через `set_thresholds(source="accept_suggested")`, с новым параметром `who`.
3. Команда CLI `thresholds` добавлена с `who=False`: у неё нет `--who`. Фаза включает `--who` (как у `gate`, `decide`); тексты и коды возврата CLI не меняются, `test_cli.py` проходит без правок.
4. Карты: `SeriesPaths.threshold_maps(sid)` = `_bearmind/thresholds/<SID>_maps.npz`. Ключи (`stages/thresholds.run`): `gsig`, `mean`, `max`, `cn_<g>`, `pnr_<g>`; `cn_mirror_<g>`, `pnr_mirror_<g>` — если `bm_qc.compute_init_maps(mirror=True)` вернул `cn_neg`. Метрики этапа: `suggested` = `{gSig, min_corr, min_pnr, method}` или `None` (тогда аномалия `no_threshold_suggestion`), `gsig_range`, `n_frames`.
5. Запасной gSig — то же правило, что у этапа: `4 if 4 in gsigs else gsigs[len(gsigs) // 2]`.
6. Выбранные пороги — из `config.resolve(paths, sid=..., mouse=...)["cnmf"]` (они могут прийти и из конфига мыши или серии). «chosen» — все три заданы.
7. «Ждёт порогов» — то же правило, что у `actions.waiting_threshold_sids(paths)`: `cnmf` в WAITING с причиной `thresholds` или «предложение есть, порогов нет» (`actions.suggested_unset`).
8. Пороги входят в параметры этапа `cnmf` (`runner.stage_params` через `cnmf_params`). Новый выбор у сессии с готовым CNMF-E пересчитывает CNMF-E, а ворота ручной инспекции снова ждут (R28). Метрики `cnmf` хранят пороги, с которыми он считал (`gSig`, `min_corr`, `min_pnr`).
9. Путь журнала выбора — новое свойство `SeriesPaths.threshold_choices_csv` (`_bearmind/threshold_choices.csv`); запись под `fileio.file_lock`.
10. В `defaults.yaml` раздел `thresholds` содержит только `gsig_range`. Новые ключи `min_blob_px` и `fallback` не входят в хэш этапа (`stage_params("thresholds")` берёт только `gsig_range` и `cnmf.tsub`), поэтому ничего не пересчитывается.
11. Видео: `bearmind.video.VideoArray(path)` (`len`, `shape`, `arr[i]` с поиском кадра на каждый запрос; кадры FFV1 все ключевые). MC-видео `SeriesPaths.mc_video(sid)` есть всегда после MC; A2 делает необязательными только mp4-превью.
12. `MainWindow` после фазы 3 заменяет заглушки по имени. Фаза убирает Thresholds из `_PHASE`; `test_placeholder_tabs_text` проверяет только Inspection.
13. Реальные данные: в `H:\Projects\_bearmind_test\series1` 6 сессий с MC-видео и картами (полная длина), только чтение; в `series_short` 2 клипа по 2000 кадров с картами и оценками, только чтение. Проверка подхвата батчем пишет только в новый корень проверки (копия `_bearmind` и `3_MC` клипов, junction `1_Raw`).

## Global Constraints

- Текст окна — по-английски; консоль — ASCII (`[OK]` / `[FAILED]` / `[WARN]`).
- Пакеты можно ставить в окружение `bearmind`, если нет конфликта с CaImAn (`python -c "import caiman"`, `pip check`); каждый пакет — в `env/bearmind-lock.txt`. Фазе пакеты не нужны.
- Окно пишет конфиги и реестр только через `bearmind/actions.py` (`config.update_yaml`, `registry.update_record`).
- Окно только показывает готовые карты (§5.6): тяжёлых вычислений по видео нет.
- Карты PNR уже сравнимы с `min_pnr` из `cnmf_dict`: в `bm_qc.compute_init_maps` PNR поделён на `sqrt(tsub)`. Пересчитывать их нельзя.
- Изображения корреляции и PNR видны целиком, с краями (§5.7 п. 3): масштаб «вписать», без обрезки.
- Каждый выбор записывается рядом с предложением автоподбора (§5.7): это данные для будущей калибровки.
- Смена порогов у сессии с готовым CNMF-E пересчитывает CNMF-E, ручная инспекция снова ждёт (R28). Окно предупреждает до записи.
- Обновление маски и счётчиков при движении ползунка — не дольше 50 мс на карту 600×600.
- Батч не ждёт человека и не останавливается из-за одной сессии; панель Monitor — ровно шесть видов.
- Тяжёлых запусков MC и CNMF-E нет (правило автора). Один CNMF-E на клипе 2000 кадров в корне проверки разрешён (задача 5, шаг 3); иначе шаг пропускается.
- Исходные данные только читаются (`G:\`, `H:\Dataset`, `H:\Projects\LT1`, включая `LT1\CalciumData\1_Raw`). `series1` и `series_short` только читаются. Проверки пишут только в `_bearmind` `H:\Projects\_bearmind_test\series_lt1` или в новые корни проверки в `H:\Projects\_bearmind_test`.
- Коммиты без строк об ИИ, только `git commit -- <пути>` (новые файлы — сначала `git add <эти пути>`); не `git add -A`. В репозитории параллельно работают другие агенты: коммитить только свои пути.

## Review Focus

- Карт нет (этап thresholds упал или ещё не дошёл): сессия видна в списке с причиной, окно не падает, кнопки принятия недоступны. Тест: задача 4 `test_session_without_maps`.
- Автоподбор не дал предложения (аномалия `no_threshold_suggestion`): ползунки стоят на стартовых значениях из конфига, это написано рядом. Тест: задача 1 `test_initial_values_without_suggestion`.
- В картах есть NaN или inf (края, мёртвые пиксели): маска и счётчики считаются, отображение не ломается. Тест: задача 1 `test_nan_in_maps`.
- Дёргание ползунка видео быстрее декодирования: окно не замирает, показан последний запрошенный кадр. Тест: задача 3 `test_rapid_seek_shows_last_request`.
- Accept for all sessions of this mouse не трогает сессии с уже сделанным выбором и исключённые. Тесты: задача 2 `test_apply_to_mouse_skips_chosen`, `test_apply_to_mouse_skips_excluded`.
- Сессия с готовым CNMF-E: отказ в вопросе — конфиг не меняется. Тест: задача 4 `test_cnmf_done_asks_before_write`.
- Батч ждёт порогов: после Accept CNMF-E этой сессии стартует без перезапуска батча. Проверка: задача 5, шаг 3.

## Решения плана (моё решение, автор может отменить)

- «Облачко» (blob) — связная (8-связность) группа пикселей, прошедших оба порога, площадью не меньше `thresholds.min_blob_px` (по умолчанию 2). «Single pixels» = групп из одного пикселя / всех групп. «Blobs on mirrored maps» — оценка ложных облачков по картам перевёрнутого по знаку видео (`cn_mirror_<g>`, `pnr_mirror_<g>`), если они есть.
- Стартовые значения: выбор человека, если он уже был; иначе предложение автоподбора; иначе `thresholds.fallback` из `defaults.yaml` (gSig по правилу этапа, `min_corr: 0.85`, `min_pnr: 8`). Это только положение ползунков, без записи.
- Accept for all sessions of this mouse пишет пороги только тем сессиям мыши, у которых выбора ещё нет и которые не исключены, в том числе тем, чьих карт ещё нет: тогда они не будут ждать. Сессии с выбором не меняются; окно перечисляет обе группы в вопросе до записи.
- Журнал выбора — `_bearmind/threshold_choices.csv` плюс `Decision("thresholds", "gSig=4 min_corr=0.85 min_pnr=8", who, now)` в реестре сессии. Решение `thresholds` не участвует в исключении (`scan.evaluate_excluded`) и в воротах.
- Числа на превью gSig (облачков при текущих порогах) пересчитываются, когда ползунок отпущен (задержка 150 мс), а маска и счётчики выбранного gSig — сразу: пять пересчётов на каждый шаг ползунка не укладываются в 50 мс.
- Список сессий — дерево по мышам, как на макете: строка мыши, под ней сессии с состоянием. «Accept → next» переходит к следующей сессии в состоянии `waiting`; «Next →» и «← Previous» идут по всем сессиям.

---

### Task 1: Модель порогов без Qt

**Files:**
- Create: `bearmind/threshold_model.py`
- Modify: `bearmind/defaults.yaml` (`thresholds.min_blob_px: 2`, `thresholds.fallback: {min_corr: 0.85, min_pnr: 8}`)
- Test: `tests/bearmind/test_threshold_model.py`

**Interfaces:**
- Consumes: файл карт `SeriesPaths.threshold_maps(sid)` с ключами из «Changes» п. 4; `rec.stages["thresholds"].state / metrics["suggested"] / metrics["gsig_range"] / error`; `config.resolve(paths, sid=, mouse=)["cnmf"]` и `["thresholds"]`; `rec.stages["cnmf"].state`; `actions.waiting_for`, `actions.suggested_unset`; `registry.read_all`.
- Produces:
```python
@dataclass
class ThresholdMaps:
    gsigs: list[int]
    mean: np.ndarray
    max: np.ndarray
    cn: dict[int, np.ndarray]
    pnr: dict[int, np.ndarray]
    cn_mirror: dict[int, np.ndarray]      # empty when the file has no mirrored maps
    pnr_mirror: dict[int, np.ndarray]

@dataclass
class Counters:
    n_blobs: int
    single_frac: float
    n_mirror_blobs: int | None

@dataclass
class ThresholdItem:
    sid: str
    mouse: str
    state: str          # "no_maps" | "waiting" | "chosen" | "cnmf_done" | "excluded"
    reason: str         # English, for no_maps: last line of the stage error, or "not computed yet"
    suggested: dict | None
    chosen: dict | None # {"gSig", "min_corr", "min_pnr"} from the resolved config

STATE_TEXT: dict[str, str]   # "waiting", "chosen", "CNMF-E done", "no maps", "excluded"

def load_maps(path: Path) -> ThresholdMaps
def passing_mask(cn: np.ndarray, pnr: np.ndarray, min_corr: float, min_pnr: float) -> np.ndarray   # bool; NaN/inf -> False
def counters(maps: ThresholdMaps, g: int, min_corr: float, min_pnr: float, min_blob_px: int) -> Counters
def display_u8(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray          # uint8; NaN -> 0
def gsig_thumbnail(maps: ThresholdMaps, g: int, size: int = 160) -> np.ndarray                     # display_u8(cn * pnr), shrunk keeping the aspect ratio
def threshold_items(paths) -> list[ThresholdItem]                                                  # by mouse, then sid
def initial_values(item: ThresholdItem, maps: ThresholdMaps, cfg: dict) -> tuple[int, float, float, str]
    # (g, min_corr, min_pnr, source): source "chosen" | "suggested" | "fallback"
def source_text(source: str, item: ThresholdItem) -> str
    # "Your earlier choice" | "Autotune suggestion (robust_bg): gSig 4, min_corr 0.85, min_pnr 8.0" |
    # "No suggestion — start values from the config"
def slider_range_pnr(maps: ThresholdMaps, g: int) -> float                                         # max(30, p99.9 of finite values)
```

- [ ] **Step 1: Тесты** (синтетические карты: два гауссовых «нейрона» и шум):
```python
def test_passing_mask(): ...                         # cn 0.9 / pnr 10 at 0.8 / 8 -> True; cn 0.9 / pnr 5 -> False
def test_counters_blobs_and_singles(): ...           # 2 neurons + 3 single pixels -> n_blobs == 2, single_frac == 3 / 5
def test_counters_mirror(): ...                      # mirrored maps with one blob -> n_mirror_blobs == 1; without mirrored maps -> None
def test_nan_in_maps(): ...                          # NaN on the border -> mask False there, display_u8 without NaN, counters computed
def test_load_maps_roundtrip(tmp_path): ...          # np.savez_compressed with the stage's keys (with and without cn_mirror_*) -> ThresholdMaps
def test_threshold_items_states(tmp_path): ...       # no maps / waiting (cnmf WAITING thresholds) / waiting (suggestion, no thresholds) / chosen / cnmf DONE / excluded
def test_no_maps_reason(tmp_path): ...               # thresholds FAILED "MemoryError: ..." -> reason == last line of the error; PENDING -> "not computed yet"
def test_initial_values_order(): ...                 # chosen > suggested > fallback
def test_initial_values_without_suggestion(): ...    # source "fallback"; gSig 4 for range 3..7, gSig 6 for range 5..7 (the stage's rule)
def test_source_texts_english(): ...                 # the three texts as in Interfaces; no Cyrillic; STATE_TEXT values no Cyrillic
def test_mask_speed(): ...                           # 600x600: passing_mask + counters < 50 ms (median of 20 runs)
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать (`scipy.ndimage.label` со структурой 3×3). **Step 4:** PASS; быстрый набор PASS.
- [ ] **Step 5:** Commit: `git commit -m "feat(bearmind): threshold model - maps, passing mask, blob counters" -- bearmind/threshold_model.py bearmind/defaults.yaml tests/bearmind/test_threshold_model.py`

### Task 2: Запись выбора порогов

**Files:**
- Modify: `bearmind/actions.py`, `bearmind/paths.py` (`threshold_choices_csv`), `bearmind/cli.py` (команда `thresholds` с `--who`; CLI передаёт `who` и `source="cli"`)
- Test: дополнить `tests/bearmind/test_actions.py`, `tests/bearmind/test_cli.py` (только новые тесты), `tests/bearmind/test_naming_paths.py`

**Interfaces:**
- Consumes: `config.update_yaml`, `registry.update_record`, `fileio.file_lock`, `actions.threshold_error`, `actions.check_sids`.
- Produces:
```python
# paths.py
SeriesPaths.threshold_choices_csv -> Path     # _bearmind/threshold_choices.csv

# actions.py
def set_thresholds(paths, sid: str, gsig, min_corr, min_pnr, *, who: str = "", source: str = "cli",
                   counters: Counters | None = None) -> ActionResult
def accept_suggested(paths, sids, *, who: str = "") -> ActionResult          # writes through set_thresholds(source="accept_suggested")
def apply_to_mouse(paths, mouse: str, gsig: int, min_corr: float, min_pnr: float, who: str) -> ActionResult
# ok: sessions written; warnings (sid, "already_chosen") for sessions with a choice; excluded sessions are skipped
# with warnings (sid, "excluded_session")
```
- `set_thresholds` пишет: конфиг сессии `{"cnmf": {"gSig", "min_corr", "min_pnr"}}`; `Decision("thresholds", "gSig=<g> min_corr=<c> min_pnr=<p>", who, now)`; строку в `threshold_choices_csv` под `file_lock` со столбцами `sid, mouse, when, who, source, suggested_gSig, suggested_min_corr, suggested_min_pnr, suggested_method, gSig, min_corr, min_pnr, n_blobs, single_frac, n_mirror_blobs` (без `counters` — пусто). Неверный порог → `ValueError` с текстом CLI до любой записи.
- Источники: `cli`, `accept_suggested`, `window`, `window_mouse`.
- Тексты для окна: `failure_text` уже покрывает коды; для предупреждения `already_chosen` вкладка пишет «<sid> already has thresholds; it was not changed.».

- [ ] **Step 1: Тесты:**
```python
def test_set_thresholds_writes_choice_log(series): ...      # CSV row with the suggestion and the choice, source "window"; Decision("thresholds", "gSig=4 min_corr=0.85 min_pnr=8", who)
def test_set_thresholds_positional_call_unchanged(series): ... # set_thresholds(paths, S1, 4, 0.85, 8) -> config written, CSV source "cli", who ""
def test_choice_log_concurrent_appends(series): ...         # 2 threads x 20 writes -> 40 rows, one header
def test_accept_suggested_logs_source(series): ...          # CSV source "accept_suggested"; suggested_* == chosen
def test_apply_to_mouse_skips_chosen(series): ...           # S1 has a choice -> warnings [("S1", "already_chosen")], S2 written
def test_apply_to_mouse_skips_excluded(series): ...         # excluded S3 -> warnings [("S3", "excluded_session")], config of S3 not written
def test_threshold_choices_path(tmp_path): ...              # SeriesPaths(root).threshold_choices_csv == root / "_bearmind" / "threshold_choices.csv"
# test_cli.py
def test_cli_thresholds_set_logs_source_cli(series): ...    # thresholds SERIES --who tester set S1 ... -> CSV source "cli", who "tester"; output text unchanged
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS; `test_cli.py` PASS без правок старых тестов.
- [ ] **Step 5:** Commit: `git commit -m "feat(bearmind): threshold choices logged next to the suggestion" -- bearmind/actions.py bearmind/paths.py bearmind/cli.py tests/bearmind/test_actions.py tests/bearmind/test_cli.py tests/bearmind/test_naming_paths.py`

### Task 3: Виджет MC-видео с ползунком

**Files:**
- Create: `bearmind/gui/video_view.py`
- Test: `tests/bearmind/gui/test_video_view.py` (крошечный FFV1 через фикстуру `make_avi`)

**Interfaces:**
- Consumes: `bearmind.video.VideoArray(path)` — `len()`, `shape`, `arr[i]`.
- Produces:
```python
class VideoView(QWidget):
    frame_shown = Signal(int)
    slider: QSlider
    play_button: QPushButton             # "▶" / "❚❚", tooltip "Play" / "Pause"
    position_label: QLabel               # "frame 1234 / 18000 · 00:41.1"
    def __init__(self, parent=None): ...
    def set_video(self, path: Path | None, fps: float, contrast: tuple[float, float] | None = None) -> None
    def seek(self, index: int) -> None   # does not block; the worker decodes only the latest request
    def current_index(self) -> int
```
- Ползунок 0…T−1. Кнопка проигрывает с FPS сессии; если декодирование не успевает, кадры пропускаются, время не отстаёт.
- Кэш последних 64 кадров. Контраст — по процентилям 1 и 99.5 изображения `mean` из карт (передаёт вкладка).
- Видео нет или не открывается → серый прямоугольник и текст «MC video not found: <path>».

- [ ] **Step 1: Тесты:**
```python
def test_shows_requested_frame(qapp, make_avi, tmp_path): ...       # frame with known values -> pixels of the shown image match
def test_rapid_seek_shows_last_request(qapp, make_avi, tmp_path): ... # 50 seeks in a row -> last frame_shown == last index
def test_missing_video_message(qapp, tmp_path): ...                  # label text == "MC video not found: <path>"
def test_play_advances(qapp, make_avi, tmp_path): ...
def test_position_label_format(qapp, make_avi, tmp_path): ...        # fps 20, frame 30 -> "frame 31 / <T> · 00:01.5"
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать (`QThread` с очередью из одного элемента). **Step 4:** PASS.
- [ ] **Step 5:** Commit: `git commit -m "feat(bearmind): MC video view with a non-blocking scrub slider" -- bearmind/gui/video_view.py tests/bearmind/gui/test_video_view.py`

### Task 4: Вкладка Thresholds

**Files:**
- Create: `bearmind/gui/thresholds_tab.py`
- Modify: `bearmind/gui/app.py` (вкладка Thresholds строится при открытии серии на месте заглушки; Thresholds убран из `_PHASE`)
- Test: `tests/bearmind/gui/test_thresholds_tab.py`; поправить `tests/bearmind/gui/test_app.py` (`test_placeholder_tabs_text` — только Inspection)

**Interfaces:**
- Consumes: `threshold_model.*`, `actions.set_thresholds / apply_to_mouse / current_user / failure_text`, `VideoView`, `config.resolve`, `SeriesPaths.mc_video / threshold_maps`.
- Produces:
```python
class ThresholdsTab(QWidget):
    status_message = Signal(str)
    session_list: QTreeWidget            # part 1: mouse rows, session rows with the state text
    video: VideoView                     # part 2
    gsig_strip: QWidget                  # part 3: one thumbnail per gSig, "gSig 4 · 212"
    corr_view: QLabel                    # part 4: correlation image with the red mask, fit to the widget
    pnr_view: QLabel
    min_corr_slider: QSlider             # 0.00..1.00, step 0.01, with a spin box
    min_pnr_slider: QSlider              # 0..slider_range_pnr, step 0.1, with a spin box
    counters_label: QLabel               # part 5
    source_label: QLabel                 # part 6
    prev_button, next_button, reset_button, accept_button, accept_mouse_button: QPushButton   # part 7
    def __init__(self, series: Path, parent=None, confirm=None, message=None): ...
    # confirm(text, yes="Write", no="Cancel") -> bool; message(title, text): replaced in tests
    def refresh(self) -> None                    # session states; the shown maps reload only when the session changes
    def show_session(self, sid: str) -> None
    def set_gsig(self, g: int) -> None
    def set_thresholds(self, min_corr: float, min_pnr: float) -> None
    def accept_and_next(self) -> None
    def accept_for_mouse(self) -> None
    def current(self) -> tuple[str, int, float, float]     # (sid, g, min_corr, min_pnr)
```
- Часть 1: список по мышам с состояниями `STATE_TEXT`; у `no_maps` — причина. Открытие вкладки показывает первую сессию в состоянии `waiting`.
- Часть 2: `VideoView` с MC-видео сессии (`SeriesPaths.mc_video`).
- Часть 3: полоса превью gSig (`gsig_thumbnail`) с подписью «gSig <g> · <облачков>». Щелчок выбирает gSig, выбранное превью выделено. Числа обновляются, когда ползунок отпущен.
- Часть 4: корреляция и PNR выбранного gSig рядом, каждая со своим ползунком и спинбоксом. Поверх обеих — красным пиксели, прошедшие оба порога. Масштаб «вписать», края видны.
- Часть 5: «blobs <n> · single pixels <p> % · blobs on mirrored maps <m>» (без зеркальных карт — последняя часть не показывается).
- Часть 6: `source_text`.
- Часть 7: кнопки. Для сессий без карт и исключённых принятие недоступно. Accept → next пишет через `actions.set_thresholds(..., who=current_user(), source="window", counters=...)` и переходит к следующей `waiting`.
- Часть 8: сессия с готовым CNMF-E — перед записью `confirm("CNMF-E of <SID> is already done. With the new thresholds it will be recomputed, and manual inspection will be needed again. Write?", yes="Write", no="Cancel")`.
- Accept for all sessions of this mouse: перед записью `confirm` с перечнем «Will get these thresholds: ...» и «Keep their own: ...»; затем `actions.apply_to_mouse` с `source="window_mouse"` в журнале.
- Вкладка обновляет список состояний по таймеру `monitor.refresh_ms`, только когда она видна.

- [ ] **Step 1: Тесты** (offscreen; серия с синтетическими картами для 3 сессий двух мышей и крошечными MC-видео):
```python
def test_session_list_states(qapp, series_maps): ...                   # texts "waiting", "chosen", "CNMF-E done", "no maps: not computed yet", "excluded"
def test_opens_on_first_waiting(qapp, series_maps): ...
def test_initial_sliders_from_suggestion(qapp, series_maps): ...       # source_label startswith "Autotune suggestion (robust_bg)"
def test_slider_updates_overlay_and_counters(qapp, series_maps): ...   # higher min_pnr -> n_blobs not larger than before; counters_label text changed
def test_gsig_click_switches_maps(qapp, series_maps): ...
def test_accept_and_next_writes_and_moves(qapp, series_maps): ...      # config of S1 written, CSV source "window", current()[0] == next waiting sid
def test_accept_for_mouse(qapp, series_maps): ...                      # confirm lists both groups; sessions with a choice unchanged
def test_cnmf_done_asks_before_write(qapp, series_maps): ...            # confirm -> False: config unchanged; the text as in Interfaces
def test_session_without_maps(qapp, series_maps): ...                    # reason visible, accept buttons disabled
def test_whole_image_visible(qapp, series_maps): ...                     # shown pixmap no larger than the widget, aspect ratio kept
def test_thresholds_tab_texts_are_english(qapp, series_maps): ...        # labels, buttons, tree items, confirm texts: no Cyrillic
def test_window_thresholds_tab_replaces_placeholder(qapp, tmp_series): ... # MainWindow tabs.widget(2) is ThresholdsTab; Monitor "Thresholds: N sessions →" switches to index 2
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS; быстрый набор PASS.
- [ ] **Step 5:** Commit: `git commit -m "feat(bearmind): thresholds tab - gSig strip, corr/PNR sliders, counters" -- bearmind/gui/thresholds_tab.py bearmind/gui/app.py tests/bearmind/gui/test_thresholds_tab.py tests/bearmind/gui/test_app.py`

### Task 5: Проверка на реальных сериях

**Files:**
- Create: `docs/superpowers/specs/<дата>-phase4-check.md` (`<дата>` — день выполнения, `YYYY-MM-DD`), снимки в `docs/superpowers/specs/phase4-check/` (каждый < 300 КБ)

- [ ] **Step 1: Просмотр без записи.** Открыть окно offscreen на `H:\Projects\_bearmind_test\series1` (только чтение: ничего не принимать). Снять вкладку на `NOF_H01_1D` и `RFC_F01_1D`. Убедиться, что файлы `series1\_bearmind` не изменились (хэш до и после).
- [ ] **Step 2: Задержки.** На той же серии замерить задержку обновления при движении ползунков (медиана 20 изменений) и время перехода к следующей сессии.
- [ ] **Step 3: Подхват батчем.** Создать корень `H:\Projects\_bearmind_test\phase4_short`: junction `1_Raw` → `H:\Projects\_bearmind_test\series_short\1_Raw`, копии `_bearmind` и `3_MC` из `series_short` (только чтение источника). Выполнить `status`: crop, mc, thresholds должны быть DONE и актуальны; иначе шаг пропустить (батч стал бы заново делать MC) и записать «не проверено». Убрать пороги одной сессии из её конфига в копии и выполнить `rerun <корень> <SID> --from cnmf`. Запустить `run <корень> --sessions <SID>` в фоне, дождаться «waits» в клетке Thresholds, принять пороги во вкладке. CNMF-E этой сессии должен стартовать сам не позже чем через `poll_s` (30 с). Это один CNMF-E на 2000 кадров (разрешён правилом автора); после него остановить батч кнопкой Stop after current stage.
- [ ] **Step 4: Отчёт.** Снимки, задержки, время подхвата, сравнение счётчиков при предложенных порогах и при порогах, которыми считала задача 17 ядра, строки журнала `threshold_choices.csv` из корня проверки, что не проверено.
- [ ] **Step 5:** Commit: `git add docs/superpowers/specs/<дата>-phase4-check.md docs/superpowers/specs/phase4-check` и `git commit -m "docs: phase 4 check on real series" -- docs/superpowers/specs/<дата>-phase4-check.md docs/superpowers/specs/phase4-check`
