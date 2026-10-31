# BEARMiND фаза 4: окно порогов CNMF-E — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Вкладка «Пороги»: человек проходит сессии и для каждой подтверждает или правит gSig, min_corr и min_pnr по готовым картам. Выбор пишется в конфиг сессии рядом с тем, что предлагал автоподбор, и ждущий батч сразу его подхватывает.

**Architecture:** Модель `bearmind/threshold_model.py` без Qt читает карты порогов (`_bearmind/thresholds/<SID>_maps.npz`, задача 9 ядра), считает маску «оба порога пройдены» и счётчики. Виджет `gui/video_view.py` показывает MC-видео с ползунком, кадры декодирует рабочий поток. Вкладка `gui/thresholds_tab.py` собирает всё вместе и пишет выбор через `actions.set_thresholds`.

**Tech Stack:** Python 3.12, PySide6 6.11.2, NumPy, SciPy (`ndimage.label`), PyAV через `bearmind.video.VideoArray`; pytest с `QT_QPA_PLATFORM=offscreen`.

**Spec:** `docs/superpowers/specs/2026-10-30-bearmind-batch-pipeline-design.md` — §4 п. 4, §5.6 (картинки порогов), §5.7 (окно порогов), §5.13. Опирается на план фазы 2 `docs/superpowers/plans/2026-10-31-bearmind-phase2-monitor.md` (`actions.py`, `texts.py`, `gui/app.py`, `qapp`).

## Global Constraints

- Все ограничения фазы 2 действуют: новых пакетов нет, окно по-русски, консоль ASCII, записи только через `actions`, коммиты без строк об ИИ и только `git commit -- <пути>`.
- Окно только показывает готовые карты (§5.6): тяжёлых вычислений по видео здесь нет.
- Карты PNR уже сравнимы с `min_pnr` из `cnmf_dict`: в `bm_qc.compute_init_maps` PNR поделён на `sqrt(tsub)`. Пересчитывать их нельзя.
- Изображения корреляции и PNR видны целиком, с краями (§5.7 п. 3): масштаб «вписать», без обрезки.
- Каждый выбор записывается рядом с предложением автоподбора (§5.7): это данные для будущей калибровки.
- Смена порогов у сессии с готовым CNMF-E пересчитывает CNMF-E, а ручная инспекция снова ждёт (правило R28 ядра). Окно предупреждает об этом до записи.
- Обновление маски и счётчиков при движении ползунка — не дольше 50 мс на кадр 600×600.

## Review Focus

- Карт нет (этап thresholds упал или ещё не дошёл): сессия видна в списке с причиной, окно не падает, кнопки принятия недоступны. Тест: задача 4 `test_session_without_maps`.
- Автоподбор не дал предложения (аномалия `no_threshold_suggestion`): ползунки стоят на стартовых значениях из конфига, это написано рядом с ними. Тест: задача 1 `test_initial_values_without_suggestion`.
- В картах есть NaN или inf (края, мёртвые пиксели): маска и счётчики считаются, отображение не ломается. Тест: задача 1 `test_nan_in_maps`.
- Дёргание ползунка видео быстрее декодирования: окно не замирает, показывается последний запрошенный кадр. Тест: задача 3 `test_rapid_seek_shows_last_request`.
- Батч в это время ждёт порогов: после «Принять» CNMF-E этой сессии стартует без перезапуска батча. Тест: задача 5 `step 3` (реальный прогон).

## Решения плана (моё решение, автор может отменить)

- «Облачко» — связная (8-связность) группа пикселей, прошедших оба порога, площадью не меньше `thresholds.min_blob_px` (по умолчанию 2). «Доля одиночных шумовых пикселей» = групп из одного пикселя / всех групп. Дополнительно — «облачков на зеркальных картах»: оценка ложных облачков по картам перевёрнутого по знаку видео (`cn_mirror_<g>`, `pnr_mirror_<g>`), если они есть.
- Стартовые значения: выбор человека, если он уже был; иначе предложение автоподбора; иначе `thresholds.fallback` из `defaults.yaml` (`gSig` — 4, если он есть в диапазоне, иначе середина диапазона; `min_corr: 0.85`, `min_pnr: 8`). Это только положение ползунков, без записи.
- «Принять для всех сессий мыши» пишет пороги только тем сессиям мыши, у которых выбора ещё нет и которые не исключены. Сессии с уже сделанным выбором не меняются; окно их перечисляет.
- Журнал выбора — `_bearmind/threshold_choices.csv`, плюс `Decision("thresholds", "gSig=4 min_corr=0.85 min_pnr=8", who)` в реестре сессии.

---

### Task 1: Модель порогов без Qt

**Files:**
- Create: `bearmind/threshold_model.py`
- Modify: `bearmind/defaults.yaml` (`thresholds.min_blob_px: 2`, `thresholds.fallback: {min_corr: 0.85, min_pnr: 8}`)
- Test: `tests/bearmind/test_threshold_model.py`

**Interfaces:**
- Consumes: файл карт (`SeriesPaths.threshold_maps(sid)`): ключи `gsig`, `mean`, `max`, `cn_<g>`, `pnr_<g>`, необязательные `cn_mirror_<g>`, `pnr_mirror_<g>`; `rec.stages["thresholds"].metrics["suggested"]` = `{gSig, min_corr, min_pnr, method}` или `None`; `config.resolve(...)["cnmf"]` (выбранные пороги); `rec.stages["cnmf"].state`.
- Produces:
```python
@dataclass
class ThresholdMaps:
    gsigs: list[int]
    mean: np.ndarray
    max: np.ndarray
    cn: dict[int, np.ndarray]
    pnr: dict[int, np.ndarray]
    cn_mirror: dict[int, np.ndarray]      # пустой, если зеркальных карт нет
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
    reason: str         # по-русски, почему нет карт (ошибка этапа, ещё не посчитаны)
    suggested: dict | None
    chosen: dict | None # {"gSig", "min_corr", "min_pnr"} из конфига сессии

def load_maps(path: Path) -> ThresholdMaps
def passing_mask(cn: np.ndarray, pnr: np.ndarray, min_corr: float, min_pnr: float) -> np.ndarray   # bool; NaN -> False
def counters(maps: ThresholdMaps, g: int, min_corr: float, min_pnr: float, min_blob_px: int) -> Counters
def display_u8(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray          # uint8; NaN -> 0
def gsig_thumbnail(maps: ThresholdMaps, g: int, size: int = 160) -> np.ndarray                     # display_u8(cn*pnr), уменьшено с сохранением пропорций
def threshold_items(paths) -> list[ThresholdItem]                                                  # по mouse, затем sid
def initial_values(item: ThresholdItem, maps: ThresholdMaps, cfg: dict) -> tuple[int, float, float, str]   # (g, min_corr, min_pnr, источник: "chosen"|"suggested"|"fallback")
def slider_range_pnr(maps: ThresholdMaps, g: int) -> float                                         # верх ползунка PNR: max(30, p99.9 по finite)
```

- [ ] **Step 1: Тесты** (синтетические карты: два гауссовых «нейрона» и шум):
```python
def test_passing_mask(): ...                         # пиксель с cn 0.9 / pnr 10 при порогах 0.8 / 8 -> True; cn 0.9 / pnr 5 -> False
def test_counters_blobs_and_singles(): ...           # 2 нейрона + 3 одиночных пикселя -> n_blobs == 2, single_frac == 3/5
def test_counters_mirror(): ...                      # зеркальные карты с одним облачком -> n_mirror_blobs == 1; без зеркальных -> None
def test_nan_in_maps(): ...                          # NaN по краю -> маска False там, display_u8 без NaN, counters считаются
def test_load_maps_roundtrip(tmp_path): ...          # np.savez_compressed с ключами этапа -> ThresholdMaps
def test_threshold_items_states(tmp_series): ...     # нет карт / ждёт / выбрано / cnmf DONE / исключена
def test_initial_values_order(): ...                 # chosen > suggested > fallback
def test_initial_values_without_suggestion(): ...    # источник "fallback", gSig 4 при диапазоне 3..7, gSig 6 при диапазоне 5..7
def test_mask_speed(): ...                           # 600x600: passing_mask + counters < 50 мс (медиана 20 запусков)
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать (`scipy.ndimage.label` со структурой 3×3). **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): threshold model - maps, passing mask, blob counters`.

### Task 2: Запись выбора порогов

**Files:**
- Modify: `bearmind/actions.py`, `bearmind/cli.py` (CLI передаёт `who` и `source="cli"`)
- Test: дополнить `tests/bearmind/test_actions.py`

**Interfaces:**
- Consumes: `actions.set_thresholds` из фазы 2, `config.update_yaml`, `registry.update_record`, `fileio.file_lock` (для CSV).
- Produces:
```python
def set_thresholds(paths, sid: str, gsig: int, min_corr: float, min_pnr: float,
                   who: str = "", source: str = "cli", counters: Counters | None = None) -> ActionResult
def apply_to_mouse(paths, mouse: str, gsig: int, min_corr: float, min_pnr: float, who: str) -> ActionResult
# ok — записанные сессии; warnings (sid, "already_chosen") — сессии с уже сделанным выбором
```
- `set_thresholds` пишет: конфиг сессии `{"cnmf": {"gSig", "min_corr", "min_pnr"}}`; `Decision("thresholds", "gSig=<g> min_corr=<c> min_pnr=<p>", who)`; строку в `_bearmind/threshold_choices.csv` под `file_lock` со столбцами `sid, mouse, when, who, source, suggested_gSig, suggested_min_corr, suggested_min_pnr, suggested_method, gSig, min_corr, min_pnr, n_blobs, single_frac`.
- `accept_suggested` (фаза 2) пишет через `set_thresholds` с `source="accept_suggested"`.

- [ ] **Step 1: Тесты:**
```python
def test_set_thresholds_writes_choice_log(series): ...      # CSV: строка с предложением и выбором; Decision в реестре
def test_choice_log_concurrent_appends(series): ...         # 2 потока по 20 записей -> 40 строк, заголовок один
def test_apply_to_mouse_skips_chosen(series): ...           # у S1 выбор есть -> warnings [("S1", "already_chosen")], S2 записана
def test_apply_to_mouse_skips_excluded(series): ...
def test_cli_thresholds_set_logs_source_cli(series): ...
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + `test_cli.py` PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): threshold choices logged next to the suggestion`.

### Task 3: Виджет MC-видео с ползунком

**Files:**
- Create: `bearmind/gui/video_view.py`
- Test: `tests/bearmind/gui/test_video_view.py` (крошечный FFV1 через фикстуру `make_avi` из `tests/bearmind/conftest.py`)

**Interfaces:**
- Consumes: `bearmind.video.VideoArray(path)` — `len()`, `shape`, `arr[i]` (поиск кадра на каждый запрос; кадры FFV1 все ключевые).
- Produces:
```python
class VideoView(QWidget):
    frame_shown = Signal(int)
    def __init__(self, parent=None): ...
    def set_video(self, path: Path | None, fps: float, contrast: tuple[float, float] | None = None) -> None
    def seek(self, index: int) -> None       # не блокирует; рабочий поток декодирует только последний запрос
    def current_index(self) -> int
```
- Ползунок 0…T−1 и подпись «кадр 1234 / 11849 · 01:02.4». Кнопка «Пуск/пауза» проигрывает с FPS сессии. Если декодирование не успевает, кадры пропускаются, время не отстаёт.
- Кэш последних 64 кадров. Контраст — по процентилям 1 и 99.5 изображения `mean` из карт, его передаёт вкладка через `contrast`.
- Видео нет или не открывается → серый прямоугольник и текст «MC-видео не найдено: <путь>».

- [ ] **Step 1: Тесты:**
```python
def test_shows_requested_frame(qapp, make_avi, tmp_path): ...       # кадр с известными значениями -> пиксели в показанном изображении совпадают
def test_rapid_seek_shows_last_request(qapp, make_avi, tmp_path): ... # 50 seek подряд -> последний frame_shown == последний индекс
def test_missing_video_message(qapp, tmp_path): ...
def test_play_advances(qapp, make_avi, tmp_path): ...
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать (`QThread` с очередью из одного элемента). **Step 4:** PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): MC video view with a non-blocking scrub slider`.

### Task 4: Вкладка «Пороги»

**Files:**
- Create: `bearmind/gui/thresholds_tab.py`
- Modify: `bearmind/gui/app.py` (вкладка «Пороги» между «Скан» и «Мониторинг / сводка»)
- Test: `tests/bearmind/gui/test_thresholds_tab.py`

**Interfaces:**
- Consumes: `threshold_model.*`, `actions.set_thresholds / apply_to_mouse / current_user`, `VideoView`, `config.resolve`.
- Produces:
```python
class ThresholdsTab(QWidget):
    def __init__(self, series: Path, parent=None, confirm=None, message=None): ...
    def show_session(self, sid: str) -> None
    def set_gsig(self, g: int) -> None
    def set_thresholds(self, min_corr: float, min_pnr: float) -> None
    def accept_and_next(self) -> None
    def accept_for_mouse(self) -> None
    def current(self) -> tuple[str, int, float, float]     # (sid, g, min_corr, min_pnr)
```
- Слева — список сессий по мышам с состоянием: «ждёт», «выбрано», «CNMF-E уже посчитан», «нет карт: <причина>», «исключена».
- Сверху — `VideoView` с MC-видео сессии.
- Полоса миниатюр gSig (`gsig_thumbnail`) с подписью «gSig 4 · облачков 212». Щелчок выбирает gSig, выбранная миниатюра выделена.
- Ниже рядом — корреляция и PNR выбранного gSig. У каждой свой ползунок со спинбоксом: min_corr 0.00–1.00 с шагом 0.01; min_pnr от 0 до `slider_range_pnr` с шагом 0.1. Поверх обеих — красным пиксели, прошедшие оба порога. Масштаб «вписать».
- Счётчики под картинками: «облачков 212 · одиночных пикселей 18 % · на зеркальных картах 9».
- Рядом с ползунками — источник стартовых значений: «предложение автоподбора (robust_bg)», «ваш прошлый выбор» или «предложения нет — стартовые значения из конфига».
- Кнопки: «← Предыдущая», «Следующая →», «Сбросить к предложению», «Принять → следующая», «Принять для всех сессий мыши». Для сессий без карт принятие недоступно.
- Сессия с готовым CNMF-E: перед записью `confirm("CNMF-E этой сессии уже посчитан. С новыми порогами он пересчитается, а ручная инспекция понадобится снова. Записать?")`.
- «Принять для всех сессий мыши»: перед записью `confirm` со списком сессий, которые получат пороги, и сессий, которые не изменятся.

- [ ] **Step 1: Тесты** (offscreen; серия с синтетическими картами для 3 сессий двух мышей и крошечными MC-видео):
```python
def test_session_list_states(qapp, series_maps): ...
def test_initial_sliders_from_suggestion(qapp, series_maps): ...
def test_slider_updates_overlay_and_counters(qapp, series_maps): ...   # повысили min_pnr -> облачков не больше, чем было
def test_gsig_click_switches_maps(qapp, series_maps): ...
def test_accept_and_next_writes_and_moves(qapp, series_maps): ...      # конфиг S1 записан, показана S2
def test_accept_for_mouse(qapp, series_maps): ...
def test_cnmf_done_asks_before_write(qapp, series_maps): ...            # confirm -> False: конфиг не изменился
def test_session_without_maps(qapp, series_maps): ...                    # причина видна, кнопки принятия недоступны
def test_whole_image_visible(qapp, series_maps): ...                     # картинка вписана: её размер на экране не больше размера виджета, пропорции сохранены
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): thresholds tab - gSig strip, corr/PNR sliders, counters`.

### Task 5: Проверка на реальной серии

**Files:**
- Create: `docs/superpowers/specs/<дата>-phase4-check.md`, снимки в `docs/superpowers/specs/phase4-check/` (каждый < 300 КБ)

- [ ] **Step 1:** Открыть вкладку offscreen на копии `H:\Projects\_bearmind_test\series1` с картами порогов (задача 17 ядра). Снять вкладку на NOF_H01_1D и RFC_F01_1D.
- [ ] **Step 2:** Замерить задержку обновления при движении ползунков (медиана 20 изменений) и время перехода к следующей сессии.
- [ ] **Step 3:** На отдельной копии серии сбросить пороги одной сессии (`rerun --from cnmf` и удаление её порогов из конфига сессии), запустить батч, дождаться «ждёт порогов», принять пороги во вкладке. Батч должен запустить CNMF-E этой сессии сам.
- [ ] **Step 4:** Отчёт: снимки, задержки, сравнение счётчиков при предложенных порогах и при порогах, которыми считала задача 17, что не проверено.
- [ ] **Step 5:** Commit: `docs: phase 4 check on a real series`.
