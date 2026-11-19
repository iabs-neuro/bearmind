# LT1, правки 19.11 (раунд 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** правки автора от 19.11 (пп. 33–45 списка правок LT1): Monitor, облегчённые estimates, галка exports на инспекции, плохие кадры в MC.

**Architecture:** правки GUI и сервера инспекции переносятся в `bearmind-batch` без остановки батча; правки этапов (CNMF-E пишет облегчённый файл, MC меняет плохие кадры) — при одной остановке, вместе с Rescan LT1_J05_1D и пересчётом LT1_J10_2D.

**Tech Stack:** Python 3.10 (conda env `bearmind`), PySide6, Bokeh (страница инспекции), CaImAn 1.13.2, pytest.

**Spec:** `docs/superpowers/specs/2026-11-11-lt1-feedback.md`, пп. 33–45 (ответы автора 19.11); задача 15 плана `docs/superpowers/plans/2026-11-16-lt1-fixes-round2.md` (плохие кадры) — её текст обязателен для задачи 7 здесь, кроме изменений, названных ниже.

## Перед началом

- Работайте только в worktree `C:\Users\User\PycharmProjects\bearmind-fixes`, ветка `bearmind-fixes`. В основной папке `C:\Users\User\PycharmProjects\bearmind` идёт батч серии `H:\Projects\LT1\CalciumData`; там ничего не меняйте.
- Данные серии только для чтения (`H:\Projects\LT1\CalciumData`, `G:`, `H:\Dataset`).

## Решения плана (автор может отменить)

- **Р1. «you» → «expert»** во всех текстах окна, сервера инспекции и CLI, которые обращаются к человеку («waits for you» → «waits for the expert», «Waiting for you» → «Waiting for the expert»). Пометка «(time waiting for you not counted)» убирается без замены.
- **Р2. Жёлтый — «в очереди».** Ячейка этапа жёлтая, если этап ещё не начат и батч дойдёт до него без эксперта: сессия не исключена и не «cannot be scheduled», а между текущим этапом сессии и этим этапом нет ожидания человека (пороги для CNMF-E, ручная инспекция для gate). Предупреждения (сейчас жёлтые `#fff3b0`) становятся оранжевыми, чтобы жёлтый значил одно.
- **Р3. Три колонки порогов.** Колонка Thresholds Monitor заменяется на `gSig`, `min_corr`, `min_pnr` — значения из конфига сессии (выбор эксперта). У каждой колонки своё предупреждение: значение отличается от других сессий этой мыши (gSig — не равен самому частому gSig остальных сессий; `min_corr` — дальше 0,03 от медианы остальных; `min_pnr` — дальше 3). Допуски — `checks.thresholds_vs_mouse: {min_corr: 0.03, min_pnr: 3.0}` в `defaults.yaml` (не хэшируются). Нужны 2 и больше других сессий с выбором. На LT1 (снимок 18.11) отмечает 0 / 1 / 8 сессий. Предупреждения этапа карт (`no_threshold_suggestion`, `temp_not_removed`) остаются в подсказке всех трёх ячеек и красят колонку gSig.
- **Р4. Облегчённые estimates.** Облегчённый файл `4_Estimates/<SID>_estimates_light.pickle`: тот же объект без непринятых компонент (`Estimates.select_components(use_object=True, save_discarded_components=False)`; проверить на CaImAn 1.13.2, что `discarded_components` не остаются в объекте). В объекте атрибут `light_of = {"name": <имя полного файла>, "size": <байт>, "mtime_ns": <...>}`; файл годен, только если они совпадают с полным файлом. Галка конфига — новый верхний раздел `estimates: {light: true}` (вне `cnmf`: раздел `cnmf` входит в хэш CNMF-E, и новый ключ там сделал бы устаревшими все 84 CNMF-E).
- **Р5. Инспекция на облегчённом файле.** Сервер берёт облегчённый файл, если он годен и у сессии нет сохранённого прогресса и ручного результата. Сессия, начатая на полном файле, остаётся на полном: номера компонент прогресса относятся к нему. Кэш страницы различает источник.
- **Р6. Галка exports** на странице инспекции, по умолчанию выключена. Выключена — Save пишет только `manual_estimates.pickle`; включена — как сейчас (плюс `traces.csv`, `filters/`, `.mat`).
- **Р7. Плохие кадры: автоматически.** Детектор полос (`StripeMeter` задачи 15 раунда 2) работает в каждом прогоне MC. Если он нашёл кадры и `bad_frames_auto` (новый верхний ключ конфига, по умолчанию `true`, не хэшируется) включён, MC заменяет их интерполяцией в том же прогоне и пишет предупреждение «N frames replaced»; выключен — только предупреждение. Явный список `bad_frames` в конфиге сессии работает как в задаче 15 и входит в хэш MC. Хэши готовых сессий не меняются.

## Global Constraints

- Тексты программы — английские; документация — русская (STE).
- Вывод консоли — только ASCII (`[OK]`, `[FAILED]`).
- Коммиты без Co-Authored-By и без упоминаний ИИ. `git log --format='%(trailers)' <base>..HEAD` должен печатать пустоту.
- Хэши этапов готовых сессий LT1 не меняются: `tests/bearmind/test_lt1_snapshot.py` зелёный после каждой задачи.
- Тесты Qt: offscreen; наведение — `QTest.mouseMove`; шрифты — `QT_QPA_FONTDIR=C:\Windows\Fonts`.
- Python: `C:\Users\User\anaconda3\envs\bearmind\python.exe`; быстрый набор: `python -m pytest -q -m "not slow and not realdata"`.

## Review Focus

1. Сессия, у которой полный estimates пересчитан после облегчения: старый облегчённый файл не должен попасть на инспекцию.
2. Сессия на середине ручной инспекции (есть прогресс): после перезапуска сервера страница открывает тот же источник и тот же прогресс.
3. Колонки порогов у мыши, где выбор есть только у одной-двух сессий: предупреждения нет, ячейка не падает.
4. «В очереди» у сессии с исключением, с ошибкой этапа, с ожиданием порогов: не жёлтая.
5. MC на сессии без полос с `bad_frames_auto: true`: результат и хэш те же, что без детектора.

---

### Task 1: «you» → «expert» (Р1)

**Files:** `bearmind/gui/*.py`, `bearmind/monitor.py`, `bearmind/actions.py`, `bearmind/cli.py`, `bearmind/inspect_server.py`, `bm_examinator.py` (только тексты для человека), их тесты.

- [ ] Тест: ни один видимый текст окна (заголовки, подписи, подсказки, сообщения Monitor/Thresholds/Inspection/Scan) не содержит слова `you`/`your` (регистронезависимо, по границе слова). Образец — английский тест текстов в `tests/bearmind/gui/test_thresholds_tab.py`.
- [ ] Тест: `_eta_text` не содержит «not counted».
- [ ] Замените тексты; комментарии в коде не трогайте.
- [ ] Быстрый набор → 0 failed. Коммит.

### Task 2: Плашки времени в Monitor (п. 41)

**Files:** `bearmind/gui/monitor_tab.py`, `bearmind/monitor.py`, тесты Monitor.

- [ ] Один ряд цветных плашек крупным шрифтом (не меньше 13 pt, жирный) под заголовком батча: состояние батча (как сейчас чип), «<SID> · <этап> · <время этапа>», «heartbeat <возраст> ago», «Stage left: ~<…>», «Total left: ~<…>». Цвета плашек — спокойные фоны палитры вкладки; heartbeat старше `START_GRACE_S`×2 — красная плашка.
- [ ] Оценка текущего этапа: `monitor.stage_eta_s(records, rec, stage, now, rates)` — секунды до конца этапа идущей сессии. Скорость — медиана по готовым этапам сессий с близкими `n_frames` (±25 %) и `fps` (±10 %), если их не меньше 2; иначе общая медиана `_rates`. Та же уточнённая скорость идёт в общий `_eta`. Без оценки — «unknown».
- [ ] Тесты: синтетический реестр (сессии 10 мин/20 Гц и 40 мин/30 Гц с разным временем CNMF-E) — оценка этапа берёт свою группу; плашки видны и в одном ряду (`y` одинаковый).
- [ ] Быстрый набор → 0 failed. Коммит.

### Task 3: «В очереди» — жёлтый (Р2)

**Files:** `bearmind/monitor.py` (`_cell`, `CELL_BG`, легенда), `bearmind/gui/monitor_tab.py`, тесты.

- [ ] Новый уровень ячейки `queued` по Р2; текст ячейки «queued»; подсказка «runs without the expert when the batch reaches it». Предупреждения — оранжевые (ячейки и `LEVEL_STYLE`), легенда с «queued».
- [ ] Тесты: CNMF-E с выбранными порогами и готовыми картами — queued; без порогов — не queued; autoinspect за ручной инспекцией — не queued; gate с `gate.enabled: false` — queued; исключённая сессия и сессия «cannot be scheduled» — не queued.
- [ ] Быстрый набор → 0 failed. Коммит.

### Task 4: Три колонки порогов (Р3)

**Files:** `bearmind/monitor.py`, `bearmind/gui/monitor_tab.py`, `bearmind/defaults.yaml` (`checks.thresholds_vs_mouse`), тесты.

- [ ] Колонки `gSig`, `min_corr`, `min_pnr` вместо Thresholds, узкие (по содержимому, `min_corr` как «0.93», `min_pnr` как «18.5»). Без выбора — «waits» (фиолетовый) во всех трёх, если батч ждёт порогов; иначе «–».
- [ ] Предупреждение колонки по Р3; подсказка ячейки: значение, медиана (или самый частый gSig) остальных сессий мыши, сколько их.
- [ ] Тесты: мышь из 4 сессий, одна с `min_pnr` дальше 3 от медианы — жёлтая только её ячейка `min_pnr`; мышь с выбором у двух сессий — без предупреждений; текст не число в конфиге — ячейка пустая, таблица целая.
- [ ] Быстрый набор → 0 failed. Коммит.

### Task 5: Облегчённые estimates (Р4)

**Files:** Create `bearmind/light_estimates.py`, `tests/bearmind/test_light_estimates.py`; modify `bearmind/paths.py` (`estimates_light`), `bearmind/stages/cnmf.py` (после записи полного файла: облегчённый при `estimates.light`, иначе удалить старый облегчённый), `bearmind/defaults.yaml` (`estimates: {light: true}`), `bearmind/gui/inspection_tab.py` (кнопка «Make light estimates»), `bearmind/cli.py` (`light-estimates SERIES [SID ...]`).

- [ ] `make_light(full: Path) -> Path` (атомарная запись, `light_of`), `light_path_if_valid(paths, sid) -> Path | None`, `make_all(paths, progress=None) -> (made, skipped, failed)`.
- [ ] Кнопка на вкладке Inspection делает облегчённые файлы для всех сессий с готовым CNMF-E, у которых их нет или они устарели; работает в фоне; в конце пишет, сколько сделано и сколько места сэкономлено.
- [ ] Тесты: облегчённый объект содержит только принятые компоненты (A, C, S, YrA, SNR_comp, r_values, neurons_sn) в том же порядке, `idx_components` = все, `idx_components_bad` пуст; после перезаписи полного файла `light_path_if_valid` → None; `estimates.light` не меняет хэш CNMF-E (`test_lt1_snapshot`).
- [ ] Быстрый набор → 0 failed. Два коммита: (а) модуль, кнопка, CLI, `paths`, `defaults.yaml`; (б) `stages/cnmf.py` — он переносится только при остановке батча.

### Task 6: Инспекция на облегчённом файле и галка exports (Р5, Р6)

**Files:** `bearmind/inspect_server.py`, `bm_examinator.py` (если нужно), тесты сервера.

- [ ] Источник страницы по Р5; в логе сервера — какой файл открыт.
- [ ] Галка «exports» рядом с Save, по умолчанию выключена; выключена — `_queue_side_files` не вызывается.
- [ ] Тесты: с годным облегчённым файлом и без прогресса — открыт облегчённый; с прогрессом — полный; Save без галки — нет `traces.csv`; с галкой — есть.
- [ ] Быстрый набор → 0 failed. Коммит.

### Task 7: Плохие кадры в MC (п. 34, Р7)

Задача 15 плана раунда 2 целиком, плюс:

- [ ] `bad_frames_auto` (верхний ключ, `true` по умолчанию, не хэшируется). Первый проход MC (чтение сырых кадров в BigTIFF) кормит `StripeMeter`; если найдены кадры вне явного списка и `bad_frames_auto` включён — второй проход чтения с заменой этих кадров, затем MC как обычно. Метрики: `bad_frames` (явные), `auto_bad_frames` (заменённые автоматически), `suspect_frames`, `n_suspect_frames`.
- [ ] Проверка `mc_suspect_frames`: заменённые автоматически — `warning` «N frames replaced automatically: <диапазоны>»; не заменённые — `warning` как в задаче 15.
- [ ] Тест: синтетический фильм с одной сдвинутой полосой и `bad_frames_auto: true` — кадр заменён, предупреждение есть; с `false` — не заменён.
- [ ] Быстрый набор и `test_lt1_snapshot` → 0 failed. Коммит.

### Gated (после ответа автора)

- **Порог template corr** — по галерее `.superpowers/sdd/2026-11-19-lt1/template-corr-gallery/`.
- **Вкладка параметров (п. 45)** — сначала дизайн.

## Перенос в bearmind-batch

1. Задачи 1–4 и 6 (окно, сервер инспекции) — без остановки, после быстрого набора на слитом дереве. Автору перезапустить окно и сервер инспекции.
2. Задачи 5 (часть `stages/cnmf.py`) и 7 — при остановке батча (`bearmind stop`, ждать конца текущего CNMF-E). Затем: Rescan LT1_J05_1D (п. 33); `bad_frames` 64642–64651 в конфиг LT1_J10_2D (п. 34); `recheck`; запуск батча.
