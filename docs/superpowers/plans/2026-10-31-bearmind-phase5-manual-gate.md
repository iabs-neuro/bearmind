# BEARMiND фаза 5: ручная инспекция — быстрое окно и сервер на серию — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ручная инспекция открывается из вкладки Inspection окна BEARMiND в браузере, на одном долгоживущем сервере инспекции на серию, который заранее готовит следующую сессию. Окно Bokeh (`bm_examinator`) быстро отвечает на Delete, Merge, выбор другого нейрона и Save results (цели — в «Решениях плана»); решения окна (что удаляется и склеивается) не меняются. Save results сам записывает решение `cleaned`, и батч сразу берёт сессию в автоинспекцию. Весь текст окон на английском.

**Architecture:**
- `bm_examinator.make_app` получает параметры обвязки: готовую сессию (`PreparedSession`), функции сохранения от сервера, полную длину трейсов. Внутренности обработчиков меняются: один рабочий объект `Estimates` и откат по записи изменения вместо полных копий; график меняется по месту (видимые строки и смещения), ось времени отправляется один раз, трейсы прорежены мин-макс до ≤ 2000 точек и в полном разрешении при приближении; после Merge пересчитывается только склеенный нейрон. Решения проверяет эталон, записанный на коде до фазы.
- `bearmind/inspect_server.py` — Bokeh-сервер одной серии (`python -m bearmind inspect SERIES [SID]`): замок `_bearmind/inspect.lock`, сессия по адресу `/?sid=<SID>`, фоновая подготовка следующей ждущей сессии (загрузка, таймстемпы из реестра, контуры с кэшем в `5_Inspection/<SID>/`, подсказка о волне), запись результата и решения `cleaned`, выход после `inspection.idle_exit_s` без открытых вкладок.
- `bearmind/gui/inspection_tab.py` — вкладка Inspection: сессии с готовым CNMF-E (столбцы CNMF-E, Inspection, Autoinspect, Export из `monitor.series_view`), кнопки Manual inspection и Not needed, строка состояния сервера. Сервер запускается отсоединённым процессом через `launcher`.

**Tech Stack:** Python 3.12, Bokeh 3.10.0 и tornado 6.5 (уже в окружении `bearmind`), CaImAn 1.13.2 (`Estimates`, `manual_merge`), PySide6 6.11.2, psutil; pytest, `urllib` для проверки сервера.

**Spec:** `docs/superpowers/specs/2026-10-30-bearmind-batch-pipeline-design.md` — §3 (ручная инспекция до автоинспекции), §4 пп. 6–7, §5.8, §5.9, §5.13. **Замеры скорости и макет (источник истины, одобрены автором):** `docs/superpowers/plans/2026-11-06-phase5-inspection-explainer.html` (таблица «Что изменит фаза 5», части 1–7 макета). Решения автора: память `author-decisions-2026-11-06` (медленные места: ответ на действия, сохранение и переход к следующей сессии, открытие сессии; рекомендации 2–4 приняты 2026-11-07). Опирается на ядро (ворота `bearmind/stages/gate.py`, правило R28), волну A1 (`make_app`, исправление «не того нейрона», `tests/test_examinator_callbacks.py`; журнал `.superpowers/sdd/2026-11-06-bearmind-author-fixes/progress.md`), фазу 2 (`actions.py`, `launcher.py`, `monitor.py`, `gui/app.py`) и фазу 3 (замена заглушки вкладки по имени). Выполняется после волны A2.

## Changes on 2026-11-07

План написан 2026-10-31, до разбора автором и до замеров. Что изменено.

По решениям автора:
- «Логику окна инспекции не менять, только обвязка» заменено одобренным объёмом (рекомендация 2): внутренности обработчиков Delete sel., Merge sel., Show sel., Reset view, Revert change, Discard all, сортировки, выбора нейрона и сохранения можно менять ради скорости. Решения окна не меняются: какие компоненты удаляются и склеиваются и что попадает в сохранённый файл. Это доказывает эталонный тест (задача 1). `tests/test_examinator_callbacks.py` остаётся зелёным (ожидаемые списки строк не меняются) и дополняется.
- Цели скорости — три медленных места, которые назвал автор: ответ на действие (Delete, Merge, переход к другому нейрону), Save results и переход к следующей сессии, открытие сессии. «До» — замеры страницы фазы 5, «после» — цели в «Решениях плана»; задача 7 мерит тем же способом и кладёт числа рядом.
- Процесс на сессию заменён одним долгоживущим сервером на серию с фоновой подготовкой следующей сессии (рекомендация 3): импорт CaImAn и driada (9.9 с) — один раз на серию. Замок — на серию (`_bearmind/inspect.lock`), а не на сессию.
- Подсказка «looks like a calcium wave» показывается только в окне инспекции (рекомендация 4): баннер на странице сессии. В Monitor её нет: ни седьмого вида панели, ни проверки в `checks.py`, ни метрики этапа CNMF-E. Её считает сервер при подготовке сессии. Прежняя задача «метрика wave_frac в CNMF-E» убрана.
- Весь текст окон по-английски: вкладка Inspection, кнопки Manual inspection и Not needed, сообщения сервера и страницы. Тесты проверяют отсутствие кириллицы.
- Пакеты можно ставить в окружение `bearmind`, если нет конфликта с CaImAn (запись в `env/bearmind-lock.txt`). План их не требует.
- Батч не ждёт: ждёт только сессия в воротах. Сбой сервера инспекции батч не трогает.
- Ошибку «не тот нейрон после удаления» исправила волна A1 (`594dcbc`); её тесты остаются.

По реальному коду:
1. `make_app(fname, default_fps=20, bkapp_kwargs=None)` уже есть (A1); `ExamineCells` = `show(make_app(...))`. `output_notebook()` вызывается при импорте модуля. Параметры обвязки добавляются к `make_app` по имени; `ExamineCells` (путь ноутбука до фазы 6) работает как раньше, `output_notebook()` переносится внутрь `ExamineCells`.
2. `ae_launch` импортирует `bm_examinator.LoadEstimates`: этап autoinspect грузит оценки этой функцией. Её сигнатура и поведение по умолчанию сохраняются; добавляется необязательный параметр `time`.
3. `bkapp_kwargs['end_frame']` по умолчанию 0, и `EstimatesToSrcFast` режет `C[:, 0:0]` — пустые трейсы; ноутбук всегда передавал `end_frame`. Сервер передаёт `end_frame=None` — полная длина.
4. Save results пишет `<имя>final_estimates.pickle` рядом с входом, а `SaveResults` — `traces.csv`, папку фильтров и `_session.mat` тоже рядом с входом, то есть в `4_Estimates`. В режиме сервера результат идёт в `SeriesPaths.manual_result(sid)`, промежуточные — в `5_Inspection/<SID>/progress/`, побочные файлы — в `5_Inspection/<SID>/`.
5. Сжатие: `estimates_compression` исправлено в main (`6c73765`, `fbd1ad3`), но этап autoinspect (`stages/autoinspect.check_aligned`) отказывается от любого сжатого входа: старое сжатие от нового не отличить. Поэтому результат по-прежнему сохраняется без сжатия, при любом `compress_estimates`.
6. Ручная инспекция идёт до автоинспекции, на `4_Estimates/<SID>_estimates.pickle` (§5.8, §5.9): там нет `metrics_df`, ML-вероятностей и кэшей автоинспекции. Сервер открывает окно в режиме `legacy`; параметры ML для `capcan` (прежний план брал `ml_model_path`, `ml_threshold`) не нужны.
7. Склеенные таймстемпы сессии уже строит `bearmind.stages.export._stitched_timestamps(paths, rec)`. Он становится публичным `bearmind.timestamps.session_timestamps(paths, rec)`, экспорт вызывает его.
8. Ворота: `actions.gate_cleaned(paths, sid, who)` (код `manual_result_missing`), `actions.gate_not_needed(paths, sids, who, force=False)` (код `already_passed`), `actions.waiting_gate_sids(paths)`, `actions.gate_waits(paths, rec)`, `actions.gate_passed(rec)`. R28 проверяет `gate.run`: время решения против `cnmf.finished`, mtime результата против файла оценок.
9. Отсоединённый процесс запускает `launcher.start_detached` — сейчас только батч, с проверкой живого батча. Его Popen-часть выносится в `launcher.spawn_detached`; батч и сервер инспекции идут через неё.
10. В Monitor кнопка «Manual inspection: N sessions →» уже переключает на вкладку Inspection (`open_tab("Inspection")`), «Inspection not needed» — в меню Actions. Monitor не меняется; кнопки Manual inspection и Not needed — на вкладке Inspection (на макете они стоят под частью таблицы).
11. Тесты обработчиков — `tests/test_examinator_callbacks.py` (A1): помощник `_App` ищет кнопки по подписи и единственный `ColumnDataSource` со столбцом `idx`, `storage` берёт из колбэка Delete, `shown()` читает `src.data["idx"]`, `partial()` — `storage.estimates_partial`.
12. Отложенные пункты окна из журнала A1 и аудита `docs/code_improvement_ideas.md`: Revert после Reset view возвращает приближённый вид к полному списку; сортировка `capcan` после Merge падает с KeyError; панель метрик показывает удалённый нейрон; кэши компонент (`metrics_df`, реконструкции, `asp_cache`, `feedback_dict`) не перенумерованы после Merge; побочные файлы Save results включают удалённые нейроны. Решение по каждому — в «Решениях плана».
13. Новые пути в `SeriesPaths`: `inspection_lock` (`_bearmind/inspect.lock`), `inspection_progress_dir(sid)` (`5_Inspection/<SID>/progress`), `inspection_cache(sid)` (`5_Inspection/<SID>/<SID>_inspect_cache.npz`).
14. Задачи: эталон и обвязка (1), скорость обработчиков (2), сохранение (3), сервер (4), вкладка (5), волна (6), проверка (7). Прежние задачи «кнопка в сводке» и «wave_frac в CNMF-E» убраны.

## Global Constraints

- Решения окна не меняются: для одной и той же последовательности щелчков сохранённый файл совпадает с эталоном задачи 1 (`idx_components`, `idx_components_bad`, `A`, `C`, `YrA`, `S`, `SNR_comp`, `r_values`). Ожидаемые списки в `tests/test_examinator_callbacks.py` не меняются; помощники `_App` можно приспособить к новому хранению видимых строк.
- `bm_examinator.LoadEstimates(name, default_fps=20)` без нового параметра ведёт себя как раньше: её вызывает этап autoinspect через `ae_launch`.
- Путь ноутбука `ExamineCells(fname, default_fps, bkapp_kwargs)` до фазы 6 работает как раньше; отличие одно — `output_notebook()` вызывается в нём, а не при импорте.
- Без лимита кадров: трейсы на всю длину сессии (часовая — 107 107 кадров); прореживание — только для показа.
- Результат — только `SeriesPaths.manual_result(sid)`, без сжатия, атомарно (`.partial` → `os.replace`). Промежуточные — `5_Inspection/<SID>/progress/`, побочные — `5_Inspection/<SID>/`. В `4_Estimates` и в текущую папку ничего не пишется.
- Таймстемпы трейсов — склеенные таймстемпы сессии из реестра (`timestamps.session_timestamps`), в секундах от первого кадра.
- Один сервер инспекции на серию. Батч от него не зависит и ждёт только решения ворот своей сессии.
- Подсказка о волне — только на странице инспекции. `monitor.py`, `checks.py`, этап CNMF-E и панель предупреждений (ровно шесть видов) фаза не меняет.
- Текст окон по-английски; консоль — ASCII (`[OK]` / `[FAILED]` / `[WARN]`).
- Пакеты — без конфликта с CaImAn, каждый в `env/bearmind-lock.txt`.
- Тяжёлых запусков MC и CNMF-E нет (правило автора). Замеры — на готовых оценках и на синтетике; автоинспекция и экспорт клипа 2000 кадров (≈ 70 с и 10 с) разрешены.
- Исходные данные только читаются (`G:\`, `H:\Dataset`, `H:\Projects\LT1`, включая `LT1\CalciumData\1_Raw`). `series_short` и `series1` только читаются. Проверки пишут только в новые корни проверки в `H:\Projects\_bearmind_test` или в `_bearmind` `series_lt1`.
- Коммиты без строк об ИИ, только `git commit -- <пути>` (новые файлы — сначала `git add <эти пути>`); не `git add -A`. В репозитории параллельно работают другие агенты: коммитить только свои пути.

## Review Focus

- Решения не изменились: та же последовательность щелчков даёт тот же файл. Тест: задача 1 `test_golden_click_sequence` (эталон записан до правок), повторяется после задач 2 и 3.
- Save results нажали дважды, или сохранили, закрыли и открыли снова: в реестре последнее решение `cleaned`, файл результата один и последний. Тест: задача 4 `test_second_final_save_overwrites`.
- Браузер закрыли без сохранения: решения нет, сессия ждёт, сервер выходит через `idle_exit_s` и удаляет замок. Тест: задача 4 `test_server_exits_when_idle`.
- CNMF-E пересчитали, пока окно открыто: сохранение отказано с сообщением, ничего не записано. Тест: задача 4 `test_save_after_cnmf_rerun_is_stale`.
- Та же сессия во второй вкладке браузера: страница отказа называет открытую вкладку. Тест: задача 4 `test_second_tab_same_session_refused`.
- Второй сервер той же серии: отказ с кодом 3 и адресом первого; вкладка Inspection берёт работающий. Тесты: задача 4 `test_second_server_refused`, задача 5 `test_open_reuses_running_server`.
- Часовая сессия: трейсы полной длины, объём данных ограничен. Тесты: задача 1 `test_full_length_traces`, задача 2 `test_delete_payload_small_on_hour_session`.
- Этап autoinspect по-прежнему грузит оценки через `LoadEstimates`. Тест: задача 1 `test_load_estimates_default_unchanged`.
- Подготовленная следующая сессия берётся из текущего списка ждущих: решённая за это время пропускается. Тест: задача 4 `test_next_session_skips_decided`.

## Решения плана (моё решение, автор может отменить)

**Цели скорости.** «До» — Python-время со страницы фазы 5 (`measured` на 2 000 кадров, `tiled copy` на 18 000 и 107 107). Мерить «после» тем же способом: настоящие обработчики на документе в памяти, без браузера; объём — размер сообщения `PATCH-DOC`, собранного из событий документа. Перерисовку в браузере проверяет автор при приёмке.

| Действие | До | Цель |
|---|---|---|
| Delete sel. (вся цепочка кнопки), 107 107 кадров | 0.92 с; 249 МБ в браузер | ≤ 0.15 с; ≤ 1 МБ |
| Delete sel., 2 000 кадров | 0.13 с; 4.8 МБ | ≤ 0.05 с; ≤ 0.1 МБ |
| Merge sel., 107 107 кадров | 5.90 с | ≤ время самого `Estimates.manual_merge` + 0.5 с; ≤ 1 МБ |
| Переход к другому нейрону | 0.05 мс Python; браузер не мерили | ≤ 5 мс Python; в браузер — только выделение и панель метрик, столбцы данных не отправляются |
| Save results до свободного окна, 107 107 кадров | 14.56 с | ≤ 0.5 с (pickle 0.19 с и решение); остальное в фоне |
| Следующая сессия после Save results, подготовлена | 9.9 с запуска + 0.6–2.9 с подготовки | ≤ 1 с |
| Первое открытие сессии на работающем сервере, 107 107 кадров | 2.92 с (+ 9.9 с запуска на каждую сессию) | ≤ 1.5 с; запуск сервера один раз на серию |

**Как ускоряется** (строки таблицы страницы):
- Delete / Merge / Show / Revert не копируют `Estimates` целиком. Рабочий объект один; запись изменения хранит то, что изменение заменило (номера компонент; у Merge — ссылки на прежние массивы, которые `manual_merge` заменяет), и Revert возвращает их.
- График меняется по месту: Delete и Show меняют только список видимых строк и столбец вертикальных смещений; трейсы и контуры не пересылаются.
- Ось времени отправляется один раз, а не по копии на нейрон. Трейс показывается не более чем `inspection.trace_points` (2000) точками (мин-макс по корзинам). Решение плана сверх страницы: при приближении по времени сервер присылает видимый кусок в полном разрешении (не более 2000 точек на трейс), иначе мелкие события пропадают из вида.
- Merge пересчитывает контур только склеенной компоненты; остальное время — сам `manual_merge` CaImAn.
- Выбор нейрона подсвечивает трейс без перерисовки остальных; сервер включает WebGL (`enable_gpu_backend`, ключ `inspection.webgl: true`). Плавность в браузере проверяет автор.
- Save results: pickle и решение `cleaned` сразу, CSV, фильтры и `.mat` — в фоновом потоке.
- Контуры считаются один раз на файл оценок и лежат в кэше `5_Inspection/<SID>/<SID>_inspect_cache.npz` (ключ — размер и mtime файла оценок). Страница предлагала считать их сразу после CNMF-E, рядом с оценками. Здесь их считает сервер при подготовке сессии: этап CNMF-E (его правит A2) и его хэш не меняются.

**Отложенные пункты окна** (журнал A1, аудит `docs/code_improvement_ideas.md`):
- Revert после Reset view возвращает приближённый вид к полному списку — **входит**. Это поведение окна в обычной работе (режим `legacy`). Revert change всей цепочкой кнопки возвращает вид до последнего изменения, в том числе приближённый.
- KeyError сортировки `capcan` после Merge — **входит**. Причина — пересборка вида через `EstimatesToSrcFast`; Merge по месту её убирает. Склеенная строка получает NaN в столбцах метрик `capcan` и стоит в сортировке последней.
- Панель метрик показывает удалённый нейрон — **входит**. После Delete, Merge, Revert change и Discard all панель пишет «Select a neuron.», пока не выбран новый нейрон.
- Кэши компонент (`metrics_df`, реконструкции, `asp_cache`, `feedback_dict`) не перенумерованы после Merge — **не входит**. Они бывают только у файла после автоинспекции в режиме `capcan`. Сервер открывает только выход CNMF-E в режиме `legacy`, а поправки `feedback` в новый процесс не входят (§5.8). После фазы 6 режим `capcan` на таких файлах никто не вызывает. Страховка: в режиме сервера `make_app` отказывается от входа с `metrics_df` («This file is already autoinspected; open 4_Estimates instead.»). Запись в `docs/code_improvement_ideas.md` остаётся открытой.
- Побочные файлы Save results включают удалённые нейроны — **входит**: запись побочных файлов переписывается всё равно. В них идут только компоненты сохранённого `idx_components`, а при наличии `metrics_df` — без строк `delete == 1`. Батч эти файлы не читает (экспорт берёт pickle).

**Сервер и вкладка:**
- Порт — `inspection.port` (0 = свободный); адрес пишется в замок. Выход без открытых вкладок через `inspection.idle_exit_s` (600 с).
- Порядок сессий — по мыши и sid, как в Monitor; следующая — первая ждущая после текущей по `actions.waiting_gate_sids`.
- Manual inspection доступна для одной выделенной неисключённой сессии с DONE `cnmf`. Если ворота уже пройдены, вкладка спрашивает: «Manual inspection of <SID> already passed. Saving a new result makes autoinspection and export run again, and the exported neurons can change. Open anyway?».
- «Cleaned» записывается само, когда на странице нажали Save results. Отдельной кнопки нет.
- Подсказка о волне: доля кадров, в которых активна не меньше `inspection.wave.active_frac` (0.3) доли принятых компонент; компонента активна, если её `C` выше `median + k_mad × 1.4826 × MAD` своего трейса (`k_mad` = 3). Баннер, если доля выше `inspection.wave.warn_frac`; порог выбирает задача 6 по уже посчитанным оценкам.

---

### Task 1: Эталон решений и обвязка `make_app`

**Files:**
- Create: `tests/test_examinator_golden.py`, `tests/data/examinator_golden.npz` (пишется на коде до правок)
- Modify: `bm_examinator.py` (`LoadEstimates(..., time=None)`, `PreparedSession`, `prepare_session`, параметры `make_app`, `output_notebook()` в `ExamineCells`), `bearmind/timestamps.py` (`session_timestamps`), `bearmind/stages/export.py` (вызывает `session_timestamps`)
- Test: `tests/test_examinator_serve.py`, дополнить `tests/bearmind/test_timestamps.py`

**Interfaces:**
- Produces:
```python
def LoadEstimates(name, default_fps=20, time: np.ndarray | None = None)   # time given: get_timestamps is not called

@dataclass
class PreparedSession:            # what make_app needs; built off the server's IOLoop thread
    fname: str
    estimates: "Estimates"        # .time set
    data: dict                    # first view of the traces (decimated)
    contours: list | None

def prepare_session(fname, time=None, default_fps=20, bkapp_kwargs=None, contours=None) -> PreparedSession

def make_app(fname, default_fps=20, bkapp_kwargs=None, *, prepared: PreparedSession | None = None,
             final_save=None, progress_save=None, refuse_autoinspected: bool = False):
# final_save(estimates) -> str: called by "Save results" instead of the old writes; the returned English
#   text is shown in the page's status line. None: the old notebook behaviour.
# progress_save(estimates) -> str: the same for "Save progress".
# refuse_autoinspected: an input with metrics_df shows "This file is already autoinspected; open 4_Estimates instead."

# bearmind/timestamps.py
def session_timestamps(paths, rec) -> Timestamps   # moved from stages/export._stitched_timestamps
```
- `bkapp_kwargs['end_frame'] = None` — полная длина (`EstimatesToSrcFast` ставит `C.shape[1]`).

- [ ] **Step 1: Эталон на коде до правок.** Написать `tests/test_examinator_golden.py`: две синтетические сессии (помощник `_estimates` из `tests/test_examinator_callbacks.py`: 8 компонент на 300 кадров; и 40 компонент на 2000 кадров), одна последовательность щелчков: Delete, Merge, Show sel., Delete в приближённом виде, Revert change, Merge, Discard all, Delete, Merge, Save results. Режим записи (`BM_GOLDEN_WRITE=1`) сохраняет в `tests/data/examinator_golden.npz` массивы сохранённого файла. Записать эталон на текущем коде и закоммитить его до любой правки `bm_examinator.py`.
- [ ] **Step 2: Тесты обвязки:**
```python
# tests/test_examinator_golden.py
def test_golden_click_sequence(tmp_path): ...       # saved idx_components, idx_components_bad equal; A, C, YrA, S, SNR_comp, r_values equal (floats: rtol 1e-12)
# tests/test_examinator_serve.py
def test_import_has_no_notebook_side_effect(): ...  # import bm_examinator in a fresh subprocess with bokeh.io.output_notebook patched to raise -> exit code 0
def test_examine_cells_still_calls_output_notebook(tmp_path, monkeypatch): ...
def test_load_estimates_default_unchanged(tmp_path, monkeypatch): ... # no time -> get_timestamps called once with (name, n_frames, default_fps); time given -> not called, est.time is that array
def test_prepared_session_used(tmp_path, monkeypatch): ... # make_app(prepared=...) does not call LoadEstimates
def test_final_save_hook_called_and_text_shown(tmp_path): ... # "Save results" -> final_save(est) called once; status Div text == the returned text; nothing written next to the input
def test_progress_save_hook(tmp_path): ...
def test_refuse_autoinspected(tmp_path): ...        # input with metrics_df and refuse_autoinspected=True -> page shows the refusal, no buttons
def test_full_length_traces(tmp_path): ...          # 100 000 frames, end_frame None -> the time axis covers frame 0..99 999; no empty traces
# tests/bearmind/test_timestamps.py
def test_session_timestamps_matches_export(make_series, tmp_path): ... # same result as the export stage used before
```
- [ ] **Step 3:** FAIL (кроме эталона). **Step 4:** Реализовать. **Step 5:** PASS; `tests/test_examinator_callbacks.py`, `tests/bearmind/test_stage_export.py` и быстрый набор PASS.
- [ ] **Step 6:** Commits: сначала `git commit -m "test(examinator): golden record of the window's decisions" -- tests/test_examinator_golden.py tests/data/examinator_golden.npz`, затем `git commit -m "refactor(examinator): prepared session, save hooks, timestamps from the registry" -- bm_examinator.py bearmind/timestamps.py bearmind/stages/export.py tests/test_examinator_serve.py tests/bearmind/test_timestamps.py`

### Task 2: Быстрые обработчики окна

**Files:**
- Modify: `bm_examinator.py` (обработчики Delete / Merge / Show / Reset view / Revert / Discard / сортировка / выбор; `decimate_minmax`; хранение видимых строк)
- Test: дополнить `tests/test_examinator_callbacks.py`; создать `tests/test_examinator_speed.py` (`@pytest.mark.slow`)

**Interfaces:**
- Produces:
```python
def decimate_minmax(y: np.ndarray, n_out: int = 2000) -> tuple[np.ndarray, np.ndarray]
# (frame index of every kept point, values): the min and the max of each bin; len <= n_out; y shorter than n_out is returned whole
```
- Хранилище окна: `storage.estimates` — единственный рабочий объект; `storage.shown` — номера компонент на экране в порядке строк (полный список или приближённый); `storage.undo` — запись последнего изменения. Помощники теста `_App.shown()` и `_App.partial()` читают видимые строки так, как их хранит новый код.

- [ ] **Step 1: Тесты:**
```python
# tests/test_examinator_callbacks.py (the existing tests keep their expected lists)
def test_revert_button_chain_keeps_zoom(tmp_path): ...      # Show [2, 3, 4, 5]; Delete row 1; click("Revert change") (whole chain) -> shown() == [2, 3, 4, 5]
def test_metrics_panel_cleared_after_change(tmp_path): ...  # select row 2 -> panel names it; Delete -> panel text == "Select a neuron."; same after Merge, Revert change, Discard all
def test_delete_does_not_copy_estimates(tmp_path, monkeypatch): ... # count copy.deepcopy calls on Estimates objects: 0 for Delete, Merge, Show sel., Revert change
def test_delete_sends_no_traces(tmp_path): ...              # document events of a Delete contain no change of trace, time or contour columns
def test_time_axis_sent_once(tmp_path): ...                 # the first document holds one time array of length <= trace_points, not one per neuron
def test_zoom_sends_full_resolution_window(tmp_path): ...   # x range 10..20 s -> traces inside it have one point per frame (<= 2000 points)
def test_merge_recomputes_one_contour(tmp_path, monkeypatch): ... # get_contours receives one column of A during Merge
def test_capcan_sort_after_merge(tmp_path): ...             # capcan input with metrics_df (synthetic): Merge, then sort by a capcan metric -> no exception; merged row last
def test_decimate_minmax(): ...                             # 10 000 points -> <= 2000; global min and max kept; 500 points -> unchanged
# tests/test_examinator_speed.py (slow; synthetic 194 components, 608x608)
def test_delete_payload_small_on_hour_session(tmp_path): ... # 107 107 frames: PATCH-DOC of a Delete <= 1 MB; Python time of the whole chain <= 0.15 s
def test_delete_fast_short(tmp_path): ...                   # 2 000 frames: <= 0.05 s, <= 0.1 MB
def test_merge_time(tmp_path): ...                          # 107 107 frames: Merge chain <= manual_merge time + 0.5 s, PATCH-DOC <= 1 MB
def test_select_is_cheap(tmp_path): ...                     # selection change: <= 5 ms Python, no data columns in the events
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS; `test_golden_click_sequence` PASS без правки эталона; быстрый набор PASS; `pytest -m slow tests/test_examinator_speed.py` PASS.
- [ ] **Step 5:** Commit: `git commit -m "perf(examinator): undo without full copies, in-place plot updates, decimated traces" -- bm_examinator.py tests/test_examinator_callbacks.py tests/test_examinator_speed.py`

### Task 3: Быстрое сохранение и побочные файлы

**Files:**
- Modify: `bm_examinator.py` (`write_side_files` вместо прямого вызова `SaveResults` в режиме сервера; `SaveResults` для ноутбука остаётся), `bearmind/paths.py` (`inspection_progress_dir`, `inspection_cache`, `inspection_lock`)
- Test: дополнить `tests/test_examinator_serve.py`, `tests/bearmind/test_naming_paths.py`

**Interfaces:**
- Produces:
```python
def write_side_files(estimates, folder: Path, base: str, sigma: int = 3) -> None
# <base>traces.csv, <base>filters/filter_NNN.tif, <base>filters_session.mat in `folder`; components: idx_components,
# minus rows with metrics_df["delete"] == 1 when metrics_df exists
def kept_for_side_files(estimates) -> np.ndarray

# paths.py
SeriesPaths.inspection_lock -> Path                 # _bearmind/inspect.lock
SeriesPaths.inspection_progress_dir(sid) -> Path    # 5_Inspection/<SID>/progress
SeriesPaths.inspection_cache(sid) -> Path           # 5_Inspection/<SID>/<SID>_inspect_cache.npz
```
- Результат пишется без сжатия при любом `compress_estimates` (причина — «Changes» п. 5).

- [ ] **Step 1: Тесты:**
```python
def test_side_files_skip_deleted(tmp_path): ...       # metrics_df delete == 1 for component 3 -> traces CSV and .mat have no component 3
def test_side_files_names(tmp_path): ...              # base "NOF_H01_1D_" -> NOF_H01_1D_traces.csv, NOF_H01_1D_filters/, NOF_H01_1D_filters_session.mat in folder
def test_final_save_uncompressed(tmp_path): ...       # bkapp_kwargs {"compress_estimates": True}, server-style final_save -> len(SNR_comp) == A.shape[1], no idx_components_bad_original
def test_inspection_paths(tmp_path): ...
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS; эталон PASS.
- [ ] **Step 5:** Commit: `git commit -m "feat(examinator): side files of the manual result in the background, kept components only" -- bm_examinator.py bearmind/paths.py tests/test_examinator_serve.py tests/bearmind/test_naming_paths.py`

### Task 4: Сервер инспекции на серию и команда `inspect`

**Files:**
- Create: `bearmind/inspect_server.py`
- Modify: `bearmind/cli.py` (команда `inspect SERIES [SID] [--port N] [--no-browser]`), `bearmind/defaults.yaml` (раздел `inspection`)
- Test: `tests/bearmind/test_inspect_server.py` (в том числе команда CLI `inspect`); `tests/bearmind/test_cli.py` не меняется

**Interfaces:**
- Consumes: `bm_examinator.prepare_session / make_app / write_side_files`; `timestamps.session_timestamps`; `SeriesPaths.estimates / manual_result / inspection_dir / inspection_progress_dir / inspection_cache / inspection_lock`; `actions.gate_cleaned / waiting_gate_sids / current_user`; `registry.load_record`; `config.resolve(...)["inspection"]`; `bokeh.server.server.Server`; psutil.
- Produces:
```python
@dataclass
class ServerState:
    alive: bool
    url: str | None
    pid: int | None
    current: list[str]        # sessions open in a browser tab
    prepared: str | None      # the session prepared in the background

def server_state(paths) -> ServerState               # lock JSON {"pid", "create_time", "url", "current", "prepared"}; alive = PID alive with that create_time
def session_url(url: str, sid: str) -> str           # "<url>/?sid=<SID>"
def session_time_s(paths, rec) -> np.ndarray         # len == rec.n_frames; seconds from the first frame
def serve(series_root: Path, sid: str | None = None, port: int | None = None, open_browser: bool = True,
          idle_exit_s: float | None = None) -> int   # 0 normal exit; 2 bad input; 3 already running
class Preparer:                                      # one background thread
    def prepare(self, sid: str) -> None
    def take(self, sid: str) -> PreparedSession      # prepared one, or prepared now
```
- `defaults.yaml`: `inspection: {port: 0, idle_exit_s: 600, webgl: true, trace_points: 2000, wave: {k_mad: 3.0, active_frac: 0.3, warn_frac: null}}`.
- Перед запуском: живой сервер этой серии → `[FAILED] an inspection server of this series is already running: <url>`, код 3. `SID` задан, а `cnmf` не DONE → `[FAILED] CNMF-E of <SID> is not done yet`, код 2.
- Сервер: `Server({"/": app}, port=..., address="localhost", allow_websocket_origin=[...])`; печать `[OK] inspection server: <url>`; при `open_browser` — `webbrowser.open(session_url(url, sid))`. Страница без `sid` показывает список ждущих сессий ссылками.
- Документ сессии: `make_app(prepared=Preparer.take(sid), final_save=..., progress_save=..., refuse_autoinspected=True)`, `bkapp_kwargs` `{"mode": "legacy", "end_frame": None, "enable_gpu_backend": webgl}`. Сверху своя строка страницы: имя сессии, баннер волны (задача 6), после сохранения кнопка «Next: <SID> →» или «No more sessions wait for manual inspection.». «Next» заменяет содержимое того же документа подготовленной сессией и ставит в подготовку следующую.
- `final_save`: если размер или mtime файла оценок либо `cnmf.finished` изменились с открытия → ничего не писать, вернуть «Result is outdated: CNMF-E was recomputed after this session was opened. Open it again.» и напечатать `[WARN]`. Иначе: атомарно записать `manual_result(sid)` без сжатия, вызвать `actions.gate_cleaned(paths, sid, who)`, поставить `write_side_files` в фоновый поток, вернуть «Saved. Decision “cleaned” recorded for <SID>. You can close this tab.». Сбой побочных файлов пишется в лог серии и строкой на странице; решение остаётся.
- `progress_save`: копия в `inspection_progress_dir(sid)` с датой в имени.
- Вторая вкладка браузера с той же сессией: «<SID> is already open in another browser tab.».
- Без открытых вкладок `idle_exit_s` секунд → сервер останавливается, замок удаляется, код 0.
- CLI `inspect`: живой сервер этой серии → `[OK] inspection server is running: <url>` и браузер на `session_url` (если не `--no-browser`), код 0; иначе `serve(...)` в этом процессе.

- [ ] **Step 1: Тесты** (серия с DONE `cnmf` и синтетическими оценками на месте `4_Estimates`; сервер в потоке с `open_browser=False`, порт 0):
```python
def test_session_time_s(make_series, tmp_path): ...          # bonsai timestamps: length n_frames, starts at 0.0
def test_refuses_without_cnmf(series_est): ...               # serve(sid=S2) with cnmf PENDING -> 2
def test_server_serves_page(series_est): ...                 # GET session_url(url, S1) -> 200, "bokeh" in the body; lock has url
def test_final_save_records_cleaned(series_est): ...         # final_save -> manual_result exists, uncompressed; latest gate decision "cleaned"; nothing in 4_Estimates
def test_second_final_save_overwrites(series_est): ...       # two saves -> one file (the last), latest decision "cleaned"
def test_save_after_cnmf_rerun_is_stale(series_est): ...     # after opening, cnmf.finished moved forward -> no file, no decision, the outdated text returned
def test_second_tab_same_session_refused(series_est): ...
def test_second_server_refused(series_est): ...              # serve() while one is alive -> 3, its url in the output
def test_server_exits_when_idle(series_est): ...             # tab opened and closed -> serve returned 0 within idle_exit_s + 5 s, no lock
def test_next_session_prepared_in_background(series_est): ... # after S1 opens, server_state(...).prepared == S2 within 30 s; "Next" builds S2 in <= 1 s
def test_next_session_skips_decided(series_est): ...         # S2 marked not needed while S1 is open -> Next opens S3
def test_contour_cache_reused(series_est, monkeypatch): ...  # second prepare of S1 does not call get_contours; touching the estimates file invalidates the cache
def test_page_texts_are_english(series_est): ...             # page title, status texts, refusals: no Cyrillic
def test_cli_inspect_help(): ...                             # python -m bearmind inspect --help -> 0
def test_cli_inspect_reuses_running_server(series_est, monkeypatch): ... # alive server -> "[OK] inspection server is running", code 0, serve not called
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS; быстрый набор PASS.
- [ ] **Step 5:** Commit: `git commit -m "feat(bearmind): one inspection server per series, next session prepared, save records cleaned" -- bearmind/inspect_server.py bearmind/cli.py bearmind/defaults.yaml tests/bearmind/test_inspect_server.py`

### Task 5: Вкладка Inspection

**Files:**
- Create: `bearmind/gui/inspection_tab.py`
- Modify: `bearmind/launcher.py` (`spawn_detached`, `inspect_command`; `start_detached` идёт через `spawn_detached`), `bearmind/gui/app.py` (вкладка Inspection на месте заглушки; заглушек больше нет)
- Test: `tests/bearmind/gui/test_inspection_tab.py`, дополнить `tests/bearmind/test_launcher.py`; поправить `tests/bearmind/gui/test_app.py` (`test_placeholder_tabs_text` → `test_no_placeholder_tabs_left`)

**Interfaces:**
- Consumes: `monitor.series_view`, `monitor.RecordCache`, `monitor.Row / Cell`, `inspect_server.server_state / session_url`, `actions.gate_not_needed / gate_passed / current_user / failure_text`, `MonitorTab._ask_text`, `CELL_BG`, `CELL_INK`, `EXCLUDED_INK`.
- Produces:
```python
# launcher.py
def spawn_detached(cmd: list[str], paths, log_name: str) -> int     # the Popen part of start_detached; output appended to _bearmind/logs/<log_name>
def inspect_command(series_root: Path) -> list[str]                 # [python.exe, "-m", "bearmind", "inspect", str(series_root), "--no-browser"]

# gui/inspection_tab.py
class InspectionTab(QWidget):
    status_message = Signal(str)
    table: QTableView                    # "Session", "CNMF-E", "Inspection", "Autoinspect", "Export"
    inspect_button: QPushButton          # "Manual inspection"
    not_needed_button: QPushButton       # "Not needed"
    hint_label: QLabel                   # "Opens in your browser. This window stays free."
    server_label: QLabel                 # "Inspection server: not running" | "starting (about 10 s)…" | "running at <url> · next prepared: <SID>"
    def __init__(self, series: Path, parent=None, confirm=None, message=None, spawn=None, open_url=None): ...
    # spawn(cmd) -> pid (default launcher.spawn_detached); open_url(url) (default webbrowser.open): replaced in tests
    def refresh(self) -> None
    def open_inspection(self, sid: str) -> None
    def selected_sids(self) -> list[str]
```
- Строки: неисключённые сессии с DONE `cnmf` (клетки — из `monitor.series_view`, тот же текст, цвет и подсказка), по мыши и sid.
- Manual inspection: одна выделенная сессия с DONE `cnmf`. Сервер жив → `open_url(session_url(url, sid))`. Не жив → `spawn(inspect_command(series))`, статус «Starting the inspection server (about 10 s)…», ждать адреса в замке до 30 с, затем открыть; не дождались → сообщение «The inspection server did not start; see _bearmind/logs/inspect_console.log». Пройденные ворота → вопрос из «Решений плана».
- Not needed: выделенные сессии; подтверждение текстом Monitor (`_ask_text("inspection_not_needed", ...)`); `actions.gate_not_needed`; отказы — `failure_text`.
- Обновление — по таймеру `monitor.refresh_ms`, только когда вкладка видна.

- [ ] **Step 1: Тесты:**
```python
def test_rows_are_sessions_with_cnmf_done(qapp, series4): ...
def test_columns_match_mockup(qapp, series4): ...            # ["Session", "CNMF-E", "Inspection", "Autoinspect", "Export"]
def test_inspect_button_rules(qapp, series4): ...            # one selected with cnmf DONE -> enabled; none / two / cnmf not done -> disabled
def test_open_starts_server_once(qapp, series4): ...         # not running: spawn called with inspect_command; lock with url appears -> open_url(session_url(url, S1)); second open: spawn not called
def test_open_reuses_running_server(qapp, series4): ...      # live server -> spawn not called, open_url called
def test_server_did_not_start_message(qapp, series4): ...    # no url within the wait -> message mentions inspect_console.log
def test_open_already_passed_asks(qapp, series4): ...        # confirm -> False: open_url not called
def test_not_needed_writes_decision(qapp, series4): ...      # confirm -> True: latest gate decision "not_needed"; already passed -> failure_text("already_passed", "") shown
def test_server_label_states(qapp, series4): ...
def test_inspection_tab_texts_are_english(qapp, series4): ...
def test_no_placeholder_tabs_left(qapp, tmp_series): ...     # tabs 1..3 are ScanTab, ThresholdsTab, InspectionTab; Monitor "Manual inspection: N sessions →" switches to index 3
# tests/bearmind/test_launcher.py
def test_spawn_detached_used_by_start(tmp_path, monkeypatch): ... # start_detached delegates to spawn_detached with "runner_console.log"; existing launcher tests unchanged
def test_inspect_command(monkeypatch): ...                   # python, not pythonw; ends with ["inspect", <series>, "--no-browser"]
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS; быстрый набор PASS.
- [ ] **Step 5:** Commit: `git commit -m "feat(bearmind): inspection tab opens the series' inspection server" -- bearmind/gui/inspection_tab.py bearmind/launcher.py bearmind/gui/app.py tests/bearmind/gui/test_inspection_tab.py tests/bearmind/test_launcher.py tests/bearmind/gui/test_app.py`

### Task 6: Подсказка «looks like a calcium wave» в окне инспекции

**Files:**
- Create: `bearmind/wave.py`
- Modify: `bearmind/inspect_server.py` (подготовка сессии считает долю, страница показывает баннер), `bearmind/defaults.yaml` (`inspection.wave.warn_frac`)
- Test: `tests/bearmind/test_wave.py`, дополнить `tests/bearmind/test_inspect_server.py`

**Interfaces:**
- Produces:
```python
def wave_frac(C: np.ndarray, idx: np.ndarray, k_mad: float = 3.0, active_frac: float = 0.3) -> float
# share of frames in which at least active_frac of the components in idx are above median + k_mad * 1.4826 * MAD of their own trace
def wave_text(frac: float, active_frac: float) -> str
# "Looks like a calcium wave: in 1.2 % of frames more than 30 % of the components fire at once."
```
- Баннер показывается, если `warn_frac` задан и доля выше него. Monitor, `checks.py`, этап CNMF-E и реестр не меняются.
- Порог `warn_frac`: посчитать `wave_frac` по всем уже посчитанным выходам CNMF-E, какие есть на машине (только чтение, новых прогонов CNMF-E нет), выбрать так, чтобы на сессиях, которые автор считает нормальными, баннера не было, и записать в отчёт задачи 7, на скольких сессиях он основан. Пока порога нет — `null`, баннера нет.

- [ ] **Step 1: Тесты:**
```python
def test_wave_frac_synthetic(): ...            # 50 traces, all active in 10 frames of T -> wave_frac == 10 / T
def test_wave_frac_flat_traces(): ...          # zero MAD -> no division error, 0.0
def test_wave_text_english(): ...
def test_wave_banner_only_in_page(series_est): ... # warn_frac below the session's frac -> banner in the page; monitor.series_view(...) cells and warnings contain no "wave"
def test_no_banner_without_threshold(series_est): ...
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать; выбрать порог по имеющимся оценкам. **Step 4:** PASS; быстрый набор PASS.
- [ ] **Step 5:** Commit: `git commit -m "feat(bearmind): calcium-wave hint inside the inspection window" -- bearmind/wave.py bearmind/inspect_server.py bearmind/defaults.yaml tests/bearmind/test_wave.py tests/bearmind/test_inspect_server.py`

### Task 7: Проверка на реальных сессиях и замеры «до / после»

**Files:**
- Create: `docs/superpowers/specs/<дата>-phase5-check.md` (`<дата>` — день выполнения, `YYYY-MM-DD`)

- [ ] **Step 1: Корень проверки.** Создать `H:\Projects\_bearmind_test\phase5_short`: junction `1_Raw` → `H:\Projects\_bearmind_test\series_short\1_Raw`, копии `_bearmind`, `3_MC`, `4_Estimates` из `series_short` (источник только читается). Выполнить `status`: crop, mc, thresholds, cnmf должны быть DONE и актуальны. Иначе шаги 2–3 пропустить (батч стал бы заново делать MC и CNMF-E) и записать «не проверено».
- [ ] **Step 2: Сервер и ворота.** Запустить `python -m bearmind inspect <корень> NOF_H01_1D --no-browser` в фоне. Страница `session_url` отвечает 200. Вызвать сохранение программно (как в тестах задачи 4). Проверить `manual_result`, решение `cleaned` и подготовленную следующую сессию.
- [ ] **Step 3: Батч после ворот.** `run <корень> --once`: автоинспекция берёт `5_Inspection/<SID>/<SID>_manual_estimates.pickle` (проверить по реестру и логу сессии), экспорт проходит. Это автоинспекция и экспорт клипа 2000 кадров, MC и CNMF-E не запускаются.
- [ ] **Step 4: Замеры тем же способом, что на странице.** `series_short\5_Inspection\NOF_H01_1D\NOF_H01_1D_processed.pickle` и `series_short\4_Estimates` (Merge), только чтение; копии, продолженные во времени до 18 000 и 107 107 кадров, — в памяти. Измерить Delete, Merge, выбор нейрона, Save results до свободного окна, переход к подготовленной сессии, первое открытие, объём `PATCH-DOC`. Таблица «до (страница) / после / цель» с метками measured / tiled copy.
- [ ] **Step 5: Длинная сессия.** CNMF-E ради проверки не запускать. Если на машине есть готовые оценки длиннее 90 000 кадров, открыть их (только чтение): время подготовки, пик памяти, длина оси времени равна числу кадров. Иначе записать «не проверено»; длину покрывают `test_full_length_traces` и `test_delete_payload_small_on_hour_session`.
- [ ] **Step 6: Инструкция для автора** (протокол STE, по-русски, названия кнопок по-английски): открыть вкладку Inspection, нажать Manual inspection, удалить компоненту, склеить две, нажать Save results, нажать Next, увидеть `cleaned` в Monitor. Отдельно: оценить плавность выбора нейрона и прокрутки в браузере (WebGL). Это самая сильная непроведённая проверка: браузер автоматически не проверяется.
- [ ] **Step 7:** Commit: `git commit -m "docs: phase 5 check on real sessions, speed before and after" -- docs/superpowers/specs/<дата>-phase5-check.md`
