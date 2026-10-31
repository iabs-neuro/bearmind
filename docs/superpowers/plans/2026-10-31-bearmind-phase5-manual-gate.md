# BEARMiND фаза 5: ворота ручной инспекции — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ручная инспекция (нынешнее окно Bokeh `bm_examinator.ExamineCells`) открывается из сводки для выбранной сессии. Она читает `4_Estimates/<SID>_estimates.pickle`, сохраняет результат в `5_Inspection/<SID>/` и сама записывает решение «почищено». Батч сразу подхватывает сессию для автоинспекции.

**Architecture:** В `bm_examinator.py` меняется только обвязка. Появляется режим `serve=True`: функция возвращает Bokeh-приложение вместо `show()`. Пути сохранения и таймстемпы передаются параметрами, а `output_notebook()` больше не вызывается при импорте. Новый модуль `bearmind/inspect_app.py` поднимает Bokeh-сервер на свободном порту отдельным процессом (`python -m bearmind inspect SERIES SID`), открывает браузер и после финального сохранения вызывает `actions.gate_cleaned`. Вкладка мониторинга (фаза 2) получает кнопку «Ручная инспекция».

**Tech Stack:** Python 3.12, Bokeh 3.10 (уже в окружении `bearmind`), CaImAn 1.13.2 (`Estimates`), PySide6 6.11.2; pytest, `urllib` для проверки сервера.

**Spec:** `docs/superpowers/specs/2026-10-30-bearmind-batch-pipeline-design.md` — §3 (ручная инспекция до автоинспекции), §4 пп. 6–7, §5.9, §5.13. Опирается на ядро (ворота `stages/gate.py`, правило R28: «почищено» действует, только если результат новее оценок CNMF-E) и на план фазы 2 (`actions.py`, `gui/monitor_tab.py`).

## Global Constraints

- Логику окна инспекции не менять: отображение, сортировки, удаление компонент, режимы `legacy` и `capcan` остаются как есть (§3, §5.9 «остаётся нынешнее окно Bokeh, меняется обвязка»).
- Ноутбучный путь (`ExamineCells(fname, ...)` без новых параметров) ведёт себя как раньше, кроме одного: `output_notebook()` вызывается внутри функции, а не при импорте модуля.
- Без лимита кадров: трейсы показываются на всю длину сессии (часовая сессия — 107 107 кадров).
- Результат инспекции — только `SeriesPaths.manual_result(sid)` = `5_Inspection/<SID>/<SID>_manual_estimates.pickle`; промежуточные сохранения — в `5_Inspection/<SID>/progress/`. В текущую папку ничего не пишется.
- Таймстемпы трейсов — склеенные таймстемпы сессии из реестра в секундах от первого кадра, а не поиск `*timestamp.csv` по `CONFIG['ROOT']`.
- Новых пакетов нет. Окно по-русски, консоль ASCII. Коммиты без строк об ИИ, только `git commit -- <пути>`.

## Review Focus

- Финальное сохранение нажали дважды, или сохранили, закрыли и открыли снова: в реестре последнее решение «почищено», файл результата один и последний. Тест: задача 2 `test_second_final_save_overwrites`.
- Браузер закрыли без сохранения: решения нет, сессия по-прежнему ждёт инспекции, сервер завершился. Тест: задача 2 `test_server_stops_after_last_session`.
- CNMF-E сессии пересчитали, пока инспекция открыта: сохранение старого результата не проходит ворота (R28). Окно инспекции сообщает, что результат устарел. Тест: задача 2 `test_save_after_cnmf_rerun_is_stale`.
- Инспекцию одной сессии открыли второй раз, пока первая открыта: второй сервер не запускается, сообщение называет адрес первого. Тест: задача 3 `test_second_inspection_same_session_refused`.
- Часовая сессия: окно открывается, трейсы полной длины. Тест: задача 5, шаг 2.

## Решения плана (моё решение, автор может отменить)

- «Почищено» записывается само, когда в окне Bokeh нажали «Save results». Отдельной кнопки в сводке для этого нет.
- Инспекция идёт отдельным процессом с Bokeh-сервером на `localhost` и открывается в браузере по умолчанию. Окно BEARMiND не блокируется.
- Подсказка «похоже на кальциевую волну» (§5.9, «желательно, не обязательно») — задача 4. Автор может её отложить, остальные задачи от неё не зависят.

---

### Task 1: Обвязка `ExamineCells` для запуска с сервера

**Files:**
- Modify: `bm_examinator.py`
- Test: `tests/bearmind/test_examinator_serve.py`

**Interfaces:**
- Produces:
```python
def ExamineCells(fname, default_fps=20, bkapp_kwargs=None, *, serve: bool = False,
                 final_path: str | None = None, progress_dir: str | None = None,
                 time: np.ndarray | None = None, on_final_save=None):
    # serve=False: как раньше (output_notebook() + show(bkapp)), возвращает None
    # serve=True: возвращает bkapp(doc); output_notebook() и show() не вызываются
    # final_path: куда пишет "Save results" (по умолчанию прежнее имя рядом с fname)
    # progress_dir: куда пишет "Save progress" (по умолчанию прежнее место)
    # time: секунды для каждого кадра C; если задано, get_timestamps не вызывается
    # on_final_save(path: str): вызывается после записи final_path и .mat
```
- Запись `final_path` атомарная: `<final_path>.partial` → `os.replace`.
- Ручной результат сохраняется **без сжатия** (`compress=False`, независимо от `bkapp_kwargs`). Причина: сжатие (`estimates_compression`) берёт `SNR_comp[:N]` и `r_values[:N]` вместо значений хороших компонент. Тогда признаки `caiman_snr` и `caiman_r_score` ML-модели автоинспекции перепутаются, а этап autoinspect откажется от такого входа (защита задачи 19 ядра; находка 2026-11-01, `docs/superpowers/specs/ml-filter-2026-11-01/findings.md`).

- [ ] **Step 1: Тесты** (фикстура `tiny_estimates`: `caiman.source_extraction.cnmf.estimates.Estimates` с 5 компонентами 32×32 и 200 кадрами; поля `A, C, b, f, YrA, dims, imax`):
```python
def test_import_has_no_notebook_side_effect(monkeypatch): ...   # import bm_examinator в свежем subprocess: bokeh.io output_notebook не вызван
def test_serve_returns_app(tiny_estimates, tmp_path): ...       # ExamineCells(..., serve=True) -> callable; bkapp(Document()) строит корень без ошибок
def test_time_param_used(tiny_estimates, tmp_path, monkeypatch): ...   # get_timestamps не вызван; длина оси времени == C.shape[1]
def test_final_save_path_and_callback(tiny_estimates, tmp_path): ...    # вызвать колбэк кнопки "Save results" из документа -> final_path есть, on_final_save(path) вызван
def test_final_save_uncompressed(tiny_estimates, tmp_path): ...         # bkapp_kwargs={'compress_estimates': True} -> в final_path len(SNR_comp) == A.shape[1], нет idx_components_bad_original
def test_progress_dir(tiny_estimates, tmp_path): ...
def test_full_length_traces(tmp_path): ...                      # 100 000 кадров -> данные трейса в ColumnDataSource полной длины
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. Минимальные правки: параметры, ветка `serve`, перенос `output_notebook()`. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `refactor(examinator): serve mode with explicit save paths and timestamps`.

### Task 2: Сервер инспекции и команда `inspect`

**Files:**
- Create: `bearmind/inspect_app.py`
- Modify: `bearmind/cli.py` (команда `inspect SERIES SID [--port N] [--no-browser]`)
- Test: `tests/bearmind/test_inspect_app.py`

**Interfaces:**
- Consumes: `ExamineCells(serve=True, ...)` из задачи 1; `SeriesPaths.estimates / manual_result / inspection_dir`; `timestamps.read_timestamps / stitch` по `rec.parts`; `actions.gate_cleaned`; `actions.current_user`; параметры автоинспекции из `config.resolve` (`ml_model_path`, `ml_threshold`) для режима `capcan`.
- Produces:
```python
def session_time_s(paths, rec) -> np.ndarray          # длина == rec.n_frames; секунды от первого кадра
def build_app(paths, sid: str, who: str):             # bkapp для Bokeh-сервера
def serve(series_root: Path, sid: str, who: str, port: int = 0, open_browser: bool = True,
          idle_exit_s: float = 10.0) -> int           # код возврата CLI
def inspection_lock(paths, sid) -> Path               # 5_Inspection/<SID>/inspect.lock: {"pid", "create_time", "url"}
```
- Перед запуском: `cnmf` не DONE → `[FAILED] CNMF-E of <SID> is not done yet`, код 2. Живой `inspect.lock` этой сессии → `[FAILED] inspection of <SID> is already open: <url>`, код 3.
- Сервер: `bokeh.server.server.Server({"/": app}, port=port, address="localhost", allow_websocket_origin=[...])`. Печать `[OK] inspection of <SID>: <url>`; браузер — `webbrowser.open(url)`. Последний сеанс браузера закрыт и новых нет `idle_exit_s` секунд → сервер останавливается, замок удаляется, код 0.
- `on_final_save`: `actions.gate_cleaned(paths, sid, who)`. Если ворота отказали (результат старее оценок, R28), в окне Bokeh появляется сообщение «Результат устарел: CNMF-E пересчитан после открытия. Откройте инспекцию заново.», а в консоли — `[WARN]`.

- [ ] **Step 1: Тесты** (серия с DONE cnmf и `tiny_estimates` на месте `4_Estimates`):
```python
def test_session_time_s(series_est): ...                     # из таймстемпов bonsai; длина n_frames, начинается с 0
def test_refuses_without_cnmf(series_est): ...               # код 2
def test_server_serves_page(series_est): ...                 # serve(open_browser=False) в потоке; GET url -> 200, "bokeh" в теле
def test_final_save_records_cleaned(series_est): ...         # колбэк сохранения -> manual_result есть, последнее решение gate "cleaned"
def test_second_final_save_overwrites(series_est): ...
def test_save_after_cnmf_rerun_is_stale(series_est): ...     # после открытия cnmf.finished сдвинули вперёд -> решения нет, сообщение
def test_server_stops_after_last_session(series_est): ...    # сеанс открыт и закрыт -> serve вернул 0 за idle_exit_s + 5 с, замка нет
def test_lock_refuses_second(series_est): ...                # код 3, url первого в сообщении
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): manual inspection server; final save records the gate decision`.

### Task 3: Кнопка «Ручная инспекция» в сводке

**Files:**
- Modify: `bearmind/gui/monitor_tab.py`
- Test: дополнить `tests/bearmind/gui/test_actions_ui.py`

**Interfaces:**
- Consumes: `inspect_app.inspection_lock`, `launcher.runner_command` как образец команды (`[python, -m, bearmind, inspect, SERIES, SID]`).
- Produces: кнопка «Ручная инспекция» в панели кнопок фазы 2; метод `MonitorTab.open_inspection(sid) -> None`; параметр `spawn=None` в `MonitorTab.__init__` (по умолчанию `subprocess.Popen`, подменяется в тестах).
- Доступна, если выделена одна неисключённая сессия с DONE `cnmf`.
- Живой замок инспекции этой сессии → сообщение «Инспекция NOF_H01_1D уже открыта: <url>» и `webbrowser.open(url)`, новый процесс не запускается.
- Статусная строка: «Инспекция NOF_H01_1D открывается в браузере…».
- Сессии, которые ждут ручной инспекции, с предупреждением «похоже на кальциевую волну» (задача 4) стоят в своей мыши первыми.

- [ ] **Step 1: Тесты:**
```python
def test_inspection_button_enabled_rules(qapp, series3): ...
def test_open_inspection_spawns_command(qapp, series3): ...        # spawn получил [..., "inspect", <серия>, "S1"]
def test_second_inspection_same_session_refused(qapp, series3): ... # живой замок -> spawn не вызван, сообщение с url
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): open manual inspection from the summary`.

### Task 4: Подсказка «похоже на кальциевую волну»

**Files:**
- Modify: `bearmind/stages/cnmf.py` (метрика `wave_frac` в QC), `bearmind/checks.py`, `bearmind/defaults.yaml` (`checks.cnmf_wave_frac: {warn: <значение>}`), `bearmind/texts.py`
- Test: дополнить `tests/bearmind/test_checks.py`, `tests/bearmind/test_stage_cnmf.py`

**Interfaces:**
- Produces: метрика этапа cnmf `wave_frac` — доля кадров, в которых активна не меньше чем `checks.wave_active_frac` (по умолчанию 0.3) доля принятых компонент. Компонента активна в кадре, если её `C` выше `median + 3 × 1.4826 × MAD` своего трейса. Проверка `cnmf_wave_frac` выше порога → «предупреждение» с текстом «похоже на кальциевую волну: в 1.2 % кадров вспыхивает больше 30 % компонент».
- Метрика не входит в хэш этапа; у записей, посчитанных раньше, проверки просто нет.
- Порог `warn` выбрать по уже посчитанным сессиям (новых прогонов CNMF-E не делать) так, чтобы на них не было ложных предупреждений, и записать в отчёт задачи 5, на чём он основан.

- [ ] **Step 1: Тесты:**
```python
def test_wave_frac_synthetic(): ...        # 50 трейсов, в 10 кадрах активны все -> wave_frac == 10 / T
def test_wave_check_levels(): ...
def test_wave_check_missing_metric(): ...  # старый DONE без wave_frac -> проверки нет, ошибок нет
```
- [ ] **Step 2:** FAIL. **Step 3:** Реализовать. **Step 4:** PASS + быстрый набор.
- [ ] **Step 5:** Commit: `feat(bearmind): calcium-wave hint after CNMF-E`.

### Task 5: Проверка на реальных сессиях

**Files:**
- Create: `docs/superpowers/specs/<дата>-phase5-check.md`

- [ ] **Step 1:** На копии `series_short` (сессии по ~2000 кадров с готовым CNMF-E) открыть инспекцию первой сессии через `python -m bearmind inspect ... --no-browser`. Страница отвечает 200. Вызвать финальное сохранение программно (как в тестах задачи 2). Запустить батч: автоинспекция должна взять `5_Inspection/<SID>/<SID>_manual_estimates.pickle` (проверить по реестру и логу), экспорт — пройти.
- [ ] **Step 2:** Длинная сессия: CNMF-E ради этой проверки не запускать (правило автора 2026-10-31). Если на машине уже есть готовые оценки сессии длиннее 90 000 кадров, открыть инспекцию по ним: время загрузки, пик памяти, длина оси времени равна числу кадров. Если таких оценок нет — записать «не проверено»; длину трейсов покрывает тест задачи 1 `test_full_length_traces`.
- [ ] **Step 3:** Посчитать `wave_frac` по всем сессиям с готовым CNMF-E, какие есть (без новых прогонов CNMF-E), обосновать порог задачи 4 и сказать, на скольких сессиях он основан.
- [ ] **Step 4:** В отчёт — короткая инструкция для автора по протоколу STE: что открыть в браузере и что нажать, чтобы проверить окно руками (удалить компоненту, сохранить, увидеть «почищено» в сводке). Автоматически это не проверяется.
- [ ] **Step 5:** Commit: `docs: phase 5 check on real sessions`.
