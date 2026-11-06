# BEARMiND: исправления автора (волна A1) — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** пять исправлений автора в готовом батч-конвейере: окно ExamineCells не путает нейроны после правок, группы слияния автоинспекции объединяют только соседние компоненты, одно правило экспорта во всех трёх путях, воспроизводимая автоинспекция, пул MC из 16 процессов.

**Architecture:** правки в существующих модулях, без новых пакетов. `bm_examinator.py` получает `make_app` (сборщик документа Bokeh) ради теста без браузера. В `auto_inspector.py` шаг слияний выносится в `assign_merge_groups`, а группы строятся как компоненты связности графа пар. Экспорт bearmind и `batch_export` вызывают `export_estimates_data.ml_approved_components`. Зерно шума driada идёт из `autoinspect.seed` до каждого `Neuron(...)`. Размер пула MC берётся из нового раздела `pool`, который не входит ни в один хэш этапа.

**Tech Stack:** Python 3.12, CaImAn 1.13.2, Bokeh 3.10, driada 1.2.2, pandas, SciPy, pytest. Окружение conda `bearmind`.

**Spec:** the author's decisions recorded in `docs/superpowers/specs/ml-filter-2026-11-01/findings.md` (section "Обновление 2026-11-06") plus this list (раздел «Decisions» ниже, дословно).

## Decisions

1. ExamineCells (bm_examinator.py, Bokeh): after a delete or merge, the next selection deletes/merges the wrong neuron. Cause (verified by reading): `del_callback` and `merge_callback` (around lines 1043-1177) set `src_partial.data` to the full current list but never update `storage.estimates_partial`, so the next `sel_inds` are mapped through the stale partial list; `revert_callback` (around 1225) restores `prev_data_partial = storage.prev_data` (typo: should be `storage.prev_data_partial`). Fix it now, with a regression test that drives the real callbacks headless. `ExamineCells(fname, ...)` builds a nested `bkapp(doc)` and calls `show(bkapp)`; the plan may add a minimal refactor that exposes the doc builder (e.g. `make_app(...)` returning the doc function, `ExamineCells` keeps its signature and behaviour) so a test can build it on `bokeh.document.Document()` and trigger button callbacks. Keep the fix minimal: a later phase will speed the window up (do not optimise now). Find a small estimates input for the test (a synthetic CaImAn Estimates or a pickle under H:\Projects\_bearmind_test\series_short\4_Estimates, read-only) and say which.
2. Merge grouping in autoinspection glues distant cells. `auto_inspector.metrics_to_decision` (around lines 1820-1866) loops over each correlation cluster (`corr_groups`, connected components of the correlation graph) and gives `merge = cluster_id` to every neuron that has at least one partner within `pxlthr_distance_boundary` (FBD <= 15 px) and |dSNR| <= d_snr_thr; so two close pairs far apart in one cluster become one merged component (findings.md item 4: 194 of 2506 groups not FBD-connected, an RFC group of 256). Fix: keep the pairwise rule and the delete logic (same loop order, same delete decisions) but record merge edges; after the loop and the existing deleted-neurons cleanup, each connected component (size >= 2) of the merge-edge graph among non-deleted neurons gets its own merge id, unique across clusters; a neuron left alone gets merge 0. Tests on synthetic metrics_df/FBD: two far pairs in one cluster -> two groups; chain A-B-C -> one group; a member deleted later drops out; delete column identical to the old code on the same input. Then a read-only re-evaluation on stored artifacts of the May 2026 corpus (the investigation scripts are in `docs/superpowers/specs/ml-filter-2026-11-01/`: `merge_spread_corpus.py`, `old_corpus.py` show where the artifacts live, e.g. `*_metrics_with_decisions.csv`, `*_FBD.npy`); recompute only the grouping from stored inputs, report groups before/after, max centre spread, number not FBD-connected (must become 0), and add the numbers to findings.md. No autoinspection or CNMF re-runs. Never write into H:\Dataset or any source folder.
3. One export rule in all three paths: `export_estimates_data.ml_approved_components` ((P >= threshold or decision == 'from_merge') and delete != 1), already used by `export_estimates_data.main()` and the notebook. Switch `batch_export.export_one` (lines ~90-115) and `bearmind/stages/export.py::_ml_filter` to it. Rename `bearmind.stages.COMPONENT_RULE` from "ml_probability" to a name for the new rule (choose one, e.g. "ml_or_merged_not_deleted"); it is part of the export stage hash (bearmind/runner.py stage_params, "export"), so finished sessions re-export on the next run: say so in `docs/superpowers/specs/2026-11-01-operator-notes.md` (Russian, STE style). Update `rule_counts` and its docstring (duplicates exported and merge products dropped should now be 0) and the tests in tests/bearmind/test_stage_export.py, test_runner.py, test_e2e_real.py that pin the old rule. Merged components exist only after task 2's grouping fix is in, so order task 2 before task 3.
4. Autoinspection reproducible: driada adds 1e-8 noise to each trace without a seed (`calcium_preprocessing(seed=None)`, found via autoinspection; locate the exact call site under auto_inspector.py / ae_launch.py / their imports). Add a fixed seed from config (`autoinspect.seed`, default 0 in bearmind/defaults.yaml; it becomes part of the autoinspect stage params/hash). Test: two runs of the affected function on the same trace give identical output; with the seed, the stage params change. 
5. MC pool of 16 processes instead of 31 (measured earlier: same speed, less memory). Find where the MC pool size comes from (bearmind/defaults.yaml `mc` section, bearmind/stages/mc.py, bearmind/pool.py) and set 16; fold this one-line config change into task 4 or a small task.

## Что показала разведка перед планом (важно для задачи 1)

Причина из решения 1 проверена прогоном без браузера (скрипт в scratch-папке: синтетические оценки, документ `bokeh.document.Document`, кнопки нажимаются через `Button._trigger_event(ButtonClick(...))`).

- **Устаревший `storage.estimates_partial` после delete и merge в окне не проявляется.** Каждая кнопка вызывает цепочку обработчиков: Delete — `del_callback`, `restore_callback`, `sort_callback` (bm_examinator.py:1575-1600). `restore_callback` сразу ставит `storage.estimates_partial = storage.estimates`. Последовательности «удалить, удалить», «слить, удалить», «приблизить, удалить», «удалить, отменить, удалить» удаляют верные компоненты. Опечатка в `revert_callback` тоже маскируется: после него идёт `restore_callback`. Исправить оба места всё равно нужно: каждый обработчик должен оставлять согласованное состояние сам, а не за счёт соседа по цепочке.
- **Подтверждённая ошибка 1: второе «Show sel.»** внутри приближенного вида показывает другие нейроны. `show_callback` переводит номера строк через полный список `estimates.idx_components`, а строки показывают частичный список. Прогон: вид [4, 5, 6], выбраны строки 1 и 2, ожидалось [5, 6], показано [4, 5].
- **Подтверждённая ошибка 2: после delete и merge окно выделяет все строки** (`src_partial.selected.indices = np.arange(...)`). Повторное нажатие Delete или Merge без нового выбора удаляет или сливает все оставшиеся нейроны. TapTool в Bokeh 3.10 работает в режиме `toggle` (bokeh.js: `update_through_toggle`, новый выбор = попадание минус текущий выбор). Поэтому первый клик по нейрону после правки снимает выделение со всех строк и ничего не выбирает. Панель метрик при этом показывает прежний нейрон. Shift+клик добавляет нейрон к «всем».
- Задача 1 исправляет всё это: синхронизация из решения 1, опечатка, обе подтверждённые ошибки. Проверка в браузере не выполнялась (см. «Не проверено»).

## Global Constraints

- Окружение `bearmind` (C:\Users\User\anaconda3\envs\bearmind\python.exe, CaImAn 1.13.2, Py 3.12). Команды ниже запускаются из корня репозитория через `conda run -n bearmind ...`: exe без активированного окружения не загружает DLL torch.
- Быстрый набор `conda run -n bearmind python -m pytest -q -m "not slow and not realdata" tests` остаётся зелёным (сейчас 347 passed); после каждой задачи — 347 + новые тесты, 0 failed.
- NO heavy MotionCorrection or CNMF-E runs (author's rule: we change only the wrapper; if a check needs real data, use the 2000-frame series H:\Projects\_bearmind_test\series_short read-only, or skip and say so).
- Новые пакеты можно ставить в окружение `bearmind`, если они не конфликтуют с CaImAn (этому плану они не нужны).
- Весь текст программы, окон и консоли — по-английски; вывод в консоль только ASCII (`[OK]` / `[FAILED]`).
- Комментарии в коде объясняют, почему код такой; истории правок в них нет.
- Коммиты только `git commit -q -m "<msg>" -- <пути>`; новые файлы сначала `git add -- <путь>`. Никаких `git add -A`, `git commit -a`, строк `Co-Authored-By` и других отметок ИИ. Не пушить.
- В репозитории работают другие агенты: не трогать `BEARMiND_full_pipeline.ipynb` и `INFO_promts.txt`, коммитить только свои пути.
- Исходные данные (G:, H:\Dataset, H:\Projects\LT1, папки 1_Raw) только читаются. Series_short тоже только читается: этот план ничего туда не пишет и батч на ней не запускает.
- Задача 2 не запускает автоинспекцию и CNMF: группировка пересчитывается по сохранённым файлам.

## Review Focus

1. **Delete или Merge нажали второй раз без нового выбора** (двойной клик) — больше ничего не удаляется и не сливается. Тест `test_no_selection_after_a_change` — в задаче 1.
2. **Сессия с одной компонентой** (`corr_groups` = None) или без кластеров корреляции — шаг слияний не падает, `merge` = 0 у всех. Тест `test_single_component_has_no_group` — в задаче 2.
3. **Пара ровно на пороге** (FBD == 15, |dSNR| == 10) — сливается, как в старом правиле (`<=`); FBD 15.0001 — не сливается. Тест `test_pair_at_the_cutoff` — в задаче 2.
4. **NaN в `caiman_snr` внутри кластера** — решения delete те же, что у старого кода. Случайные входы с NaN — в `test_random_inputs_keep_old_deletes`, задача 2.
5. **Processed pickle старого прогона без столбцов `decision` и `delete`** — экспорт берёт правило по вероятности, без KeyError. Тест `test_rule_without_decision_columns` — в задаче 3.

## Структура файлов

```
bm_examinator.py                 # make_app; ExamineCells = show(make_app(...)); 4 правки в обработчиках
auto_inspector.py                # assign_merge_groups (шаг слияний); seed до Neuron(...)
ae_launch.py                     # run_auto_inspection(seed=0)
batch_export.py                  # export_one -> ml_approved_components
bearmind/stages/__init__.py      # COMPONENT_RULE = "ml_or_merged_not_deleted"
bearmind/stages/export.py        # _ml_filter -> ml_approved_components; docstring rule_counts
bearmind/stages/mc.py            # mc_pool_size(cfg)
bearmind/defaults.yaml           # autoinspect.seed: 0; раздел pool: mc_processes: 16
tests/test_examinator_callbacks.py   # новый
tests/test_merge_groups.py           # новый
tests/test_auto_inspector.py         # тесты зерна
tests/bearmind/test_stage_export.py, test_runner.py, test_e2e_real.py, test_stage_autoinspect.py,
tests/bearmind/test_config.py, test_stage_mc.py
docs/superpowers/specs/ml-filter-2026-11-01/merge_regroup.py, merge_regroup_out.txt, findings.md
docs/superpowers/specs/2026-11-01-operator-notes.md
```

---

### Task 1: ExamineCells — частичный список, выбор после правки, второе приближение

**Files:**
- Modify: `bm_examinator.py:411-1659` (`ExamineCells` -> `make_app` + обёртка), `:1075-1079` (`del_callback`), `:1173-1177` (`merge_callback`), `:1190-1195` (`show_callback`), `:1228-1229` (`revert_callback`)
- Create: `tests/test_examinator_callbacks.py`

**Interfaces:**
- Produces: `bm_examinator.make_app(fname, default_fps=20, bkapp_kwargs=None) -> Callable[[bokeh.document.Document], None]` — тело нынешнего `ExamineCells` без `show`, возвращает вложенный `bkapp`. `ExamineCells(fname, default_fps=20, bkapp_kwargs=None)` сохраняет сигнатуру и docstring и делает `show(make_app(fname, default_fps=default_fps, bkapp_kwargs=bkapp_kwargs))`.

Вход теста — синтетический CaImAn `Estimates` (8 гауссовых футпринтов на 48x48, 300 кадров), собранный в самом тесте. Pickle из series_short не нужен. Такой набор уже проверен прогоном: окно строится в режиме `legacy`, merge работает.

- [ ] **Step 1: Тесты** в `tests/test_examinator_callbacks.py`

Первой строкой `import torch  # noqa: F401` с комментарием, как в `tests/test_merge_alignment.py` (shm.dll). Вспомогательные функции:

```python
def _estimates(folder) -> Path:
    """8 round footprints on 48x48, 300 frames; enough for delete, merge, zoom and revert."""
    # Estimates(A=csc(A), C=C, b=None, f=None, R=YrA, dims=(48, 48)); A[:, k] = Gaussian sigma 2 at
    # ((8 + 10*(k % 4)), (10 + 20*(k // 4))), raveled order "F"; C[k] = sparse spikes (rng(0), p 0.03)
    # convolved with exp(-t/5) + 0.01*k; YrA = 0.01*normal; S = zeros; bl = c1 = zeros(8); neurons_sn = ones(8);
    # g = full((8, 1), 0.9); idx_components = arange(8); idx_components_bad = []; SNR_comp = arange(8) + 1;
    # r_values = linspace(0.5, 0.9, 8); imax = ones((48, 48)); cnmf_dict = CNMFParams(params_dict={"p": 1, "fr": 20}).to_dict()
    # written to folder / "NOF_H01_1D_estimates.pickle" (the window takes the session id from the name)

class _App:  # make_app(...)(Document()) and handles to what the user clicks
    # buttons: {label: Button}; src: the only ColumnDataSource whose data has "idx" (src_partial);
    # storage: buttons["Delete sel."]._event_callbacks["button_click"][0].keywords["storage"]
    def select(self, rows): self.src.selected.indices = list(rows)
    def click(self, label): b = self.buttons[label]; b._trigger_event(ButtonClick(b))   # whole chain
    def first(self, label): b = self.buttons[label]; b._event_callbacks["button_click"][0](ButtonClick(b))  # own callback only
    def shown(self): return [int(i) for i in self.src.data["idx"]]
    def partial(self): return [int(i) for i in self.storage.estimates_partial.idx_components]

def _open(tmp_path) -> _App:  # make_app(str(_estimates(tmp_path)), 20, {"mode": "legacy", "end_frame": 300})
```

`end_frame` передаётся явно: по умолчанию окно берёт `end_frame=0` и строит пустые трейсы (старое поведение, не трогаем).

```python
def test_partial_list_matches_the_view_after_each_callback(tmp_path):
    """Each callback leaves storage.estimates_partial equal to the rows it shows, without restore_callback."""
    app = _open(tmp_path)
    app.select([2]); app.first("Delete sel.")
    assert app.partial() == app.shown() == [0, 1, 3, 4, 5, 6, 7]
    app.select([0, 1]); app.first("Merge sel.")
    assert app.partial() == app.shown() == [1, 2, 3, 4, 5, 6]   # CaImAn renumbers; the merged one is 6
    app.click("Discard all")
    app.select([2, 3, 4, 5]); app.click("Show sel.")
    app.select([1]); app.click("Delete sel.")
    app.first("Revert change")
    assert app.partial() == app.shown() == [2, 3, 4, 5]   # the zoomed view before the delete


def test_delete_merge_zoom_pick_the_selected_rows(tmp_path):
    app = _open(tmp_path)
    steps = [([2], "Delete sel.", [0, 1, 3, 4, 5, 6, 7]),
             ([2], "Delete sel.", [0, 1, 4, 5, 6, 7]),
             ([0, 1], "Merge sel.", [2, 3, 4, 5, 6]),     # CaImAn renumbers; the merged one is 6
             ([0], "Delete sel.", [3, 4, 5, 6]),
             ([1, 2, 3], "Show sel.", [4, 5, 6]),
             ([1, 2], "Show sel.", [5, 6]),               # rows of the zoomed view, not of the full list
             ([0], "Delete sel.", [3, 4, 6])]
    for rows, label, expected in steps:
        app.select(rows); app.click(label)
        assert app.shown() == expected, (rows, label)


def test_revert_restores_the_last_change(tmp_path):
    app = _open(tmp_path)
    app.select([2, 3, 4, 5]); app.click("Show sel.")
    app.select([1]); app.click("Delete sel.")
    assert app.shown() == [0, 1, 2, 4, 5, 6, 7]
    app.select([3]); app.click("Delete sel.")
    app.click("Revert change")
    assert app.shown() == [0, 1, 2, 4, 5, 6, 7]
    app.select([3]); app.click("Delete sel.")
    app.select([0, 1]); app.click("Merge sel.")
    assert app.shown() == [0, 3, 4, 5, 6]
    app.click("Revert change")
    assert app.shown() == [0, 1, 2, 5, 6, 7]
    app.select([0]); app.click("Delete sel.")
    assert app.shown() == [1, 2, 5, 6, 7]


def test_no_selection_after_a_change(tmp_path):
    """A second click without a new selection changes nothing (all rows used to stay selected)."""
    app = _open(tmp_path)
    app.select([2]); app.click("Delete sel.")
    assert list(app.src.selected.indices) == []
    app.click("Delete sel.")
    assert app.shown() == [0, 1, 3, 4, 5, 6, 7]
    app.select([0, 1]); app.click("Merge sel.")
    assert list(app.src.selected.indices) == []
    app.click("Merge sel.")
    assert app.shown() == [1, 2, 3, 4, 5, 6]


def test_examine_cells_shows_the_app(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(bm_examinator, "show", shown.append)
    bm_examinator.ExamineCells(str(_estimates(tmp_path)), default_fps=20,
                               bkapp_kwargs={"mode": "legacy", "end_frame": 300})
    assert len(shown) == 1 and callable(shown[0])
    shown[0](Document())  # builds without a server
```

- [ ] **Step 2:** Run: `conda run -n bearmind python -m pytest tests/test_examinator_callbacks.py -v` → FAIL: `make_app` нет (AttributeError).
- [ ] **Step 3: Выделить `make_app`.** Переименовать `def ExamineCells` в `def make_app`, последнюю строку `show(bkapp)` заменить на `return bkapp`. Добавить новый `ExamineCells` с прежним docstring (см. Interfaces). Других изменений в этом шаге нет.
- [ ] **Step 4:** Run тот же файл → ожидаемо падают ровно три теста: `test_partial_list_matches_the_view_after_each_callback` (после первого `first("Delete sel.")` частичный список ещё из 8 компонент), `test_delete_merge_zoom_pick_the_selected_rows` (на шаге `([1, 2], "Show sel.")` показано [4, 5]), `test_no_selection_after_a_change` (выделены все строки). `test_revert_restores_the_last_change` и `test_examine_cells_shows_the_app` проходят. Если картина другая — остановиться и доложить, ничего не чинить наугад.
- [ ] **Step 5: Четыре правки в обработчиках**, каждая с комментарием «почему»:
  - `del_callback` и `merge_callback`: после `storage.estimates = copy.deepcopy(estimates)` добавить `storage.estimates_partial = copy.deepcopy(estimates)`: строки вида теперь показывают полный список. Вместо `src_partial.selected.indices = np.arange(...)` поставить `src_partial.selected.indices = []`. Причина: с выбором всех строк повторное нажатие действует на все нейроны, а клик в режиме `toggle` снимает выбор.
  - `show_callback`: обе генерации (`estimates_partial.idx_components = ...` и `part_to_total_mapping`) переводят номера строк через `estimates_partial.idx_components` — список, который показывают строки, а не `estimates.idx_components`.
  - `revert_callback`: `prev_data_partial = storage.prev_data_partial`.
  Больше ничего не менять: ускорение окна — в следующей фазе.
- [ ] **Step 6:** Run `tests/test_examinator_callbacks.py` → 5 passed; затем быстрый набор → 0 failed.
- [ ] **Step 7:** Commit:

```bash
git add -- tests/test_examinator_callbacks.py
git commit -q -m "fix(examinator): each callback keeps the partial list in sync; no select-all after a change; zoom maps rows of the zoomed view" -- bm_examinator.py tests/test_examinator_callbacks.py
```

### Task 2: Группы слияния по соседству и пересчёт на корпусе мая 2026

**Files:**
- Modify: `auto_inspector.py:1820-1868` (шаг слияний в `metrics_to_decision`)
- Create: `tests/test_merge_groups.py`
- Create: `docs/superpowers/specs/ml-filter-2026-11-01/merge_regroup.py`, `docs/superpowers/specs/ml-filter-2026-11-01/merge_regroup_out.txt`
- Modify: `docs/superpowers/specs/ml-filter-2026-11-01/findings.md`

**Interfaces:**
- Produces: `auto_inspector.assign_merge_groups(metrics_df: pd.DataFrame, FBD: np.ndarray, pxlthr_distance_boundary: float = 15, d_snr_thr: float = 10) -> pd.DataFrame`. Вход: столбцы `component_idx`, `corr_groups`, `caiman_snr`, `delete` (решения «мозга»); метки индекса — номера строк FBD. Функция меняет `delete` в переданной таблице, как это делал старый код, ставит `merge` в каждой строке (создаёт столбец, если его нет) и возвращает таблицу отсортированной по индексу. `metrics_to_decision` при `enable_merge` вызывает её вместо старого цикла; дальше, как раньше, `sort_index` и `return`.

Алгоритм:
1. Цикл по кластерам и парам — тот же, что в строках 1822-1860: та же сортировка `sort_values('corr_groups', ascending=False, inplace=True)`, тот же порядок, те же условия, то же удаление SNR-дубля (`delete = 1`). Отличие одно: там, где старый код ставил `merge = cluster_id` для `index_1`, теперь записывается ребро `{component_idx_1, component_idx_2}`.
2. После цикла — прежняя чистка (`merge = 0` у удалённых) не нужна. Вместо неё: граф на строках с `delete == 0`, рёбра — записанные пары, у которых оба конца не удалены. Компоненты связности размером >= 2 получают `merge` = 1..K по возрастанию наименьшего `component_idx` группы; остальные строки — `merge` = 0. Для связности подойдёт `scipy.sparse.csgraph.connected_components`.
3. В docstring: группа — цепочка пар «граница <= порога и |dSNR| <= порога» внутри одного кластера корреляций; прежняя группа (весь кластер) склеивала далёкие клетки (findings.md, пункт 4).

- [ ] **Step 1: Тесты** в `tests/test_merge_groups.py` (первой строкой `import torch  # noqa: F401`, как в `tests/test_merge_alignment.py`).

Эталон старого кода — функция `_old_merge_step(metrics_df, FBD, pxlthr_distance_boundary, d_snr_thr)`. Это дословная копия строк 1822-1865 `auto_inspector.py` на коммите 63dfe9c (`git show 63dfe9c:auto_inspector.py`): в начале `metrics_df["merge"] = 0` (как строка 1739), в конце `sort_index`. `component_idx` в тестах не совпадает с номером строки (как в данных: это значения `idx_components`).

```python
def _df(snr, groups, delete=None, ids=None):
    n = len(snr)
    return pd.DataFrame({"component_idx": ids if ids is not None else np.arange(n) * 2 + 1,
                         "corr_groups": groups, "caiman_snr": snr,
                         "delete": delete if delete is not None else [0] * n})

def _fbd(n, close):  # close: {(i, j): distance}; all other pairs inf, as for uncorrelated pairs
    ...

def test_two_far_pairs_in_one_cluster_are_two_groups():
    out = assign_merge_groups(_df([5.0] * 4, [1.0] * 4), _fbd(4, {(0, 1): 5, (2, 3): 5}))
    assert out["merge"].tolist() == [1, 1, 2, 2] and out["delete"].tolist() == [0] * 4

def test_chain_is_one_group():
    out = assign_merge_groups(_df([5.0] * 3, [1.0] * 3), _fbd(3, {(0, 1): 5, (1, 2): 5, (0, 2): 40}))
    assert out["merge"].tolist() == [1, 1, 1]

def test_member_deleted_later_drops_out():
    # C (row 2) is linked to B, then deleted as the dim copy of D (SNR 7 vs 30): A-B stay a group
    out = assign_merge_groups(_df([5.0, 6.0, 7.0, 30.0], [1.0] * 4),
                              _fbd(4, {(0, 1): 5, (1, 2): 5, (2, 3): 5}))
    assert out["delete"].tolist() == [0, 0, 1, 0] and out["merge"].tolist() == [1, 1, 0, 0]

def test_groups_are_numbered_across_clusters():
    out = assign_merge_groups(_df([5.0] * 4, [2.0, 2.0, 1.0, 1.0]), _fbd(4, {(0, 1): 5, (2, 3): 5}))
    assert out["merge"].tolist() == [1, 1, 2, 2]

def test_pair_at_the_cutoff():
    out = assign_merge_groups(_df([5.0, 15.0, 5.0, 5.0], [1.0] * 4), _fbd(4, {(0, 1): 15.0, (2, 3): 15.0001}))
    assert out["merge"].tolist() == [1, 1, 0, 0] and out["delete"].tolist() == [0] * 4

def test_single_component_has_no_group():
    out = assign_merge_groups(_df([3.0], [None]), np.full((1, 1), np.inf))
    assert out["merge"].tolist() == [0] and out["delete"].tolist() == [0]

def test_random_inputs_keep_old_deletes():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n = 40
        snr = rng.uniform(1, 30, n); snr[rng.random(n) < 0.1] = np.nan
        df = _df(snr, rng.integers(0, 4, n).astype(float), (rng.random(n) < 0.2).astype(int))
        fbd = rng.uniform(0, 40, (n, n)); fbd = np.minimum(fbd, fbd.T); np.fill_diagonal(fbd, np.inf)
        old = _old_merge_step(df.copy(), fbd, 15, 10)
        new = assign_merge_groups(df.copy(), fbd, 15, 10)
        assert new["delete"].tolist() == old["delete"].tolist(), seed
        for gid, g in new[new["merge"] > 0].groupby("merge"):
            assert len(g) >= 2 and (g["delete"] == 0).all()
            assert g["merge"].nunique() == 1 and old.loc[g.index, "merge"].nunique() == 1
            assert (old.loc[g.index, "merge"] != 0).all()     # inside one old group
            assert _fbd_connected(fbd, g.index.to_numpy(), 15)  # every new group is FBD-connected
        # every kept pair that passes the pair rule ends in one group
        kept = new.index[new["delete"] == 0]
        for i in kept:
            for j in kept:
                if (i < j and new.at[i, "corr_groups"] == new.at[j, "corr_groups"] != 0
                        and fbd[i, j] <= 15 and abs(snr[i] - snr[j]) <= 10):
                    assert new.at[i, "merge"] == new.at[j, "merge"] != 0

def test_metrics_to_decision_uses_the_new_groups():
    df = _df([5.0] * 4, [1.0] * 4).assign(area=10.0)
    out = metrics_to_decision(df, None, None, _fbd(4, {(0, 1): 5, (2, 3): 5}),
                              deletion_rules=["area<1"], brain="thresholds")
    assert out["merge"].tolist() == [1, 1, 2, 2]
```

`_fbd_connected(fbd, rows, thr)` — обход в ширину по `fbd[np.ix_(rows, rows)] <= thr`, как в `merge_spread_corpus.groups_of`.

- [ ] **Step 2:** Run: `conda run -n bearmind python -m pytest tests/test_merge_groups.py -v` → FAIL (ImportError: `assign_merge_groups`).
- [ ] **Step 3:** Реализовать `assign_merge_groups` и вызов из `metrics_to_decision` (см. Interfaces и алгоритм). Комментарий над вызовом сменить с `(UNCHANGED)` на одну строку о том, что группа — компонента связности пар.
- [ ] **Step 4:** Run тот же файл → 8 passed; быстрый набор → 0 failed.
- [ ] **Step 5: Скрипт пересчёта** `docs/superpowers/specs/ml-filter-2026-11-01/merge_regroup.py` (только чтение; первой строкой `import torch  # noqa: F401`, затем `sys.path` на корень репозитория и `from auto_inspector import assign_merge_groups`).
  - Аргументы: `<корень корпуса> <серия series_short>`. Сессии корпуса — как в `merge_spread_corpus.py`: `<DS>/7_output/inspection_artifacts_*` и `<DS>/4_Estimates_inspections/inspection_artifacts_*` для DS в CC, RFC, FOF. Сессии series_short — `5_Inspection/<SID>/`. Папка без `*_metrics_with_decisions.csv`, `*_FBD.npy` или `*_FCD.npy` пропускается строкой `[SKIP] <папка>: <чего нет>`.
  - Пороги — из `<SID>_metadata.json` → `autoinspection_config` (`ml_threshold`, `pxlthr_distance_boundary`, `d_snr_thr`), если файл есть; иначе 0.72 / 15 / 10.
  - Решения мозга восстанавливаются из csv: `delete_brain = (P < ml_threshold) | (любой столбец failed_* == 1)`, где P — `pd.to_numeric(ml_keep_probability, errors="coerce")`, NaN не считается удалением (гибридный мозг: правила-пороги ставят P = NaN, ML удаляет при P < порога).
  - `new = assign_merge_groups(csv[["component_idx", "corr_groups", "caiman_snr"]] + delete_brain, FBD, ...)`. Проверка: `new["delete"]` совпадает с сохранённым `delete` построчно. Строка на сессию: `[OK] <DS> <SID> delete identical` или `[FAILED] ... <n> rows differ`.
  - Группы «до» — по сохранённому `merge`, «после» — по `new["merge"]`; в обоих случаях только строки `delete == 0` и группы >= 2 (как в `merge_spread_corpus.groups_of`). Для группы: размер, связность по FBD <= порога, разброс центров = max FCD внутри группы.
  - Итоговые строки на каждый DS, на ALL (корпус) и на series_short: сессий; сессий с совпавшим delete; групп до -> после; компонент в группах до -> после; самая большая группа до -> после; групп не связных по FBD до -> после; макс. разброс центров до -> после; групп с разбросом > 30 px и > 100 px до -> после.
  - Вывод — ASCII, в консоль и в `merge_regroup_out.txt` рядом со скриптом. Ничего не писать в папки данных.
- [ ] **Step 6:** Run: `conda run -n bearmind python docs/superpowers/specs/ml-filter-2026-11-01/merge_regroup.py H:\Dataset\CalciumData H:\Projects\_bearmind_test\series_short`. Ожидаемо:
  - 106 сессий корпуса, у всех `[OK] delete identical`;
  - «до» повторяет `merge_spread_out.txt`: CC 1463 групп, 83 несвязных; FOF 329 / 18; RFC 714 / 93; ALL 2506 / 194, 4936 компонент в несвязных группах из 10930;
  - «после»: несвязных групп 0 во всех строках.
  Если хоть одно из этого не так — остановиться и доложить с выводом скрипта. Не подгонять восстановление решений и не менять код под ожидаемые числа.
- [ ] **Step 7: findings.md.**
  - В пункт 4 раздела «Итог» дописать: «Исправлено: см. раздел «Обновление <дата>: группы слияния по соседству».»
  - В конец файла — новый раздел `## Обновление <дата выполнения, ГГГГ-ММ-ДД>: группы слияния по соседству`. Один абзац: что изменилось (группа — компонента связности пар FBD <= 15 px и |dSNR| <= 10 среди неудалённых; удаления SNR-дублей прежние); что пересчитано (только группировка по сохранённым csv, FBD и FCD; автоинспекция не запускалась); где скрипт и вывод.
  - Затем таблица с числами из `merge_regroup_out.txt`: строки CC, FOF, RFC, корпус, series_short; столбцы — те, что в итоговых строках скрипта.
  - Одна строка о последствии: после повторной автоинспекции число продуктов слияния в сессии равно числу групп «после».
- [ ] **Step 8:** Commit:

```bash
git add -- tests/test_merge_groups.py docs/superpowers/specs/ml-filter-2026-11-01/merge_regroup.py docs/superpowers/specs/ml-filter-2026-11-01/merge_regroup_out.txt
git commit -q -m "fix(autoinspect): a merge group is a connected set of close pairs, not a whole correlation cluster; corpus re-evaluation" -- auto_inspector.py tests/test_merge_groups.py docs/superpowers/specs/ml-filter-2026-11-01/merge_regroup.py docs/superpowers/specs/ml-filter-2026-11-01/merge_regroup_out.txt docs/superpowers/specs/ml-filter-2026-11-01/findings.md
```

### Task 3: Одно правило экспорта в трёх путях

**Files:**
- Modify: `bearmind/stages/__init__.py:9-11`, `bearmind/stages/export.py:19-66`, `batch_export.py:90-115`
- Modify: `tests/bearmind/test_stage_export.py`, `tests/bearmind/test_runner.py:1251`, `tests/bearmind/test_e2e_real.py:55-72`
- Modify: `docs/superpowers/specs/2026-11-01-operator-notes.md`, `docs/superpowers/specs/ml-filter-2026-11-01/findings.md`

**Interfaces:**
- Consumes: `export_estimates_data.ml_approved_components(metrics_df, threshold) -> set` (не меняется); группы слияния из задачи 2.
- Produces: `bearmind.stages.COMPONENT_RULE = "ml_or_merged_not_deleted"`. `export._ml_filter(est)` возвращает те же `(idx, ml_info)`, но `idx` — компоненты `est.idx_components` из `ml_approved_components`. `batch_export.export_one` экспортирует те же компоненты.

- [ ] **Step 1: Тесты** — правки в `tests/bearmind/test_stage_export.py`:
  - Комментарий в `_est`: строка 0 — SNR-дубль (delete, P выше порога), не экспортируется; строка 4 — продукт слияния (P = NaN), экспортируется.
  - `test_export_matches_old_structure`: `idx = np.array([2, 4])`; `rule = {"component_rule": "ml_or_merged_not_deleted", "n_exported": 2, "n_snr_duplicates_exported": 0, "n_merge_products_dropped": 0}`; `res.metrics == {"n_neurons": 2, "median_snr": 8.0, **rule}` (медиана `caiman_snr` строк 2 и 4: 7.0 и 9.0; комментарий поправить).
  - `test_rule_counts_on_series_short_shape`: ожидание `{"component_rule": "ml_or_merged_not_deleted", "n_exported": 94, "n_snr_duplicates_exported": 0, "n_merge_products_dropped": 0, "median_snr": 5.0}` (93 с P >= 0.72 + 1 продукт слияния; дубль не экспортируется); docstring — про новое правило.
  - `test_components_none_of_which_pass_the_rule`: перед записью pickle сделать строку 4 обычной (`est.metrics_df.loc[4, "decision"] = "ok"`, `est.metrics_df.loc[4, "ml_keep_probability"] = 0.3`). Иначе продукт слияния проходит при любом пороге.
  - `_check_empty_export`: `meta["component_rule"] in ("ml_or_merged_not_deleted", "idx_components")`.
  - Новые тесты:

```python
def test_rule_without_decision_columns():
    """A processed pickle from an older run: P only, no decision / delete columns."""
    est = SimpleNamespace(idx_components=np.arange(3), autoinspection_config={"ml_threshold": 0.72},
                          metrics_df=pd.DataFrame({"component_idx": [0, 1, 2],
                                                   "ml_keep_probability": [0.9, 0.5, np.nan]}))
    assert export._ml_filter(est)[0].tolist() == [0]
    assert _counts(est) == {"component_rule": "ml_or_merged_not_deleted", "n_exported": 1,
                            "n_snr_duplicates_exported": None, "n_merge_products_dropped": None,
                            "median_snr": None}


def test_all_three_export_paths_use_one_rule(tmp_path, monkeypatch):
    import batch_export
    est = _est()
    est.name = SID  # extract_session_name reads it
    assert old.ml_approved_components(est.metrics_df, 0.5) == {2, 4}
    assert export._ml_filter(est)[0].tolist() == [2, 4]
    folder = tmp_path / "a" / "b"  # export_one also looks for feedback one folder up
    folder.mkdir(parents=True)
    path = folder / f"{SID}_processed.pickle"
    path.write_bytes(pickle.dumps(est))
    monkeypatch.setattr(batch_export, "get_fps", lambda name: 19.76)
    assert batch_export.export_one(path)["neurons"] == 2
    with np.load(folder / f"{SID}_data.npz") as d:
        assert sorted(d["component_indices"].tolist()) == [2, 4]
```

  - `tests/bearmind/test_runner.py:1251`: `"ml_probability"` → `"ml_or_merged_not_deleted"`.
  - `tests/bearmind/test_e2e_real.py::test_export_outputs`: после чтения `meta` добавить `assert meta["component_rule"] in ("ml_or_merged_not_deleted", "idx_components")`. Если правило новое: `meta["n_snr_duplicates_exported"] == 0` и `meta["n_merge_products_dropped"] == 0`. Это realdata-тест: на series_short он пройдёт только после повторного прогона серии, а план её не запускает.
- [ ] **Step 2:** Run: `conda run -n bearmind python -m pytest tests/bearmind/test_stage_export.py tests/bearmind/test_runner.py -q -k "export"` → FAIL: правило старое (`component_rule` = `ml_probability`, экспорт [0, 2]).
- [ ] **Step 3: Реализация.**
  - `bearmind/stages/__init__.py`: `COMPONENT_RULE = "ml_or_merged_not_deleted"`. Комментарий над ним: правило — `export_estimates_data.ml_approved_components`, оно входит в параметры этапа export, поэтому новое имя заново запускает каждый экспорт.
  - `export._ml_filter`: `ok = old.ml_approved_components(df, threshold)`; docstring: «export_estimates_data.ml_approved_components, the rule of all three export paths; without feedback».
  - `rule_counts`: код подсчёта не меняется. Docstring: правило — `ml_approved_components` (`'idx_components'`, когда вероятностей ML нет). При нём `n_snr_duplicates_exported` и `n_merge_products_dropped` равны 0. Счётчики остаются проверкой: ненулевое значение значит, что processed-файл нарушает допущения правила. Фразу про медиану SNR из `metrics_df` оставить.
  - `batch_export.export_one`: `ml_approved_indices = ml_approved_components(df, threshold)` (импорт из `export_estimates_data`) вместо маски `>=`. Печать: `print(f"  Export rule (P >= {threshold} or merged, not deleted): {n_before_ml_filter} -> {len(component_indices)} (-{n_ml_rejected})")`.
- [ ] **Step 4:** Run тот же набор → PASS; быстрый набор → 0 failed.
- [ ] **Step 5: Заметки оператора.** В конец `docs/superpowers/specs/2026-11-01-operator-notes.md` добавить раздел (текст дословно):

```markdown
## После обновления 2026-11: правило экспорта

Экспорт выбирает нейроны по правилу `ml_or_merged_not_deleted`. Нейрон попадает в экспорт, если выполнены оба условия:

- вероятность ML не ниже порога автоинспекции, или нейрон получен слиянием (`decision: from_merge`);
- автоинспекция не пометила нейрон на удаление (`delete` не равно 1).

Старое правило `ml_probability` проверяло только вероятность. С ним в экспорт попадали тусклые копии соседних нейронов (SNR-дубли), а нейроны от слияний терялись. Новое правило одно для `run_bearmind`, `export_estimates_data.py`, ноутбука и `batch_export.py`.

Имя правила входит в параметры этапа export. Поэтому следующий запуск заново сделает экспорт всех готовых сессий.

1. Запустите `run_bearmind run <СЕРИЯ>`.
2. Дождитесь конца прогона. Экспорт одной сессии занимает около 10 с. MC, пороги и CNMF-E не повторяются.
3. Откройте `_bearmind/summary.csv`. Числа в столбце `export_n_neurons` изменятся.

В json экспорта поля `n_snr_duplicates_exported` и `n_merge_products_dropped` теперь равны 0. Если в каком-то файле там не 0, сообщите разработчику.
```

- [ ] **Step 6: findings.md:** в раздел из задачи 2 добавить строку: `batch_export.export_one` и этап экспорта bearmind теперь используют `ml_approved_components`, правило (в) таблицы «Числа по правилам»; имя правила в json — `ml_or_merged_not_deleted`.
- [ ] **Step 7:** Commit:

```bash
git commit -q -m "fix(export): one export rule (ml_approved_components) in batch_export and the bearmind export stage" -- bearmind/stages/__init__.py bearmind/stages/export.py batch_export.py tests/bearmind/test_stage_export.py tests/bearmind/test_runner.py tests/bearmind/test_e2e_real.py docs/superpowers/specs/2026-11-01-operator-notes.md docs/superpowers/specs/ml-filter-2026-11-01/findings.md
```

### Task 4: Зерно шума driada в автоинспекции

Шум ставит `driada.experiment.signal_preprocessing.calcium_preprocessing(ca, seed=None)`: `ca += default_rng(seed).random(len(ca)) * 1e-08`. Его вызывает `Neuron.__init__(..., seed=None)` (driada/experiment/neuron.py:249). В репозитории `Neuron(...)` создают три места, все без `seed`: `auto_inspector.py:147` (`get_neuron_with_spikes`), `:243` и `:263` (`_optimize_kinetics_hybrid`). С зерном 0 два прогона дают бит-в-бит одно и то же (замер задачи 19 batch-core).

**Files:**
- Modify: `auto_inspector.py` (`get_neuron_with_spikes`, `_optimize_kinetics_hybrid`, `get_single_neuron_metrics`, `get_multineuron_metrics`, `estimates_to_metrics`, `transform_metrics_df_indices`)
- Modify: `ae_launch.py:171-232` (сигнатура и docstring `run_auto_inspection`), `:402-421` и `:492-500` (передача), `:537-556` (`autoinspection_config`)
- Modify: `bearmind/defaults.yaml` (раздел `autoinspect`)
- Modify: `tests/test_auto_inspector.py`, `tests/bearmind/test_stage_autoinspect.py`, `tests/bearmind/test_config.py`, `docs/superpowers/specs/2026-11-01-operator-notes.md`

**Interfaces:**
- Produces: параметр `seed: int | None = 0` у `get_neuron_with_spikes`, `_optimize_kinetics_hybrid`, `get_single_neuron_metrics`, `get_multineuron_metrics`, `estimates_to_metrics`, `transform_metrics_df_indices`, `ae_launch.run_auto_inspection`. Каждая функция передаёт его дальше по цепочке до `Neuron(..., seed=seed)`. `None` возвращает старый непредсказуемый шум. Одно зерно у всех трейсов: результат трейса не зависит от его места в списке. Поэтому метрики продукта слияния, пересчитанные в `transform_metrics_df_indices`, тоже воспроизводимы. `est.autoinspection_config["seed"]` = переданное значение. Конфиг: `autoinspect.seed: 0` (входит в `stage_params("autoinspect")` через `cfg["autoinspect"]` и доходит до `run_auto_inspection(**params)` без правок этапа).

- [ ] **Step 1: Тесты.** В `tests/test_auto_inspector.py` (первой строкой `import torch  # noqa: F401`, как в `tests/test_merge_alignment.py`):

```python
def _trace(seed, n=2000):  # calcium-like: sparse spikes, exp decay 1 s at 20 fps, noise, scaled to 0..1
    ...

def test_every_neuron_gets_the_seed(monkeypatch):
    seen = []

    class Recording(auto_inspector.Neuron):
        def __init__(self, *a, **k):
            seen.append(k.get("seed", "missing"))
            super().__init__(*a, **k)

    monkeypatch.setattr(auto_inspector, "Neuron", Recording)
    # threads, not loky processes: the patched class must be the one that runs
    monkeypatch.setattr(auto_inspector, "get_parallelization_backend", lambda: "threading")
    traces = np.stack([_trace(1), _trace(2), np.random.default_rng(3).random(2000)])
    auto_inspector.get_multineuron_metrics(traces, fps=20, include_heavy=True, event_method="wavelet",
                                           n_iter=3, hybrid_kinetics=True, seed=7)
    assert seen and set(seen) == {7}


def test_event_metrics_are_reproducible():
    kw = dict(fps=20, include_heavy=True, event_method="wavelet", n_iter=3, hybrid_kinetics=True, seed=0)
    a = auto_inspector.get_single_neuron_metrics(_trace(1), **kw)
    b = auto_inspector.get_single_neuron_metrics(_trace(1), **kw)
    assert a.keys() == b.keys()
    for k in a:  # arrays: np.array_equal(equal_nan=True); floats: equal or both NaN; others ==
        assert _same(a[k], b[k]), k


def test_seed_defaults_to_zero():
    import inspect
    import ae_launch
    for f in (auto_inspector.get_neuron_with_spikes, auto_inspector.get_single_neuron_metrics,
              auto_inspector.get_multineuron_metrics, auto_inspector.estimates_to_metrics,
              auto_inspector.transform_metrics_df_indices, ae_launch.run_auto_inspection):
        assert inspect.signature(f).parameters["seed"].default == 0, f.__name__


def test_run_auto_inspection_forwards_the_seed(tmp_path, monkeypatch):
    import ae_launch

    class Stop(Exception):
        pass

    seen = {}

    def fake(est, fps, **kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(ae_launch, "estimates_to_metrics", fake)
    p = tmp_path / "NOF_H01_1D_estimates.pickle"
    p.write_bytes(pickle.dumps(SimpleNamespace(C=np.zeros((1, 10)), idx_components=np.array([0]))))
    with pytest.raises(Stop):
        ae_launch.run_auto_inspection(str(p), fps=20, brain="thresholds", seed=5)
    assert seen["seed"] == 5


SHORT = Path(r"H:\Projects\_bearmind_test\series_short\4_Estimates\NOF_H03_3D_estimates.pickle")

@pytest.mark.slow
@pytest.mark.realdata
@pytest.mark.skipif(not SHORT.exists(), reason="series_short not on this machine")
def test_autoinspection_twice_gives_identical_decisions():
    """Production parameters (bearmind defaults.yaml autoinspect), fps 30 = round(29.784); nothing saved."""
    # params = load_defaults()["autoinspect"] without model_path, + ml_model_path=<repo>/production_models/ebm_v9_iter8.pkl,
    # save_artifacts=False; r1, r2 = two run_auto_inspection(str(SHORT), fps=30, **params)
    pd.testing.assert_frame_equal(r1["estimates"].metrics_df, r2["estimates"].metrics_df)
    assert np.array_equal(r1["estimates"].idx_components, r2["estimates"].idx_components)
    assert r1["estimates"].autoinspection_config["seed"] == 0
```

  Правки в тестах bearmind:
  - `tests/bearmind/test_stage_autoinspect.py::test_passes_production_parameters`: добавить `assert kw["seed"] == 0`.
  - Новый тест в том же файле:

```python
def test_seed_is_an_autoinspect_input(tmp_path):
    from bearmind import runner
    paths, rec, cfg = _setup(tmp_path)
    p0 = runner.stage_params("autoinspect", rec, cfg, paths)
    cfg1 = copy.deepcopy(cfg); cfg1["autoinspect"]["seed"] = 1
    p1 = runner.stage_params("autoinspect", rec, cfg1, paths)
    assert p0["autoinspect"]["seed"] == 0
    assert config.params_hash(p0) != config.params_hash(p1)
```

  - `tests/bearmind/test_config.py::test_defaults_int_keys_and_values`: `assert d["autoinspect"]["seed"] == 0`.
- [ ] **Step 2:** Run: `conda run -n bearmind python -m pytest tests/test_auto_inspector.py tests/bearmind/test_stage_autoinspect.py tests/bearmind/test_config.py -q -m "not slow and not realdata"` → FAIL (`seed` не параметр; `Recording` видит `"missing"`; в конфиге нет `seed`).
- [ ] **Step 3: Реализация.**
  - Передать `seed` по цепочке (Interfaces). Над `Neuron(...)` в `get_neuron_with_spikes` — комментарий-причина: driada добавляет к трейсу шум 1e-8 для разрыва равенств; без зерна признаки событий и P меняются от прогона к прогону.
  - `run_auto_inspection`: `seed: int | None = 0` в группе «Metrics extraction parameters»; строка в docstring; передать в `estimates_to_metrics(...)` и `transform_metrics_df_indices(...)`; `'seed': seed` в `autoinspection_config` (оба места: обычное и для 0 компонент).
  - `bearmind/defaults.yaml`, раздел `autoinspect`: `seed: 0` с комментарием: «seed of driada's tie-breaking noise (1e-8 per frame): with a fixed seed two runs give the same decisions; null: unseeded». Комментарий в начале файла уже говорит, что новый ключ раздела заново запускает этап.
- [ ] **Step 4:** Run тот же набор → PASS; быстрый набор → 0 failed.
- [ ] **Step 5:** Run один раз: `conda run -n bearmind python -m pytest tests/test_auto_inspector.py -v -m "slow and realdata"` → `test_autoinspection_twice_gives_identical_decisions` PASSED. Это два прогона автоинспекции на 2000 кадрах, без MC и CNMF-E; series_short только читается, ничего не сохраняется. Записать время в отчёт задачи.
- [ ] **Step 6: Заметки оператора** — в раздел из задачи 3 добавить подраздел (дословно):

```markdown
### Автоинспекция

В автоинспекции два изменения:

- Новый ключ `autoinspect.seed` (по умолчанию 0) задаёт шум, который библиотека driada добавляет к трейсам. С одним значением ключа два прогона на одних данных дают одинаковые решения. Раньше число нейронов менялось от прогона к прогону (117 или 118 на NOF_H01_1D).
- Группа слияния объединяет только соседние компоненты: каждую компоненту группы с другой связывает цепочка пар, у которых границы ближе 15 px. Раньше в одну группу попадали далёкие клетки из одного кластера корреляций.

Ключ `seed` входит в параметры этапа autoinspect. Поэтому следующий запуск заново сделает автоинспекцию и экспорт всех готовых сессий. MC, пороги и CNMF-E не повторяются. Ручную инспекцию повторять не нужно: автоинспекция берёт тот же вход.

Не меняйте `seed` без причины. Другое значение даёт другой, но тоже воспроизводимый результат и заново запускает автоинспекцию всех сессий.
```

- [ ] **Step 7:** Commit:

```bash
git commit -q -m "fix(autoinspect): seed driada's tie-breaking noise (autoinspect.seed, default 0) so runs are reproducible" -- auto_inspector.py ae_launch.py bearmind/defaults.yaml tests/test_auto_inspector.py tests/bearmind/test_stage_autoinspect.py tests/bearmind/test_config.py docs/superpowers/specs/2026-11-01-operator-notes.md
```

### Task 5: Пул MC из 16 процессов

Сейчас размер пула MC — `max(1, n_cpus - 1)` (`bearmind/stages/mc.py:158`), на рабочей машине 31. В раздел `mc` ключ класть нельзя по двум причинам. Раздел целиком уходит в `MotionCorrect(**mcfg)` (неизвестный аргумент). И он входит в хэш этапа MC (`runner.stage_params`, "mc"): смена ключа заново запустила бы MC и все следующие этапы у всех сессий. Размер пула результат не меняет: сдвиги при пулах 31, 16 и 8 совпали до 0.000 px, пул 16 работал так же быстро, как 31, при меньшей памяти (docs/superpowers/specs/2026-11-01-e2e-report.md, строки 13 и 129). Поэтому нужен новый раздел `pool`, которого нет ни в одном хэше.

**Files:**
- Modify: `bearmind/defaults.yaml` (новый раздел `pool`), `bearmind/stages/mc.py:158`
- Modify: `tests/bearmind/test_stage_mc.py`, `docs/superpowers/specs/2026-11-01-operator-notes.md`

**Interfaces:**
- Produces: `bearmind.stages.mc.mc_pool_size(cfg: dict) -> int` = `max(1, min(int(cfg["pool"]["mc_processes"]), int(cfg["machine"]["n_cpus"]) - 1))`; `mc.run` вызывает `pool.start_pool(mc_pool_size(cfg))`. Конфиг: `pool: {mc_processes: 16}`.

- [ ] **Step 1: Тесты** в `tests/bearmind/test_stage_mc.py`:

```python
def test_mc_pool_size():
    assert config.load_defaults()["pool"]["mc_processes"] == 16
    size = lambda n_cpus: mc_stage.mc_pool_size({"pool": {"mc_processes": 16}, "machine": {"n_cpus": n_cpus}})
    assert (size(32), size(8), size(1)) == (16, 7, 1)


def test_mc_starts_the_pool_with_the_configured_size(tmp_path, make_avi, monkeypatch):
    class Stop(Exception):
        pass

    paths, rec, cfg = _setup(tmp_path, make_avi)
    cfg["machine"]["n_cpus"] = 32
    sizes = []

    def start_pool(n):
        sizes.append(n)
        raise Stop

    monkeypatch.setattr(mc_stage, "_write_tif", lambda *a, **k: None)
    monkeypatch.setattr(mc_stage.pool, "start_pool", start_pool)
    with pytest.raises(Stop):
        mc_stage.run(paths, rec, cfg)
    assert sizes == [16]
    assert not (tmp_path / "tmp").exists() or not any((tmp_path / "tmp").iterdir())


def test_pool_size_is_not_an_mc_input(tmp_path, make_avi):
    from bearmind import runner
    paths, rec, cfg = _setup(tmp_path, make_avi)
    other = {**cfg, "pool": {"mc_processes": 4}}
    assert runner.stage_params("mc", rec, cfg, paths) == runner.stage_params("mc", rec, other, paths)
```

- [ ] **Step 2:** Run: `conda run -n bearmind python -m pytest tests/bearmind/test_stage_mc.py -q -m "not slow"` → FAIL (`pool` нет в defaults; `mc_pool_size` нет).
- [ ] **Step 3: Реализация.** В `bearmind/defaults.yaml` перед разделом `runner` — раздел (комментарий дословно):

```yaml
# Worker processes of CaImAn pools. Not stage parameters (no stage hash includes them): rigid MC corrects
# each of its ceil(T/500) splits on its own, and pools of 31, 16 and 8 gave the same shifts (to 0.000 px);
# 16 ran as fast as 31 with less memory (docs/superpowers/specs/2026-11-01-e2e-report.md). At most n_cpus - 1.
pool:
  mc_processes: 16
```

  В `mc.py` — `mc_pool_size` (Interfaces) и вызов в `run`.
- [ ] **Step 4:** Run: `conda run -n bearmind python -m pytest tests/bearmind/test_stage_mc.py tests/bearmind/test_config.py -q -m "not realdata"` → PASS. Медленные тесты MC здесь крошечные: синтетический фильм 300 кадров 72x72, пул 2. Затем быстрый набор → 0 failed.
- [ ] **Step 5: Заметки оператора** — в конец файла (дословно):

```markdown
## Размер пула MC

Этап MC запускает 16 рабочих процессов CaImAn. Это значение ключа `pool.mc_processes`. На машине с N логическими процессорами процессов будет не больше N - 1.

Чтобы задать другой размер для серии, впишите `pool: {mc_processes: 8}` в `_bearmind/config/series.yaml`. Размер пула не входит в параметры этапа MC: сдвиги от него не зависят. Поэтому смена размера не запускает MC заново.
```

- [ ] **Step 6:** Commit:

```bash
git commit -q -m "perf(mc): pool of 16 workers (pool.mc_processes), outside the MC stage hash" -- bearmind/defaults.yaml bearmind/stages/mc.py tests/bearmind/test_stage_mc.py docs/superpowers/specs/2026-11-01-operator-notes.md
```

---

## Финальная проверка

- [ ] Run: `conda run -n bearmind python -m pytest -q -m "not slow and not realdata" tests` → 347 + новые тесты passed, 0 failed.
- [ ] Run: `git log --format='%(trailers)' <коммит этого плана>..HEAD` → пустой вывод (нет отметок ИИ).
- [ ] Run: `git status --short` → изменены только `BEARMiND_full_pipeline.ipynb` и `INFO_promts.txt` (чужие, как до начала).

## Не проверено и не входит в план

- **Окно в браузере.** Сильнейшая непроведённая проверка задачи 1: автор открывает ExamineCells на `series_short\4_Estimates\NOF_H01_1D_estimates.pickle` (ноутбук, режим `capcan`), кликает нейрон, удаляет, кликает следующий, удаляет. Затем делает два «Show sel.» подряд. Тесты гоняют обработчики на сервере. Поведение выбора в браузере проверено только чтением bokeh.js.
- **Выбор всех строк после «Discard all» и «Show sel.»** остаётся. После Show это рабочий приём: приблизить выбранные нейроны и удалить их разом. Задача 1 снимает выбор только после Delete и Merge.
- **Режим `capcan` после Merge.** `merge_callback` пересобирает вид через `EstimatesToSrcFast`: исчезают столбцы метрик и цвета ML. Сортировка по метрике capcan после слияния падает с KeyError в `sort_callback`. Это прежнее поведение, оно не входит в задачу 1. Его исправит фаза ускорения окна.
- **Повторный прогон серий.** План не запускает батч. `test_e2e_real.py` на series_short начнёт проходить только после `run_bearmind run` серии: автоинспекция и экспорт пересчитаются из-за нового ключа `seed` и нового имени правила.
- **Модель EBM** обучена на признаках из файлов со старым сжатием (findings.md, «Обновление 2026-11-06»). Этот план её не касается.
