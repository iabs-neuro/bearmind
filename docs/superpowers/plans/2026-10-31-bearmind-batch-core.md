# BEARMiND batch core (фазы 0–1) — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** конвейер без окон, который берёт папку серии (bonsai или legacy) и доводит каждую сессию до экспорта `npz`/`json`/`mat` по конфигам, с реестром, проверками, логами и продолжением после сбоя; запуск из командной строки.

**Architecture:** новый пакет `bearmind/` в корне репозитория. Он владеет именами, путями, конфигами, реестром, сканом, видео, кропом и запуском этапов, а алгоритмы берёт из существующих модулей (`ae_launch.run_auto_inspection`, `bm_qc`, `export_estimates_data`) и из CaImAn 1.13.2. Состояние серии живёт в её папке `_bearmind/`; каждый этап — функция `(paths, record, cfg) -> StageResult`, а планировщик выбирает следующий готовый этап и пишет итог в реестр.

**Tech Stack:** Python 3.12, CaImAn 1.13.2, NumPy 2, PyAV (FFV1, H.264), OpenCV, PyYAML, driada 1.x, interpret, pytest. Окружение conda `bearmind`.

**Spec:** `docs/superpowers/specs/2026-10-30-bearmind-batch-pipeline-design.md` (одобрен 2026-10-31). Разведка кода: `docs/superpowers/specs/2026-10-30-bearmind-recon.md`.

**Следующие планы** (пишутся после этой фазы, на её интерфейсах): мониторинг и сводка; окно скана; окно порогов; ворота ручной инспекции (окно Bokeh); чистка и инструкция для студентов.

## Global Constraints

- CaImAn 1.13.2, Python 3.12, NumPy ≥ 2, PyTorch вместо TensorFlow, OpenBLAS; окружение `bearmind`.
- Папка `1_Raw/` и все исходные файлы только читаются; пайплайн никогда не пишет и не удаляет в них.
- Хранимое промежуточное видео — только `3_MC/<SID>_MC.avi` (FFV1, uint8). TIF не создаются как хранимые выходы. Кропнутое видео и memmap — временные файлы в `machine.temp_dir`, удаляются после этапа (и при ошибке).
- Никаких лимитов на число кадров; сессия до ~108 000 кадров проходит целиком.
- `fr` CNMF-E = точный FPS сессии при любом `tsub`; `fr` на `tsub` не делится.
- FPS: точный (по таймстемпам, с отбросом выбросов) везде; в автоинспекцию передаётся `round(fps)`; в метаданные пишутся оба.
- Автоинспекция по умолчанию: `brain='hybrid'`, модель `production_models/ebm_v9_iter8.pkl` (путь от корня репозитория), `ml_threshold=0.72`, `n_iter=3`, `detect_corner_artifacts=False`, `correlation_method='spearman'`, `cthr=0.35`, `corr_thr=0.4`, `pxlthr_distance_boundary=15`, `d_snr_thr=10`, `enable_merge=True`, `include_event_based=True`, `include_heavy=True`, `event_method='wavelet'`, `hybrid_kinetics=True`.
- Поля и структура `npz` и `mat` экспорта не меняются; в `json` поля только добавляются.
- Имена папок серии: `0_TimeStamps`, `1_Raw`, `3_MC`, `4_Estimates`, `5_Inspection/<SID>`, `6_Export/npz`, `6_Export/filters`, `Preview`, `_bearmind`.
- ID сессии разбирает только `bearmind.naming`; никаких срезов строк по позициям.
- Батч не ждёт человека, кроме ворот ручной инспекции (ждёт только эта сессия).
- Вывод в консоль — только ASCII (`[OK]` / `[FAILED]`), без эмодзи и экзотического Юникода.
- Коммиты — без строк `Co-Authored-By` и других отметок ИИ.

## Review Focus

1. **Имя папки сессии не подходит под шаблон** (`NOF_H01_1D_part2`, строчные буквы, лишние суффиксы) — сессия не должна пропасть молча: аномалия `unrecognized` с путём. Тест — в задаче 6.
2. **Кончилось место на диске во время MC или CNMF-E** — этап «упал» с понятным сообщением, временные файлы удалены, батч идёт к следующей сессии. Тест — в задаче 15.
3. **Процесс убит посреди этапа** (выключили свет) — при перезапуске этап в состоянии `running` считается незавершённым, недописанные выходы не принимаются за готовые (запись через `.partial` и переименование). Тест — в задаче 15.
4. **Студент запустил батч дважды на одной серии** — второй процесс отказывается с понятным сообщением (файл-замок с PID). Тест — в задаче 15.
5. **Пробелы и кириллица в пути к серии** — всё работает. Тест — в задачах 1 и 6 (`tmp_path / "серия 1"`).

---

## Фаза 0. Окружение и CaImAn 1.13.2

### Task 0.1: Окружение `bearmind` на CaImAn 1.13.2

**Files:**
- Modify: `environment.yml`, `requirements-pip.txt` (закрепить точные версии)
- Create: `env/bearmind-lock.txt` (вывод `conda list --export -n bearmind`)
- Create: `tests/test_environment.py` (не `test_env.py`: такое имя уже занято в корне)

**Interfaces:**
- Produces: окружение `bearmind`, в котором импортируются `caiman` (1.13.2), `driada`, `interpret`, `cv2`, `av`, `yaml`, `torch`.

- [x] **Step 1:** Окружение `bearmind_cm113` собрано по `environment.yml` + `requirements-pip.txt` (2026-10-31): NumPy 2.5.4, CaImAn 1.13.2, driada 1.2.2, interpret 0.7.8, torch 2.14.1, OpenCV 5.0.0; `scipy.linalg.inv` и `numpy.linalg.lstsq` не падают.
- [ ] **Step 2: Тест окружения**

```python
def test_versions():
    import caiman, numpy
    assert caiman.__version__.startswith("1.13.2")
    assert int(numpy.__version__.split(".")[0]) >= 2

def test_linalg_does_not_crash():
    import numpy as np, scipy.linalg as sl
    a = np.random.default_rng(0).random((80, 80))
    assert sl.inv(a).shape == (80, 80)
    assert np.linalg.lstsq(a, a[:, 0], rcond=None)[0].shape == (80,)

def test_ffv1_codec_available():
    import av
    assert "ffv1" in av.codecs_available and "libx264" in av.codecs_available

def test_production_model_loads():
    import joblib, numpy as np
    m = joblib.load("production_models/ebm_v9_iter8.pkl")
    assert m.predict_proba(np.zeros((2, len(m.feature_names_in_)))).shape == (2, 2)
```

- [ ] **Step 3:** Run: `conda run -n bearmind_cm113 python -m pytest tests/test_environment.py -v` → Expected: 4 passed.
- [ ] **Step 4:** Удалить старое окружение `bearmind` (CaImAn 1.11.3, собрано в этой же сессии) и переименовать: `conda env remove -n bearmind -y` → `conda rename -n bearmind_cm113 bearmind`. Повторить Step 3 с `-n bearmind`.
- [ ] **Step 5:** Закрепить точные версии верхнего уровня в `environment.yml`/`requirements-pip.txt`, сохранить `env/bearmind-lock.txt`. Commit: `env: CaImAn 1.13.2 environment, pinned versions and env tests`.

### Task 0.2: Старый код на NumPy 2 и CaImAn 1.13.2

**Files:**
- Modify: `bm_batch_routines.py:669`, `bm_examinator.py:1671,1713,1725` (`np.Inf` → `np.inf`), `capcan_validation.py:50-54` (`sep=';'`), прочие места, которые падают в тестах.
- Test: существующие `tests/`, `test_event_stats.py`

**Interfaces:**
- Produces: старые функции (`DoMotionCorrection`, `DoCNMF`, `run_auto_inspection`, `bm_qc.*`) запускаются в `bearmind` — нужны для эталонных сравнений (задачи 0.4, 18).

- [ ] **Step 1:** Run: `conda run -n bearmind python -m pytest tests/ test_event_stats.py -q` → зафиксировать падения.
- [ ] **Step 2:** Исправить только совместимость (NumPy 2, изменённые API CaImAn 1.13) и баг разделителя в `capcan_validation.load_fps_data`. Поведение не менять.
- [ ] **Step 3:** Run тот же набор → Expected: всё зелёное, включая `test_event_stats.py`.
- [ ] **Step 4:** Run: `conda run -n bearmind python -c "import ae_launch, bm_examinator, bm_batch_routines, export_estimates_data, capcan_validation"` → без ошибок.
- [ ] **Step 5:** Commit: `fix: run legacy modules on NumPy 2 / CaImAn 1.13.2; fps table separator in capcan_validation`.

### Task 0.3: Разведка памяти на CaImAn 1.13.2

**Files:**
- Create: `docs/superpowers/specs/2026-10-31-caiman-memory.md`
- Create (временные, вне репо): скрипты замеров в scratch-папке

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-10-31-caiman-memory-prelim.md` (разведка по 1.11.3: горячие точки, множители, минимальный набор правок); скрипты замеров scratch `memrecon/`.
- Produces: для задач 8 и 10 — подтверждённые на 1.13.2 правки обёртки (поток для кропа, сдвигов MC, memmap, PNR) и коэффициент памяти воркера патча `patch_gb_per_mframe` для `tsub=1` и `tsub=2` (ГБ на миллион кадров); ответ, материализует ли `compute_W` в 1.13.2 весь фильм (если да — это в TODO, не в этой фазе).

Решения автора (2026-10-31): детерминизм CNMF-E не навязывать — небольшие различия между машинами допустимы, ядра и память использовать максимально в разумных пределах; out-of-core подмена `compute_W` разрешена, но перенесена в TODO.

- [ ] **Step 1:** Перепроверить на исходниках 1.13.2 каждую горячую точку отчёта (#1–#7): та же ли строка, тот же ли множитель; что из этого в 1.13.2 уже исправлено.
- [ ] **Step 2:** Повторить замеры скриптами `memrecon/` в `bearmind`: воркер патча (T = 2000/4000/8000, `tsub` 1 и 2), `motion_correct` при `splits_rig=ceil(T/500)`, `save_movie=False`, `compute_W`.
- [ ] **Step 3:** Отчёт `docs/superpowers/specs/2026-10-31-caiman-memory.md`: таблица «этап → пик ГБ → формула от T» для 1.13.2, обновлённый `patch_gb_per_mframe`. Если что-то расходится с текстом задач 8 и 10 — обновить задачи в этом плане.
- [ ] **Step 4:** Commit: `docs: CaImAn 1.13.2 memory profile and decisions for MC/CNMF-E stages`.

### Task 0.4: Насколько CaImAn 1.13.2 меняет результаты

**Files:**
- Create: `docs/superpowers/specs/2026-10-31-caiman-1.11-vs-1.13.md`

- [ ] **Step 1:** Скопировать 2 сессии `H:\Dataset\CalciumData\NOF` во временную папку. Прогнать старые `DoMotionCorrection` + `DoCNMF` (`fr`=FPS, `tsub`=1, одинаковые пороги) в окружении `caiman` (1.11.3) и в `bearmind` (1.13.2).
- [ ] **Step 2:** Сравнить: сдвиги MC (макс. |разница|), число компонент, сопоставление футпринтов (корреляция > 0.9), корреляция трейсов сопоставленных компонент.
- [ ] **Step 3:** Записать таблицу различий в отчёт. Commit: `docs: measured differences CaImAn 1.11.3 vs 1.13.2`.

---

## Фаза 1. Конвейер без окон

### Структура файлов

```
bearmind/
  __init__.py, __main__.py   # __main__ вызывает cli.main()
  naming.py                  # разбор ID сессии (перенос из корневого naming.py)
  paths.py                   # раскладка серии, все имена файлов
  defaults.yaml              # все значения по умолчанию
  config.py                  # машина / серия / мышь / сессия, слияние, хэши
  registry.py                # SessionRecord, атомарная запись JSON
  timestamps.py              # форматы таймстемпов, FPS, склейка, запись
  video.py                   # чтение сырья по сессии, FFV1, mp4, VideoArray
  scan.py                    # поиск сессий, аномалии
  crop.py                    # поле зрения, рамка мыши
  checks.py                  # проверки после этапов
  provenance.py              # git, версии, хэши файлов
  stages/{__init__,crop,mc,thresholds,cnmf,gate,autoinspect,export}.py
  runner.py                  # планировщик, продолжение, логи, пульс, замок
  cli.py
naming.py                    # остаётся тонкой прослойкой: from bearmind.naming import *
tests/bearmind/              # тесты пакета; conftest.py с генератором крошечных серий
```

### Task 1: Имена и пути

**Files:**
- Create: `bearmind/__init__.py`, `bearmind/naming.py`, `bearmind/paths.py`
- Modify: `naming.py` → прослойка `from bearmind.naming import *`
- Test: `tests/bearmind/test_naming_paths.py`

**Interfaces:**
- Produces:
  - `naming.SESSION_PATTERN: str` (как сейчас), `session_id(text: str) -> str | None`, `mouse_id(sid: str) -> str`; старые `extract_session_id`, `extract_base_session` остаются.
  - `paths.SeriesPaths(root: Path)` (frozen dataclass) с методами: `raw_root`, `timestamps_root`, `mc_video(sid)`, `mc_shifts(sid)`, `estimates(sid)`, `inspection_dir(sid)`, `manual_result(sid)` (`5_Inspection/<sid>/<sid>_manual_estimates.pickle`), `export_npz(sid)`, `export_json(sid)`, `export_timestamps(sid)`, `export_filters(sid)`, `preview(sid, kind)` (`Preview/<sid>_<kind>.mp4`; kinds: `raw`, `raw_dff`, `mc`, `mc_dff`, `cnmf_qc`), `residual_png(sid)`, `registry_file(sid)` (`_bearmind/registry/<sid>.json`), `series_config`, `mouse_config(mouse)`, `session_config(sid)` (`_bearmind/config/...yaml`), `threshold_maps(sid)` (`_bearmind/thresholds/<sid>_maps.npz`), `logs_dir`, `session_log(sid)`, `heartbeat`, `lock`, `summary_csv`, `rel(p) -> str` (путь относительно корня серии, с `/`).

- [ ] **Step 1: Тесты**

```python
def test_session_id_and_mouse():
    assert session_id("x/NOF_H01_1D_0.avi") == "NOF_H01_1D"
    assert session_id("HOS_D04_1D_2T") == "HOS_D04_1D_2T"
    assert session_id("readme.txt") is None
    assert mouse_id("LNOF_J01_1D") == "LNOF_J01"

def test_paths_layout(tmp_path):
    p = SeriesPaths(tmp_path / "серия 1")
    assert p.mc_video("NOF_H01_1D").relative_to(p.root).as_posix() == "3_MC/NOF_H01_1D_MC.avi"
    assert p.export_filters("NOF_H01_1D").relative_to(p.root).as_posix() == "6_Export/filters/NOF_H01_1D_filters.mat"
    assert p.preview("NOF_H01_1D", "mc_dff").name == "NOF_H01_1D_mc_dff.mp4"
    assert p.rel(p.estimates("NOF_H01_1D")) == "4_Estimates/NOF_H01_1D_estimates.pickle"
```

- [ ] **Step 2:** Run: `conda run -n bearmind python -m pytest tests/bearmind/test_naming_paths.py -v` → FAIL (модуля нет).
- [ ] **Step 3:** Реализовать; перенести тело `naming.py` в `bearmind/naming.py`.
- [ ] **Step 4:** Run тот же тест и весь `tests/` → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): session naming and series path layout`.

### Task 2: Конфиги

**Files:**
- Create: `bearmind/defaults.yaml`, `bearmind/config.py`
- Test: `tests/bearmind/test_config.py`

**Interfaces:**
- Consumes: `SeriesPaths` (задача 1).
- Produces: `load_defaults() -> dict`; `machine_config() -> dict` (файл `~/.bearmind/machine.yaml`, создаётся при первом вызове: `temp_dir` = `%TEMP%/bearmind`, `n_cpus`, `ram_gb` из psutil, `cnmf_ram_budget_gb` = 0.8 × `ram_gb`); `resolve(paths, sid=None, mouse=None) -> dict` (глубокое слияние: defaults < series < mouse < session; плюс ключ `machine`); `update_yaml(path, patch: dict) -> None` (атомарно); `params_hash(params: dict) -> str` (sha256 от `json.dumps(sort_keys=True, default=str)`).

`defaults.yaml` содержит разделы с точными значениями:
- `scan`: `max_mismatch_frac: 0.02`, `gap_factor: 2.0`, `fps_series_dev: 0.05`.
- `crop`: `n_sample_frames: 300`, `margin_px: 10`, `min_fov_frac: 0.05`.
- `mc`: из ячейки ноутбука `mc_dict` без `write_mp4` и `memory_fact` (`pw_rigid: false`, `max_shifts: [35, 35]`, `gSig_filt: [8, 8]`, `strides: [48, 48]`, `overlaps: [24, 24]`, `max_deviation_rigid: 15`, `border_nan: copy`, `niter_rig: 1`, `use_cuda: false`); `splits_rig` не хранится — этап MC ставит `ceil(T/500)`; `memory_fact` выброшен: это параметр патчей CNMF, MotionCorrect его игнорирует (разведка памяти).
- `memory`: `patch_gb_per_mframe: {1: 245, 2: 163}` (ГБ на миллион кадров на воркер патча; по 1.11.3, уточняется задачей 0.3).
- `thresholds`: `gsig_range: [3, 4, 5, 6, 7]`.
- `cnmf`: словарь ячейки ноутбука (`decay_time: 1`, `method_init: corr_pnr`, `K: null`, `merge_thr: 0.8`, `p: 1`, `tsub: 1`, `ssub: 1`, `rf: 40`, `stride: 25`, `only_init: true`, `nb: 0`, `nb_patch: 0`, `method_deconvolution: oasis`, `low_rank_background: null`, `update_background_components: true`, `normalize_init: false`, `center_psf: true`, `ssub_B: 2`, `ring_size_factor: 1.5`, `del_duplicates: true`, `border_pix: 5`, `min_SNR: 2.5`, `rval_thr: 0.95`, `use_cnn: false`); `gSig`, `min_corr`, `min_pnr` — из конфига сессии; `gSiz = 4*gSig + 1`; `fr` = FPS сессии.
- `autoinspect`: значения из Global Constraints.
- `gate`: `enabled: true`.
- `checks`: `mc_max_shift_px: {warn: 15, error: 30}`, `mc_template_corr: {warn: 0.6}`, `cnmf_n_components_vs_median: {warn: 2.0, error: 3.0}`, `cnmf_missed_cells: {warn: 20}`, `auto_deleted_frac: {warn: 0.8, error: 0.95}`, `export_n_neurons: {warn: 30, error: 5}`, `export_median_snr: {warn: 3.0}`, `min_sessions_for_relative: 3`. (Начальные значения; калибруются в задаче 17.)
- `preview`: `crf: 23`, `speed: 4`, `tbin: 2`.

- [ ] **Step 1: Тесты**

```python
def test_merge_order(tmp_path):
    p = SeriesPaths(tmp_path)
    update_yaml(p.series_config, {"cnmf": {"merge_thr": 0.7}})
    update_yaml(p.mouse_config("NOF_H01"), {"crop": {"box": [400, 420]}})
    update_yaml(p.session_config("NOF_H01_1D"), {"cnmf": {"gSig": 5, "min_corr": 0.85, "min_pnr": 8}})
    cfg = resolve(p, sid="NOF_H01_1D")
    assert cfg["cnmf"]["merge_thr"] == 0.7 and cfg["cnmf"]["gSig"] == 5
    assert cfg["crop"]["box"] == [400, 420]
    assert cfg["autoinspect"]["pxlthr_distance_boundary"] == 15
    assert cfg["cnmf"]["tsub"] == 1 and "fr" not in cfg["cnmf"]

def test_hash_stable_and_sensitive():
    assert params_hash({"a": 1, "b": [1, 2]}) == params_hash({"b": [1, 2], "a": 1})
    assert params_hash({"a": 1}) != params_hash({"a": 2})
```

- [ ] **Step 2–4:** FAIL → реализовать → PASS (`pytest tests/bearmind/test_config.py -v`).
- [ ] **Step 5:** Commit: `feat(bearmind): layered configs with production defaults`.

### Task 3: Реестр сессий

**Files:**
- Create: `bearmind/registry.py`
- Test: `tests/bearmind/test_registry.py`

**Interfaces:**
- Produces:
  - `STAGES = ("crop", "mc", "thresholds", "cnmf", "gate", "autoinspect", "export")`
  - `class State(str, Enum)`: `PENDING, RUNNING, DONE, FAILED, SKIPPED, WAITING`
  - `@dataclass Check(stage: str, name: str, level: str, value: float | None, message: str)`; `level ∈ {"ok", "warning", "error"}`
  - `@dataclass Anomaly(code: str, level: str, message: str, auto_fixed: bool = False, data: dict = {})`
  - `@dataclass Decision(what: str, value: str, who: str, when: str)`
  - `@dataclass RawPart(folder: str, chunks: list[str], n_frames: int, shape: tuple[int, int], timestamps_file: str | None, ts_format: str | None, start_ms: float | None, metadata_file: str | None, drop_chunk_tail: bool = False, drop_last_frame: bool = False)`
  - `@dataclass StageRecord(state: State = PENDING, params_hash: str | None = None, started: str | None = None, finished: str | None = None, outputs: dict[str, str] = {}, metrics: dict = {}, checks: list[Check] = [], error: str | None = None)`
  - `@dataclass SessionRecord(sid: str, mouse: str, mode: str, parts: list[RawPart], n_frames: int, fps: float | None, fps_info: dict, anomalies: list[Anomaly], decisions: list[Decision], excluded: bool, stages: dict[str, StageRecord])`
  - `load_record(paths, sid) -> SessionRecord`, `save_record(paths, rec) -> None` (атомарно: `.tmp` + `os.replace`), `all_records(paths) -> list[SessionRecord]` (сортировка по sid).

- [ ] **Step 1: Тест** — круговая запись/чтение со всеми полями; `save_record` при исключении в середине записи не портит старый файл (смоделировать падение `json.dump` через monkeypatch); `all_records` на пустой серии → `[]`.
- [ ] **Step 2–4:** FAIL → реализовать (dataclasses + `asdict`, enum как строка) → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): session registry with atomic writes`.

### Task 4: Таймстемпы и FPS

**Files:**
- Create: `bearmind/timestamps.py`
- Test: `tests/bearmind/test_timestamps.py`

**Interfaces:**
- Produces:
  - `@dataclass Timestamps(raw: np.ndarray, t_ms: np.ndarray, fmt: str)`; `fmt ∈ {"bonsai_ticks", "ucla_ms"}`; `raw` — исходные значения строк (int64 тики или мс), `t_ms` — float64 мс (тики/1e4; для UCLA — мс от старта + `start_epoch_ms`, если известен).
  - `read_timestamps(path: Path, start_epoch_ms: float | None = None) -> Timestamps` — формат определяется сам: одна колонка целых без заголовка → bonsai; заголовок `Frame Number,Time Stamp (ms),Buffer Index` → UCLA; иначе `ValueError("unknown timestamp format: <path>")`.
  - `@dataclass FpsEstimate(fps: float, n_outliers: int, n_gaps: int, max_gap_ms: float, median_interval_ms: float)`
  - `robust_fps(t_ms: np.ndarray, gap_factor: float = 2.0) -> FpsEstimate` — правило: отбросить интервалы ≤ 0; `m` = медиана оставшихся; оставить интервалы в `[0.5m, 1.5m]`; `fps = 1000 / mean(оставшихся)`; `n_outliers` = отброшенные; `n_gaps` = интервалы `> gap_factor*m`.
  - `stitch(parts: list[Timestamps]) -> Timestamps` — по времени начала; паузы сохраняются.
  - `write_timestamps(ts: Timestamps, frame_idx: np.ndarray, path: Path) -> None` — в исходном формате: bonsai — колонка `raw[frame_idx]` без заголовка; UCLA — тот же заголовок, `Frame Number` заново с 0, мс от начала первой части.

- [ ] **Step 1: Тесты**

```python
def test_bonsai_fps_with_gap(tmp_path):
    ticks = 638331574485491072 + np.arange(1000) * 506000          # 50.6 ms -> 19.763 fps
    ticks[500:] += 30_000_000                                        # 3 s disconnect gap
    f = tmp_path / "S_Mini_TS.csv"; np.savetxt(f, ticks, fmt="%d")
    ts = read_timestamps(f)
    est = robust_fps(ts.t_ms)
    assert ts.fmt == "bonsai_ticks"
    assert abs(est.fps - 19.763) < 0.01 and est.n_gaps == 1 and est.max_gap_ms > 3000

def test_ucla_negative_start(tmp_path):
    f = tmp_path / "timeStamps.csv"
    f.write_text("Frame Number,Time Stamp (ms),Buffer Index\n0,-14,0\n1,22,0\n2,53,0\n3,86,0\n")
    ts = read_timestamps(f, start_epoch_ms=1728989165178.0)
    assert ts.fmt == "ucla_ms" and ts.raw[0] == -14 and abs(robust_fps(ts.t_ms).fps - 30.0) < 1.5

def test_roundtrip_bonsai_subset(tmp_path): ...  # write_timestamps(ts, np.arange(10, 20)) -> read back equals raw[10:20]

def test_unknown_format(tmp_path):
    f = tmp_path / "x.csv"; f.write_text("a;b\n1;2\n")
    with pytest.raises(ValueError, match="unknown timestamp format"):
        read_timestamps(f)
```

- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): timestamp formats, robust fps, stitching`.

### Task 5: Видео: чтение сессии, FFV1, mp4, массив поверх AVI

**Files:**
- Create: `bearmind/video.py`
- Modify: `bm_qc.py:36-45` (`open_movie` оборачивает `.avi` в `VideoArray`)
- Test: `tests/bearmind/test_video.py`; `tests/bearmind/conftest.py` (фикстура `make_avi(path, frames, codec="FFV1")` на PyAV)

**Interfaces:**
- Consumes: `RawPart`, `SessionRecord` (задача 3).
- Produces:
  - `@dataclass CropBox(top: int, left: int, height: int, width: int)` (здесь, чтобы не было циклического импорта с `crop.py`).
  - `video_info(path: Path) -> tuple[int, int, int, float]` (n_frames, h, w, fps).
  - `last_frame_readable(path: Path) -> bool`.
  - `class SessionReader(series_root: Path, rec: SessionRecord, crop: CropBox | None = None)`: `.n_frames`, `.shape`; `iter_chunks(chunk: int = 1000) -> Iterator[np.ndarray]` (uint8, (n, h, w); порядок частей и кусков из реестра; применяет `drop_chunk_tail`, `drop_last_frame`, обрезку до `rec.n_frames`, кроп); `sample(n: int, seed: int = 0) -> np.ndarray`.
  - `write_ffv1(chunks: Iterable[np.ndarray], path: Path, fps: float) -> int` — PyAV, `ffv1`, `gray`; пишет в `path.with_suffix(".partial.avi")` и переименовывает в конце; возвращает число кадров.
  - `class VideoArray(path: Path)`: `.shape`, `.dtype`, `__getitem__` для `int` и `slice` по времени (FFV1 — каждый кадр ключевой, произвольный доступ через seek), `__len__`.
  - `write_preview_mp4(chunks: Iterable[np.ndarray], path: Path, fps: float, crf: int = 23, speed: int = 4, tbin: int = 2) -> None` — libx264, yuv420p, кадры усредняются по `tbin`.

- [ ] **Step 1: Тесты**

```python
def test_ffv1_lossless(tmp_path):
    frames = np.random.default_rng(0).integers(0, 256, (37, 24, 32), dtype=np.uint8)
    n = write_ffv1([frames[:20], frames[20:]], tmp_path / "x.avi", fps=20.0)
    assert n == 37 and np.array_equal(VideoArray(tmp_path / "x.avi")[0:37], frames)
    assert not list(tmp_path.glob("*.partial*"))

def test_session_reader_rules(tmp_path, make_avi):
    # 3 chunks of 10 frames whose tails duplicate (drop_chunk_tail) -> 28 frames
    ...
    assert reader.n_frames == 28 and sum(c.shape[0] for c in reader.iter_chunks(7)) == 28

def test_crop_applied(...)            # shape == (crop.height, crop.width), values match slice of source
def test_videoarray_random_access(...)  # VideoArray(p)[15] == frames[15]
```

- [ ] **Step 2–4:** FAIL → реализовать → PASS. Плюс `tests/test_bm_qc.py` остаётся зелёным.
- [ ] **Step 5:** Commit: `feat(bearmind): session reader, FFV1 writer, previews, VideoArray`.

### Task 6: Скан серии

**Files:**
- Create: `bearmind/scan.py`
- Test: `tests/bearmind/test_scan.py`; фикстура `make_series(tmp_path, layout)` в `conftest.py` строит крошечные серии bonsai/legacy из `make_avi` и таймстемпов.

**Interfaces:**
- Consumes: задачи 1–5.
- Produces: `detect_mode(folder: Path) -> str | None` (`"bonsai"`: внутри `<SID>_<n>.avi`; `"legacy"`: путь вида `<SID>/<YYYY_MM_DD>/<HH_MM_SS>/Miniscope/<n>.avi`); `scan_series(series_root: Path) -> list[SessionRecord]` — пишет реестр; для известных сессий сохраняет `decisions`, `excluded` и `stages`; создаёт `series_config`, если его нет.

Правила скана (коды аномалий и уровни):
- Части сессии группируются по `session_id(имя папки)`. Папка с AVI, чьё имя не разбирается → `unrecognized` (error) с путём; такие записи в реестр не идут, но попадают в `_bearmind/scan_unrecognized.csv`.
- Таймстемпы: bonsai — `<SID>_Mini_TS.csv` / `<SID>_timestamp.csv` в папке сессии, иначе в `0_TimeStamps/`; legacy — `Miniscope/timeStamps.csv`, `start_ms` из `<HH_MM_SS>/metaData.json` → `recordingStartTime.msecSinceEpoch`. Нет файла → `no_timestamps` (error, `excluded=True`).
- Нет `metaData.json` (legacy) → `no_metadata` (warning).
- Последний кадр последнего куска не читается → `drop_last_frame=True`, `broken_last_frame` (warning, auto_fixed).
- Лишних кадров ровно `n_chunks − 1` → `drop_chunk_tail=True`, `chunk_tail_frames` (warning, auto_fixed).
- Иначе кадры ≠ строки: `|diff|/n ≤ scan.max_mismatch_frac` → обрезка до общей длины с начала, `frames_ts_mismatch` (warning, auto_fixed, `data={"diff":…, "where":…}`); больше → `frames_ts_mismatch` (error, `excluded=True`).
- Несколько частей → склейка по `start_ms` (или по первому таймстемпу), `split_recording` (warning, `data={"gaps_s": [...]}`).
- `fps_info.n_gaps > 0` → `dropped_frames` (warning); FPS отличается от медианы серии больше чем на `scan.fps_series_dev` или от номинального `frameRate` legacy → `fps_outlier` (warning).

- [ ] **Step 1: Тесты**

```python
def test_bonsai_nested_and_ts_dir(make_series): ...      # 1_Raw/HM_NOF_1D/NOF_H01_1D/, ts in 0_TimeStamps -> mode bonsai, n_frames ok
def test_legacy_split_is_stitched(make_series): ...      # two <HH_MM_SS> folders -> 1 record, 2 parts, anomaly split_recording, n_frames == sum
def test_missing_timestamps_excluded(make_series): ...   # anomaly no_timestamps level error, excluded True
def test_chunk_tail_rule(make_series): ...               # surplus == n_chunks-1 -> drop_chunk_tail True
def test_large_mismatch_excluded(make_series): ...
def test_unrecognized_folder_reported(make_series): ...  # "nof_h01_1d_part2" -> in scan_unrecognized.csv, not silently lost
def test_rescan_keeps_decisions(make_series): ...        # excluded flag and a Decision survive a second scan
def test_cyrillic_space_root(make_series): ...           # series root "серия 1" works
```

- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): series scan for bonsai and legacy layouts with anomalies`.

### Task 7: Автокроп (рамка мыши)

**Files:**
- Create: `bearmind/crop.py`, `bearmind/stages/__init__.py`, `bearmind/stages/crop.py`
- Test: `tests/bearmind/test_crop.py`

**Interfaces:**
- Consumes: `SessionReader.sample`, `CropBox`, конфиги.
- Produces:
  - `detect_fov(frames: np.ndarray, min_fov_frac: float = 0.05) -> tuple[int, int, int, int] | None` — (top, left, height, width) освещённой области: среднее по кадрам → гауссово сглаживание σ=5 → порог Оцу → наибольшая связная компонента → её ограничивающий прямоугольник; `None`, если площадь < `min_fov_frac` кадра.
  - `mouse_box_size(fovs: list[tuple[int,int,int,int]], frame_shape: tuple[int,int], margin: int) -> tuple[int, int]` — поэлементный максимум (h, w) по сессиям + 2·margin, не больше кадра.
  - `place_box(fov, size, frame_shape) -> CropBox` — рамка заданного размера с центром в центре поля зрения, сдвинутая внутрь кадра.
  - `stages.StageResult(outputs: dict[str, str], metrics: dict, anomalies: list[Anomaly])` (в `stages/__init__.py`).
  - `stages.crop.detect(paths, rec, cfg) -> StageResult` (поле зрения сессии в `metrics["fov"]`) и `stages.crop.finalize_mouse(paths, mouse, records, cfg) -> dict[str, CropBox | None]` — пишет `crop.box` в конфиг мыши (если там ещё нет; поздние сессии получают тот же размер) и `crop.box_pos` в конфиги сессий; сессия без поля зрения → `None` (целый кадр) + аномалия `fov_not_found` (warning: «размер кадра отличается от остальных сессий мыши»).

- [ ] **Step 1: Тесты**

```python
def test_detect_fov_bright_disk():   # synthetic disk on dark background -> bbox within 3 px
def test_no_fov_on_uniform_noise():  # -> None
def test_box_same_size_moves_with_fov():
    s = mouse_box_size([(100, 120, 300, 310), (110, 90, 290, 300)], (608, 608), margin=10)
    a = place_box((100, 120, 300, 310), s, (608, 608)); b = place_box((110, 90, 290, 300), s, (608, 608))
    assert (a.height, a.width) == (b.height, b.width) == s and (a.top, a.left) != (b.top, b.left)
def test_box_clipped_inside_frame():  # FOV at the edge -> box fully inside frame, size unchanged
def test_finalize_keeps_existing_mouse_size(tmp_path): ...  # second call with a new session keeps crop.box
```

- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): auto crop with fixed per-mouse box size`.

### Task 8: Этап MC

**Files:**
- Create: `bearmind/stages/mc.py`
- Test: `tests/bearmind/test_stage_mc.py`

**Interfaces:**
- Consumes: `SessionReader`, `CropBox` из конфига сессии, `write_ffv1`, `write_preview_mp4`, `bm_qc.make_dff_video`, решения задачи 0.3.
- Produces: `run(paths, rec, cfg) -> StageResult`: `outputs = {"mc_video", "mc_shifts", "preview_raw", "preview_raw_dff", "preview_mc", "preview_mc_dff"}`; `metrics = {"max_shift_px", "mean_template_corr", "n_frames", "crop": [top, left, h, w] | None}`.

Порядок внутри этапа (по разведке памяти, `2026-10-31-caiman-memory-prelim.md`, точки #2, #4, #5, #9; задача 0.3 подтверждает на 1.13.2). Ни один шаг не держит в памяти всё видео:
1. Кропнутая сессия потоком (`SessionReader.iter_chunks`) пишется во временный BigTIFF uint8 через `tifffile.TiffWriter` в `machine.temp_dir/<sid>/`.
2. `MotionCorrect([tmp_tif], dview, **mc)` с `splits_rig = ceil(T/500)` и `motion_correct(save_movie=False)` — нужны только сдвиги и шаблон (сдвиги не зависят от `splits_rig` и `save_movie` при `niter_rig=1`, замерено).
3. Сдвиги применяются кусками по 1000 кадров той же функцией CaImAn, что внутри `apply_shifts_movie` (`apply_shift_iteration` с теми же `border_nan`, интерполяцией), кадры переводятся в uint8 через `np.clip(np.rint(x), 0, 255)` и потоком пишутся в FFV1 (`write_ffv1`) и в mp4-превью `mc`.
4. Сдвиги → `3_MC/<sid>_MC_shifts.npy`; `max_shift_px` = макс. |сдвиг|; `mean_template_corr` = средняя корреляция 200 равномерно взятых кадров с шаблоном MC.
5. Превью: `raw` (сырое без кропа, потоком), `mc`; dF/F-версии обоих через `bm_qc.make_dff_video` (его память ограничена `max_mem_gb`).
6. Временные файлы удаляются в `finally`; кластер останавливается в `finally`.

- [ ] **Step 1: Тест** на синтетике (300 кадров 64×64, известный сдвиг ±3 px): выходы существуют; `max_shift_px` ≈ 3 (±0.5); кадры FFV1 = `np.clip(np.rint(apply_shifts_movie(...)), 0, 255)` того же входа (побитово); временная папка пуста после этапа; при исключении внутри MotionCorrect (monkeypatch) временная папка тоже пуста и исключение пробрасывается.
- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): motion correction stage with FFV1 output and previews`.

### Task 9: Этап картинок для порогов

**Files:**
- Create: `bearmind/stages/thresholds.py`
- Test: `tests/bearmind/test_stage_thresholds.py`

**Interfaces:**
- Consumes: `VideoArray(mc_video)`, `bm_qc.compute_init_maps(movie, gSig, tsub=cfg.cnmf.tsub, max_mem_gb=…)`, `bm_qc.suggest_init_thresholds`.
- Produces: `run(paths, rec, cfg) -> StageResult`: файл `threshold_maps(sid)` (npz: `gsig`, `cn_<g>`, `pnr_<g>`, `mean`, `max` для каждого g из `thresholds.gsig_range`); `metrics["suggested"] = {"gSig": int, "min_corr": float, "min_pnr": float}`. Предложение не пишется в конфиг сессии само — только командой `thresholds accept-suggested` (задача 16).

- [ ] **Step 1: Тест** на синтетике `tests/test_bm_qc.synthetic_movie`, записанной в FFV1: npz содержит ключи для всех gSig; `suggested.gSig` ∈ диапазона.
- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): threshold maps stage`.

### Task 10: Этап CNMF-E

**Files:**
- Create: `bearmind/stages/cnmf.py`
- Test: `tests/bearmind/test_stage_cnmf.py`

**Interfaces:**
- Consumes: `mc_video`, конфиг сессии (`gSig`, `min_corr`, `min_pnr`), `rec.fps`, решения задачи 0.3, `bm_qc.cnmf_qc`.
- Produces: `cnmf_params(cfg, fps) -> dict` (словарь CaImAn: базовый `cnmf` + `gSig=(g,g)`, `gSiz=(4g+1, 4g+1)`, `min_corr`, `min_pnr`, `fr=fps`); `run(paths, rec, cfg) -> StageResult`: `outputs = {"estimates", "preview_cnmf_qc", "residual_png"}`; `metrics = {"n_components", "missed_cells", "median_snr", "gSig", "min_corr", "min_pnr"}`. Если порогов в конфиге сессии нет — `raise WaitingForInput("thresholds")` (класс в `stages/__init__.py`), планировщик ставит `WAITING`.

Порядок (по разведке памяти, точки #1, #3, #6, #7; задача 0.3 подтверждает на 1.13.2):
- FFV1 → временный C-order memmap CaImAn в `machine.temp_dir` полосами по столбцам (тот же файл, что дал бы `cm.save_memmap`, побитово; без загрузки целиком).
- Пул на этапе патчей: `patch_workers(T, tsub, cfg) -> int` = `floor(machine.cnmf_ram_budget_gb / (patch_gb_per_mframe[tsub] * T / 1e6))`, ограничить `[1, machine.n_cpus - 1]`; `patch_gb_per_mframe` из задачи 0.3 (по 1.11.3: ≈ 245 ГБ при `tsub=1`, ≈ 163 ГБ при `tsub=2` на миллион кадров), `cnmf_ram_budget_gb` по умолчанию = 0.8 × `ram_gb` (решение автора). Размер пула пишется в метаданные (он влияет на результат).
- `compute_W`: штатный (out-of-core подмена — в TODO; если на 1.13.2 он не влезает, кадр 608×608 на часовой сессии пока не обрабатывается без кропа — записать это в отчёт задачи 17).
- `imax` в estimates (нужен окну ручной инспекции) — потоковым двухпроходным PNR, не `correlation_pnr(images[::5])`.
- `CNMF(n_processes, dview, params=CNMFParams(params_dict=cnmf_params(...)))` → `fit` → `evaluate_components` → в estimates: `est.cnmf_dict = cnmf_params`, `est.mc_video = paths.rel(mc_video)` (относительный путь; абсолютного `tif_name` нет) → pickle через `.partial` → `bm_qc.cnmf_qc(est, movie=memmap, out_base=…)`: QC-видео и карту остатков переместить в `Preview/`, `missed_cells` = `residual_excess_outside` из его JSON → удалить memmap в `finally`.

- [ ] **Step 1: Тесты**

```python
def test_fr_equals_fps_regardless_of_tsub():
    cfg = load_defaults(); cfg["cnmf"].update(gSig=4, min_corr=0.85, min_pnr=8, tsub=2)
    p = cnmf_params(cfg, fps=19.76)
    assert p["fr"] == 19.76 and p["tsub"] == 2 and p["gSiz"] == (17, 17)

def test_waiting_without_thresholds(tmp_path): ...  # raises WaitingForInput

def test_cnmf_on_synthetic(tmp_path): ...  # synthetic_movie(6 cells) in FFV1 -> n_components >= 4, estimates pickle has mc_video relative path, temp dir empty

def test_patch_workers_budget():
    cfg = {"machine": {"n_cpus": 32, "cnmf_ram_budget_gb": 153}, "memory": {"patch_gb_per_mframe": {1: 245, 2: 163}}}
    assert patch_workers(108000, 1, cfg) == 5 and patch_workers(108000, 2, cfg) == 8 and patch_workers(2000, 1, cfg) == 31

def test_memmap_strips_identical_to_save_memmap(tmp_path): ...  # small FFV1 -> our memmap bytes == cm.save_memmap bytes on the same frames
```

- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): CNMF-E stage with fr = session fps and QC previews`.

### Task 11: Ворота ручной инспекции (без окна)

**Files:**
- Create: `bearmind/stages/gate.py`
- Test: `tests/bearmind/test_stage_gate.py`

**Interfaces:**
- Produces: `GATE_DECISIONS = ("cleaned", "not_needed")`; `decide(paths, sid, decision: str, who: str) -> None` (пишет `Decision("manual_inspection", decision, who, now)` и ставит этап `gate` в `DONE`; `cleaned` требует существующего `manual_result(sid)`); `run(paths, rec, cfg) -> StageResult` — если `gate.enabled` = false → `DONE` с `metrics={"gate": "disabled"}`; если решения нет → `raise WaitingForInput("manual_inspection")`; иначе `DONE`. `input_estimates(paths, rec) -> Path` — `manual_result`, если решение `cleaned`, иначе `estimates`.

- [ ] **Step 1: Тесты:** без решения → `WaitingForInput`; `not_needed` → DONE и `input_estimates` = `4_Estimates/...`; `cleaned` без файла → `ValueError`; ворота выключены → DONE.
- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): manual inspection gate (CLI decisions)`.

### Task 12: Этап автоинспекции

**Files:**
- Create: `bearmind/stages/autoinspect.py`
- Modify: `ae_launch.py` — `run_auto_inspection(..., artifacts_dir: str | None = None)`: если задан, артефакты пишутся прямо в эту папку (без подпапки `inspection_artifacts_<sid>`); остальное поведение без изменений.
- Test: `tests/bearmind/test_stage_autoinspect.py`

**Interfaces:**
- Consumes: `gate.input_estimates`, `rec.fps`, `cfg["autoinspect"]`.
- Produces: `run(paths, rec, cfg) -> StageResult`: вызывает `run_auto_inspection(str(input), fps=round(rec.fps), session_name=rec.sid, artifacts_dir=str(paths.inspection_dir(sid)), ml_model_path=str(<repo>/production_models/ebm_v9_iter8.pkl), **остальные параметры из cfg)`; `outputs = {"processed": <rel path of *_processed.pickle>}`; `metrics = {"n_initial", "n_final", "deleted_frac", "autoinspection_fps": round(rec.fps)}`.

- [ ] **Step 1: Тесты:** (а) монкипатч `run_auto_inspection` — проверить, что передан `fps == round(19.76) == 20`, `pxlthr_distance_boundary == 15`, `brain == "hybrid"`, абсолютный существующий путь модели, `artifacts_dir` = `5_Inspection/<sid>`; (б) интеграционный (маркер `slow`): синтетические estimates из задачи 10 → processed-pickle в `5_Inspection/<sid>/`.
- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): autoinspection stage with production parameters`.

### Task 13: Происхождение и экспорт

**Files:**
- Create: `bearmind/provenance.py`, `bearmind/stages/export.py`
- Test: `tests/bearmind/test_stage_export.py`

**Interfaces:**
- Consumes: `export_estimates_data.extract_data`, `build_metadata`, `export`, `export_filters_mat`; `timestamps.write_timestamps`; реестр.
- Produces:
  - `provenance.collect(paths, rec, cfg) -> dict`: `{"code": {"git_commit", "git_dirty"}, "packages": {"caiman", "numpy", "driada", "interpret", "python"}, "model": {"path", "sha256"}, "inputs": [rel paths raw parts], "mode", "fps": {"value", "source", "n_outliers", "max_gap_ms"}, "stage_params": {stage: params}, "decisions": [...]}`.
  - `stages.export.run(paths, rec, cfg) -> StageResult`: ML-фильтр как в `batch_export.export_one` (порог из `est.autoinspection_config["ml_threshold"]`), без feedback; `metadata["fps"] = rec.fps` (точный), `metadata["autoinspection_fps"]`, `metadata["provenance"]`, `metadata["timestamps_file"]`; пишет `6_Export/npz/<sid>_data.npz`, `_metadata.json`, `_timestamps.csv` (кадры экспорта в исходном формате), `6_Export/filters/<sid>_filters.mat`; `metrics = {"n_neurons", "median_snr"}` (`median_snr` — медиана `est.SNR_comp` по экспортированным компонентам).

- [ ] **Step 1: Тесты:** на processed-pickle из задачи 12 (или собранном вручную объекте estimates с `metrics_df`): ключи `npz` совпадают с набором ключей старого `export()` на том же объекте; в `json` есть все старые ключи + `autoinspection_fps`, `provenance`; `fps` = 19.76 (не 20); число строк `_timestamps.csv` = `C.shape[1]`; `.mat` читается `scipy.io.loadmat`.
- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): single export with exact fps, aligned timestamps, provenance`.

### Task 14: Проверки после этапов

**Files:**
- Create: `bearmind/checks.py`
- Test: `tests/bearmind/test_checks.py`

**Interfaces:**
- Consumes: `cfg["checks"]`, `StageRecord.metrics` всех сессий серии.
- Produces: `evaluate(stage: str, metrics: dict, series_metrics: list[dict], cfg: dict) -> list[Check]` — абсолютные пороги (`mc_max_shift_px`, `mc_template_corr` (ниже порога — предупреждение), `cnmf_missed_cells`, `auto_deleted_frac`, `export_n_neurons`, `export_median_snr`) и относительный (`cnmf_n_components_vs_median`, только если сессий с этим этапом ≥ `min_sessions_for_relative`); сообщение вида `"2950 components, 4.8x series median (612)"`.

- [ ] **Step 1: Тесты:** сдвиг 18 px → warning, 31 px → error; 2950 компонент при медиане 612 → error с «4.8x» в сообщении; при 2 сессиях относительной проверки нет; 4 нейрона → error.
- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): per-stage checks with absolute and series-relative thresholds`.

### Task 15: Планировщик и запуск

**Files:**
- Create: `bearmind/runner.py`
- Test: `tests/bearmind/test_runner.py`

**Interfaces:**
- Consumes: всё выше; этапы как `STAGE_FUNCS = {"crop": stages.crop.detect, "mc": stages.mc.run, "thresholds": stages.thresholds.run, "cnmf": stages.cnmf.run, "gate": stages.gate.run, "autoinspect": stages.autoinspect.run, "export": stages.export.run}`.
- Produces:
  - `stage_hash(rec, stage, cfg) -> str` — `params_hash` от параметров этапа и хэшей всех предыдущих этапов (изменение выше по цепочке делает этап и всё ниже устаревшим).
  - `next_ready(records, paths) -> tuple[str, str] | None` — сначала меньший индекс этапа, потом порядок сессий; этап готов, если сессия не `excluded`, все предыдущие этапы `DONE`, а сам этап не `DONE` с актуальным хэшем; `crop.finalize_mouse` вызывается, как только у всех неисключённых сессий мыши готов `crop.detect`.
  - `run(series_root: Path, until: str | None = None, sids: list[str] | None = None) -> dict` — цикл: `next_ready` → состояние `RUNNING` + запись → вызов этапа → `DONE` + выходы + метрики + `checks.evaluate` / `WaitingForInput` → `WAITING` / любое другое исключение → `FAILED` с трейсбеком в лог сессии и `SKIPPED` для нижних этапов этой сессии; выход, когда готовой работы нет; итог `{"done", "failed", "waiting", "skipped"}`.
  - Замок `_bearmind/run.lock` (PID + время; занятый живым процессом → `RuntimeError("series is already being processed by PID <n>")`; чужой мёртвый PID — замок перехватывается).
  - Пульс `_bearmind/heartbeat.json` (`pid`, `time`, `current`, счётчики) — при каждом переходе этапа и раз в 30 с из фонового потока.
  - Логи: `logging` в `_bearmind/logs/series.log` и `_bearmind/logs/<sid>.log`.
  - При старте все этапы в состоянии `RUNNING` переводятся в `PENDING` (процесс был убит).
  - `reset_from(paths, sid: str, stage: str, who: str) -> None` — переанализ: этап и все ниже → `PENDING`, `Decision("rerun_from", stage, who, now)`. Старые выходы остаются на месте, пока этап не перезапишет их через `.partial` + переименование.
  - `summary(paths) -> pandas.DataFrame` и запись `_bearmind/summary.csv`: по строке на сессию, столбцы — состояние и ключевое число каждого этапа, пороги, худший уровень проверки.

- [ ] **Step 1: Тесты** (этапы подменяются заглушками через `STAGE_FUNCS`):

```python
def test_order_and_resume(series_with_3_sessions): ...      # second run() does nothing; changing cnmf param re-runs cnmf..export only
def test_failure_isolated(series_with_3_sessions): ...      # mc raises for one sid -> FAILED, its later stages SKIPPED, others DONE
def test_waiting_does_not_block_others(...): ...            # gate WAITING for s1, s2/s3 reach export
def test_stale_running_reset(...): ...                      # registry has RUNNING -> treated as pending, partial file ignored
def test_disk_full_cleans_temp(...): ...                    # stage raises OSError(28) -> FAILED with message, temp dir empty
def test_second_runner_refused(...): ...                    # live PID in lock -> RuntimeError
def test_summary_csv(...): ...                              # one row per session, columns per stage
def test_reset_from_reruns_only_downstream(...): ...        # reset_from(s1, "cnmf") -> next run re-runs cnmf..export for s1 only
```

- [ ] **Step 2–4:** FAIL → реализовать → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): scheduler with resume, isolation, lock, heartbeat, summary`.

### Task 16: Командная строка

**Files:**
- Create: `bearmind/cli.py`, `bearmind/__main__.py`, `run_bearmind.bat` (запуск `conda run -n bearmind python -m bearmind %*`)
- Test: `tests/bearmind/test_cli.py`

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int`; команды:
  - `scan SERIES`
  - `run SERIES [--until STAGE] [--sessions SID ...]`
  - `status SERIES` (печать сводки ASCII + `_bearmind/summary.csv`)
  - `thresholds SERIES accept-suggested [--sessions ...]` · `thresholds SERIES set SID --gsig G --min-corr X --min-pnr Y`
  - `gate SERIES (SID ... | --all-waiting) not-needed` · `gate SERIES SID cleaned`
  - `decide SERIES SID (exclude | include | accept-mismatch)`
  - `rerun SERIES SID ... --from STAGE` (вызывает `runner.reset_from`)
  - `who` по умолчанию — имя пользователя Windows; `--who NAME` переопределяет.

- [ ] **Step 1: Тесты:** каждая команда на крошечной серии (этапы-заглушки) меняет реестр/конфиг как ожидается; неизвестная серия → код возврата 2 и сообщение `[FAILED] series folder not found: …`.
- [ ] **Step 2–4:** FAIL → реализовать (argparse) → PASS.
- [ ] **Step 5:** Commit: `feat(bearmind): command line interface`.

### Task 17: Сквозной прогон на реальных данных и калибровка проверок

**Files:**
- Create: `tests/bearmind/test_e2e_real.py` (маркер `realdata`; пропускается, если нет `BEARMIND_TEST_SERIES`)
- Modify: `bearmind/defaults.yaml` (раздел `checks` — по результатам)
- Create: `docs/superpowers/specs/2026-11-xx-e2e-report.md`

- [ ] **Step 1:** Собрать тестовую серию в `H:\Projects\_bearmind_test\series1` (копии): 3 сессии `H:\Dataset\CalciumData\NOF` (вложенная раскладка), 2 сессии legacy `H:\Dataset\CalciumData\RFC`, `LT1_K02_1D` (аномалия кадров). Оригиналы не трогать.
- [ ] **Step 2:** `scan` → проверить аномалии (у `LT1_K02_1D` — `frames_ts_mismatch`); `run --until thresholds`; `thresholds accept-suggested`; `run --until gate`; `gate --all-waiting not-needed`; `run`.
- [ ] **Step 3:** Проверить: у каждой неисключённой сессии есть все выходы `6_Export`, превью, сводка; повторный `run` ничего не делает.
- [ ] **Step 4:** Одна часовая сессия `G:\Projects\LNOF\CalciumData` (копия) целиком при `tsub=1`: проходит все этапы, пик памяти записан (сравнить с бюджетом задачи 0.3).
- [ ] **Step 5:** По распределению метрик выставить начальные пороги `checks`; записать отчёт; commit: `test: end-to-end run on real subsets; calibrated check thresholds`.

### Task 18: Проверки эквивалентности

**Files:**
- Create: `tests/bearmind/test_equivalence.py` (маркеры `realdata`, `slow`)

- [ ] **Step 1: MC.** Одна сессия NOF: старый путь (кроп той же рамкой в TIF → `DoMotionCorrection`) и этап MC. Сдвиги совпадают (`np.allclose(atol=1e-6)`), кадры отличаются не больше чем на 1 уровень серого (округление вместо усечения).
- [ ] **Step 2: CNMF-E.** Тот же MC-вход: старый `DoCNMF` (`fr`=FPS, те же параметры) и этап CNMF-E **с одинаковым размером пула и одинаковыми настройками потоков BLAS** (разведка показала, что от них зависит результат). Критерий (детерминизм не требуется, решение автора 2026-10-31): ≥ 98% компонент каждой стороны имеют пару с корреляцией футпринтов > 0.95 и трейсов > 0.95. Отдельно записать в отчёт, как меняется результат при пуле 4 против 31 (цена ограничения памяти).
- [ ] **Step 3: ML.** Найти локально `*_processed.pickle` со столбцом `ml_keep_probability` (например, под `H:\Dataset\CalciumData\*`); пересчитать вероятности моделью в `bearmind` по его `metrics_df` → `np.allclose(atol=1e-6)`. Если таких файлов нет — записать «не проверено» в отчёт задачи 17.
- [ ] **Step 4:** Схема экспорта: ключи `npz` и `mat` совпадают со старым экспортом того же processed-pickle; ключи `json` — надмножество.
- [ ] **Step 5:** Commit: `test: equivalence of MC, CNMF-E, ML and export schema with the old code`.
