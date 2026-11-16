# BEARMiND: правки LT1, раунд 2 — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** правки, которые автор одобрил 16.11 по прогону LT1: обрезанная строка таймстемпов LT1_J05_1D, проверка сдвига MC без чёрных кадров, fps вне хэша MC, три мелочи п. 22, превью в конце очереди батча, одно правило «кто ждёт пороги», три новые возможности вкладки Thresholds, заметка про VirtualDub, список плохих кадров LT1_J10_2D.

**Architecture:** правки в существующих модулях. Два новых модуля: `bearmind/seed_suggest.py` (подсказка порогов без CaImAn; `bm_qc` реэкспортирует её функции) и `bearmind/bad_frames.py` (замена плохих кадров интерполяцией и детектор полос). Хэш ни одного готового этапа не меняется, кроме двух намеренных: LT1_J05_1D после рескана (задача 7) и LT1_J10_2D после записи плохих кадров (задача 15). Это держит тест-снимок реестра LT1 (задача 1). Задачи 1–6 не трогают ничего, что импортирует работающий батч; задачи 7–16 переносятся одной остановкой батча.

**Tech Stack:** Python 3.12, CaImAn 1.13.2, NumPy 2, OpenCV, PyAV, PySide6, pytest; окружение conda `bearmind`.

**Spec:** `docs/superpowers/specs/2026-11-11-lt1-feedback.md`, пункты 3, 4, 7, 8, 16, 17, 18, 22, 24, 25, 27, 28 и решения автора 16.11. Причины и предложения:
- п. 16 и 24 — `C:\Users\User\PycharmProjects\bearmind\.superpowers\sdd\2026-11-16-lt1\j05-timeline-report.md`;
- п. 25 — `C:\Users\User\PycharmProjects\bearmind\.superpowers\sdd\2026-11-16-lt1\j10-stripes-report.md`;
- числа п. 17, 18, 22 — `docs/data-flow/index.html` (разделы 3, 4, 7) и `docs/data-flow/scripts/dfr_firstframe.json`.

Проект тестов J05 лежал в scratch (`...\scratchpad\j05\proposed_test\test_truncated_ts_row.py`). Scratch временный, поэтому тесты перенесены в задачу 7 целиком.

## Перед началом

- Работайте только в worktree `C:\Users\User\PycharmProjects\bearmind-fixes`, ветка `bearmind-fixes`. В основной папке `C:\Users\User\PycharmProjects\bearmind` идёт батч серии `H:\Projects\LT1\CalciumData`. Там ничего не меняйте до раздела «Перенос».
- Ветка `bearmind-fixes` начата с 8e82c3d. Ветка `bearmind-batch` ушла на один коммит документации (5a36c46, список правок). План от него не зависит; слияние — в разделе «Перенос».
- Автору, до задачи 7: не сохранять ручную инспекцию LT1_J05_1D до рескана по варианту A. Сохранённый результат унесёт неверную ось `.time`, а рескан выбросит его.
- Автору, до задачи 15: отчёт J10 советует придержать LT1_J10_2D (исключить), чтобы CNMF-E не прошёл по фильму с полосами. Это решение автора, не задача плана.

## Решения плана (автор может отменить)

**Р1. п. 22б, Accept.** Решение `accept_mismatch` хранит отпечаток расхождений сессии: место и разницу каждого `frames_ts_mismatch`, например `end:+107`. Новое расхождение (другая разница или другая часть) решением не покрыто, и сессия снова исключена. Старое значение `true` не считается, как у `stitch_confirmed`; в реестре LT1 таких решений нет (проверено 16.11). Где на самом деле лишние кадры, по файлам узнать нельзя: у кадра видео нет своего номера или времени. Поэтому обрезка остаётся с конца, а вопрос в окне это говорит (вкладка Scan уже говорит, Monitor — после задачи 3).

**Р2. п. 18, fps вне хэша MC.** Новые параметры MC — `{"mc", "box", "box_pos"}`. Все 84 записи MC в LT1 хранят параметры с `fps`. Запись, чей сохранённый хэш совпадает с хэшем «параметры без fps плюс её собственный сохранённый fps», сохраняет этот хэш (`runner._mc_legacy`). Так готовые MC и все этапы после них остаются актуальными, а смена fps сессии их больше не трогает. Любой новый прогон MC пишет параметры без `fps`.

**Р3. п. 17, чёрные кадры.** MC не пересчитывается. Новая команда `recheck` переписывает метрики и проверки готовых MC по файлу сдвигов и нескольким кадрам видео MC. Кадры смотрятся от самого большого сдвига вниз, до первого нечёрного: читаются только чёрные кадры, чей сдвиг иначе попал бы в максимум. Чёрный кадр — среднее ниже `checks.mc_black_frame_mean` = 0.5. Все 10 чёрных кадров LT1 имеют среднее не выше 0.0002. Самые тёмные кадры с изображением — 1.42–1.46, и MC их выравнивает (сдвиг ≤ 4.1 px; `dfr_firstframe.json`).

**Р4. п. 25, детектор полос — проверка MC, не скана.** Скан не читает пиксели (только заголовки и таймстемпы), а MC и так читает каждый сырой кадр. Метрика считается на сырых полных кадрах до кропа и до замены, как в отчёте J10 (пороги проверены там на 3 233 673 кадрах). Список плохих кадров входит в параметры MC, только если он не пуст. Кроп и сырое превью его не видят: кроп пересчитался бы со всей цепочкой и перенёс бы рамку, а сырое превью честно показывает запись.

**Р5. п. 28, «по требованию».** Значения методов считаются в фоне для показанной пары (сессия, gSig), когда её показывают, и кэшируются по (сессия, gSig, файл карт). Заранее для всех сессий ничего не считается.

**Р6. п. 8, одно правило.** Правило `waits_for_thresholds` переезжает в `runner` (иначе цикл импорта: `actions` импортирует `runner`); `actions` реэкспортирует его под прежним именем. По нему считают сердцебиение батча, `status`, `summary.csv` и итог `run`. Свой цикл ожидания runner не меняет: если перестать ждать по устаревшей записи WAITING, `run` мог бы закончиться раньше, чем истечёт `poll_s`. Monitor и вкладка Thresholds уже идут по правилу (`monitor._cell`, `NO_LONGER_WAITS`).

**Р7. п. 3.** «После основных этапов» и «когда батчу нечего делать» — одно правило: побочный этап предлагается, только если ни один основной этап ни одной сессии не готов. Это верно и тогда, когда все основные этапы ждут человека.

**Р8. п. 22в.** Этап автоинспекции передаёт настоящую ось времени из таймстемпов (`session_time_s`), как страница инспекции. Запасную ось `bm_examinator.get_timestamps` (`linspace(0, n // fps, n)`) план не трогает: ею пользуется путь ноутбука. Готовые `processed.pickle` тестовых серий не пересчитываются: экспорт эту ось не читает, а в LT1 автоинспекции ещё не было.

**Р9. п. 16, A и B.** Общая часть A и B — `read_timestamps` отбрасывает обрезанную строку и ставит флаг; скан пишет предупреждение. Для A больше кода не нужно: скан уже режет `n_frames` до числа строк. Поэтому задача 7 переносится при любом ответе. Гейт стоит на рескане: J05_1D пересчитывается с кропа (около 3 ч 25 мин батча), только когда автор выбрал A. Без рескана страница инспекции J05_1D откажет с текстом «rescan the series», а экспорт запишет 47 308 строк с предупреждением. Вариант B требует своей задачи: план её не содержит.

**Р10. п. 4, «сколько впереди».** Число сессий, чей кроп, MC или карты батч сделает раньше карт этой сессии, по порядку `runner.next_ready`: основные этапы по очереди, внутри этапа по имени сессии. Область прогона (`--sessions`, `--until`) окно не знает и считает всю серию. Тест сверяет число с самим `next_ready`.

## Global Constraints

- Окружение `bearmind` (`C:\Users\User\anaconda3\envs\bearmind\python.exe`, CaImAn 1.13.2, Python 3.12). Команды — из корня worktree через `conda run -n bearmind ...`: без активированного окружения DLL torch не грузятся.
- Быстрый набор: `conda run -n bearmind python -m pytest -q -m "not slow and not realdata" tests`. На 8e82c3d он выбирает 1280 тестов (`1280/1312 tests collected (32 deselected)`, 16.11). После каждой задачи — прежнее число плюс новые тесты, 0 failed.
- Проверки на живой серии: `BEARMIND_LT1_SERIES=H:\Projects\LT1\CalciumData`, затем `conda run -n bearmind python -m pytest -q -m realdata tests/bearmind/test_lt1_snapshot.py`. Они только читают H: и копируют файлы во временную папку теста.
- Ни одна правка не делает готовые этапы серии устаревшими (`params_hash`), кроме намеренных: LT1_J05_1D после задачи 7 и рескана, LT1_J10_2D после задачи 15 и `bad-frames`. Каждая задача, которая трогает хэшируемое (`runner.stage_params`, `defaults.yaml`, разделы конфигов, `n_frames` скана), прогоняет `tests/bearmind/test_lt1_snapshot.py`, и тест остаётся зелёным без правок.
- Текст программы, окон, консоли, комментарии и сообщения коммитов — по-английски. Вывод в консоль — только ASCII (`[OK]`, `[WARN]`, `[FAILED]`). Руководство студента (`docs/students/`) и заметки оператора — по-русски, в стиле STE; `tests/test_student_guide.py` это проверяет.
- Комментарии объясняют, почему код такой. Истории правок в них нет.
- Никаких тяжёлых прогонов MC и CNMF-E (правило автора). Чтение сырых кадров только для чтения разрешено там, где задача его называет.
- `H:\Projects\LT1\CalciumData` и все исходные данные только читаются. Тесты не пишут в H:.
- Коммиты: новые файлы — `git add -- <путь>`; затем `git commit -q -m "<msg>" -- <пути>`. Без `git add -A` и `commit -a`. В сообщениях нет `Co-Authored-By`, «Claude», «Generated with» и других упоминаний ИИ. Не пушить. В `bearmind-batch` переносить только по разделу «Перенос» и с согласия автора.
- В основной папке лежат чужие неотслеживаемые файлы (`INFO_promts.txt` и др.): не трогать.
- Перенос: задачи 1–6 — без остановки батча; задачи 7–16 — при остановке; задачи 15 и 16 — только после «да» автора.

## Review Focus

1. **Последняя строка Bonsai целая, но без перевода строки, или файл кончается одиночным `\r`.** Ничего не отбрасывается, кадры не теряются. Тесты `test_bonsai_complete_last_row_without_line_end_kept` и `test_bonsai_last_row_with_lone_cr_kept` — задача 7.
2. **fps сессии изменился после MC** (новое правило fps, рескан). MC и карты остаются актуальными; CNMF-E пересчитывается, потому что `fr` — его вход. Тест `test_fps_change_keeps_mc_and_maps_current` — задача 8.
3. **Несколько чёрных кадров наверху списка сдвигов, тёмный нечёрный кадр, сессия из одних чёрных кадров.** Пропускаются только чёрные кадры; падения нет. Тесты `test_several_black_frames`, `test_dark_frame_counts`, `test_all_frames_black` — задача 9.
4. **Человек переключил сессию или gSig, или батч заменил карты, пока считались методы.** Кнопки показывают значения только показанной сессии, gSig и файла карт. Тесты `test_method_values_follow_the_shown_session` и `test_method_values_recomputed_for_new_maps` — задача 14.
5. **Пороги заданы в конфиге мыши, а запись CNMF-E всё ещё WAITING «waiting for thresholds».** Сессия не ждёт ни в сердцебиении, ни в `status`, ни в `summary.csv`. Тест `test_mouse_config_thresholds_are_no_wait` — задача 12.

## Карта файлов

| Файл | Задачи | Что меняется |
|---|---|---|
| `tests/data/lt1_snapshot/` (новый) | 1 | 9 записей реестра LT1 и их конфиги |
| `tests/bearmind/lt1_snapshot.py`, `test_lt1_snapshot.py` (новые) | 1, 7, 8, 9, 12, 15 | копия серии, проверка «готовые этапы актуальны» |
| `docs/students/series-workflow.md` | 2, 3, 4, 5, 11, 14 | короткие абзацы по-русски |
| `bearmind/scan.py`, `scan_view.py` | 3, 7 | отпечаток Accept; предупреждение об обрезанной строке |
| `bearmind/actions.py` | 3, 9, 12, 15 | `accept_mismatch`, `recheck_mc`, реэкспорт правила, `set_bad_frames` |
| `bearmind/gui/monitor_tab.py` | 3 | текст вопроса Accept |
| `bearmind/threshold_model.py` | 4, 5, 14 | заметка «что делает батч», выбор других сессий, методы |
| `bearmind/gui/thresholds_tab.py` | 5, 14 | кнопки **From session** и трёх методов |
| `docs/superpowers/specs/2026-11-16-mc-template-corr/` (новая) | 6 | предложение порога template corr |
| `bearmind/timestamps.py` | 7, 10 | `dropped_row`; `session_time_s` переезжает сюда |
| `bearmind/runner.py` | 8, 11, 12, 15 | параметры MC, порядок побочных этапов, счётчики, `bad_frames` |
| `bearmind/stages/mc.py`, `bearmind/checks.py`, `bearmind/defaults.yaml` | 9, 15, 16 | сдвиг без чёрных кадров; полосы; порог corr |
| `bearmind/cli.py` | 3, 9, 12, 15 | `recheck`, `bad-frames`, ячейки `status` |
| `ae_launch.py`, `bearmind/stages/autoinspect.py`, `bearmind/inspect_server.py` | 10 | ось времени автоинспекции |
| `bearmind/seed_suggest.py` (новый), `bm_qc.py` | 13 | подсказка порогов без CaImAn |
| `bearmind/bad_frames.py` (новый), `bearmind/video.py`, `bearmind/monitor.py` | 15 | замена кадров, детектор полос |
| `docs/superpowers/specs/2026-11-01-operator-notes.md` | 9, 11, 15 | `recheck`, порядок превью, `bad-frames` |

## Задачи, перенос и гейты

| Задача | Пункт | Перенос | Гейт |
|---|---|---|---|
| 1. Снимок реестра LT1 | глобальное ограничение | без остановки | — |
| 2. VirtualDub в руководстве | 7 | без остановки | — |
| 3. Accept привязан к расхождению | 22б | без остановки | — |
| 4. Thresholds: что делает батч | 4 | без остановки | — |
| 5. Thresholds: **From session** | 27 | без остановки | — |
| 6. Предложение порога template corr | 22а (анализ) | без остановки | ответ автора нужен для 16 |
| 7. Обрезанная строка таймстемпов | 16, 24 | остановка | рескан — только при варианте A |
| 8. fps вне хэша MC | 18 | остановка | — |
| 9. Сдвиг MC без чёрных кадров, `recheck` | 17 | остановка | — |
| 10. Ось `est.time` | 22в | остановка | — |
| 11. Превью в конце очереди | 3 | остановка | — |
| 12. Одно правило «ждёт порогов» | 8 | остановка | — |
| 13. Подсказка порогов без CaImAn | 28 (подготовка) | остановка | — |
| 14. Кнопки методов | 28 | остановка (идёт с 13) | — |
| 15. Плохие кадры и детектор полос | 25 | остановка | «да» автора |
| 16. Новый порог template corr | 22а (код) | остановка | ответ автора на задачу 6 |

Задачи 15 и 16 стоят последними. Если ответа автора к остановке нет, остановка переносит задачи 7–14, а 15 и 16 ждут следующей.

---

### Task 1: Снимок реестра LT1 и тест «готовые этапы не устарели»

**Files:**
- Create: `tests/data/lt1_snapshot/registry/<SID>.json`, `tests/data/lt1_snapshot/config/series.yaml`, `config/mice/<MOUSE>.yaml`, `config/sessions/<SID>.yaml`
- Create: `tests/bearmind/lt1_snapshot.py` (помощник, не тест), `tests/bearmind/test_lt1_snapshot.py`

**Interfaces:**
- Produces (в `tests/bearmind/lt1_snapshot.py`):
  - `SNAPSHOT: Path` — `tests/data/lt1_snapshot`.
  - `SNAPSHOT_SIDS = ("LT1_J01_1D", "LT1_J03_2D", "LT1_J05_1D", "LT1_J06_1D", "LT1_J10_1D", "LT1_J10_2D", "LT1_J17_2D", "LT1_J24_3D", "LT1_J52_1D")`.
  - `MACHINE = {"temp_dir": "C:/bearmind_test_temp", "n_cpus": 32, "ram_gb": 191.2, "cnmf_ram_budget_gb": 152.96}` — значения машины батча (`~/.bearmind/machine.yaml`, 16.11). Бюджет выбирает `ssub_B`, а это вход хэша CNMF-E.
  - `copy_series(bearmind_dir: Path, root: Path) -> SeriesPaths` — копирует `<bearmind_dir>/registry/*.json` и всё дерево `<bearmind_dir>/config` в `root/_bearmind/`, создаёт `root/1_Raw`, пишет `MACHINE` в `config.machine_file()` (папка `BEARMIND_HOME` теста). Для каждого основного этапа DONE создаёт пустой файл на месте каждого обязательного выхода (`runner.REQUIRED`, путь из `outputs`): тогда `StageView.current` и правило `waits_for_thresholds` судят по хэшам, как в настоящей серии. Источник только читается.
  - `stale_finished(paths) -> list[tuple[str, str]]` — `(sid, stage)` каждого основного этапа в состоянии DONE или RUNNING, чей сохранённый `params_hash` не равен `runner.StageView(paths).hash(rec, stage)`; по возрастанию.
  - `live_bearmind_dir() -> Path | None` — `Path(os.environ["BEARMIND_LT1_SERIES"]) / "_bearmind"`, если переменная задана.

Сессии снимка: J01_1D (часовая, CNMF-E готов), J03_2D (10 мин, CNMF-E готов), J05_1D (обрезанная строка), J06_1D (CNMF-E идёт или готов), J10_1D (сосед J10_2D), J10_2D (полосы), J17_2D и J24_3D (чёрный кадр 0, ошибка «max shift»), J52_1D (ждёт порогов, порогов нет). Проверено 16.11 на копии всех 84 записей: код 8e82c3d воспроизводит все 301 хэш этапов DONE и RUNNING. Расходятся только 43 устаревшие записи CNMF-E «waiting for thresholds», где пороги уже выбраны (это п. 8, задача 12); тест их не сравнивает.

- [ ] **Step 1: Скопируйте снимок.** Только чтение из H:. Файл, который батч сейчас заменяет, может отказать: повторите.

```python
import shutil
from pathlib import Path
from bearmind.naming import mouse_id
SRC = Path(r"H:\Projects\LT1\CalciumData\_bearmind")
DST = Path("tests/data/lt1_snapshot")
SIDS = ["LT1_J01_1D", "LT1_J03_2D", "LT1_J05_1D", "LT1_J06_1D", "LT1_J10_1D", "LT1_J10_2D", "LT1_J17_2D",
        "LT1_J24_3D", "LT1_J52_1D"]
for rel in [f"registry/{s}.json" for s in SIDS] + [f"config/sessions/{s}.yaml" for s in SIDS] \
        + [f"config/mice/{m}.yaml" for m in sorted({mouse_id(s) for s in SIDS})] + ["config/series.yaml"]:
    if (SRC / rel).is_file():
        (DST / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SRC / rel, DST / rel)
```

- [ ] **Step 2: Тесты** в `tests/bearmind/test_lt1_snapshot.py`. Докстринг модуля: источник (`H:\Projects\LT1\CalciumData\_bearmind`), дата копии, зачем тест (глобальное ограничение плана).

```python
def test_snapshot_finished_stages_stay_current(tmp_path):
    paths = copy_series(SNAPSHOT, tmp_path / "lt1")
    recs, broken = registry.read_all(paths)
    assert broken == [] and sorted(r.sid for r in recs) == sorted(SNAPSHOT_SIDS)
    assert sum(r.stages[s].state is State.DONE for r in recs for s in STAGES) >= 30  # not vacuous
    assert stale_finished(paths) == []
    view = runner.StageView(paths)
    assert all(view.current(r, "mc") for r in recs)  # the placeholder outputs make "current" judge by hashes

def test_snapshot_check_sees_a_changed_input(tmp_path):
    paths = copy_series(SNAPSHOT, tmp_path / "lt1")
    config.update_yaml(paths.mouse_config("LT1_J03"), {"mc": {"max_shifts": [30, 30]}})
    assert ("LT1_J03_2D", "mc") in stale_finished(paths)

@pytest.mark.realdata
def test_live_series_finished_stages_stay_current(tmp_path):
    src = live_bearmind_dir()
    if src is None:
        pytest.skip("BEARMIND_LT1_SERIES is not set")
    assert stale_finished(copy_series(src, tmp_path / "lt1")) == []
```

- [ ] **Step 3: Запустите.** `conda run -n bearmind python -m pytest -q tests/bearmind/test_lt1_snapshot.py` → 2 passed, 1 skipped. Затем с `BEARMIND_LT1_SERIES` и `-m realdata` → 1 passed. Если хэш хоть одного этапа не совпал, остановитесь и доложите: снимок или значения машины не те, что у батча.
- [ ] **Step 4: Быстрый набор** → выбрано 1282 (+2), 0 failed.
- [ ] **Step 5: Коммит.** `git add -- tests/data/lt1_snapshot tests/bearmind/lt1_snapshot.py tests/bearmind/test_lt1_snapshot.py`; `git commit -q -m "Tests: LT1 registry snapshot; finished stages must keep their hashes" -- tests/data/lt1_snapshot tests/bearmind/lt1_snapshot.py tests/bearmind/test_lt1_snapshot.py`

---

### Task 2: VirtualDub 1.10 не открывает видео MC (п. 7)

**Files:**
- Modify: `docs/students/series-workflow.md` — после абзаца «**Не удаляйте** видео MC…» (раздел 1, около строки 85)

- [ ] **Step 1: Добавьте абзац** (смысл точно этот; форма — STE, фразы шагов до 20 слов):

> Видео `3_MC/<SID>_MC.avi` записано кодеком FFV1 без потерь. VirtualDub 1.10 такие файлы не открывает. Смотрите видео на вкладке **Thresholds** (**MC video**), в VirtualDub2 или в ffplay из пакета FFmpeg.

Имена VirtualDub2 и ffplay пишите обычным текстом: тест руководства требует, чтобы каждый текст в обратных кавычках был текстом программы. Пример команды можно дать только в блоке ```` ``` ````: `ffplay "<СЕРИЯ>\3_MC\<SID>_MC.avi"`.

- [ ] **Step 2: Запустите** `conda run -n bearmind python -m pytest -q tests/test_student_guide.py` → PASS.
- [ ] **Step 3: Коммит.** `git commit -q -m "Students guide: VirtualDub 1.10 cannot open the FFV1 MC video; VirtualDub2 or ffplay can" -- docs/students/series-workflow.md`

---

### Task 3: Accept привязан к расхождению (п. 22б)

**Files:**
- Modify: `bearmind/scan.py` (`evaluate_excluded`, новые `mismatch_fingerprint`, `mismatch_accepted`), `bearmind/actions.py:415-425` (`accept_mismatch`), `bearmind/scan_view.py:226`, `bearmind/gui/monitor_tab.py:524-525` (текст вопроса), `docs/students/series-workflow.md` (раздел «Принять несовпадение кадров»)
- Test: `tests/bearmind/test_scan.py`, `tests/bearmind/test_actions.py`, `tests/bearmind/gui/test_actions_ui.py:352`

**Interfaces:**
- Produces:
  - `scan.mismatch_fingerprint(rec) -> str` — то, о чём решение Accept: все `frames_ts_mismatch` скана (любой уровень, без `data["stage"]`) как `f"{a.data.get('where', 'end')}:{a.data['diff']:+d}"`, по возрастанию, через `|`; `""`, если их нет. Уровень не участвует: `evaluate_excluded` понижает принятые ошибки до warning на месте, и второй вызов должен видеть тот же отпечаток.
  - `scan.mismatch_accepted(rec) -> bool` — отпечаток не пуст и равен значению последнего решения `accept_mismatch`.
  - `actions.accept_mismatch(paths, sids, who) -> ActionResult` — та же сигнатура и те же предупреждения; значение решения — `scan.mismatch_fingerprint(rec)` свежей записи внутри `update_record` (по образцу `confirm_stitch`), затем `rec.excluded = scan.evaluate_excluded(rec)`.

- [ ] **Step 1: Тесты.**

```python
# tests/bearmind/test_scan.py
def _mismatch_rec(diff, where="end"):
    return SessionRecord(sid="NOF_H01_1D", mouse="NOF_H01", mode="bonsai", parts=[], n_frames=30, fps=20.0,
                         fps_info={}, decisions=[], excluded=False, stages={},
                         anomalies=[Anomaly("frames_ts_mismatch", "error", "x", data={"diff": diff, "where": where})])

def test_accept_covers_only_the_mismatch_it_saw():
    rec = _mismatch_rec(107)
    assert scan.mismatch_fingerprint(rec) == "end:+107"
    rec.decisions.append(Decision("accept_mismatch", "end:+107", "u", "t"))
    assert scan.evaluate_excluded(rec) is False
    assert scan.evaluate_excluded(rec) is False  # levels were lowered by the first call: still accepted
    rec.anomalies = [Anomaly("frames_ts_mismatch", "error", "x", data={"diff": 200, "where": "end"})]
    assert scan.evaluate_excluded(rec) is True  # files replaced, another mismatch: not covered

def test_accept_written_as_true_does_not_count():
    rec = _mismatch_rec(107)
    rec.decisions.append(Decision("accept_mismatch", "true", "u", "t"))
    assert scan.evaluate_excluded(rec) is True

def test_accept_survives_a_rescan_of_unchanged_files(make_series, tmp_path):
    root = make_series(tmp_path / "s", ts_rows=25)  # 30 frames, 25 rows: 5 / 25 = 20 %, an error
    paths = SeriesPaths(root)
    rec = scan.scan_series(root)[0]
    assert rec.excluded
    actions.accept_mismatch(paths, [rec.sid], "u")
    assert load_record(paths, rec.sid).decisions[-1].value == "end:+5"
    assert not load_record(paths, rec.sid).excluded
    assert not scan.scan_series(root)[0].excluded
```

В `tests/bearmind/gui/test_actions_ui.py:352` замените ожидание `"the session is used as it is"` новым текстом (Step 3).

- [ ] **Step 2: Запустите** `conda run -n bearmind python -m pytest -q tests/bearmind/test_scan.py tests/bearmind/test_actions.py tests/bearmind/gui/test_actions_ui.py` → FAIL (`mismatch_fingerprint` нет; решение пишет `true`).
- [ ] **Step 3: Реализуйте.**
  - `evaluate_excluded`: условие `latest_decision(rec, "accept_mismatch") == "true"` → `mismatch_accepted(rec)`. Докстринг: решение покрывает только расхождение, которое человек видел.
  - `scan_view.py:226`: `accepted = a.code == "frames_ts_mismatch" and scan.mismatch_accepted(rec)`.
  - Monitor, вопрос `accept_mismatch`: `f"The frame/timestamp mismatch of {who} will be accepted: the extra frames or timestamp rows at the end are dropped; the files stay unchanged."`
  - Руководство, раздел «Принять несовпадение кадров», после первого абзаца: «Решение относится только к этому несовпадению: к его месту и разнице. Если после замены файлов разница станет другой, программа снова исключит сессию. Тогда примите несовпадение ещё раз.»
- [ ] **Step 4: Запустите** те же тесты и `tests/test_student_guide.py` → PASS; быстрый набор → 0 failed.
- [ ] **Step 5: Коммит.** `git commit -q -m "Scan: Accept covers only the mismatch it was given (place and size)" -- bearmind/scan.py bearmind/actions.py bearmind/scan_view.py bearmind/gui/monitor_tab.py docs/students/series-workflow.md tests/bearmind/test_scan.py tests/bearmind/test_actions.py tests/bearmind/gui/test_actions_ui.py`

---

### Task 4: Вкладка Thresholds говорит, что делает батч и сколько сессий впереди (п. 4)

**Files:**
- Modify: `bearmind/threshold_model.py` (`_item`, `threshold_items`; новые `sessions_ahead`, `queue_text`, `batch_activity`), `docs/students/series-workflow.md` (раздел 4)
- Test: `tests/bearmind/test_threshold_model.py`

**Interfaces:**
- Consumes: `monitor.runner_state(paths) -> monitor.RunnerState` (поля `alive`, `status`, `current: (sid, stage) | None`, `stopping`), `monitor.STAGE_TITLE`.
- Produces:
  - `tm.sessions_ahead(view, records, rec) -> int` — Р10. Считаются другие сессии, которые не исключены и могут быть запланированы (`view.problem(r) is None`): (а) каждая, у которой кроп или MC не актуален и не заблокирован (`view.blocked`); (б) каждая с именем меньше `rec.sid`, у которой карты не актуальны и не заблокированы.
  - `tm.queue_text(ahead: int) -> str`: `"The batch makes these maps next."` при 0; `"1 session comes first in the batch."`; `f"{n} sessions come first in the batch."`.
  - `tm.batch_activity(rs) -> str`: живой батч с `current` → `f"Now: {title} of {sid}."`, где `title` — `monitor.STAGE_TITLE[stage]`, а для `thresholds` — `"maps"`; при `rs.stopping` в конце `" The batch stops after it."`; живой батч без `current` → `"Now: the batch waits for a person."`; не живой → `"The batch is not running: press Start on the Monitor tab."`
  - `tm.threshold_items(paths, view=None, records=None, runner_state=None)` — `None`: `monitor.runner_state(paths)`.
  - Состояние `no_maps` с причиной `not computed yet`: `reason` прежний; `note = f"No maps yet. {queue_text(n)} {batch_activity(rs)}"`; `text = "no maps: next"` при 0, иначе `f"no maps: {n} ahead"`. Состояние `out_of_date`, которое батч пересчитает сам (`_stale` вернул `True`): к прежней фразе дописываются `queue_text(n)` и `batch_activity(rs)` через пробел. Остальные состояния не меняются.

Порядок без задачи 11: при включённых превью батч делает их раньше карт, и число будет меньше настоящего. В LT1 превью выключены (`series.yaml`).

- [ ] **Step 1: Тесты** (фикстура `series` из `tests/bearmind/test_cli.py`, помощник `make_current`).

```python
def test_sessions_ahead_follows_the_batch_order(series):
    p = series.paths
    _set(p, S1, "thresholds", State.PENDING)
    make_current(p, S1, "crop", "mc")          # S1: its maps next
    make_current(p, S2, "crop")                # S2: MC first, then maps
    make_current(p, S3, "crop", "mc", "thresholds")
    recs = registry.read_all(p)[0]
    by = {r.sid: r for r in recs}
    assert tm.sessions_ahead(runner.StageView(p), recs, by[S1]) == 1
    # the same number from the batch itself: the stages next_ready offers before S1's maps
    before = set()
    while True:
        recs = registry.read_all(p)[0]
        job = runner.next_ready(recs, p, until="thresholds")
        if job == (S1, "thresholds"):
            break
        before.add(job[0])
        make_current(p, job[0], job[1])
    assert len(before - {S1}) == 1

def test_no_maps_note_says_what_the_batch_does(series):
    p = series.paths
    _set(p, S1, "thresholds", State.PENDING)
    make_current(p, S1, "crop", "mc")
    make_current(p, S2, "crop")
    make_current(p, S3, "crop", "mc", "thresholds")
    rs = monitor.RunnerState(True, "running", 1, 1.0, (S2, "mc"), 5.0)
    it = {i.sid: i for i in tm.threshold_items(p, runner_state=rs)}[S1]
    assert it.reason == "not computed yet"
    assert it.note == "No maps yet. 1 session comes first in the batch. Now: MC of NOF_H01_2D."
    assert it.text == "no maps: 1 ahead"

def test_no_maps_note_when_the_batch_is_not_running(series):
    p = series.paths
    _set(p, S1, "thresholds", State.PENDING)
    rs = monitor.RunnerState(False, "stopped", None, None, None, None)
    note = {i.sid: i for i in tm.threshold_items(p, runner_state=rs)}[S1].note
    assert note.endswith("The batch is not running: press Start on the Monitor tab.")
```

`make_current` создаёт пустые выходы только для этапов, которые отмечает: у S1 этап карт не отмечен, файла карт нет, и `_item` даёт `no_maps`.

- [ ] **Step 2: Запустите** `conda run -n bearmind python -m pytest -q tests/bearmind/test_threshold_model.py` → FAIL (`sessions_ahead` нет).
- [ ] **Step 3: Реализуйте** функции из Interfaces в `threshold_model.py`. `threshold_items` считает `sessions_ahead` только для элементов `no_maps` и `out_of_date` (остальным число не нужно) и передаёт число и `rs` в `_item`. `thresholds_tab.py` не меняется: он уже зовёт `threshold_items` на каждом обновлении, а заметка входит в `_maps_sig`, поэтому текст на месте карт обновляется сам.
- [ ] **Step 4: Руководство,** раздел 4, после абзаца про неактивную **Accept → next**: «Если карт ещё нет, на месте карт написано, сколько сессий батч сделает раньше и чем он занят сейчас.»
- [ ] **Step 5: Запустите** тесты модели, `tests/bearmind/gui/test_thresholds_tab.py`, `tests/test_student_guide.py` → PASS; быстрый набор → 0 failed.
- [ ] **Step 6: Коммит.** `git commit -q -m "Thresholds tab: a session without maps says what the batch does and how many sessions come first" -- bearmind/threshold_model.py docs/students/series-workflow.md tests/bearmind/test_threshold_model.py`

---

### Task 5: Кнопка **From session**: пороги другой сессии этой мыши (п. 27)

**Files:**
- Modify: `bearmind/threshold_model.py` (новые `mouse_choices`, `fit_gsig`; `missing_gsig_note` через `fit_gsig`; `source_text`), `bearmind/gui/thresholds_tab.py` (кнопка и меню, `apply_from_session`), `docs/students/series-workflow.md` (раздел 4)
- Test: `tests/bearmind/test_threshold_model.py`, `tests/bearmind/gui/test_thresholds_tab.py`

**Interfaces:**
- Produces:
  - `tm.mouse_choices(items: list[ThresholdItem], sid: str) -> list[tuple[str, dict]]` — `(other_sid, chosen)` других сессий той же мыши с выбранными порогами (`item.chosen` не `None`), по имени.
  - `tm.fit_gsig(maps: ThresholdMaps, g: int) -> tuple[int, str]` — `(g, "")`, если у `g` есть карты, иначе `(config.fallback_gsig(maps.usable_gsigs), f"gSig {g} has no map; showing {fb}")`. `missing_gsig_note` берёт текст отсюда (один текст на оба места).
  - `tm.source_text(source, item, chosen_from="", values=None)` — новый вид источника `f"session:{sid}"`: `f"Thresholds of {sid}: gSig {g}, min_corr {value_text(c, CORR_DECIMALS)}, min_pnr {value_text(p, PNR_DECIMALS)}"` по `values`. Прежние виды не меняются.
  - Вкладка: `from_session_button: QToolButton` с текстом `"From session"` и меню (`InstantPopup`); пункт меню — `f"{sid}: gSig {g}, min_corr {…}, min_pnr {…}"`. `apply_from_session(other_sid: str) -> None` делает то же, что `_reset`: `_apply(g, c, p)` с `g` из `fit_gsig`, затем `self._source, self._note = f"session:{other_sid}", note` и `_show_source()`. Ничего не пишет. Кнопка в ряду с **Reset to suggestion**; активна, когда карты показаны и список не пуст; иначе подсказка `f"No other session of {mouse} has thresholds"`. Меню собирается в `_update_buttons` из `self._items`.

- [ ] **Step 1: Тесты** (фикстуры `series_maps`, `make_tab`, `_yaml` из `test_cli`). S1 и S2 — мышь NOF_H01, S3 — NOF_H02.

```python
def test_from_session_takes_another_sessions_choice(qapp, series_maps, make_tab):
    p = series_maps.paths
    config.update_yaml(p.session_config(S2), {"cnmf": {"gSig": 5, "min_corr": 0.9, "min_pnr": 12.5}})
    before = p.session_config(S1).read_bytes() if p.session_config(S1).exists() else None
    tab = make_tab(p)
    tab.show_session(S1)
    assert [a.text() for a in tab.from_session_button.menu().actions()] == \
        ["NOF_H01_2D: gSig 5, min_corr 0.90, min_pnr 12.5"]
    tab.apply_from_session(S2)
    assert tab.current() == (S1, 5, 0.9, 12.5)
    assert tab.source_label.text().startswith("Thresholds of NOF_H01_2D: gSig 5")
    after = p.session_config(S1).read_bytes() if p.session_config(S1).exists() else None
    assert after == before  # nothing written until Accept

def test_from_session_lists_only_the_same_mouse(qapp, series_maps, make_tab):
    p = series_maps.paths
    config.update_yaml(p.session_config(S3), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    tab = make_tab(p)
    tab.show_session(S1)
    assert not tab.from_session_button.isEnabled()
    assert tab.from_session_button.toolTip() == "No other session of NOF_H01 has thresholds"

def test_from_session_gsig_without_map_falls_back(qapp, series_maps, make_tab):
    p = series_maps.paths  # maps of gSig 3, 4, 5
    config.update_yaml(p.session_config(S2), {"cnmf": {"gSig": 7, "min_corr": 0.9, "min_pnr": 12.5}})
    tab = make_tab(p)
    tab.show_session(S1)
    tab.apply_from_session(S2)
    assert tab.current() == (S1, 4, 0.9, 12.5)
    assert "gSig 7 has no map; showing 4" in tab.source_label.text()
```

В `test_threshold_model.py`: `mouse_choices` на трёх элементах (две сессии одной мыши, одна с `chosen`; сессия другой мыши с `chosen`) → один кортеж.

- [ ] **Step 2: Запустите** `conda run -n bearmind python -m pytest -q tests/bearmind/test_threshold_model.py tests/bearmind/gui/test_thresholds_tab.py` → FAIL.
- [ ] **Step 3: Реализуйте** по Interfaces. `self._source_values` хранит `chosen` выбранной сессии для `source_text`.
- [ ] **Step 4: Руководство,** раздел 4, после абзаца про **Reset to suggestion**: «Кнопка **From session** подставляет пороги другой сессии этой мыши, если там они уже выбраны. Пороги внутри мыши обычно похожи. Запись — только по **Accept → next**.»
- [ ] **Step 5: Запустите** тесты шага 2 и `tests/test_student_guide.py` → PASS; быстрый набор → 0 failed.
- [ ] **Step 6: Коммит.** `git commit -q -m "Thresholds tab: From session takes the thresholds of another session of the mouse" -- bearmind/threshold_model.py bearmind/gui/thresholds_tab.py docs/students/series-workflow.md tests/bearmind/test_threshold_model.py tests/bearmind/gui/test_thresholds_tab.py`

---

### Task 6: Предложение нового порога template corr (п. 22а, анализ)

Код порога не меняется: автор сначала смотрит предложение (решение 16.11, Р3). Код — задача 16.

**Files:**
- Create: `docs/superpowers/specs/2026-11-16-mc-template-corr/template_corr.py`, `template_corr_out.txt`, `template_corr.png`, `proposal.md`

- [ ] **Step 1: Скрипт** `template_corr.py`. Читает только H: и пишет только в свою папку.
  - Входы для всех 84 сессий: `_bearmind/registry/<SID>.json` (`stages.mc.metrics.mean_template_corr`), `3_MC/<SID>_MC_shifts.npy` (99-й перцентиль покадрового `max |shift|`: он не чувствует чёрный кадр 0), `_bearmind/thresholds/<SID>_maps.npz` (`mean`; яркость поля — медиана `mean` по пикселям не темнее четверти медианы, правило тёмной зоны `seed_mask.valid_mask`).
  - Кандидаты: абсолютные пороги 0.50, 0.55, 0.60, 0.65, 0.70; правило мыши «ниже медианы мыши больше чем на 0.05 / 0.10» (для мышей с ≥ 3 сессиями).
  - `template_corr_out.txt`: для каждого кандидата — сколько сессий отмечено и какие; медианы по мышам; ранговая корреляция corr с яркостью и с p99 сдвига.
  - `template_corr.png`: гистограмма corr с линиями кандидатов; corr по мышам; corr против яркости поля; corr против p99 сдвига.
- [ ] **Step 2: Запустите** `conda run -n bearmind python -I docs/superpowers/specs/2026-11-16-mc-template-corr/template_corr.py`. Проверка: при пороге 0.70 отмечена 21 сессия, минимум corr 0.605 (числа 16.11). Если иначе, объясните в `proposal.md` (реестр изменился).
- [ ] **Step 3: `proposal.md`** по-русски, в стиле STE. Сверху — рекомендация с числами и вопрос автору: какой порог (или правило) брать. Дальше таблица кандидатов, рисунок и честная оговорка: калибровка 0.7 сделана на 6 ярких сессиях (сдвиг на 2 px давал 0.57–0.75, `defaults.yaml:119-121`), а у тусклых мышей LT1 corr ниже без признаков движения. Последний абзац: после ответа — задача 16 и `recheck` при остановке; MC не пересчитывается.
- [ ] **Step 4: Коммит.** `git add -- docs/superpowers/specs/2026-11-16-mc-template-corr`; `git commit -q -m "docs: MC template corr over LT1 and the proposed threshold" -- docs/superpowers/specs/2026-11-16-mc-template-corr`
- [ ] **Step 5:** покажите `proposal.md` автору через оператора. Задача 16 ждёт ответа.

---

### Task 7: Обрезанная последняя строка таймстемпов (п. 16, 24)

**Files:**
- Modify: `bearmind/timestamps.py:15-55` (`Timestamps`, `read_timestamps`), `bearmind/scan.py` (`_read_part`), `bearmind/scan_view.py` (`ANOMALY_TITLE`)
- Test: `tests/bearmind/test_timestamps.py`, `tests/bearmind/test_inspect_server.py`, `tests/bearmind/test_scan.py`, `tests/bearmind/test_lt1_snapshot.py`

**Interfaces:**
- Produces:
  - `Timestamps.dropped_row: str | None = None` — текст отброшенной последней строки.
  - Правило Bonsai в `read_timestamps`: если строк не меньше двух, текст файла (после `rstrip(" \t")`) не кончается на `"\n"` или `"\r"`, и последнее значение не больше предыдущего, последняя строка отбрасывается и попадает в `dropped_row`. Обрезанное N-значное число всегда меньше предыдущего N-значного, поэтому условие точное; целая строка без перевода строки остаётся. Формат UCLA не меняется (отчёт J05, «Открыто», п. 3).
  - Аномалия скана `ts_last_row_incomplete`: уровень `warning`, `auto_fixed=True`, `data={"file": ts_file, "row": row, "digits": len(row), "digits_before": len(str(ts.raw[-1]))}`, текст `f"last timestamp row of {ts_file} is incomplete ({len(row)} digits instead of {k}, no line end); left out"`.
  - `scan_view.ANOMALY_TITLE["ts_last_row_incomplete"] = "last timestamp row incomplete"`.
  - Дальше работает прежний код: скан режет `n_frames` до числа строк (`scan.py:264-281`), а `session_timestamps` даёт инспекции и экспорту на строку меньше.

- [ ] **Step 1: Тесты.** В `tests/bearmind/test_timestamps.py`:

```python
BASE = 639281122689862784  # LT1_J05_1D's first tick (18 digits, 100 ns since year 1)
STEP = 506172  # 50.6 ms, the session's median interval

def _ticks(n):
    return BASE + np.arange(n, dtype=np.int64) * STEP

def _write_cut(path, n):
    """n rows as Bonsai writes them (CRLF), the last cut to its first 16 digits and without a line end, as in
    LT1_J05_1D_Mini_TS.csv."""
    t = _ticks(n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("".join(f"{v}\r\n" for v in t[:-1]) + str(t[-1])[:16]).encode("ascii"))
    return t

def test_bonsai_incomplete_last_row_left_out(tmp_path):
    t = _write_cut(tmp_path / "S_Mini_TS.csv", 50)
    ts = read_timestamps(tmp_path / "S_Mini_TS.csv")
    assert ts.raw.tolist() == t[:49].tolist() and np.all(np.diff(ts.t_ms) > 0)
    assert ts.dropped_row == str(t[-1])[:16]

def test_bonsai_complete_last_row_without_line_end_kept(tmp_path):
    t = _ticks(50)
    (tmp_path / "S.csv").write_bytes(("".join(f"{v}\r\n" for v in t[:-1]) + str(t[-1])).encode("ascii"))
    ts = read_timestamps(tmp_path / "S.csv")
    assert ts.raw.tolist() == t.tolist() and ts.dropped_row is None

def test_bonsai_last_row_with_lone_cr_kept(tmp_path):
    t = _ticks(50)
    (tmp_path / "S.csv").write_bytes(("".join(f"{v}\r\n" for v in t[:-1]) + f"{t[-1]}\r").encode("ascii"))
    assert read_timestamps(tmp_path / "S.csv").raw.tolist() == t.tolist()

@pytest.mark.parametrize("end", ["\n", "\r\n"])
def test_complete_files_read_as_before(tmp_path, end):
    t = _ticks(50)
    (tmp_path / "S.csv").write_bytes("".join(f"{v}{end}" for v in t).encode("ascii"))
    ts = read_timestamps(tmp_path / "S.csv")
    assert ts.raw.tolist() == t.tolist() and ts.dropped_row is None
```

В `tests/bearmind/test_inspect_server.py`, рядом с `test_session_time_s` (импортировать `_write_cut` из `test_timestamps`):

```python
def test_cut_row_stays_out_of_inspection_and_export(make_series, tmp_path):
    """The scan, the inspection time axis and the rows the export writes (session_timestamps) all leave the
    incomplete row out."""
    # 60 frames: one row short is 1.7 %, below max_mismatch_frac 0.02, so the session stays included
    root = make_series(tmp_path / "s", chunk_frames=20)  # bonsai: 3 chunks of 20 frames
    _write_cut(root / "0_TimeStamps" / "NOF_H01_1D_Mini_TS.csv", 60)
    rec = scan.scan_series(root)[0]
    paths = SeriesPaths(root)
    assert rec.n_frames == 59  # frame 60 has no valid timestamp
    t = inspect_server.session_time_s(paths, rec)
    assert len(t) == 59 and np.all(np.diff(t) > 0) and t[-1] < 3.0  # 58 intervals of 50.6 ms
    raw = session_timestamps(paths, rec).raw
    assert len(raw) == 59 and np.all(np.diff(raw) > 0)
    a = {x.code: x for x in rec.anomalies}
    assert a["ts_last_row_incomplete"].level == "warning" and a["ts_last_row_incomplete"].auto_fixed
    assert a["ts_last_row_incomplete"].data["digits"] == 16
    assert a["frames_ts_mismatch"].level == "warning" and a["frames_ts_mismatch"].data["diff"] == 1
    assert not rec.excluded
```

В `tests/bearmind/test_lt1_snapshot.py` (только чтение H:):

```python
@pytest.mark.realdata
def test_live_timestamp_files_lose_a_row_only_in_j05_1d():
    src = live_bearmind_dir()
    if src is None:
        pytest.skip("BEARMIND_LT1_SERIES is not set")
    paths = SeriesPaths(src.parent)
    recs, _ = registry.read_all(paths)
    dropped = []
    for r in recs:
        for p in r.parts:
            f = paths.root / p.timestamps_file
            ts = read_timestamps(f)
            rows = [ln for ln in f.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
            assert len(ts.raw) + (ts.dropped_row is not None) == len(rows), r.sid
            if ts.dropped_row is not None:
                dropped.append(r.sid)
    assert len(recs) == 84 and dropped == ["LT1_J05_1D"]
```

Последний тест доказывает: рескан не меняет `n_frames`, а значит и хэши, ни у одной сессии LT1, кроме J05_1D.

- [ ] **Step 2: Запустите** `conda run -n bearmind python -m pytest -q tests/bearmind/test_timestamps.py tests/bearmind/test_inspect_server.py tests/bearmind/test_scan.py` → FAIL: 2 теста J05 и тесты `dropped_row` (охранные тесты целой строки проходят и до правки).
- [ ] **Step 3: Реализуйте** по Interfaces. Комментарий у правила: почему строка обрезана (не дописался последний буфер файла при остановке записи; у LT1_J05_1D размер файла ровно 231 × 4096 байт) и почему её не восстанавливают (это были бы выдуманные данные в экспорте).
- [ ] **Step 4: Запустите** тесты шага 2, `tests/bearmind/test_scan_gaps.py`, `tests/bearmind/test_scan_view.py`, `tests/bearmind/test_lt1_snapshot.py` → PASS. С `BEARMIND_LT1_SERIES`: `-m realdata tests/bearmind/test_lt1_snapshot.py` → PASS. Быстрый набор → 0 failed.
- [ ] **Step 5: Коммит.** `git commit -q -m "Timestamps: a Bonsai last row cut while being written is left out; the scan says so" -- bearmind/timestamps.py bearmind/scan.py bearmind/scan_view.py tests/bearmind/test_timestamps.py tests/bearmind/test_inspect_server.py tests/bearmind/test_lt1_snapshot.py`

Гейт: рескан серии после переноса — только если автор выбрал A (раздел «Перенос», шаг 6).

---

### Task 8: fps вне хэша MC (п. 18)

**Files:**
- Modify: `bearmind/runner.py:257-259` (`stage_params("mc")`), `bearmind/runner.py:302-309` (`_chain`; новый `_mc_legacy`)
- Test: `tests/bearmind/test_runner.py`, `tests/bearmind/test_lt1_snapshot.py`

**Interfaces:**
- Produces:
  - `stage_params("mc", rec, cfg, paths) -> {"mc": cfg["mc"], "box": c.get("box"), "box_pos": c.get("box_pos", _UNSET)}` — без `fps`.
  - `runner._mc_legacy(rec, params: dict, upstream: list[str], h: str) -> tuple[dict, str]` — Р2. Есть сохранённая запись MC с ключом `fps` в параметрах, и `config.params_hash({"params": {**params, "fps": old.params["fps"]}, "upstream": upstream}) == old.params_hash` → вернуть эти параметры и этот хэш. Иначе `(params, h)`. `_chain` вызывает его для этапа `mc`. Совпадение хэша — единственное условие: любая другая разница уже его ломает.
  - Комментарий: MC берёт fps только для заголовка AVI; запись, чей хэш построен с её собственным fps, хранит его, пока он совпадает, поэтому готовые MC и всё после них остаются актуальными.

- [ ] **Step 1: Тесты.**

```python
# tests/bearmind/test_lt1_snapshot.py
def test_fps_change_keeps_mc_and_maps_current(tmp_path):
    paths = copy_series(SNAPSHOT, tmp_path / "lt1")
    update_record(paths, "LT1_J03_2D", lambda r: setattr(r, "fps", r.fps + 0.01))
    rec = load_record(paths, "LT1_J03_2D")
    view = runner.StageView(paths)
    assert view.current(rec, "mc") and view.current(rec, "thresholds")
    assert not view.current(rec, "cnmf")  # fr is an input of CNMF-E

def test_legacy_mc_hash_needs_its_own_fps(tmp_path):
    paths = copy_series(SNAPSHOT, tmp_path / "lt1")
    def other_fps(r):
        r.stages["mc"].params["fps"] = 12.5
    update_record(paths, "LT1_J03_2D", other_fps)
    assert ("LT1_J03_2D", "mc") in stale_finished(paths)

# tests/bearmind/test_runner.py
def test_mc_params_have_no_fps(series):
    _run(series)
    rec = load_record(series.paths, S1)
    assert set(rec.stages["mc"].params) == {"mc", "box", "box_pos"}
```

Тест-снимок задачи 1 — главный тест задачи: все 84 записи MC LT1 хранят `fps`, и все должны остаться актуальными.

- [ ] **Step 2: Запустите** `conda run -n bearmind python -m pytest -q tests/bearmind/test_lt1_snapshot.py tests/bearmind/test_runner.py` → FAIL (`test_mc_params_have_no_fps`, `test_fps_change_keeps_mc_and_maps_current`).
- [ ] **Step 3: Реализуйте** по Interfaces.
- [ ] **Step 4: Запустите** тесты шага 2 → PASS; `-m realdata tests/bearmind/test_lt1_snapshot.py` с `BEARMIND_LT1_SERIES` → PASS; быстрый набор → 0 failed. `test_stage_hash_inputs` проходит без правок: в нём новые записи.
- [ ] **Step 5: Коммит.** `git commit -q -m "Runner: fps is not an input of MC; records hashed with their own fps stay current" -- bearmind/runner.py tests/bearmind/test_runner.py tests/bearmind/test_lt1_snapshot.py`

---

### Task 9: Проверка «max shift» без чёрных кадров и команда `recheck` (п. 17)

**Files:**
- Modify: `bearmind/stages/mc.py` (новые `max_shift_without_black`, `shift_metrics`; `run`, строки 175-180), `bearmind/checks.py` (текст `mc_max_shift_px`), `bearmind/defaults.yaml` (`checks.mc_black_frame_mean: 0.5` с комментарием из Р3), `bearmind/actions.py` (`RecheckRow`, `recheck_mc`), `bearmind/cli.py` (`recheck`), `docs/superpowers/specs/2026-11-01-operator-notes.md`
- Create: `tests/bearmind/test_mc_black_frames.py`
- Test: `tests/bearmind/test_checks.py`, `tests/bearmind/test_cli.py`, `tests/bearmind/test_config.py`, `tests/bearmind/test_lt1_snapshot.py`, `tests/bearmind/test_stage_mc.py` (если там закреплены ключи метрик)

**Interfaces:**
- Produces:
  - `mc.max_shift_without_black(shifts: np.ndarray, frame_at: Callable[[int], np.ndarray], black_mean: float) -> tuple[float | None, list[int]]` — `(наибольший max |сдвиг| по нечёрным кадрам, оставленные чёрные кадры по возрастанию)`. Кадры по убыванию сдвига (`np.argsort(-mag, kind="stable")`, `mag = np.abs(shifts).max(axis=1)`); кадр со средним `< black_mean` пропускается; первый нечёрный даёт ответ. Все кадры чёрные → `(None, все)`.
  - `mc.shift_metrics(shifts, video, black_mean: float) -> dict` → `{"max_shift_px": float | None, "max_shift_all_px": float, "black_frames_left_out": list[int]}`; `video[i]` — кадр (`VideoArray`). `run` кладёт эти три ключа в метрики вместо прежнего `max_shift_px`; `black_mean` — `cfg["checks"]["mc_black_frame_mean"]`. Кадр судится по видео MC: сдвиг с `border_nan: copy` оставляет нулевой кадр нулевым.
  - `checks.evaluate("mc", …)`: сообщение `mc_max_shift_px` — `"max shift {v} px"`, а при непустом `black_frames_left_out` с добавкой `f", {n} black frame(s) left out"`, затем пороги, как сейчас. Шаблон `"max shift {v} px"` остаётся буквально: его цитирует руководство.
  - `actions.RecheckRow(sid: str, before: str, after: str, note: str)` (dataclass; `before`/`after` — худший уровень проверок MC; `note` — почему пропущена).
  - `actions.recheck_mc(paths, sids=None) -> list[RecheckRow]` — для каждой сессии с MC в состоянии DONE (исключённой тоже: её проверки видны в окне): читает `paths.mc_shifts(sid)` и `VideoArray(paths.mc_video(sid))`, считает `shift_metrics`, затем `checks.evaluate("mc", metrics, [], cfg)` с конфигом сессии. Пишет в запись MC только `metrics` и `checks`, через `registry.update_record`, и только если `params_hash` и `finished` записи не изменились с чтения. Нет файла сдвигов или видео → строка с `note="MC outputs missing"`, запись не трогается. Импорт `bearmind.stages.mc` — внутри функции.
  - CLI: `recheck SERIES [--sessions SID ...]`, справка `"re-evaluate the checks of finished MC stages from their outputs (MC does not run again)"`. Вывод: строка на каждую сессию, где уровень изменился (`f"[OK] {sid}: {before} -> {after}"`), пропущенные — `f"[WARN] {sid}: {note}"`, в конце `f"[OK] MC checks re-evaluated for {n} session(s)"`.

- [ ] **Step 1: Тесты** в `tests/bearmind/test_mc_black_frames.py`:

```python
def _frames(means, shape=(4, 4)):
    return [np.full(shape, m, np.uint8) for m in means]

def test_black_frame_with_the_largest_shift_is_left_out():
    shifts = np.array([[35.7, 0], [2.0, 1.0], [0.5, 6.7], [0, 0]], np.float32)
    v, left = mc.max_shift_without_black(shifts, _frames([0, 40, 40, 40]).__getitem__, 0.5)
    assert v == pytest.approx(6.7) and left == [0]

def test_dark_frame_counts():
    shifts = np.array([[35.7, 0], [2.0, 1.0]], np.float32)
    assert mc.max_shift_without_black(shifts, _frames([2, 40]).__getitem__, 0.5)[1] == []

def test_several_black_frames():
    shifts = np.array([[30, 0], [0, 33], [5, 0], [1, 0]], np.float32)
    v, left = mc.max_shift_without_black(shifts, _frames([0, 0, 40, 0]).__getitem__, 0.5)
    assert v == pytest.approx(5.0) and left == [0, 1]

def test_all_frames_black():
    shifts = np.array([[3, 0], [1, 0]], np.float32)
    assert mc.max_shift_without_black(shifts, _frames([0, 0]).__getitem__, 0.5) == (None, [0, 1])

def test_only_frames_down_to_the_answer_are_read():
    shifts = np.array([[35, 0], [9, 0], [1, 0], [2, 0]], np.float32)
    frames, read = _frames([0, 40, 40, 40]), []
    mc.max_shift_without_black(shifts, lambda i: read.append(i) or frames[i], 0.5)
    assert read == [0, 1]

def test_recheck_rewrites_only_metrics_and_checks(series, make_avi):
    """Frame 0 black with a 35.7 px shift; the rest 6.7 px at most: error -> ok; hash, params and the other
    stages untouched."""

def test_recheck_skips_missing_outputs(series):
    """No shifts file: the row says so; the record is unchanged."""
```

Тело двух последних — по фикстуре `series` из `test_cli`: MC S1 через `make_current`, затем запись настоящих `paths.mc_shifts(S1)` (`np.save`) и `paths.mc_video(S1)` (`make_avi`, кадр 0 нулевой), в записи MC — проверка уровня error. После `actions.recheck_mc`: `metrics["max_shift_px"] == pytest.approx(6.7)`, `black_frames_left_out == [0]`, уровень `ok`, а `params_hash`, `params`, `finished` и остальные этапы — как до вызова.

В `test_checks.py`:

```python
def test_max_shift_message_names_left_out_frames():
    c = by_name(evaluate("mc", {"max_shift_px": 6.7, "black_frames_left_out": [0]}, [], CFG))["mc_max_shift_px"]
    assert c.level == "ok" and c.message.startswith("max shift 6.7 px, 1 black frame left out (")
```

В `test_config.py`: `d["checks"]["mc_black_frame_mean"] == 0.5`. В `test_cli.py`: `recheck` на той же сессии печатает `[OK] NOF_H01_1D: error -> ok` и `[OK] MC checks re-evaluated for 1 session(s)`. В `test_lt1_snapshot.py` (только чтение):

```python
@pytest.mark.realdata
def test_live_black_frames(tmp_path):
    """The nine LT1 sessions whose MC error or warning came from a black frame 0 drop to at most 10 px;
    LT1_J10_2D keeps its 27.1 px (frame 64646 is a striped frame, not a black one)."""
    src = live_bearmind_dir()
    if src is None:
        pytest.skip("BEARMIND_LT1_SERIES is not set")
    paths = SeriesPaths(src.parent)
    for sid in ("LT1_J17_2D", "LT1_J18_2D", "LT1_J58_2D", "LT1_J53_2D", "LT1_J57_2D", "LT1_J30_2D",
                "LT1_J24_3D", "LT1_J54_2D", "LT1_J24_2D"):
        m = mc.shift_metrics(np.load(paths.mc_shifts(sid)), VideoArray(paths.mc_video(sid)), 0.5)
        assert m["black_frames_left_out"] == [0] and m["max_shift_px"] <= 10, sid
    m = mc.shift_metrics(np.load(paths.mc_shifts("LT1_J10_2D")), VideoArray(paths.mc_video("LT1_J10_2D")), 0.5)
    assert m["black_frames_left_out"] == [] and m["max_shift_px"] == pytest.approx(27.1, abs=0.05)
```

- [ ] **Step 2: Запустите** `conda run -n bearmind python -m pytest -q tests/bearmind/test_mc_black_frames.py tests/bearmind/test_checks.py tests/bearmind/test_config.py tests/bearmind/test_cli.py` → FAIL.
- [ ] **Step 3: Реализуйте** по Interfaces. `checks` не хэшируется ни одним этапом (`defaults.yaml:1-4`), поэтому новый ключ не трогает хэши; тест-снимок это подтверждает.
- [ ] **Step 4: Заметки оператора:** короткий раздел «Перепроверить MC без пересчёта» (по-русски, STE): когда нужен `run_bearmind recheck <СЕРИЯ>` (сменился порог проверки MC; записи MC до этой правки), что он читает, что пишет, что MC не повторяется.
- [ ] **Step 5: Запустите** тесты шага 2, `tests/bearmind/test_lt1_snapshot.py`, медленный `tests/bearmind/test_stage_mc.py -m "not realdata"` (синтетический фильм) → PASS; `-m realdata tests/bearmind/test_lt1_snapshot.py` с `BEARMIND_LT1_SERIES` → PASS; быстрый набор и `tests/test_student_guide.py` → 0 failed.
- [ ] **Step 6: Коммит.** `git add -- tests/bearmind/test_mc_black_frames.py`; `git commit -q -m "MC: the max shift check leaves out black frames; recheck rewrites finished MC checks without MC" -- bearmind/stages/mc.py bearmind/checks.py bearmind/defaults.yaml bearmind/actions.py bearmind/cli.py docs/superpowers/specs/2026-11-01-operator-notes.md tests/bearmind/test_mc_black_frames.py tests/bearmind/test_checks.py tests/bearmind/test_config.py tests/bearmind/test_cli.py tests/bearmind/test_lt1_snapshot.py tests/bearmind/test_stage_mc.py`

---

### Task 10: Ось `est.time` в processed.pickle из таймстемпов (п. 22в)

**Files:**
- Modify: `bearmind/timestamps.py` (сюда переезжает `session_time_s`), `bearmind/inspect_server.py:203-207` (импорт вместо определения), `ae_launch.py:152-200, 343` (параметр `time_s`), `bearmind/stages/autoinspect.py:48-62`
- Test: `tests/bearmind/test_stage_autoinspect.py`

**Interfaces:**
- Produces:
  - `timestamps.session_time_s(paths, rec) -> np.ndarray` — тело и докстринг из `inspect_server` без изменений. `inspect_server` делает `from bearmind.timestamps import session_time_s`, поэтому `inspect_server.session_time_s` остаётся.
  - `ae_launch.run_auto_inspection(..., time_s: np.ndarray | None = None)` — ось кадров в секундах, передаётся в `LoadEstimates(str(estimates_path), default_fps=fps, time=time_s)`. `None` — прежнее поведение (путь ноутбука). Имя `time_s`, не `time`: не затенять модуль `time`.
  - `autoinspect.run` передаёт `time_s=session_time_s(paths, rec)`. Таймстемпы не читаются → этап падает с ошибкой чтения, как упал бы экспорт.

- [ ] **Step 1: Тесты.** `_setup` в `test_stage_autoinspect.py` получает часть с файлом таймстемпов: `RawPart(folder="1_Raw/NOF_H01_1D", chunks=[], n_frames=10, shape=(4, 4), timestamps_file="0_TimeStamps/NOF_H01_1D_Mini_TS.csv", ts_format="bonsai_ticks", start_ms=None, metadata_file=None)` и файл из 10 строк с шагом 506000 тиков.

```python
def test_processed_time_axis_comes_from_the_timestamps(tmp_path, monkeypatch):
    paths, rec, cfg = _setup(tmp_path)
    seen = {}
    def fake(estimates_path, fps=None, **kw):
        seen.update(kw)
        _write_processed(paths)
        return _result(3, 0, 3, _processed_df(3, 0, 0))
    monkeypatch.setattr(ae_launch, "run_auto_inspection", fake)
    autoinspect.run(paths, rec, cfg)
    assert np.allclose(seen["time_s"], np.arange(10) * 0.0506)

def test_run_auto_inspection_gives_the_axis_to_load_estimates(tmp_path, monkeypatch):
    got = {}
    est = SimpleNamespace(idx_components=np.array([], int))
    monkeypatch.setattr(ae_launch, "LoadEstimates", lambda name, default_fps=20, time=None: got.update(time=time) or est)
    t = np.arange(5) * 0.05
    ae_launch.run_auto_inspection("x_estimates.pickle", fps=20, time_s=t, save_artifacts=False)
    assert got["time"] is t
```

Ветка «0 компонент» во втором тесте может требовать больше полей `est`; добавьте их или возьмите `_fake_inspection` из того же файла.

- [ ] **Step 2: Запустите** `conda run -n bearmind python -m pytest -q tests/bearmind/test_stage_autoinspect.py tests/bearmind/test_inspect_server.py` → FAIL.
- [ ] **Step 3: Реализуйте** по Interfaces.
- [ ] **Step 4: Запустите** тесты шага 2, `tests/test_auto_inspector.py`, `tests/bearmind/test_lt1_snapshot.py` → PASS; быстрый набор → 0 failed.
- [ ] **Step 5: Коммит.** `git commit -q -m "Autoinspect: est.time is the session's timestamp axis, not a linspace of the rounded fps" -- bearmind/timestamps.py bearmind/inspect_server.py ae_launch.py bearmind/stages/autoinspect.py tests/bearmind/test_stage_autoinspect.py`

---

### Task 11: Превью после основных этапов (п. 3)

**Files:**
- Modify: `bearmind/runner.py` (`next_ready`, `StageView.first_ready`, новый `_offer_order`, докстринг модуля), `bearmind/registry.py:18-20` (комментарий `SIDE_STAGES`), `docs/students/series-workflow.md` (абзац про порядок батча, около строки 34), `docs/superpowers/specs/2026-11-01-operator-notes.md:78-93`
- Test: `tests/bearmind/test_runner.py:1630-1710`, `tests/bearmind/test_cli.py:703-722`

**Interfaces:**
- Produces:
  - `runner._offer_order(until) -> tuple[str, ...]` — основные этапы из `_offered(until)` в порядке `STAGES`, затем побочные в порядке `ALL_STAGES`.
  - `next_ready` обходит `_offer_order(until)`: побочный этап сессии предлагается, только если ни один основной этап ни одной сессии не готов (Р7).
  - `StageView.first_ready(rec)` — первый готовый этап сессии в том же порядке.
  - `registry.ALL_STAGES` не меняется: это порядок столбцов и записей, а не очереди.

- [ ] **Step 1: Тесты.** Замените `test_previews_on_run_right_after_parent`:

```python
def test_previews_run_after_the_main_stages(series):
    s = series
    _on(s)
    res = _run(s)
    assert s.stubs.calls == [(sid, st) for st in STAGES for sid in SIDS] + _sides_only()
    assert res == {"done": 27, "failed": 0, "waiting": 0, "skipped": 0, "not_recorded": []}

def test_previews_run_while_people_are_awaited(series):
    """No session has thresholds: CNMF-E waits for a person, and the MC previews run meanwhile."""
```

Во втором тесте включите превью (`_on(s)`), снимите пороги (фикстура пишет `THRESH` в конфиги сессий: перепишите каждый файл `s.paths.session_config(sid)` пустым) и запустите `_run(s, until="cnmf")` (`once=True` по умолчанию). Проверьте: все три `(sid, "mc_preview")` есть в `calls` и стоят после всех `thresholds` и всех попыток `cnmf`; CNMF-E каждой сессии в состоянии WAITING «waiting for thresholds» (заглушка `_cnmf` бросает `WaitingForInput`). Остальные тесты превью в этом блоке проверяйте по новому порядку. В `test_status_marks_out_of_date_previews` ожидания меняются: строка S1 кончается на `"out of date: the batch redoes autoinspect first"`, а `next_ready(...) == (S1, "autoinspect")`.

- [ ] **Step 2: Запустите** `conda run -n bearmind python -m pytest -q tests/bearmind/test_runner.py tests/bearmind/test_cli.py` → FAIL в изменённых тестах.
- [ ] **Step 3: Реализуйте** по Interfaces. Докстринг модуля runner и комментарий `SIDE_STAGES`: побочный этап идёт, когда батчу нечего делать с основными этапами; основная цепочка от него не зависит.
- [ ] **Step 4: Документы.** Руководство, после «Самый ранний этап, который можно сделать, всегда идёт первым.»: «Видео для проверки (флажок **Preview videos**) батч делает в последнюю очередь, когда основных этапов делать нечего.» Заметки оператора: перечень порядка (строки 85-93) заменить этим правилом; предупреждение строки 78 о картах порогов убрать (карты больше не ждут превью).
- [ ] **Step 5: Запустите** тесты шага 2, `tests/bearmind/test_lt1_snapshot.py`, `tests/test_student_guide.py` → PASS; быстрый набор → 0 failed.
- [ ] **Step 6: Коммит.** `git commit -q -m "Runner: preview videos only when no main stage is ready" -- bearmind/runner.py bearmind/registry.py docs/students/series-workflow.md docs/superpowers/specs/2026-11-01-operator-notes.md tests/bearmind/test_runner.py tests/bearmind/test_cli.py`

---

### Task 12: Одно правило «ждёт порогов» для сердцебиения, `status` и сводки (п. 8)

**Files:**
- Modify: `bearmind/runner.py` (сюда `THRESHOLD_KEYS` и `waits_for_thresholds`; `_session_status`, `_counts`, `_Run.refresh`, `series_state`, `_summary_row`), `bearmind/actions.py:111-127` (реэкспорт), `bearmind/cli.py` (`_row`, `cmd_status`)
- Test: `tests/bearmind/test_runner.py`, `tests/bearmind/test_cli.py`, `tests/bearmind/test_lt1_snapshot.py`

**Interfaces:**
- Produces:
  - `runner.THRESHOLD_KEYS`, `runner.waits_for_thresholds(view, rec) -> bool` — перенесены из `actions` без изменений. В `actions`: `from bearmind.runner import THRESHOLD_KEYS, waits_for_thresholds  # noqa: F401` с комментарием: одно правило, runner считает по нему тоже. `actions.waits_for_thresholds is runner.waits_for_thresholds`.
  - `runner._session_status(r, scope, stages, view) -> str | None` — RUNNING как сейчас; затем `"waiting_thresholds"`, если `"cnmf" in stages`, `scope.has(r, "cnmf")` и правило верно; записанное ожидание `what == "thresholds"` без правила не считается ожиданием (сессия дальше идёт как `pending`); остальное как сейчас.
  - `runner._counts(records, scope, stages, view) -> tuple[dict, dict]`; `_Run.refresh` передаёт свой `view`.
  - `runner.series_state(paths)` — в `waiting` записанные ожидания порогов заменены на `(sid, "cnmf", "thresholds")` каждой неисключённой сессии, где правило верно. Цикл runner (`_loop.person_waits`, `_waiting`) не меняется (Р6).
  - `runner._summary_row`: столбец `cnmf` = `"waiting"` и `waiting_for` = `"thresholds"`, если правило верно; записанное ожидание порогов без правила → `"pending"` и без `waiting_for`.
  - `cli._row(rec, on=None, stale=(), waits=False)` — ячейка CNMF-E `W` и причина `thresholds`, только если `waits`; записанное ожидание порогов при `waits=False` → `.` без причины. `cmd_status` передаёт `waits=runner.waits_for_thresholds(state.view, r)`.

- [ ] **Step 1: Тесты.**

```python
# tests/bearmind/test_lt1_snapshot.py
def test_heartbeat_counts_by_the_rule(tmp_path):
    paths = copy_series(SNAPSHOT, tmp_path / "lt1")
    recs, _ = registry.read_all(paths)
    view = runner.StageView(paths)
    counts, _ = runner._counts(recs, runner._Scope(recs, None), STAGES, view)
    rule = sorted(r.sid for r in recs if runner.waits_for_thresholds(view, r))
    assert "LT1_J52_1D" in rule and "LT1_J10_2D" not in rule  # J10_2D: a stale wait, thresholds chosen
    assert counts["waiting_thresholds"] == len(rule)

@pytest.mark.realdata
def test_live_counts_match_the_thresholds_tab(tmp_path):
    src = live_bearmind_dir()
    if src is None:
        pytest.skip("BEARMIND_LT1_SERIES is not set")
    paths = copy_series(src, tmp_path / "lt1")
    recs, _ = registry.read_all(paths)
    view = runner.StageView(paths)
    counts, _ = runner._counts(recs, runner._Scope(recs, None), STAGES, view)
    waiting = [i for i in threshold_model.threshold_items(paths, view, recs) if i.state == "waiting"]
    assert counts["waiting_thresholds"] == len(waiting)
```

В `test_runner.py` / `test_cli.py` (фикстура `series`):
- `test_stale_threshold_wait_counts_as_pending` — CNMF-E WAITING «waiting for thresholds», пороги в конфиге сессии → `_session_status == "pending"`, в `status` ячейка `.`, в строке `waiting:` нет порогов.
- `test_unrecorded_threshold_wait_counts` — кроп, MC и карты актуальны, порогов нет, CNMF-E PENDING (батч ещё не пробовал) → `"waiting_thresholds"`; `status` пишет `waiting: thresholds 1`, ячейка `W`; в `summary.csv` `cnmf == "waiting"`, `waiting_for == "thresholds"`.
- `test_mouse_config_thresholds_are_no_wait` (Review Focus 5) — пороги только в конфиге мыши, запись CNMF-E WAITING «waiting for thresholds» → не ждёт нигде: счётчик, `status`, `summary.csv`.

- [ ] **Step 2: Запустите** `conda run -n bearmind python -m pytest -q tests/bearmind/test_runner.py tests/bearmind/test_cli.py tests/bearmind/test_lt1_snapshot.py` → FAIL.
- [ ] **Step 3: Реализуйте** по Interfaces. `monitor.py` и `threshold_model.py` не меняются: они зовут `actions.waits_for_thresholds`, а это теперь тот же объект.
- [ ] **Step 4: Запустите** тесты шага 2, `tests/bearmind/test_monitor.py`, `tests/bearmind/test_threshold_model.py`, `tests/bearmind/test_actions.py` → PASS; `-m realdata tests/bearmind/test_lt1_snapshot.py` с `BEARMIND_LT1_SERIES` → PASS; быстрый набор → 0 failed.
- [ ] **Step 5: Коммит.** `git commit -q -m "Runner: heartbeat, status and summary count threshold waits by actions.waits_for_thresholds" -- bearmind/runner.py bearmind/actions.py bearmind/cli.py tests/bearmind/test_runner.py tests/bearmind/test_cli.py tests/bearmind/test_lt1_snapshot.py`

---

### Task 13: Подсказка порогов в модуле без CaImAn (подготовка п. 28)

**Files:**
- Create: `bearmind/seed_suggest.py`, `tests/bearmind/test_seed_suggest.py`
- Modify: `bm_qc.py:249-418` (функции уходят), `bm_qc.py:27` (импорт)

**Interfaces:**
- Produces: `bearmind/seed_suggest.py` с функциями `count_seeds`, `_robust`, `halo_mask`, `threshold_seeds`, `evaluate_thresholds`, `suggest_init_thresholds`. Тела и докстринги переносятся дословно; модуль импортирует только `cv2`, `numpy` и `bearmind.seed_mask.valid_mask`. Докстринг модуля: почему отдельно (вкладка Thresholds считает подсказку методов и не должна грузить CaImAn; `bm_qc` грузит его при импорте).
- `bm_qc.py`: `from bearmind.seed_suggest import (_robust, count_seeds, evaluate_thresholds, halo_mask, suggest_init_thresholds, threshold_seeds)  # noqa: F401` с комментарием рядом с импортом `valid_mask`. Вызовы через `bm_qc.<имя>` (кроп, этап thresholds, тесты) работают как раньше.

- [ ] **Step 1: Тесты** в `tests/bearmind/test_seed_suggest.py`:

```python
NAMES = ("count_seeds", "_robust", "halo_mask", "threshold_seeds", "evaluate_thresholds", "suggest_init_thresholds")

def test_bm_qc_reexports_the_suggestion():
    import bm_qc
    from bearmind import seed_suggest
    for name in NAMES:
        assert getattr(bm_qc, name) is getattr(seed_suggest, name), name

def test_suggestion_module_loads_no_caiman():
    code = "import sys, bearmind.seed_suggest; print('caiman' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"
```

- [ ] **Step 2: Запустите** `conda run -n bearmind python -m pytest -q tests/bearmind/test_seed_suggest.py` → FAIL (модуля нет).
- [ ] **Step 3: Перенесите** функции.
- [ ] **Step 4: Запустите** `tests/bearmind/test_seed_suggest.py`, `tests/test_bm_qc.py`, `tests/test_init_maps_reference.py`, `tests/bearmind/test_stage_thresholds.py`, `tests/bearmind/test_crop.py` → PASS без правок старых тестов; быстрый набор → 0 failed.
- [ ] **Step 5: Коммит.** `git add -- bearmind/seed_suggest.py tests/bearmind/test_seed_suggest.py`; `git commit -q -m "Suggestion: the threshold suggestion lives in bearmind.seed_suggest without CaImAn; bm_qc re-exports it" -- bearmind/seed_suggest.py bm_qc.py tests/bearmind/test_seed_suggest.py`

---

### Task 14: Кнопки методов автопорогов (п. 28)

**Files:**
- Modify: `bearmind/threshold_model.py` (новые `METHODS`, `suggestion_maps`, `method_values`; `source_text`), `bearmind/gui/thresholds_tab.py` (кнопки, фоновый расчёт, кэш, `apply_method`), `docs/students/series-workflow.md` (раздел 4)
- Test: `tests/bearmind/test_threshold_model.py`, `tests/bearmind/gui/test_thresholds_tab.py`

**Interfaces:**
- Consumes: `seed_suggest.suggest_init_thresholds` (задача 13), `tm.fit_gsig`, `self._loaded_sig` вкладки (заметка и `mtime`/размер файла карт).
- Produces:
  - `tm.METHODS = ("mirror_fdr", "robust_bg", "pct_margin")`.
  - `tm.suggestion_maps(maps: ThresholdMaps, g: int) -> dict` — `{"cn": maps.cn[g], "pnr": maps.pnr[g], "mean": maps.mean, "gSig": g}` и `"cn_neg"`, `"pnr_neg"` из зеркальных карт, если они есть. Так же делали проверки отчётов (`docs/data-flow/scripts/dfr_suggest.py`): подсказка по `npz` совпала с записанной в реестре (отчёт J10, `suggest.log`).
  - `tm.method_values(maps, g) -> dict[str, dict]` — `suggest_init_thresholds(suggestion_maps(maps, g), gSiz=4 * g + 1)`; для каждого метода `{"min_corr": float | None, "min_pnr": float | None, "note": str}`. Нет зеркальных карт → `mirror_fdr` с `note="no mirrored maps"`; метод без пары → его собственная `note`.
  - `tm.source_text`: вид источника `f"method:{name}"` → `f"Automatic thresholds ({name}) for gSig {g}: min_corr {…}, min_pnr {…}"`.
  - Вкладка: `method_buttons: dict[str, QPushButton]` в ряду с **Reset to suggestion** и **From session**. Текст: `f"{m}: {value_text(c, 2)} / {value_text(p, 1)}"`; пока считается — `f"{m}: …"`; без значения — `f"{m}: none"`, кнопка неактивна, подсказка — `note`. `apply_method(method: str) -> None` делает то же, что `_reset`, с источником `f"method:{method}"`; ничего не пишет.
  - Фон: `concurrent.futures.ThreadPoolExecutor(max_workers=1)` вкладки; кэш `dict[(sid, g, loaded_sig), dict]`; `QTimer` 100 мс проверяет `future.done()` и обновляет кнопки, только если ключ совпадает с показанной парой. Новый расчёт ставится, когда показаны карты (`_load_maps`) и при смене gSig, если ключа нет в кэше. `shutdown()` зовёт `executor.shutdown(wait=False, cancel_futures=True)`. Объекты Qt трогает только главный поток.

- [ ] **Step 1: Тесты** (фикстуры `series_maps`, `make_tab`, `_wait`):

```python
def test_method_buttons_show_each_method(qapp, series_maps, make_tab):
    p = series_maps.paths
    tab = make_tab(p)
    tab.show_session(S1)
    _wait(qapp, lambda: all("…" not in b.text() for b in tab.method_buttons.values()))
    g = tab.current()[1]
    maps = tm.load_maps(p.threshold_maps(S1))
    want, _ = seed_suggest.suggest_init_thresholds(tm.suggestion_maps(maps, g), gSiz=4 * g + 1)
    c, q = want["robust_bg"]["min_corr"], want["robust_bg"]["min_pnr"]
    assert tab.method_buttons["robust_bg"].text() == f"robust_bg: {tm.value_text(c, 2)} / {tm.value_text(q, 1)}"
    tab.apply_method("robust_bg")
    assert tab.current() == (S1, g, c, q)
    assert tab.source_label.text().startswith(f"Automatic thresholds (robust_bg) for gSig {g}")

def test_mirror_method_needs_mirrored_maps(qapp, series_maps, make_tab):
    tab = make_tab(series_maps.paths)
    tab.show_session(S2)  # maps without mirrors
    _wait(qapp, lambda: "…" not in tab.method_buttons["mirror_fdr"].text())
    b = tab.method_buttons["mirror_fdr"]
    assert b.text() == "mirror_fdr: none" and not b.isEnabled() and b.toolTip() == "no mirrored maps"

def test_method_values_follow_the_shown_session(qapp, series_maps, make_tab):
    """Review focus 4: S1's computation still runs when S2 is shown; its result never reaches S2's buttons,
    and S1's values come from the cache when it is shown again."""

def test_method_values_recomputed_for_new_maps(qapp, series_maps, make_tab):
    """The batch replaced S1's maps file (new mtime and content): the buttons show the new file's values."""
```

Два последних теста: подмените `tm.method_values` медленной обёрткой (`time.sleep`) и считайте вызовы. Проверьте, что кнопки S2 показывают значения S2, второй показ S1 не вызывает расчёт, а новый файл карт (`save_maps` с другими пиками BLOBS) вызывает. Тест `test_window_import_loads_no_bokeh` (`tests/bearmind/gui/test_inspection_tab.py`) уже ловит CaImAn при старте окна.

- [ ] **Step 2: Запустите** `conda run -n bearmind python -m pytest -q tests/bearmind/test_threshold_model.py tests/bearmind/gui/test_thresholds_tab.py` → FAIL.
- [ ] **Step 3: Реализуйте** по Interfaces.
- [ ] **Step 4: Руководство,** раздел 4: «Кнопки `mirror_fdr`, `robust_bg` и `pct_margin` показывают пороги трёх методов автоподбора для выбранного `gSig`. Щелчок ставит эти пороги на ползунки. Запись — только по **Accept → next**.»
- [ ] **Step 5: Запустите** тесты шага 2, `tests/bearmind/gui/test_inspection_tab.py::test_window_import_loads_no_bokeh`, `tests/test_student_guide.py` → PASS; быстрый набор → 0 failed.
- [ ] **Step 6: Коммит.** `git commit -q -m "Thresholds tab: one button per suggestion method, computed in the background for the shown gSig" -- bearmind/threshold_model.py bearmind/gui/thresholds_tab.py docs/students/series-workflow.md tests/bearmind/test_threshold_model.py tests/bearmind/gui/test_thresholds_tab.py`

---

### Task 15: Плохие кадры: замена интерполяцией до MC и детектор полос (п. 25) — гейт: «да» автора

Начинайте только после «да» автора на вариант A отчёта J10 (замена кадров 64642–64651 LT1_J10_2D и пересчёт с MC).

**Files:**
- Create: `bearmind/bad_frames.py`, `tests/bearmind/test_bad_frames.py`
- Modify: `bearmind/video.py` (`SessionReader`), `bearmind/runner.py` (`stage_params("mc")`), `bearmind/stages/mc.py` (`run`), `bearmind/checks.py` (проверка `mc_suspect_frames`), `bearmind/monitor.py:35` (`_CHECK_KIND`), `bearmind/defaults.yaml` (`checks.mc_suspect_frames: {warn: 0}`), `bearmind/actions.py` (`set_bad_frames`), `bearmind/cli.py` (`bad-frames`), `docs/superpowers/specs/2026-11-01-operator-notes.md`
- Test: `tests/bearmind/test_video.py`, `tests/bearmind/test_checks.py`, `tests/bearmind/test_cli.py`, `tests/bearmind/test_lt1_snapshot.py`, `tests/bearmind/test_stage_mc.py`

**Interfaces:**
- Produces (`bearmind/bad_frames.py`, импортирует только `numpy` и `cv2`):
  - `normalize(ranges, n_frames: int) -> list[tuple[int, int]]` — пары `[first, last]` включительно, номера кадров сессии с 0 (номера кадров видео MC). Сливает пересекающиеся и соседние, сортирует. `ValueError` с понятным текстом: не пара целых, `first > last`, выход за `[0, n_frames)`, или заменяются все кадры.
  - `parse(texts: list[str]) -> list[tuple[int, int]]` — `"64642-64651"`, `"70000"` для CLI.
  - `ranges_text(ranges) -> str` — `"64642-64651, 70000"`.
  - `interpolate(frames: Iterable[np.ndarray], ranges) -> Iterator[np.ndarray]` — поток кадров; кадры диапазона `[a, b]` заменяются линейно между кадрами `a - 1` и `b + 1`: `rint(f0 + (f1 - f0) * k / (n + 1))`, `k = 1..n`, в float32, затем uint8. Диапазон у начала сессии — копии кадра `b + 1`, у конца — копии кадра `a - 1`. Держит в памяти не больше одного диапазона кадров.
  - `class StripeMeter` с методами `add(frame: np.ndarray) -> None` и `suspects() -> list[int]`. Метрика из отчёта J10 (`s09_scan_all.py`, `s17_summary.py`, тем же `cv2`): `V` — среднее `|f[r+1] - f[r]|`, `H` — среднее `|f[:, c+1] - f[:, c]|`, `vmax` — наибольшее построчное среднее `|f[r+1] - f[r]|`. `rel = (V / max(H, 1e-3)) / median`; `vz = (vmax - median_filter(vmax, VZ_WINDOW, mode="nearest")) / max(1.4826 * MAD, 1e-3)`. Подозрительный кадр — `rel > REL_MAX` или `vz > VZ_MAX`. Константы `REL_MAX = 1.5`, `VZ_MAX = 20.0`, `VZ_WINDOW = 301`; комментарий: проверены на 84 сессиях LT1 (3 233 673 кадра), срабатывают только на 64642–64651 J10_2D.
- `SessionReader(series_root, rec, crop=None, bad_frames=(), on_raw=None)` — `on_raw(frame)` получает каждый сырой полный кадр до замены и кропа; `bad_frames` (нормализованные) заменяются в `_iter_frames`. `windows()` при непустом `bad_frames` → `ValueError`: окна читают куски выборочно, а замене нужны соседи.
- `stage_params("mc")` добавляет `"bad_frames": [[a, b], ...]` (списки из `normalize(cfg.get("bad_frames") or [], rec.n_frames)`; ключ `bad_frames` — верхний уровень конфига сессии), только если список не пуст. Ошибка `normalize` даёт «cannot be scheduled: stage parameters: …» в `status` и окне.
- `mc.run`: `SessionReader(paths.root, rec, crop=crop, bad_frames=bad, on_raw=meter.add)`; новые метрики `"bad_frames"` (список пар), `"suspect_frames"` (не больше 1000 номеров), `"n_suspect_frames"`.
- Проверка `mc_suspect_frames` в `checks.evaluate("mc", …)`: число подозрительных кадров вне `bad_frames`; уровень по `checks.mc_suspect_frames` (`warn: 0`). Текст: `f"{n} suspect frames (stripes) not replaced: {ranges_text(...)}"` или `f"{n} suspect frames, all replaced"`. `monitor._CHECK_KIND["mc_suspect_frames"] = "mc_shift"` (подсказка панели — «Check the MC video on the Thresholds tab.»).
- `actions.set_bad_frames(paths, sid, ranges, who) -> ActionResult` — проверяет `normalize` по `n_frames` записи, пишет `bad_frames` в конфиг сессии (`config.update_yaml`; пустой список удаляет ключ) и решение `Decision("bad_frames", ranges_text(...) or "none", who, now)`.
- CLI: `bad-frames SERIES SID [RANGE ...] [--clear]`; справка `"replace damaged frames of a session by interpolation before MC (MC and later stages run again)"`. Вывод: `f"[OK] {sid}: frames {ranges_text} ({n}) will be replaced; MC and every later stage of the session run again"`. Неверный диапазон → код 2 и текст `normalize`.
- Ключа `bad_frames` в `defaults.yaml` нет: его отсутствие и даёт неизменные хэши остальных сессий.

- [ ] **Step 1: Тесты** в `tests/bearmind/test_bad_frames.py`:

```python
def test_normalize_merges_and_checks():
    assert bad_frames.normalize([[5, 6], [3, 4], [10, 10]], 20) == [(3, 6), (10, 10)]
    for bad in ([[4, 3]], [[0, 20]], [[0, 19]], [["a", 2]]):
        with pytest.raises(ValueError):
            bad_frames.normalize(bad, 20)

def test_interpolate_between_neighbours():
    frames = [np.full((2, 2), 10 * i, np.uint8) for i in range(10)]
    out = list(bad_frames.interpolate(iter(frames), [(3, 5)]))
    assert [int(f[0, 0]) for f in out] == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # linear already
    frames[4] = np.full((2, 2), 255, np.uint8)  # the damaged frame
    out = list(bad_frames.interpolate(iter(frames), [(4, 4)]))
    assert int(out[4][0, 0]) == 40

def test_interpolate_at_the_ends_copies_the_neighbour():
    frames = [np.full((2, 2), v, np.uint8) for v in (200, 201, 7, 8, 9, 250)]
    out = [int(f[0, 0]) for f in bad_frames.interpolate(iter(frames), [(0, 1), (5, 5)])]
    assert out == [7, 7, 7, 8, 9, 9]

def test_stripe_meter_finds_a_displaced_band():
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(0, 255, (128, 128)).astype(np.uint8), (0, 0), 3)
    meter, n = bad_frames.StripeMeter(), 400
    for i in range(n):
        f = np.clip(base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
        if i == 250:
            f[40:60] = np.roll(f[40:60], 37, axis=1)  # a band of rows taken from the wrong place
        meter.add(f)
    assert meter.suspects() == [250]
```

В `tests/bearmind/test_video.py`: `SessionReader(..., bad_frames=[(12, 13)], on_raw=seen.append)` на `make_series` — кадры 12 и 13 потока заменены интерполяцией, `seen[12]` равен исходному кадру 12, `windows()` с `bad_frames` → `ValueError`.

В `tests/bearmind/test_lt1_snapshot.py`:

```python
def test_bad_frames_change_only_their_session(tmp_path):
    paths = copy_series(SNAPSHOT, tmp_path / "lt1")
    config.update_yaml(paths.session_config("LT1_J10_2D"), {"bad_frames": [[64642, 64651]]})
    assert stale_finished(paths) == [("LT1_J10_2D", "mc"), ("LT1_J10_2D", "thresholds")]

@pytest.mark.realdata
@pytest.mark.slow
def test_live_stripe_detector_finds_the_j10_2d_burst():
    """Reads the raw frames of LT1_J10_2D and LT1_J10_1D (about 1 min each), read-only."""
    src = live_bearmind_dir()
    if src is None:
        pytest.skip("BEARMIND_LT1_SERIES is not set")
    paths = SeriesPaths(src.parent)
    for sid, want in (("LT1_J10_2D", list(range(64642, 64652))), ("LT1_J10_1D", [])):
        meter = bad_frames.StripeMeter()
        for _ in SessionReader(paths.root, load_record(paths, sid), on_raw=meter.add).iter_chunks():
            pass
        assert meter.suspects() == want, sid
```

У J10_2D CNMF-E не DONE, поэтому в списке только MC и карты. Если к моменту задачи CNMF-E J10_2D в снимке стал DONE, ожидание дополняется `("LT1_J10_2D", "cnmf")`. Кроп J10_2D и все этапы остальных сессий в списке появиться не должны.

В `test_checks.py`: подозрительные `[64642..64651]` и `bad_frames [[64642, 64651]]` → `ok`, текст `"10 suspect frames, all replaced"`; подозрительные `[5, 6]` без `bad_frames` → `warning`, текст начинается с `"2 suspect frames (stripes) not replaced: 5-6"`. В `test_cli.py`: `bad-frames` пишет конфиг сессии и решение; `bad-frames ... 10-5` → код 2. В `test_stage_mc.py` (медленный, синтетический фильм): конфиг с `bad_frames` → метрика `bad_frames` записана, `n_suspect_frames` — целое.

- [ ] **Step 2: Запустите** `conda run -n bearmind python -m pytest -q tests/bearmind/test_bad_frames.py tests/bearmind/test_video.py tests/bearmind/test_checks.py tests/bearmind/test_cli.py tests/bearmind/test_lt1_snapshot.py` → FAIL.
- [ ] **Step 3: Реализуйте** по Interfaces.
- [ ] **Step 4: Заметки оператора:** раздел «Плохие кадры сессии» (по-русски, STE): когда (`suspect frames` в панели предупреждений), команда `run_bearmind bad-frames <СЕРИЯ> <SID> <первый>-<последний>`, что батч пересчитает (MC и всё после него), что пороги сессии остаются в её конфиге и что карты стоит посмотреть заново на вкладке **Thresholds**.
- [ ] **Step 5: Запустите** тесты шага 2, медленный `tests/bearmind/test_stage_mc.py -m "not realdata"`, `tests/test_student_guide.py` → PASS. С `BEARMIND_LT1_SERIES`: `-m realdata tests/bearmind/test_lt1_snapshot.py` (включая медленный детектор) → PASS. Быстрый набор → 0 failed.
- [ ] **Step 6: Коммит.** `git add -- bearmind/bad_frames.py tests/bearmind/test_bad_frames.py`; `git commit -q -m "MC: damaged frames listed per session are interpolated before MC; a stripe detector lists suspect frames" -- bearmind/bad_frames.py bearmind/video.py bearmind/runner.py bearmind/stages/mc.py bearmind/checks.py bearmind/monitor.py bearmind/defaults.yaml bearmind/actions.py bearmind/cli.py docs/superpowers/specs/2026-11-01-operator-notes.md tests/bearmind/test_bad_frames.py tests/bearmind/test_video.py tests/bearmind/test_checks.py tests/bearmind/test_cli.py tests/bearmind/test_lt1_snapshot.py tests/bearmind/test_stage_mc.py`

---

### Task 16: Новый порог template corr (п. 22а, код) — гейт: ответ автора на задачу 6

**Files:**
- Modify: `bearmind/defaults.yaml:119-124` (`checks.mc_template_corr` и комментарий: источник — `docs/superpowers/specs/2026-11-16-mc-template-corr/proposal.md`, решение автора и дата)
- Test: `tests/bearmind/test_config.py:38`

Если автор выбрал абсолютный порог — шаги ниже. Если правило мыши (относительное), остановитесь: ему нужен свой дизайн (`checks._components_vs_median` — образец относительной проверки).

- [ ] **Step 1:** в `test_config.py` замените ожидание `{"warn": 0.7}` выбранным значением → FAIL.
- [ ] **Step 2:** запишите значение в `defaults.yaml` → PASS.
- [ ] **Step 3:** `tests/bearmind/test_lt1_snapshot.py` и быстрый набор → PASS, 0 failed (`checks` не хэшируется).
- [ ] **Step 4: Коммит.** `git commit -q -m "Checks: MC template corr threshold <value> (author's choice from the LT1 distribution)" -- bearmind/defaults.yaml tests/bearmind/test_config.py`

После переноса старые проверки 84 сессий обновит `recheck` (раздел «Перенос», шаг 5).

---

## Перенос в bearmind-batch

Слияние и остановка батча меняют общее состояние: каждый шаг ниже — только с согласия автора.

**Без остановки (задачи 1–6), в любое время после их ревью:**
1. В основной папке: `git merge --no-edit <хэш коммита задачи 6>`. Батч не останавливается: эти задачи меняют тесты, документы, скан, модель и вкладки окна, а runner их не импортирует (проверено: процесс runner грузит `bearmind.{checks, config, crop, fileio, isolate, naming, paths, pool, provenance, registry, runner, stages, timestamps, video}`).
2. Перезапустите окно BEARMiND: вкладки грузят новый код только при старте.

**С остановкой (задачи 7–14; 15 и 16 — если готовы и одобрены):**
1. В worktree: `git merge --no-edit bearmind-batch` (документы основной ветки). Затем быстрый набор → 0 failed и `BEARMIND_LT1_SERIES=H:\Projects\LT1\CalciumData conda run -n bearmind python -m pytest -q -m realdata tests/bearmind/test_lt1_snapshot.py` → PASS. Так всё проверено до остановки, и сама остановка короткая.
2. Выберите момент: батч ждёт человека или идёт короткий этап. Остановка ждёт конца текущего этапа; CNMF-E часовой сессии идёт до 2 ч.
3. Остановите батч: **Stop after current stage** в окне или `conda run -n bearmind python -m bearmind stop H:\Projects\LT1\CalciumData`. Дождитесь **Stopped**.
4. В основной папке: `git merge --ff-only bearmind-fixes` (если `bearmind-batch` с шага 1 не сдвинулась; иначе `git merge --no-edit bearmind-fixes`).
5. `conda run -n bearmind python -m bearmind recheck H:\Projects\LT1\CalciumData`. Ожидается: у 9 сессий с чёрным кадром 0 (J17_2D, J18_2D, J58_2D, J53_2D, J57_2D, J30_2D, J24_3D, J54_2D, J24_2D) проверка MC становится `ok`; J10_2D остаётся `warning` (27,1 px). С задачей 16 — новые уровни template corr.
6. Только если автор выбрал A: `conda run -n bearmind python -m bearmind scan H:\Projects\LT1\CalciumData`. У LT1_J05_1D: 47 308 кадров, аномалия `ts_last_row_incomplete`.
7. Только с задачей 15: `conda run -n bearmind python -m bearmind bad-frames H:\Projects\LT1\CalciumData LT1_J10_2D 64642-64651`.
8. `conda run -n bearmind python -m bearmind status H:\Projects\LT1\CalciumData`. Устаревшими (`D*`) могут быть только LT1_J05_1D (с кропа, после шага 6) и LT1_J10_2D (с MC, после шага 7). Любая другая сессия с `D*` — стоп: не запускайте батч, разберите причину.
9. Запустите батч (**Start** или `run`). Перезапустите окно и сервер инспекции.
10. Через минуту: `waiting_thresholds` в `_bearmind/heartbeat.json` равен числу сессий `waiting` на вкладке **Thresholds**.

Остановка без ожидания этапа: шаги 4–9, около 10 минут (рескан 84 сессий и `recheck` — по нескольку минут).
