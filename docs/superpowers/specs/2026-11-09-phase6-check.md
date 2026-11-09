# Фаза 6: файлы окружения, установка с нуля, свежий клон, сквозной прогон и полный набор тестов

Дата: 2026-11-09. Код: ветка `bearmind-batch`, коммиты задачи 7 фазы 6 (список в конце). Машина: AMD Ryzen 9 7950X3D (16 ядер, 32 потока), 191 ГБ памяти, RTX 4070 SUPER, Windows 11. conda 24.7.1, mamba 1.5.8 (решатель libmamba). Метки: **[изм]** — измерено в этой проверке, **[вывод]** — расчёт или толкование.

## Итог

1. **Файлы окружения полные** [изм]. Все пакеты, которые импортирует программа, закреплены версиями из `env/bearmind-lock.txt`. Новый тест сверяет файлы окружения с lock и импорты программы с файлами окружения. До правки тест нашёл ровно 15 незакреплённых пакетов из таблицы аудита.
2. **Окружение собирается с нуля** [изм]. `mamba env create` во временный префикс занял 213 с, `pip install -r requirements-pip.txt` — 72 с. Переменная `OPENBLAS_CORETYPE=Sandybridge` попала в новое окружение. Быстрый набор тестов в нём: 1153 passed, 24 skipped, код 0. Версии всех закреплённых пакетов равны lock. Транзитивные пакеты: 28 новее lock, 15 с другой сборкой.
3. **Свежий клон работает** [изм]. `run_bearmind.bat --help` и `python -m bearmind status` на копии `series_short` прошли с кодом 0.
4. **Сквозной прогон из другой папки прошёл** [изм]. Батч запущен из новой пустой временной папки. Он выполнил crop, mc, thresholds, cnmf, gate, autoinspect и export сессии `NOF_H01_1D` за 690 с и вышел с кодом 0. Во временную папку и в репозиторий батч ничего не записал. `series_short` не изменилась (sha256 до и после). Экспорт побитово равен экспорту `series_short`, сделанному из корня репозитория.
5. **Полный набор тестов со `slow` прошёл со второго раза** [изм]. Первый прогон: 1174 passed, 1 failed, 31 skipped. Упал тест скорости `test_merge_time`: собственное время окна при Merge 0,503 с при пределе 0,5 с. Этот тест отдельно прошёл (0,265 с). Второй полный прогон: 1175 passed, 31 skipped, код 0. Порог не менялся. Тесты `realdata` пропущены: переменная `BEARMIND_TEST_SERIES` не задана.
6. **Найдена и исправлена ошибка запуска** [изм]. Запасной путь `run_bearmind.bat` через `conda run` не работал в Anaconda Prompt. Это тот путь, который инструкция советует в разделе 4. Причина: переменная файла запуска `CONDA_BAT` совпадала с переменной, которую читает сам `conda run`. Переменная переименована в `CONDA_CMD`, тест ловит ошибку.

**Что нужно от автора.**

- Решить, устраивает ли канал `defaults`. PyTorch со сборкой OpenBLAS есть только в канале Anaconda `pkgs/main`. Новые Anaconda и Miniconda спрашивают согласие с Terms of Service этого канала. Условия Anaconda (https://www.anaconda.com/pricing/terms-of-service-faqs, раздел 1): учебное заведение пользуется бесплатно, если оно Eligible Academic Institution и принимает Academic End User License Agreement; коммерческой организации больше 200 человек нужен платный план. Подходит ли лаборатория, решает автор. Это вопрос лицензии, а не кода.
- Самая сильная непроведённая проверка: установка на машине студента (другой процессор, без CUDA, Miniforge или новая Anaconda с вопросом Terms of Service).
- По желанию: добавить `.gitattributes` с `*.bat text eol=crlf`, чтобы клон без `core.autocrlf` получал `.bat` с CRLF (раздел «Что не проверено»).

## Что не проверено

- **Машина без CUDA.** Проверено только решение: без CUDA решатель выбирает те же 358 пакетов conda (раздел 2). PyTorch закреплён в сборке для процессора. GPU программа использует только по желанию (`cupy` для вейвлета, его в окружении нет). Установка и прогон на такой машине не проверены.
- **Другая модель процессора и `OPENBLAS_CORETYPE`.** На другом процессоре ничего не запускалось. Решатель для процессора AVX2 без AVX-512 выбирает те же пакеты. Нужен ли там обход: ядра Sandybridge работают на любом процессоре с AVX (Intel Core с 2011 года, все AMD Ryzen) [вывод]. Там BLAS может быть медленнее собственных ядер процессора. Эталоны тестов сделаны на ядрах Sandybridge, поэтому с обходом тесты сравнивают те же ядра [вывод]. На процессоре без AVX (некоторые Pentium, Celeron, Atom) эти ядра не работают: нужен другой `OPENBLAS_CORETYPE` (комментарий `environment.yml`). Есть ли ошибка Zen 4 в OpenBLAS новее 0.3.34, не проверено (TODO в `environment.yml`).
- **Вопрос Terms of Service.** На этой машине conda 24.7.1 без плагина `conda-anaconda-tos`. Текст вопроса и команда взяты со страницы Anaconda, сам вопрос не воспроизведён.
- **Miniforge и строгий приоритет каналов.** Проверено решателем: без обхода файл не решается, обход проверен на `conda config`. Полная установка с обходом не проверена.
- **Холодная установка.** Время установки измерено с тёплым кэшем пакетов. Загрузку с пустым кэшем эта проверка не измеряла.
- **Клон с `core.autocrlf=false`.** В индексе `.bat` лежат с концами строк LF. Клон без `autocrlf` получит `.bat` с LF; `cmd.exe` с такими файлами иногда не находит метки `goto`. Здесь клон получил CRLF. Защита — файл `.gitattributes` с `*.bat text eol=crlf`; в задачу он не входил.
- **Запасной путь файлов запуска на других раскладках conda.** Проверен поддельной папкой пользователя и junction на этой машине, не на машине с Miniconda или Anaconda для всех пользователей.
- **Тесты `realdata`.** Переменная `BEARMIND_TEST_SERIES` не задана: тесты `realdata` пропущены.

## 1. Перенесённые пункты

### П-15. `OPENBLAS_CORETYPE` в `environment.yml`

- Сделано. Раздел `variables:` ставит `OPENBLAS_CORETYPE: Sandybridge`. Комментарий говорит, зачем это нужно (обход Zen 4, ядра эталонов тестов). Комментарий говорит, где обход безвреден: процессоры с AVX (Intel Core с 2011 года, все AMD Ryzen). Там меняется только выбор ядер BLAS: BLAS может быть медленнее, результаты могут отличаться на округление. Процессор без AVX (некоторые Pentium, Celeron, Atom) эти ядра выполнить не может. Для него комментарий называет команду `conda env config vars set OPENBLAS_CORETYPE=Nehalem`.
- Проверка [изм]. `conda env config vars list -p <свежий префикс>` напечатал `OPENBLAS_CORETYPE = Sandybridge`. Файл `conda-meta\state` нового окружения: `{"env_vars": {"OPENBLAS_CORETYPE": "Sandybridge"}}`. mamba 1.5.8 переменную не потерял.
- Тест `test_env_file_sets_openblas_coretype`: значение в `environment.yml` равно значению в `bearmind/__init__.py` и в корневом `conftest.py`.

### П-16. Установщики, `test_env.py`, `bearmind2025.yml`, инструкция

- `test_env.py` проверяет пакеты окружения `bearmind`: 31 импорт (CaImAn, PySide6, bokeh, tornado, driada, ssqueezepy, psutil, threadpoolctl, PyAV, PyTorch, interpret и другие). `tensorflow`, `jupyter_bokeh`, `peakutils` в списке нет. При ошибке скрипт печатает `[FAILED]` и выходит с кодом 1.
- `install2_pip.bat` по-прежнему вызывает `test_env.py`. При коде 1 он печатает `[FAILED]` и останавливается. Он больше не обновляет pip: окружение сохраняет pip, который поставила conda, как в lock и в инструкции. Команда установки стала той же, что в инструкции: `pip install -r requirements-pip.txt`.
- `bearmind2025.yml` удалён (`git rm`).
- Тест `test_install_check_script_checks_the_bearmind_env`: в `test_env.py` нет старых пакетов, есть каждый импорт программы, все импорты проходят. Тест `test_installers_and_guide_use_the_env_files`: шапка `environment.yml`, оба установщика и инструкция используют одни команды и одно имя окружения.
- Проверка [изм]. `install2_pip.bat` прогнан на свежем окружении (через junction в поддельной папке пользователя, см. раздел 2): `[OK] Found`, NumPy работает, все пакеты pip уже стоят, `Results: 31 OK, 0 FAILED`, `Installation Complete!`, код 0, 18 с. `test_env.py` в окружении `bearmind`: 31 OK, код 0.

### Незакреплённые пакеты (таблица аудита, Е3–Е12)

- До правки [изм]. Новый тест `test_program_imports_are_pinned` упал со списком: bokeh, imageio-ffmpeg, ipython, ipywidgets, joblib, matplotlib, pandas, psutil, pyside6, pytorch, pytz, ssqueezepy, threadpoolctl, tifffile, tornado. Это ровно 15 пакетов аудита.
- Сделано. В `environment.yml`: pyside6 6.11.2, bokeh 3.10.0, tornado 6.5.10, threadpoolctl 3.7.0, psutil 7.2.2, pytorch 2.14.1 со сборкой `cpu_openblas*`, pandas 3.1.0, matplotlib 3.11.2, tifffile 2026.10.15, joblib 1.6.0, ipywidgets 8.1.9, ipython 9.18.0. В `requirements-pip.txt`: ssqueezepy 0.6.6, pytz 2026.6, imageio-ffmpeg 0.7.0. Версии взяты из lock.
- Тест `test_env_files_match_lock` прошёл и до правки: фазы 3–5 пакетов не добавляли (аудит, Е2). Тест различает ошибки: подмена `pytz==2026.5` и сборки `cpu_mkl*` дала FAILED с обеими строками [изм]. Подмена отменена.
- Тест `test_program_imports_are_pinned` читает импорты корневых модулей, `bearmind/` и `tests/` (модуль `ast`). Новый сторонний импорт без закрепления роняет тест. Необязательный `cupy` (GPU) не закрепляется.
- Шапка lock (О11) теперь говорит: строки `pypi_0` ставит pip из `requirements-pip.txt`, `conda create --file` их не ставит.
- `scipy` (Е15) [изм]. В lock и в обоих окружениях `scipy` показан как `pypi_0`. На деле он из conda-forge: в `conda-meta` есть запись `scipy-1.18.1-py312h8847828_2.json`, файл `INSTALLER` — `conda`. Запись conda называет путь `lib/site-packages` строчными буквами, у numpy и pandas — `Lib/site-packages`. Поэтому `conda list` не связывает папку с записью. pip `scipy` не заменял. Это записано в шапке lock.
- `python=3.12.16` (О12) оставлен: решатель нашёл эту версию в свежей сборке.

### Каналы conda: `nodefaults` или инструкция

- Факт [изм]. `mamba search --override-channels -c conda-forge "pytorch=2.14.1" --platform win-64` даёт только сборки `cpu_mkl_*` и `cuda134_mkl_*`. Сборки с OpenBLAS в conda-forge нет. `pytorch 2.14.1 cpu_openblas_py312he688b4c_0` есть только в `pkgs/main`. В живом и в свежем окружении `pytorch` и `libtorch` пришли из `defaults`.
- Решение. `nodefaults` не добавлен: без `defaults` закреплённую сборку не найти. Канал `defaults` записан в `environment.yml` явно, с комментарием. На этой машине список каналов не изменился: `defaults` и раньше приходил из `.condarc`. На машине без `defaults` в `.condarc` файл теперь сам называет нужный канал.
- Инструкция (раздел 2) теперь предупреждает о вопросе Terms of Service. Что ответить: букву a. Если студент отказался: `conda tos accept` и повторить шаг 9. Источник: страница Anaconda о плагине `conda-anaconda-tos` (https://www.anaconda.com/docs/getting-started/tos-plugin). Плагин входит в Anaconda Distribution 2025.06-0 и Miniconda 25.1-1-0. Здесь conda 24.7.1 без плагина: сам вопрос не воспроизведён.

### Файлы запуска

- Оба файла ищут окружение в одних пяти папках. Пятая — `%USERPROFILE%\.conda\envs\bearmind`: туда conda кладёт окружения пользователя, когда Anaconda стоит для всех пользователей. `bearmind_gui.bat` получил запасной путь `conda run`, как у `run_bearmind.bat`. В запасном пути консоль остаётся открытой, пока работает окно. При ошибке консоль ждёт нажатия клавиши.
- Ошибка (найдена тестом) [изм]. Старый запасной путь `run_bearmind.bat` ставил `CONDA_BAT=%CONDA_EXE%`. `conda run` читает `CONDA_BAT` как путь к `conda.bat` (`conda/utils.py`, строка 407). В Anaconda Prompt `CONDA_EXE` — это `conda.exe`, и `conda run` падал: `CondaError: Run 'conda init' before 'conda activate'`. Воспроизведено в активированной Anaconda Prompt с папкой пользователя без окружения. После переименования в `CONDA_CMD` оба файла печатают справку в той же обстановке. Без `CONDA_EXE` (двойной щелчок) запасной путь работал и раньше.
- Тесты `tests/test_launchers.py`:
  1. папки и запасной путь одинаковы в обоих файлах, переменной `CONDA_BAT` нет;
  2. поддельная папка пользователя без окружения даёт `[FAILED] conda env "bearmind" not found.`;
  3. junction `.conda\envs\bearmind` на окружение теста даёт справку `run_bearmind`; junction удаляется (`os.rmdir`), окружение не трогается;
  4. запасной путь `conda run` обоих файлов печатает справку (`run_bearmind` и `run_bearmind gui`).
- Окно [изм]. `bearmind_gui.bat` запущен как двойной щелчок. Через 1 с работал `pythonw.exe -m bearmind gui` (PID 4062556) с окном `BEARMiND`. Процесс остановлен по PID. `~\.bearmind` не изменилась.
- Инструкция: пять папок; раздел 4 описывает поиск через conda и предупреждает, что консоль закрывать нельзя.

### `.gitignore`

- Сделано: после `!/env/` добавлены `/env/*/` и `/env/pyvenv.cfg`. Отступление: одной строки `/env/*/` мало. `python -m venv env` кладёт `pyvenv.cfg` прямо в `env/`, и `git status` его показал бы.
- Проверка [изм]. `git check-ignore --no-index`: `env/Scripts/python.exe`, `env/Lib/site-packages/x.py`, `env/Include/x.h`, `env/pyvenv.cfg` скрыты; `env/bearmind-lock.txt` и `env/other-lock.txt` не скрыты. В свежем клоне `python -m venv env` дал пустой `git status --porcelain`. Тест `test_gitignore_hides_a_root_venv_not_the_lock`.

### По желанию: строки реконструкций в экспорте

`tests/test_export_after_compression.py` теперь проверяет и `reconstructions` в npz: строк столько же, сколько экспортированных компонент; нулевых строк нет; строка `k` равна реконструкции своей компоненты. Тест прошёл сразу: ошибки нет. Тест закрепляет поведение.

## 2. Установка с нуля

Команды (корень репозитория, префикс `C:\Users\User\AppData\Local\Temp\t67_bm_fresh`) [изм]:

| Шаг | Команда | Время | Результат |
|---|---|---|---|
| Создание | `mamba env create -p <префикс> -f environment.yml` | 212,8 с | код 0; каналы conda-forge, defaults |
| Решение отдельно | `mamba env create --dry-run -p <другой префикс> -f environment.yml` | 90,5 с | код 0; метаданные и решение, кэш метаданных тёплый, параллельно шёл набор тестов |
| Переменная | `conda env config vars list -p <префикс>` | — | `OPENBLAS_CORETYPE = Sandybridge` |
| pip | `<префикс>\python.exe -m pip install -r requirements-pip.txt` | 72,3 с | код 0; 69 пакетов pip |
| Установщик | `install2_pip.bat` через junction `.conda\envs\bearmind` | 18,2 с | код 0; `Results: 31 OK, 0 FAILED` |
| Тесты | `conda run -p <префикс> python -m pytest -q -m "not slow and not realdata" tests` | 1118,5 с | 1153 passed, 24 skipped, 29 deselected, 1 warning, код 0 |
| Удаление | `conda env remove -p <префикс> -y` | 17,8 с | папки нет |

Пояснения:

- Время установки [вывод]. Пакеты conda уже лежали в кэше `anaconda3\pkgs` после сборки окружения `bearmind`. На машине с пустым кэшем загрузка добавит время. Создание без решения заняло около 120 с.
- pip предупредил: `caiman 1.13.2 requires cython, numpydoc, yapf, which is not installed`. Этих пакетов нет и в lock: живое окружение такое же. На тесты это не влияет.
- 24 skipped: 22 теста без реальных сессий (как в задачах 5 и 6) и 2 теста запасного пути `conda run`. Им нужно окружение с именем `bearmind` в папке `envs` установки conda. Свежий префикс таким не является: пропуск задуман.
- Первый прогон `install2_pip.bat` через junction в папке сессии агента дал `[FAILED] driada: FileNotFoundError`. Причина — длинный путь: префикс через junction в папке сессии агента длиннее 260 знаков вместе с файлами driada. С той же поддельной папкой пользователя и настоящим коротким префиксом driada импортировался. Повтор через короткий путь `C:\Users\User\AppData\Local\Temp\t67_fh` прошёл. Это ошибка подготовки проверки, а не установщика.

Сравнение свежего окружения с lock (`conda list --export`) [изм]: 426 пакетов в обоих, наборы имён равны. 383 строки совпали. Ни один закреплённый пакет не отличается. `libopenblas` той же версии. Отличаются транзитивные пакеты:

- другая версия (28): anyio 4.15.1→4.16.0, aom 3.14.1→3.15.3, blas 2.311→2.312, click 8.5.0→8.5.1, coverage 7.17.0→7.17.2, filelock 4.2.4→4.2.9, fmt 12.2.0→12.2.1, google-api-core 2.43.2→2.44.0, google-auth 2.62.0→2.62.1, google-crc32c 1.8.0→1.9.0, jsonpointer 3.6→3.10, jupyterlab 4.6.5→4.6.6, lcms2 2.19.1→2.20, llvm-openmp 23.1.4→23.1.5, markdown 3.11.2→3.11.3, notebook 7.6.4→7.6.5, openh264 2.6.0→2.7.0, platformdirs 4.14.0→4.14.5, pydantic 2.14.2→2.14.3, pydantic-core 2.50.2→2.50.4, python-dateutil 3.0.0→4.0.0, python-fastjsonschema 2.22.2→2.22.3, pyzmq 27.2.0→27.3.0, regex 2026.9.29→2026.11.9, sdl3 3.4.18→3.4.20, soupsieve 3.0.4→4.0.1, svt-av1 4.2.0→4.3.0, typing-inspection 0.4.4→0.4.5;
- та же версия, другая сборка (15): `_libavif_api`, `blas-devel`, `certifi`, `debugpy`, `ffmpeg` (`gpl_ha77653d_905` → `gpl_h116a9b4_907`), `libabseil`, `libavif16`, `libblas`, `libcblas`, `liblapack`, `liblapacke` (сборка 11 → 12 при версии 3.11.0), `markupsafe`, `pip`, `pyyaml`, `qhull`.

FFV1 и libx264 в новой сборке ffmpeg есть: `test_ffv1_codec_available` прошёл в свежем окружении.

После сборки `environment.yml` менялся только в комментариях. Проверка [изм]: `mamba env create --dry-run` по итоговому файлу дал те же 358 пакетов conda с теми же версиями и сборками, что свежее окружение до pip. Единственное расхождение — `scipy`: `conda list` показывает его как `pypi_0` (см. раздел 1).

### Решатель для других машин

Скрипт `t67_dryruns.py`: `mamba env create --dry-run` с подменой виртуальных пакетов conda. Это проверяет только решение, не установку и не работу [изм].

| Случай | Подмена | Код | Время | Пакетов | Отличий от этой машины |
|---|---|---|---|---|---|
| эта машина | — | 0 | 84,5 с | 358 | — |
| нет CUDA | `CONDA_OVERRIDE_CUDA=""` | 0 | 74,5 с | 358 | 0 |
| процессор AVX2 без AVX-512 | `CONDA_OVERRIDE_ARCHSPEC=x86_64_v3` | 0 | 75,5 с | 358 | 0 |
| оба сразу | обе подмены | 0 | 74,9 с | 358 | 0 |
| строгий приоритет каналов | `CONDA_CHANNEL_PRIORITY=strict` | 1 | 141,5 с | — | `pytorch-2.14.1-cpu_openblas... is excluded by strict repo priority` |

- Машина без CUDA и процессор без AVX-512 получают те же пакеты conda [изм].
- Строгий приоритет каналов (умолчание Miniforge) файл не решает: PyTorch из `defaults` исключается, потому что в conda-forge тоже есть pytorch [изм]. Привязка к каналу (`defaults::pytorch`, `defaults::libtorch`) не помогла: при строгом приоритете тот же отказ, при гибком — те же 358 пакетов [изм].
- Обход [изм]. Переменная `CONDA_CHANNEL_PRIORITY=flexible` перекрывает строгий `.condarc`: с временным `.condarc` (`channel_priority: strict`, через `CONDARC`) `conda config --show channel_priority` дал `strict`, с переменной — `flexible`. Инструкция (раздел 2) и комментарий `environment.yml` теперь называют этот обход. Отдельный коммит b3f116b.

## 3. Свежий клон

Скрипт `t67_clone_check.py` (папка сессии агента) [изм]:

1. `git clone -b bearmind-batch <репозиторий> C:\Users\User\AppData\Local\Temp\t67_clone`: 1,9 с, HEAD клона 2d3f220 равен HEAD репозитория.
2. Файлы `.bat` в клоне — с концами строк CRLF (здесь `core.autocrlf=true`). `bearmind2025.yml` и `config.json` в клоне нет.
3. `run_bearmind.bat --help` из клона: код 0, 2,0 с, справка как в инструкции.
4. `series_short` скопирован целиком в `C:\Users\User\AppData\Local\Temp\t67_short_copy` (62 файла; источник только читался).
5. `python -m bearmind status <копия>` из клона: код 0, 2,0 с. `bearmind` загружен из клона. Таблица: обе сессии `D` во всех этапах, превью `DD`. `run_bearmind.bat status <копия>`: код 0.
6. `python -m venv --without-pip env` в клоне: `git status --porcelain` пуст.
7. Клон и копия удалены.

## 4. Сквозной прогон из другой папки

Фаза 6 убрала вызовы `os.chdir`: дочерние процессы этапов теперь работают в папке, из которой запущен батч. Прогон проверяет, что батч из чужой папки ничего туда не пишет и даёт тот же результат.

### Подготовка

1. `t67_make_root.py` создал корень `H:\Projects\_bearmind_test\phase6_e2e`: копии `series_short\_bearmind` (18 файлов) и `series_short\3_MC` (4 файла) с сохранением времени файлов (`shutil.copy2`) и junction `1_Raw` на `series_short\1_Raw` [изм].
2. Поиск путей в копии `_bearmind` без логов: строки `series_short` нет ни в одном файле (и в бинарных `npz`); путей с буквой диска в `json`, `yaml`, `csv` нет [изм].
3. `run_bearmind.bat previews <корень> off`: превью выключены.
4. `run_bearmind.bat rerun <корень> NOF_H01_1D --from crop --who t67`: все этапы `NOF_H01_1D` сброшены. Пороги сессии остались в её конфиге (`gSig 4`, `min_corr 0.9`, `min_pnr 11.0`).
5. `status` после сброса: у `NOF_H01_1D` все этапы `.`; у `NOF_H03_3D` cnmf и дальше `D*` (папка `4_Estimates` не копировалась). `NOF_H03_3D` вне `--sessions`, батч её не трогает.

### Прогон

Команда: `run_bearmind.bat run H:\Projects\_bearmind_test\phase6_e2e --sessions NOF_H01_1D` (файл запуска выполняет `python -m bearmind run ...`). Текущая папка — новая пустая `C:\Users\User\AppData\Local\Temp\t67_e2e_cwd`. Батч запустил скрипт `t67_e2e.py`: он опрашивал реестр раз в 2 с, ответил воротам и следил за запретными этапами. PID батча 4069008 (журнал серии) [изм].

| Этап | Начало → конец | Время | Пик памяти дерева процессов этапа |
|---|---|---|---|
| crop | 11:31:12,6 → 11:32:10,6 | 58 с | 19,1 ГБ |
| mc | 11:32:11,1 → 11:33:49,7 | 99 с | 13,1 ГБ |
| thresholds | 11:33:49,7 → 11:35:32,5 | 103 с | 59,6 ГБ |
| cnmf | 11:35:32,5 → 11:39:51,2 | 259 с | 33,3 ГБ |
| ожидание ворот | 11:39:51,2 `gate: waiting for manual_inspection`; 11:39:54,4 решение `gate ... not-needed` (код 0); 11:40:21,3 батч взял решение | 30 с | — |
| gate | 11:40:21,3 → 11:40:23,6 | 2 с | 0,0 ГБ |
| autoinspect | 11:40:23,7 → 11:42:24,6 | 121 с | 36,2 ГБ |
| export | 11:42:24,7 → 11:42:39,8 | 15 с | 4,7 ГБ |

- 11:42:39,9 журнал: `run finished: done 7, failed 0, waiting 0, skipped 0, not_recorded []`. Консоль: `[OK] this run: done 7, failed 0, skipped 0`. Батч вышел сам с кодом 0 через 689,9 с после запуска [изм].
- Ожидание ворот — 27 с после решения: батч опрашивает решения раз в 30 с (`poll_s`). Это поведение батча, не ошибка.
- Пик памяти всех процессов батча по опросу раз в 2 с — 51,4 ГБ. В дереве батча было 100 процессов; после выхода живых нет, останавливать не пришлось [изм].
- Время этапов меняется от прогона к прогону. Прогон `series_short` 2026-11-07 (журнал серии): crop 51 с, mc 81 с, thresholds 86 с, cnmf 221 с, autoinspect 76 с, export 10 с. Проверка фазы 5: autoinspect 113 с [изм]. Причину разницы эта проверка не искала.

### Проверки после прогона

| Что | Результат |
|---|---|
| Все этапы `NOF_H01_1D` | crop, mc, thresholds, cnmf, gate, autoinspect, export — `done`, ошибок нет; `mc_preview`, `cnmf_preview` — `pending` (превью выключены) [изм] |
| Запретные этапы | охрана скрипта не сработала; состояния этапов `NOF_H03_3D` до и после равны [изм] |
| Текущая папка `t67_e2e_cwd` | пуста; после проверки удалена [изм] |
| Репозиторий | размер и время всех файлов (без `.git`, вместе с игнорируемыми и `__pycache__`) не изменились. Новых файлов два: этот отчёт и `task-7-report.md`, их писал я во время прогона [изм] |
| Временная папка этапов `C:\Users\User\AppData\Local\Temp\bearmind` | пуста до и после [изм] |
| `series_short` | не изменилась (таблица ниже) [изм] |
| Результат | экспорт: 142 нейрона. `C`, `asp` и `component_indices` в `NOF_H01_1D_data.npz` побитово равны экспорту `series_short` от 2026-11-07 (тогда батч работал из корня репозитория) [изм] |

Хэши `series_short` (скрипт `t67_hash.py`: sha256 каждого файла; сводный хэш — sha256 строк «путь размер время sha256» и строк папок с их временем). «До» снято в 11:13, «после» — в 11:44: интервал покрывает свежий клон (копия `series_short`), создание корня и прогон [изм].

| Папка | Файлов | До | После |
|---|---|---|---|
| `1_Raw` | 6 | 7a74d6b6241dd6f1 | 7a74d6b6241dd6f1 |
| `3_MC` | 4 | 50a53c7a084df468 | 50a53c7a084df468 |
| `4_Estimates` | 2 | a548430dfca6feca | a548430dfca6feca |
| `5_Inspection` | 12 | 2bfa6410472badf2 | 2bfa6410472badf2 |
| `6_Export` | 8 | 51f9f269fac5da1c | 51f9f269fac5da1c |
| `_bearmind` | 18 | b574c68cb0490ad6 | b574c68cb0490ad6 |
| `Preview` | 12 | 242aa5f3b0440a25 | 242aa5f3b0440a25 |

Сравнение по файлам: добавлено 0, удалено 0, изменено 0, время папок не изменилось. Сводные хэши считаются иначе, чем в фазе 5, поэтому с её таблицей не сравниваются. Сверка по файлам: sha256 всех файлов `_bearmind`, `3_MC`, `4_Estimates`, `1_Raw`, `5_Inspection` равны файлу `docs/superpowers/specs/phase5-check/short_after.json` [изм].

Корень `phase6_e2e` оставлен для осмотра.

## 5. Полный набор тестов

Команда: `conda run -n bearmind python -m pytest -q -p no:cacheprovider tests -rs --durations=15` (без `-m`: все тесты, `slow` тоже). Код — после всех коммитов кода (HEAD b3f116b) [изм].

| Прогон | Результат | Время |
|---|---|---|
| полный, первый (11:58) | 1174 passed, 1 failed, 31 skipped, 214 warnings, код 1 | 1289 с |
| `tests/test_examinator_speed.py` отдельно | 5 passed, код 0 | 28 с |
| полный, второй (12:22) | 1175 passed, 31 skipped, 216 warnings, код 0 | 1278 с |

- Состав: 1155 быстрых тестов (1144 до задачи и 11 новых) и 20 тестов `slow`, все с синтетическими данными. 31 skipped: 22 быстрых теста без реальных сессий и 9 тестов `realdata` (`test_e2e_real` — 4, `test_equivalence` — 5). Переменная `BEARMIND_TEST_SERIES` не задана [изм].
- Падение первого прогона: `tests/test_examinator_speed.py::test_merge_time`. Строка теста: `[speed] hour merge time_s=2.587 manual_merge_s=2.083 extra_s=0.503`; предел `extra_s <= 0.5`. Отдельный прогон: `time_s=1.682 manual_merge_s=1.417 extra_s=0.265`. Фаза 5 записала `extra_s=0.088` [изм].
- Толкование [вывод]. В первом полном прогоне медленнее была и сама `manual_merge` CaImAn (2,08 с против 1,42 с). Значит, медленнее была вся машина в тот момент, а не код окна. Код окна в этой задаче не менялся. Тест скорости с запасом 3 мс зависит от нагрузки машины. Порог не менялся; решать, нужен ли запас, — автору.
- Предупреждения: почти все из `tests/test_auto_inspector.py` (во втором прогоне 215 из 216, числовые предупреждения автоинспекции), одно из `dash` (`PytestRemovedIn10Warning`), как в задачах 5 и 6.
- `~\.bearmind\machine.yaml` после обоих прогонов не изменился (время 2026-11-08 16:16).

## 6. Коммиты

| Коммит | Что |
|---|---|
| 072e159 | env: все импортируемые пакеты закреплены по lock, переменная `OPENBLAS_CORETYPE`, канал `defaults`, шапка lock, `test_env.py` для окружения `bearmind`, `install2_pip.bat`, удалён `bearmind2025.yml`, `.gitignore`, тесты окружения |
| 57269af | файлы запуска: папка `.conda\envs`, запасной путь `conda run` в обоих, исправлена ошибка `CONDA_BAT`; инструкция; `tests/test_launchers.py` |
| 2d3f220 | тест: строки реконструкций в экспорте после сжатия |
| b3f116b | инструкция и `environment.yml`: строгий приоритет каналов и обход |
| коммит с этим отчётом | `docs: phase 6 check - env files, fresh install and full test run` |

Ничего не отправлено (`git push` не выполнялся). Подписей ИИ в коммитах нет (`git log --format='%(trailers)'` пуст).
