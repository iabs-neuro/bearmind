# BEARMiND
### A pipeline for Batch Examination & Analysis of Raw Miniscopic Neural Data
<img src="/bearmind_logo.png" align="left">

BEARMiND batch-processes miniscope calcium-imaging series with CaImAn: crop, motion correction, CNMF-E, manual and automatic inspection, export.
A PySide6 window shows the batch, asks for CNMF-E thresholds and for the manual-inspection decision; the batch runs on its own.
The documentation below is in Russian, for the lab's students; run `run_bearmind.bat --help` for the commands.

BEARMiND обрабатывает записи минископа целыми сериями. Пакет `bearmind` ведёт каждую сессию серии через этапы: кроп, коррекция движения (NoRMCorre), пороги CNMF-E, CNMF-E (CaImAn), ручная инспекция, автоинспекция, экспорт. Экспорт пишет трейсы в `npz` и контуры клеток в `mat` для сопоставления клеток между днями (CellReg).

Вы работаете в окне BEARMiND: выбираете пороги и решаете, нужна ли сессии ручная инспекция. Остальное делает батч — отдельный процесс, который идёт без окна. Автоинспекция убирает плохие компоненты моделью машинного обучения и использует пакет [DRIADA](https://github.com/iabs-neuro/driada).

</br>

## Инструкция для студентов

- [Установка и первый запуск](docs/students/README.md): окружение `bearmind`, проверка установки, что делать, если окружение не найдено.
- [Работа с серией](docs/students/series-workflow.md): папка серии, вкладки Monitor, Scan, Thresholds, Inspection, сводка и что делать при сбоях.

## Коротко

1. Создайте окружение по файлам `environment.yml` и `requirements-pip.txt`.
2. Положите сырые видео в папку `1_Raw` папки серии.
3. Запустите окно двойным щелчком по `bearmind_gui.bat`.
4. Откройте серию и найдите сессии на вкладке Scan.
5. Нажмите Start на вкладке Monitor.
6. Выберите пороги CNMF-E на вкладке Thresholds: батч ждёт их у каждой сессии.
7. Решите на вкладке Inspection, нужна ли сессии ручная инспекция: батч ждёт и этого решения.

Команды без окна перечисляет `run_bearmind.bat --help`.

## Что ещё лежит в репозитории

| Папка | Что внутри |
|---|---|
| `bearmind/` | пакет батча и окна |
| `history/` | старый конвейер ноутбука и скрипты MINI3, которыми обработаны данные статьи ([history/README.md](history/README.md)) |
| `ml/` | обучение и оценка модели автоинспекции ([ml/README.md](ml/README.md)) |
| `production_models/` | рабочие модели автоинспекции |
| `docs/` | инструкции, планы и отчёты проверок |
| `experiments/`, `paper/` | исследовательские скрипты и тексты статей |

Ноутбук `BEARMiND_full_pipeline.ipynb` удалён: его заменил пакет `bearmind`. Как достать ноутбук из истории git, сказано в [history/README.md](history/README.md).

Ошибки сообщайте во вкладке *Issues* репозитория.

## References
- Pnevmatikakis, E.A. & Giovannucci, A. (2017). NoRMCorre: An online algorithm for piecewise rigid motion correction of calcium imaging data. *J. Neurosci. Methods*.
- Giovannucci, A. et al. (2019). CaImAn: An open source tool for scalable calcium imaging data analysis. *eLife*.
- Sheintuch, L. et al. (2017). Tracking the same neurons across multiple days in Ca²⁺ imaging data (CellReg). *Cell Reports*.
