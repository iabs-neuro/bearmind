# BEARMiND: установка и первый запуск

Эта инструкция для студентов лаборатории. Она описывает, как поставить BEARMiND на компьютер с Windows и проверить установку. Работа с серией записей описана в файле [series-workflow.md](series-workflow.md).

BEARMiND обрабатывает записи минископа: кроп, коррекция движения (MC), CNMF-E, ручная и автоматическая инспекция, экспорт. Вы работаете в окне BEARMiND. Тяжёлую работу делает батч — отдельный процесс, который идёт без окна.

## Содержание

1. [Что нужно на компьютере](#1-что-нужно-на-компьютере)
2. [Как создать окружение `bearmind`](#2-как-создать-окружение-bearmind)
3. [Как проверить установку](#3-как-проверить-установку)
4. [Окружение не найдено: что делать](#4-окружение-не-найдено-что-делать)

## 1. Что нужно на компьютере

- Windows 10 или новее.
- Anaconda, Miniconda или Miniforge. Файлы запуска ищут окружение в шести папках:
  `%USERPROFILE%\anaconda3\envs\bearmind`, `%USERPROFILE%\miniconda3\envs\bearmind`,
  `%ProgramData%\anaconda3\envs\bearmind`, `%ProgramData%\miniconda3\envs\bearmind`,
  `%USERPROFILE%\.conda\envs\bearmind`, `%USERPROFILE%\miniforge3\envs\bearmind`.
  Если окружение лежит в другой папке, читайте раздел 4.
- Git for Windows: им вы скачаете репозиторий.
- Интернет: окружение скачивает пакеты.
- Много оперативной памяти. Пример с машины лаборатории: CNMF-E клипа из 2000 кадров с пулом из 31 процесса занял около 29 ГБ. Это не требование для всех машин. Длинные сессии требуют больше.
- Свободное место на диске. Временная папка одной сессии может занимать до 160 ГБ. По умолчанию она лежит на системном диске C:.
- Браузер (Chrome, Edge или Firefox): в нём открывается ручная инспекция.

## 2. Как создать окружение `bearmind`

Окружение — это отдельный Python со всеми пакетами BEARMiND. Его список пакетов записан в двух файлах репозитория: `environment.yml` (пакеты conda) и `requirements-pip.txt` (пакеты pip).

Окружение `bearmind` может уже быть на компьютере. Старая версия BEARMiND создаёт окружение с тем же именем, но с другими версиями Python и CaImAn. Поэтому найденное окружение проверьте в шагах 9–11. Скрипт `test_env.py` импортирует все пакеты BEARMiND и сверяет версии Python и CaImAn.

Новая Anaconda или Miniconda в шаге 14 может спросить, принимаете ли вы условия каналов Anaconda (Terms of Service). Окружению нужен канал Anaconda: из него приходит PyTorch. На этот вопрос введите букву a (accept) и нажмите Enter. Если вы отказались, выполните `conda tos accept` и повторите шаг 14.

Если conda в шаге 14 пишет excluded by strict repo priority, в вашей conda включён строгий приоритет каналов. С ним окружение не собирается. Выполните в той же Anaconda Prompt команду ниже и повторите шаг 14:

```
set CONDA_CHANNEL_PRIORITY=flexible
```

1. Откройте Anaconda Prompt из меню «Пуск».
2. Создайте папку для репозитория: `mkdir C:\Work`. Подойдёт и любая другая ваша папка.
3. Перейдите в эту папку: `cd C:\Work`.
4. Скачайте репозиторий: `git clone -b bearmind-batch https://github.com/iabs-neuro/bearmind`.
5. Перейдите в папку репозитория: `cd bearmind`.
6. Проверьте, что в папке есть файлы `environment.yml` и `requirements-pip.txt`.
7. Выполните `conda env list`.
8. Если в списке нет строки `bearmind`, перейдите к шагу 14.
9. Включите окружение: `conda activate bearmind`.
10. Проверьте окружение: `python test_env.py`.
11. Если строка `Results: …` кончается на `0 FAILED`, окружение готово. Перейдите к разделу 3.
12. Иначе выключите окружение: `conda deactivate`.
13. Удалите окружение: `conda env remove -n bearmind`. Если conda спросит подтверждение, введите y и нажмите Enter.
14. Создайте окружение: `conda env create -f environment.yml`. Это занимает 5–15 минут.
15. Включите окружение: `conda activate bearmind`.
16. Поставьте пакеты pip: `pip install -r requirements-pip.txt`.

Если в Anaconda стоит mamba, в шаге 14 пишите `mamba` вместо `conda`. Так окружение собирается быстрее.

## 3. Как проверить установку

Сначала проверьте окружение, потом командную строку, потом окно.

1. Откройте Anaconda Prompt.
2. Перейдите в папку репозитория: `cd C:\Work\bearmind`.
3. Включите окружение: `conda activate bearmind`.
4. Выполните `python test_env.py`.
5. Проверьте, что строка `Results: …` кончается на `0 FAILED`.
6. Выполните `run_bearmind.bat --help`.
7. Проверьте, что программа напечатала список команд: `scan`, `run`, `gui`, `status` и другие.
8. Откройте папку репозитория в Проводнике.
9. Дважды щёлкните файл `bearmind_gui.bat`.
10. Проверьте, что открылось окно BEARMiND с вкладками **Monitor**, **Scan**, **Thresholds**, **Inspection**.

Строка `[FAILED]` в выводе шага 4 называет пакет, который не импортируется, или версию, которая не подходит. Чаще всего это пакет pip, например `driada`, `interpret` или `ssqueezepy`. Тогда повторите шаг 16 раздела 2, потом шаг 4. Если строка `[FAILED]` осталась, покажите вывод руководителю.

Начало вывода шага 6:

```
usage: run_bearmind [-h] command ...

Batch pipeline for calcium-imaging series. Exit codes: 0 ok, 1 partial
failure, 2 bad input, 3 series busy, 130 interrupted.

positional arguments:
  command
    scan        find sessions in the series folder and register them
    run         process the series (resumes where it stopped)
```

Пустое окно пишет **Open a series: Series > Open series…**. Это нормально: серию вы откроете позже.

Первый запуск окна создаёт файл настроек машины `~/.bearmind/machine.yaml`. В нём временная папка, число ядер и память. Без нужды файл не правьте.

Установка готова. Дальше читайте [series-workflow.md](series-workflow.md).

## 4. Окружение не найдено: что делать

Файлы запуска ищут окружение в шести папках раздела 1. Если окружения там нет, они ищут его через conda. Файлы пишут `[FAILED] conda env "bearmind" not found.`, если не нашли ни окружение, ни conda. Причин две: окружение не создано, или окружение и conda лежат в других папках.

1. Откройте Anaconda Prompt.
2. Выполните `conda env list`.
3. Найдите в списке строку `bearmind`.
4. Если строки нет, создайте окружение по разделу 2.
5. Если строка есть, запомните путь окружения в этой строке.

Если путь не из раздела 1, файлы запуска находят окружение через conda. Двойной щелчок по `bearmind_gui.bat` находит conda, только если Anaconda стоит в `%USERPROFILE%\anaconda3`. Иначе запускайте окно из Anaconda Prompt.

Miniforge: окружение в папке `%USERPROFILE%\miniforge3\envs\bearmind` файлы запуска находят сами. Вместо Anaconda Prompt в этой инструкции открывайте Miniforge Prompt.

**Внимание.** Окно BEARMiND, запущенное через conda или из Anaconda Prompt, работает вместе с окном консоли. Не закрывайте консоль, пока работаете в окне BEARMiND: окно закроется вместе с ней.

1. Откройте Anaconda Prompt.
2. Перейдите в папку репозитория.
3. Выполните `run_bearmind.bat gui`.

В Anaconda Prompt `run_bearmind.bat` находит окружение через conda. Так же работают и другие команды, например `run_bearmind.bat status <папка серии>`.

Если окно не открылось, программа пишет причину в файл `~/.bearmind/gui.log`. Покажите руководителю этот файл и вывод `conda env list`.
