# Даунсемплинг CNMF-E (`tsub`) и пороги `min_corr` / `min_pnr`: что говорят авторы CaImAn

Сбор источников 2026-11-12 по вопросу автора: «если подбирать пороги по каждому 5-му кадру и факторизацию делать так же, всё ли в порядке?»

## Вывод
1. `tsub` в CNMF-E действует только на инициализацию, то есть ровно на тот шаг, где применяются `min_corr` и `min_pnr`. Найденные клетки потом уточняются на всех кадрах.
2. CaImAn не берёт каждый k-й кадр, а усредняет блоки по `tsub` кадров. `min_pnr` он сам умножает на √tsub, а `min_corr` не меняет.
3. Карты, построенные с тем же `tsub` и тем же способом (среднее блоков), совпадают с тем, что видит инициализация. Наш `bm_qc.compute_init_maps(tsub=k)` делает именно это.
4. В демо-ноутбуке CaImAn карты для подбора порогов строятся по каждому k-му кадру (около 1000 кадров), а сама инициализация идёт с `tsub=2`. Мейнтейнер признал, что `min_corr`, подобранный так, может оказаться слишком низким. Он советует строить карты по даунсемплированным данным.
5. Количественных данных о том, как `tsub` или длина записи сдвигают пороги и число клеток, авторы не публиковали.

## Код CaImAn 1.13.2 (установлен в env bearmind)
- `caiman/source_extraction/cnmf/initialization.py:198`: `tsub: temporal downsampling factor recommended for long datasets (default 1, no downsampling)` (docstring `initialize_components`).
- `initialization.py:315-324`: `Y_ds = downscale(Y, (ssub, ssub, tsub))`; `downscale` (`:73-112`) — среднее по блокам (`block_reduce(..., np.nanmean)`).
- `initialization.py:1208-1211`: `init_neurons_corr_pnr(Y_ds, ..., min_corr=min_corr, min_pnr=min_pnr * np.sqrt(np.size(Y) / np.size(Y_ds)), ...)` — `min_pnr` масштабируется на √(tsub·ssub²), `min_corr` нет.
- `params.py` (docstring `CNMFParams.init`): `min_corr` по умолчанию 0.85, `min_pnr` 20, `ssub` 2, `tsub` 2. В документации Getting Started по умолчанию стоит 1 — источники расходятся. В конвейере `tsub` задаётся явно (сейчас 1).

## Документация и демо
- Демо `demos/notebooks/demo_pipeline_cnmfE.ipynb` (https://raw.githubusercontent.com/flatironinstitute/CaImAn/main/demos/notebooks/demo_pipeline_cnmfE.ipynb):
  - `tsub = 2  # downsampling factor in time for initialization, increase if you have memory problems`;
  - карты для подбора: `cm.summary_images.correlation_pnr(images[::max(T//1000, 1)], gSig=gsig_tmp[0], swap_dim=False)  # subsample if needed` — каждый k-й кадр, около 1000 кадров, `gSig` 3;
  - «For this data it will be at a correlation value lower bound between 0.8 and 0.9, and a pnr lower bound somewhere between 10 and 20»; «it is better to have false positives later than false negatives».
- Getting Started (https://caiman.readthedocs.io/en/latest/Getting_Started.html): «Spatial and temporal subsampling during initialization. Defaults to 1 (no compression).» «Can be set to 2 or even higher to save resources, but might impair detection/extraction quality.»

## Issue на GitHub
- #815 (https://github.com/flatironinstitute/CaImAn/issues/815#issuecomment-704539193), j-friedrich, 6 окт 2020: «the threshold for `min_corr` determined using `caiman.utils.visualization.nb_inspect_correlation_pnr` might be too low» / «because you use downscaling during the initialization (`tsub>1`, `ssub>1`).» / «Whereas `min_pnr` is automatically adjusted … `min_corr` is not.» / «Consider using `caiman.utils.visualization.nb_inspect_correlation_pnr` on the downscaled data to help setting those thresholds.»
- #1158 (https://github.com/flatironinstitute/CaImAn/issues/1158), август 2023, тот же вопрос. Мейнтейнер EricThomson: «whatever @j-friedrich said then still holds. I will need to go through and look at this and get back to you with a clear answer.» Окончательного ответа в теме нет, issue открыт.

## Статьи (текст из Europe PMC; метаданные оттуда же)
- Zhou P, Resendez SL, Rodriguez-Romaguera J, Jimenez JC, Neufeld SQ, Giovannucci A, Friedrich J, Pnevmatikakis EA, Stuber GD, Hen R, Kheirbek MA, Sabatini BL, Kass RE, Paninski L. Efficient and accurate extraction of in vivo calcium signals from microendoscopic video data. eLife 7:e28728 (2018). doi:10.7554/eLife.28728.
  - «In practice, we also use spatial and temporal decimation for improved speed, following (Friedrich et al., 2017a).» Сначала весь алгоритм идёт на прореженных данных, потом A и C поднимаются до полного разрешения, потом одна итерация на сырых данных.
  - «downsampling increases the signal-to-noise ratio and eliminates many false positives.»
  - Пороги P_min (по умолчанию 10) и L_min (0.8): «we first compute the correlation image and PNR image and then visually select very weak neurons from these two images.» «Small P_min and L_min yield more false positive neurons, but they can be removed in the intervention step.»
  - По данным пересказа (дословно не проверено): гиппокамп прорежен по времени ×2, пороги для гиппокампа 0.9 и 15.
- Giovannucci A, Friedrich J, Gunn P, Kalfon J, Brown BL, Koay SA, Taxidis J, Najafi F, Gauthier JL, Zhou P, Khakh BS, Tank DW, Chklovskii DB, Pnevmatikakis EA. CaImAn an open source tool for scalable calcium imaging data analysis. eLife 8:e38173 (2019). doi:10.7554/eLife.38173.
  - Затравки берутся из локальных максимумов «an image formed by the pointwise product between the correlation image and the peak signal-to-noise ratio image».
  - «optimal parameter setting for CaImAn online can also depend on the length of the experiment» — для более длинных записей подходят более строгие параметры.

## Оригинальный CNMF_E (MATLAB, https://github.com/zhoupc/CNMF_E)
- Демо: `ssub = 1; % spatial downsampling factor`, `tsub = 1; % temporal downsampling factor`. Раздел `%% downsample data for fast and better initialization`, затем `%% apply results to the full resolution`.
- Пороги в демо: `min_corr = 0.8`, `min_pnr = 8`. Карты строятся по всем кадрам (`frame_range = []`).
