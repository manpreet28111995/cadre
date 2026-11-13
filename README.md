# Illicit Bitcoin transaction detection under delayed labels

Code for the experiments on the Elliptic++ transaction stream: a leakage audit of the wallet features, static baselines, retraining policies under label delay, drift monitors, ablations, significance tests, explanations and cost measurements. Every table, figure and number in the manuscript is produced by the scripts in `code/`.

## Setup

Python 3.13 and the pinned packages in `requirements.txt`:

    pip install -r requirements.txt

All experiments ran on two CPU cores without a GPU. The tree learners are fixed to two threads, so keep that setting if you want to match the reported numbers.

## Data

Elliptic++ (Elmougy and Liu, KDD 2023) is downloaded with

    python code/download.py

into `data/ellipticpp`. Set `ELLPP_DIR` to use another location. The dataset is not redistributed here, and its authors state no explicit licence.

## Running everything

`code/reproduce.sh` executes the full pipeline in order. It takes many hours on two cores, mostly because of the streaming runs. The stages are:

| Stage | Scripts | Output |
|---|---|---|
| Features | `prep.py`, `delay_features.py 1 2 3 5` | `cache/base.npz`, `cache/R_delay.npz` |
| Audit and splits | `audit.py`, `split_stats.py`, `protocol_inflation.py` | `results/audit_*.csv`, `split_stats.csv`, `protocol_inflation.csv` |
| Static baselines | `run_all_tabular.sh`, `run_all_neural.sh`, `evaluate.py`, `stats.py`, `late_analysis.py` | `results/metrics_*.csv`, `cost.csv`, `significance.csv`, `late_period_ranking.csv`, `preds/` |
| Streaming experiments | `run_stream.sh`, `run_stream2.sh`, `run_stream3.sh` (call `stream.py`) | `results/stream/*.npz` |
| Aggregation and tests | `stream_eval.py`, `stream_stats.py` | `results/stream_*.csv` |
| Explanation and cost | `explain.py`, `cost_stream.py` | `results/shap_*.csv`, `stream_cost.csv` |
| Text numbers and tables | `make_numbers.py`, `make_numbers2.py`, `make_tables2.py` | `numbers.tex`, `numbers2.tex`, `tables/*.tex` |
| Figures | `fig2_data_audit.py` to `fig6_curves.py`; `figtex/fig1_framework.tex` for Figure 1 | `figures/*.png` |

Run the streaming jobs one at a time. Figure 1 is compiled with `pdflatex` and rasterised with `pdftoppm -r 500 -png -singlefile`.

A single streaming configuration can be run directly, for example

    python code/stream.py --model LGBM --fs TXSR --policy dpm --d 1 --seeds 0-4

where `--policy` is one of `static`, `every`, `slide`, `per3`, `per5`, `psi`, `c2st`, `dpm`, `oracle`, `--fs` selects the feature set (`TX`, `TXS`, `TXR`, `TXSR`) and `--d` is the label delay.

## Seeds

Stream experiments use seeds 0 to 4. The static tree baselines use seeds 0 to 9, the graph models seeds 0 to 4, and the XGBoost, Random Forest and rolling-origin runs seeds 0 to 2. Splits are fixed in `code/config.py`: training steps 1 to 29, validation 30 to 34, test 35 to 49.

## Reference results

`results/` holds the summary tables and the per-run settings of the reported experiments, and `figures/` the figures. The large per-run prediction archives are not included; rerunning the stream scripts recreates them.

Selected values to check after a rerun (LightGBM on transaction features plus causal context, fixed cutoff, label delay 1):

| Quantity | Value |
|---|---|
| F1 of LightGBM, random split / chronological split | 0.959 / 0.794 |
| F1 on steps 43 to 49, static model | 0.096 |
| F1 on steps 43 to 49, retraining at every step | 0.375 |
| F1 on steps 43 to 49, delayed-performance trigger | 0.370 |
| Trainings over the test period, every step / trigger | 15 / 7 |

## Reproducibility notes

Rerunning the feature construction gives identical arrays. Rerunning streaming configurations gives identical alarms, retraining decisions and cutoffs; predicted probabilities can differ in the sixth decimal place between runs because of floating-point reduction order in the tree learners, which does not change any value reported at the precision of the paper. Training and inference times depend on the machine and are not expected to match.
