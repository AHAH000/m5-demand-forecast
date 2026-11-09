# M5 Walmart Demand Forecasting

Forecasting 28-day daily unit sales for Walmart products using the M5 dataset (Kaggle).
Scope: a subset (one store or one category) to keep it fast and focused.

## Result
_TODO: one-line result + best chart here (put this at the top)._

## Data
Kaggle competition: `m5-forecasting-accuracy`. Files used: `sales_train_evaluation.csv`,
`calendar.csv`, `sell_prices.csv`. Raw data is not committed (Kaggle terms). Run
`python scripts/download_data.py` after setting up your Kaggle API key and accepting the
competition rules on Kaggle.

## Approach
1. EDA: hierarchy, intermittency (share of zero-sales days), weekly/yearly seasonality, events, SNAP days, price effects
2. Baselines: naive, seasonal naive (7 days), moving average
3. Classical / statistical per-series models (ETS, Croston/TSB for intermittent items)
4. Global LightGBM model on lag, rolling, calendar, event, and price features
5. Rolling-origin backtest with 28-day horizon (RMSSE, MAE, bias)
6. Business view: error by category / store / intermittency class

## Results
_TODO: table of model by category._

## Limitations
- Subset only, not the full 30k series
- No hierarchical reconciliation (unless added)
- Evaluation uses RMSSE per series, not the official weighted WRMSSE

## How to run
```bash
pip install -r requirements.txt
python scripts/download_data.py
jupyter lab
```
