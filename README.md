# naive-bayes-onnx-mt5

An MetaTrader 5 trading system built around Naive Bayes models exported to ONNX.

This repository contains:

- An MT5 Expert Advisor that loads ONNX models and manages trade entries and exits.
- Python scripts to train, evaluate, optimize, and export Naive Bayes variants.
- Historical market data, backtest outputs, and exported ONNX models for EURUSD, EURJPY, and USDJPY.

## Project Overview

The system tests multiple Naive Bayes variants for intraday forex trading:

- **Gaussian**: uses raw indicator values.
- **Static**: uses uniform discretization.
- **CPDA**: uses quantile-based discretization.

The EA can switch between model variants and includes preset-based trade management for different symbols.

## Repository Structure

- `NaiveBayesTrader.mq5` - MetaTrader 5 Expert Advisor.
- `train_model.py` - trains a Naive Bayes model and exports it to ONNX.
- `export_all_models.py` - trains and exports multiple model variants.
- `export_all_pairs.py` - exports models for multiple currency pairs.
- `export_all_variants_all_pairs.py` - exports all variants across all pairs.
- `export_all_variants_eurusd.py` - exports variants for EURUSD.
- `optimize_static_model.py` - optimization script for static model settings.
- `optimize_static_robust.py` - robustness-oriented optimization script.
- `test_static_params.py` - tests static parameter settings.
- `evaluate_table1.py` - evaluates and prepares comparison metrics.
- `inspect_static.py` - inspection helper for the static setup.
- `Backtest/` - backtest transaction histories grouped by model type.
- `*.onnx` - exported ONNX models.
- `*_H1_Data.csv` - hourly training data used by the scripts.
- `comparison_equity_curves.png` - comparison chart of equity curves.

Files related to article writing or manuscript drafting are intentionally not included in the upload set.

## Requirements

### For Python scripts

- Python 3.10 or newer
- `pandas`
- `numpy`
- `scikit-learn`
- `skl2onnx`

### For MetaTrader 5

- MetaTrader 5 terminal with ONNX support enabled
- A supported symbol: `EURUSD`, `USDJPY`, or `EURJPY`

## How to Use

### 1. Train and export models

Run one of the Python scripts, for example:

```bash
python train_model.py
```

This reads the corresponding H1 CSV data, trains a Naive Bayes pipeline, and exports an ONNX model.

### 2. Evaluate model variants

Use the evaluation script to compare model performance:

```bash
python export_all_models.py
python test_static_params.py
python evaluate_table1.py
```

### 3. Install the EA in MetaTrader 5

1. Copy `NaiveBayesTrader.mq5` into the MT5 `MQL5/Experts` folder.
2. Copy the required `.onnx` model files into the appropriate MT5 ONNX directory or the folder expected by your EA setup.
3. Compile the EA in MetaEditor.
4. Attach the EA to a supported chart timeframe.

### 4. Select model behavior

The EA supports these modes:

- `MODEL_CPDA`
- `MODEL_GAUSSIAN`
- `MODEL_STATIC`

You can either use the built-in presets or disable presets and configure the filters manually.

## Typical Workflow

1. Prepare or update the H1 CSV data.
2. Train and export ONNX models with the Python scripts.
3. Run backtests and review the files in `Backtest/`.
4. Load the desired model in MT5 through `NaiveBayesTrader.mq5`.
5. Adjust presets or manual risk parameters if needed.

## Notes

- The project is designed for forex pairs only.
- The EA uses indicator features such as RSI, ATR, MACD, and hour-of-day context.
- The repository keeps model artifacts and backtest outputs so results can be reproduced and reviewed.
