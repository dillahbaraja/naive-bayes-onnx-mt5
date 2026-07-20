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

- `src/mql5/NaiveBayesTrader.mq5` - MetaTrader 5 Expert Advisor.
- `scripts/python/` - training, export, optimization, and evaluation scripts.
- `data/h1/` - hourly training data for EURUSD, EURJPY, and USDJPY.
- `artifacts/` - exported ONNX models and generated CSV results.
- `results/figures/` - comparison charts and visual outputs.
- `Backtest/` - backtest transaction histories and MT5 reports grouped by model type.

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

Run one of the Python scripts from the repository root, for example:

```bash
python scripts/python/train_model.py
```

This reads the corresponding H1 CSV data, trains a Naive Bayes pipeline, and exports an ONNX model.

### 2. Evaluate model variants

Use the evaluation script to compare model performance:

```bash
python scripts/python/export_all_models.py
python scripts/python/test_static_params.py
python scripts/python/evaluate_table1.py
```

### 3. Install the EA in MetaTrader 5

1. Copy `NaiveBayesTrader.mq5` into the MT5 `MQL5/Experts` folder.
2. Copy the required `.onnx` model files from `artifacts/` into the appropriate MT5 ONNX directory or the folder expected by your EA setup.
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
4. Load the desired model in MT5 through `src/mql5/NaiveBayesTrader.mq5`.
5. Adjust presets or manual risk parameters if needed.

## Notes

- The project is designed for forex pairs only.
- The EA uses indicator features such as RSI, ATR, MACD, and hour-of-day context.
- The repository keeps model artifacts and backtest outputs so results can be reproduced and reviewed.
