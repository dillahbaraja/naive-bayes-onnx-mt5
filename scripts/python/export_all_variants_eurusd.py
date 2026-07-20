import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]

def main():
    csv_file = ROOT / "data" / "h1" / "EURUSD_H1_Data.csv"
    if not csv_file.exists():
        print(f"File {csv_file} tidak ditemukan!")
        return
        
    print("Membaca data EURUSD untuk melatih semua varian dengan Target Dinamis...")
    df = pd.read_csv(csv_file)
    df['Time'] = pd.to_datetime(df['Time'], format="%Y.%m.%d %H:%M")
    df.set_index('Time', inplace=True)
    
    # Target (5-bar lookahead dengan ATR * 1.5)
    lookahead = 5
    df['Future_Close'] = df['Close'].shift(-lookahead)
    dynamic_threshold = df['ATR'] * 1.5
    conditions = [
        (df['Future_Close'] > df['Close'] + dynamic_threshold),
        (df['Future_Close'] < df['Close'] - dynamic_threshold)
    ]
    df['Target'] = np.select(conditions, [1, 2], default=0)
    
    features = ['RSI', 'ATR', 'MACD_Main', 'Hour']
    df.dropna(subset=['Future_Close'] + features, inplace=True)
    
    X = df[features]
    y = df['Target']
    
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    
    # 1. Model A: Gaussian Naive Bayes (Raw Data)
    print("\nMelatih & Mengekspor Model A (Gaussian)...")
    model_a = GaussianNB()
    model_a.fit(X, y)
    onnx_a = skl2onnx.convert_sklearn(model_a, initial_types=initial_type, 
                                      options={'zipmap': False}, target_opset=12)
    with open(ROOT / "models" / "onnx" / "gaussian_eurusd.onnx", "wb") as f:
        f.write(onnx_a.SerializeToString())
    print("Sukses membuat 'gaussian_eurusd.onnx'")
    
    # 2. Model B: Static/Uniform Discretization
    print("\nMelatih & Mengekspor Model B (Statis/Uniform)...")
    pipeline_b = Pipeline([
        ('discretizer', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')),
        ('nb', BernoulliNB())
    ])
    pipeline_b.fit(X, y)
    onnx_b = skl2onnx.convert_sklearn(pipeline_b, initial_types=initial_type, 
                                      options={'zipmap': False}, target_opset=12)
    with open(ROOT / "models" / "onnx" / "static_eurusd.onnx", "wb") as f:
        f.write(onnx_b.SerializeToString())
    print("Sukses membuat 'static_eurusd.onnx'")
    
    # 3. Model C: CPDA (Quantile Discretization) - Re-export for consistency
    print("\nMelatih & Mengekspor Model C (CPDA/Quantile)...")
    pipeline_c = Pipeline([
        ('discretizer', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')),
        ('nb', BernoulliNB())
    ])
    pipeline_c.fit(X, y)
    onnx_c = skl2onnx.convert_sklearn(pipeline_c, initial_types=initial_type, 
                                      options={'zipmap': False}, target_opset=12)
    with open(ROOT / "models" / "onnx" / "cpda_eurusd.onnx", "wb") as f:
        f.write(onnx_c.SerializeToString())
    print("Sukses membuat 'cpda_eurusd.onnx'")
    
    print("\nSemua varian model EURUSD dengan Target Dinamis berhasil diperbarui!")

if __name__ == "__main__":
    main()
