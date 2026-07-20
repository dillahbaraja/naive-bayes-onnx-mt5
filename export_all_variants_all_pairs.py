import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import os
import warnings

warnings.filterwarnings('ignore')

def train_and_export(csv_file, pair_name):
    # Resolve absolute path for files relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_file)
    
    if not os.path.exists(csv_path):
        print(f"File {csv_path} tidak ditemukan, melewati...")
        return
        
    print(f"\n================ Memproses Pair: {pair_name} ================")
    print(f"Membaca {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Time'] = pd.to_datetime(df['Time'], format="%Y.%m.%d %H:%M")
    df.set_index('Time', inplace=True)
    
    # Target (5-bar lookahead)
    # Dynamic target based on volatility (1.5 * ATR)
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
    
    # --- QUANT OPTIMIZATION: WINSORIZATION (OUTLIER CLIPPING) ---
    # Extreme volatility spikes (like BOJ interventions in JPY) stretch the feature ranges.
    # For strategy='uniform' (Model B), this collapses all normal data into a single bin.
    # By clipping the features at 1% and 99% quantiles, we protect the bin resolution and
    # dramatically improve the model's predictive power for JPY pairs.
    for col in ['ATR', 'MACD_Main']:
        lower_limit = df[col].quantile(0.01)
        upper_limit = df[col].quantile(0.99)
        df[col] = df[col].clip(lower_limit, upper_limit)
        print(f"Clipped {col} for {pair_name} to range: [{lower_limit:.5f}, {upper_limit:.5f}]")
    
    X = df[features]
    y = df['Target']
    
    print("Distribusi Target Asli:")
    print(y.value_counts(normalize=True))
    
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    
    # ----------------------------------------------------
    # 1. Model A: Gaussian Naive Bayes (Raw Data)
    # ----------------------------------------------------
    print(f"[{pair_name}] Melatih & Mengekspor Model A (Gaussian)...")
    model_a = GaussianNB()
    model_a.fit(X, y)
    
    onnx_a = skl2onnx.convert_sklearn(model_a, initial_types=initial_type, 
                                      options={'zipmap': False}, target_opset=12)
    output_a = os.path.join(script_dir, f"gaussian_{pair_name.lower()}.onnx")
    with open(output_a, "wb") as f:
        f.write(onnx_a.SerializeToString())
        
    # ----------------------------------------------------
    # 2. Model B: Static/Uniform Discretization
    # ----------------------------------------------------
    print(f"[{pair_name}] Melatih & Mengekspor Model B (Statis)...")
    pipeline_b = Pipeline([
        ('discretizer', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')),
        ('nb', BernoulliNB())
    ])
    pipeline_b.fit(X, y)
    
    onnx_b = skl2onnx.convert_sklearn(pipeline_b, initial_types=initial_type, 
                                      options={'zipmap': False}, target_opset=12)
    output_b = os.path.join(script_dir, f"static_{pair_name.lower()}.onnx")
    with open(output_b, "wb") as f:
        f.write(onnx_b.SerializeToString())

    # ----------------------------------------------------
    # 3. Model C: CPDA (Quantile Discretization)
    # ----------------------------------------------------
    print(f"[{pair_name}] Melatih & Mengekspor Model C (CPDA)...")
    pipeline_c = Pipeline([
        ('discretizer', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')),
        ('nb', BernoulliNB())
    ])
    pipeline_c.fit(X, y)
    
    onnx_c = skl2onnx.convert_sklearn(pipeline_c, initial_types=initial_type, 
                                      options={'zipmap': False}, target_opset=12)
    output_c = os.path.join(script_dir, f"cpda_{pair_name.lower()}.onnx")
    with open(output_c, "wb") as f:
        f.write(onnx_c.SerializeToString())
        
    print(f"Sukses mengekspor semua varian untuk {pair_name}!")

def main():
    train_and_export("EURUSD_H1_Data.csv", "EURUSD")
    train_and_export("USDJPY_H1_Data.csv", "USDJPY")
    train_and_export("EURJPY_H1_Data.csv", "EURJPY")
    print("\n[SUKSES] Semua varian model (Natural Prior + Winsorized) berhasil dibuat!")

if __name__ == "__main__":
    main()
