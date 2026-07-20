import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]

def train_and_export(csv_file, pair_name):
    csv_path = ROOT / "data" / "h1" / csv_file
    
    if not csv_path.exists():
        print(f"File {csv_path} tidak ditemukan, melewati...")
        return
        
    print(f"\n================ Memproses Pair: {pair_name} ================")
    print(f"Membaca {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Time'] = pd.to_datetime(df['Time'], format="%Y.%m.%d %H:%M")
    df.set_index('Time', inplace=True)
    
    # Target (5-bar lookahead)
    # Dynamic target based on volatility (1.5 * ATR) to filter out noise
    lookahead = 5
    df['Future_Close'] = df['Close'].shift(-lookahead)
    
    # Calculate dynamic volatility thresholds
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
    
    # Print target distribution to check balance
    print("Distribusi Target:")
    print(y.value_counts(normalize=True))
    
    print("Melatih Model CPDA (Kuantil Dinamis)...")
    pipeline = Pipeline([
        ('discretizer', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')),
        ('nb', BernoulliNB())
    ])
    pipeline.fit(X, y)
    
    output_filename = f"cpda_{pair_name.lower()}.onnx"
    output_path = ROOT / "models" / "onnx" / output_filename
    print(f"Mengekspor ke {output_path}...")
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onnx_model = skl2onnx.convert_sklearn(pipeline, initial_types=initial_type, 
                                          options={'zipmap': False}, target_opset=12)
    
    try:
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Sukses mengekspor {output_filename}!")
    except Exception as e:
        print(f"Error menulis file {output_filename}: {str(e)}")

def main():
    train_and_export("EURUSD_H1_Data.csv", "EURUSD")
    train_and_export("USDJPY_H1_Data.csv", "USDJPY")
    train_and_export("EURJPY_H1_Data.csv", "EURJPY")
    print("\nSemua model ONNX dengan target dinamis selesai diproses!")

if __name__ == "__main__":
    main()
