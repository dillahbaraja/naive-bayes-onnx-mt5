import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
import os

def analyze_model_probabilities(csv_file, pair_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_file)
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    df['Time'] = pd.to_datetime(df['Time'], format="%Y.%m.%d %H:%M")
    df.set_index('Time', inplace=True)
    
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
    
    pipeline_b = Pipeline([
        ('discretizer', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')),
        ('nb', BernoulliNB())
    ])
    pipeline_b.fit(X, y)
    
    probs = pipeline_b.predict_proba(X)
    preds = pipeline_b.predict(X)
    
    print(f"\n--- ANALISIS PROBABILITAS MODEL STATIC ({pair_name}) ---")
    print(f"Probabilitas rata-rata: Hold={probs[:,0].mean():.4f}, Buy={probs[:,1].mean():.4f}, Sell={probs[:,2].mean():.4f}")
    print(f"Probabilitas Maksimum: Hold={probs[:,0].max():.4f}, Buy={probs[:,1].max():.4f}, Sell={probs[:,2].max():.4f}")
    print("Jumlah Prediksi Asli Model: Hold:", (preds==0).sum(), "Buy:", (preds==1).sum(), "Sell:", (preds==2).sum())

def main():
    analyze_model_probabilities("EURUSD_H1_Data.csv", "EURUSD")
    analyze_model_probabilities("USDJPY_H1_Data.csv", "USDJPY")
    analyze_model_probabilities("EURJPY_H1_Data.csv", "EURJPY")

if __name__ == "__main__":
    main()
