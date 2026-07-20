import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, f1_score
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]

def main():
    print("Membaca data EURUSD_H1_Data.csv...")
    df = pd.read_csv(ROOT / "data" / "h1" / "EURUSD_H1_Data.csv")
    df['Time'] = pd.to_datetime(df['Time'], format="%Y.%m.%d %H:%M")
    df.set_index('Time', inplace=True)
    
    # Target definition (5-bar lookahead)
    threshold = 0.0005
    lookahead = 5
    df['Future_Close'] = df['Close'].shift(-lookahead)
    conditions = [
        (df['Future_Close'] > df['Close'] + threshold),
        (df['Future_Close'] < df['Close'] - threshold)
    ]
    df['Target'] = np.select(conditions, [1, 2], default=0)
    
    features = ['RSI', 'ATR', 'MACD_Main', 'Hour']
    df.dropna(subset=['Future_Close'] + features, inplace=True)
    
    X = df[features]
    y = df['Target']
    
    # Train/Test Split (Last 1 year for test)
    cutoff_date = df.index[-1] - pd.DateOffset(years=1)
    X_train = X[X.index < cutoff_date]
    y_train = y[y.index < cutoff_date]
    X_test = X[X.index >= cutoff_date]
    y_test = y[y.index >= cutoff_date]
    
    print("Mengevaluasi Model A: Gaussian Naive Bayes (Raw Data)...")
    # Model A: Raw continuous data using GaussianNB
    model_a = GaussianNB()
    model_a.fit(X_train, y_train)
    y_pred_a = model_a.predict(X_test)
    acc_a = accuracy_score(y_test, y_pred_a)
    f1_a = f1_score(y_test, y_pred_a, average='weighted')
    
    print("Mengevaluasi Model B: Static Thresholds (Manual)...")
    # Model B: Static thresholds
    def apply_static_thresholds(data):
        df_static = pd.DataFrame(index=data.index)
        # RSI: > 70 is 2, < 30 is 0, else 1
        df_static['RSI_State'] = np.where(data['RSI'] > 70, 2, np.where(data['RSI'] < 30, 0, 1))
        # ATR: above median of training vs below
        atr_median = X_train['ATR'].median()
        df_static['ATR_State'] = np.where(data['ATR'] > atr_median, 1, 0)
        # MACD: > 0 is 1, < 0 is 0
        df_static['MACD_State'] = np.where(data['MACD_Main'] > 0, 1, 0)
        # Hour: 3 sessions
        df_static['Hour_State'] = np.where(data['Hour'] < 8, 0, np.where(data['Hour'] < 16, 1, 2))
        return df_static
    
    X_train_static = apply_static_thresholds(X_train)
    X_test_static = apply_static_thresholds(X_test)
    
    # We use BernoulliNB but data must be one-hot. For simplicity, we can use GaussianNB or one-hot encode.
    # We will one-hot encode the static states to feed into BernoulliNB for proper categorical treatment.
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_static_ohe = ohe.fit_transform(X_train_static)
    X_test_static_ohe = ohe.transform(X_test_static)
    
    model_b = BernoulliNB()
    model_b.fit(X_train_static_ohe, y_train)
    y_pred_b = model_b.predict(X_test_static_ohe)
    acc_b = accuracy_score(y_test, y_pred_b)
    f1_b = f1_score(y_test, y_pred_b, average='weighted')
    
    print("Mengevaluasi Model C: CPDA (Quantile Discretization)...")
    # Model C: CPDA
    discretizer = KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')
    X_train_cpda = discretizer.fit_transform(X_train)
    X_test_cpda = discretizer.transform(X_test)
    
    model_c = BernoulliNB()
    model_c.fit(X_train_cpda, y_train)
    y_pred_c = model_c.predict(X_test_cpda)
    acc_c = accuracy_score(y_test, y_pred_c)
    f1_c = f1_score(y_test, y_pred_c, average='weighted')
    
    print("\n================ HASIL EVALUASI TABEL 1 ================\n")
    print(f"| Model | Akurasi | F1-Score (Weighted) |")
    print(f"|-------|---------|---------------------|")
    print(f"| Model A: Gaussian (Raw) | {acc_a*100:.2f}% | {f1_a*100:.2f}% |")
    print(f"| Model B: Statis (Manual) | {acc_b*100:.2f}% | {f1_b*100:.2f}% |")
    print(f"| Model C: CPDA (Kuantil) | {acc_c*100:.2f}% | {f1_c*100:.2f}% |")
    print("\n========================================================")

if __name__ == "__main__":
    main()
