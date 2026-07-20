import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Membaca data...")
    df = pd.read_csv("EURUSD_H1_Data.csv")
    df['Time'] = pd.to_datetime(df['Time'], format="%Y.%m.%d %H:%M")
    df.set_index('Time', inplace=True)
    
    # Target (5-bar lookahead)
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
    
    # Train/Test Split (Chronological - Last 1 Year for testing)
    cutoff_date = df.index[-1] - pd.DateOffset(years=1)
    X_train = X[X.index < cutoff_date]
    y_train = y[y.index < cutoff_date]
    X_test = X[X.index >= cutoff_date]
    y_test = y[y.index >= cutoff_date]
    
    # List to store metrics
    results = []
    
    # ----------------------------------------------------
    # Model A: Gaussian Naive Bayes (Raw Data)
    # ----------------------------------------------------
    print("\n[Model A] Melatih & Mengevaluasi Gaussian...")
    model_a = GaussianNB()
    model_a.fit(X_train, y_train)
    
    # Evaluasi Train & Test
    y_pred_train_a = model_a.predict(X_train)
    y_pred_test_a = model_a.predict(X_test)
    
    results.append({
        'Model': 'Model A: Gaussian (Raw)',
        'Train_Accuracy': accuracy_score(y_train, y_pred_train_a),
        'Train_F1_Weighted': f1_score(y_train, y_pred_train_a, average='weighted'),
        'Test_Accuracy': accuracy_score(y_test, y_pred_test_a),
        'Test_F1_Weighted': f1_score(y_test, y_pred_test_a, average='weighted')
    })
    
    # Export model (dilatih pada seluruh data untuk produksi)
    print("[Model A] Mengekspor ke ONNX...")
    model_a_full = GaussianNB()
    model_a_full.fit(X, y)
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onnx_a = skl2onnx.convert_sklearn(model_a_full, initial_types=initial_type, 
                                      options={'zipmap': False}, target_opset=12)
    with open("nb_model_a_gaussian.onnx", "wb") as f:
        f.write(onnx_a.SerializeToString())
        
    # ----------------------------------------------------
    # Model B: Static/Uniform Thresholds
    # ----------------------------------------------------
    print("\n[Model B] Melatih & Mengevaluasi Statis...")
    pipeline_b = Pipeline([
        ('discretizer', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')),
        ('nb', BernoulliNB())
    ])
    pipeline_b.fit(X_train, y_train)
    
    y_pred_train_b = pipeline_b.predict(X_train)
    y_pred_test_b = pipeline_b.predict(X_test)
    
    results.append({
        'Model': 'Model B: Statis (Manual Bins)',
        'Train_Accuracy': accuracy_score(y_train, y_pred_train_b),
        'Train_F1_Weighted': f1_score(y_train, y_pred_train_b, average='weighted'),
        'Test_Accuracy': accuracy_score(y_test, y_pred_test_b),
        'Test_F1_Weighted': f1_score(y_test, y_pred_test_b, average='weighted')
    })
    
    # Export model
    print("[Model B] Mengekspor ke ONNX...")
    pipeline_b_full = Pipeline([
        ('discretizer', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')),
        ('nb', BernoulliNB())
    ])
    pipeline_b_full.fit(X, y)
    onnx_b = skl2onnx.convert_sklearn(pipeline_b_full, initial_types=initial_type, 
                                      options={'zipmap': False}, target_opset=12)
    with open("nb_model_b_static.onnx", "wb") as f:
        f.write(onnx_b.SerializeToString())

    # ----------------------------------------------------
    # Model C: CPDA (Quantile Discretization)
    # ----------------------------------------------------
    print("\n[Model C] Melatih & Mengevaluasi CPDA (Quantile)...")
    pipeline_c = Pipeline([
        ('discretizer', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')),
        ('nb', BernoulliNB())
    ])
    pipeline_c.fit(X_train, y_train)
    
    y_pred_train_c = pipeline_c.predict(X_train)
    y_pred_test_c = pipeline_c.predict(X_test)
    
    results.append({
        'Model': 'Model C: CPDA (Kuantil Dinamis)',
        'Train_Accuracy': accuracy_score(y_train, y_pred_train_c),
        'Train_F1_Weighted': f1_score(y_train, y_pred_train_c, average='weighted'),
        'Test_Accuracy': accuracy_score(y_test, y_pred_test_c),
        'Test_F1_Weighted': f1_score(y_test, y_pred_test_c, average='weighted')
    })
    
    # Export model
    print("[Model C] Mengekspor ke ONNX...")
    pipeline_c_full = Pipeline([
        ('discretizer', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')),
        ('nb', BernoulliNB())
    ])
    pipeline_c_full.fit(X, y)
    onnx_c = skl2onnx.convert_sklearn(pipeline_c_full, initial_types=initial_type, 
                                      options={'zipmap': False}, target_opset=12)
    with open("nb_model_c_cpda.onnx", "wb") as f:
        f.write(onnx_c.SerializeToString())

    # ----------------------------------------------------
    # Menyimpan & Menampilkan Hasil
    # ----------------------------------------------------
    results_df = pd.DataFrame(results)
    csv_filename = "model_evaluation_results.csv"
    results_df.to_csv(csv_filename, index=False)
    
    print("\n================ HASIL EVALUASI TRAINING ================")
    print(results_df.to_string(index=False))
    print("=========================================================")
    print(f"\nHasil evaluasi sukses disimpan ke file '{csv_filename}'")
    print("Semua model ONNX telah berhasil diperbarui!")

if __name__ == "__main__":
    main()
