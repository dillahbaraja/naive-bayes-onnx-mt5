import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    # Load data
    df = pd.read_csv("EURUSD_H1_Data.csv")
    
    # Parse time
    # Format appears to be "2020.10.06 20:00"
    df['Time'] = pd.to_datetime(df['Time'], format="%Y.%m.%d %H:%M")
    df.set_index('Time', inplace=True)
    
    print(f"Data spans from {df.index[0]} to {df.index[-1]}")
    
    # Feature Selection
    # Using independent features as discussed: Momentum (RSI), Volatility (ATR), Trend (MACD), Time (Hour)
    features = ['RSI', 'ATR', 'MACD_Main', 'Hour']
    
    # Target Creation (5-bar lookahead)
    # 1: Buy (Price goes up by > 0.0005)
    # 2: Sell (Price goes down by > 0.0005)
    # 0: Hold (Sideways)
    threshold = 0.0005
    lookahead = 5
    
    df['Future_Close'] = df['Close'].shift(-lookahead)
    
    conditions = [
        (df['Future_Close'] > df['Close'] + threshold),
        (df['Future_Close'] < df['Close'] - threshold)
    ]
    choices = [1, 2]
    df['Target'] = np.select(conditions, choices, default=0)
    
    # Drop rows with NaN (the last `lookahead` rows)
    df.dropna(subset=['Future_Close'] + features, inplace=True)
    
    X = df[features]
    y = df['Target']
    
    print("\nTarget Distribution:")
    print(y.value_counts(normalize=True))
    
    # Train/Test Split (Chronological - Last 1 Year for testing)
    cutoff_date = df.index[-1] - pd.DateOffset(years=1)
    
    X_train = X[X.index < cutoff_date]
    y_train = y[y.index < cutoff_date]
    X_test = X[X.index >= cutoff_date]
    y_test = y[y.index >= cutoff_date]
    
    print(f"\nTraining set size: {len(X_train)} ({X_train.index[0]} to {X_train.index[-1]})")
    print(f"Testing set size: {len(X_test)} ({X_test.index[0]} to {X_test.index[-1]})")
    
    # Create Pipeline with CPDA (Discretization) and Naive Bayes
    # We use encode='onehot-dense' with BernoulliNB to perfectly mimic Naive Bayes over discrete states
    pipeline = Pipeline([
        ('discretizer', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile')),
        ('nb', BernoulliNB())
    ])
    
    print("\nTraining Model...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating Model...")
    y_pred = pipeline.predict(X_test)
    
    print("\nClassification Report on Test Data:")
    print(classification_report(y_test, y_pred, target_names=['Hold (0)', 'Buy (1)', 'Sell (2)']))
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    # Define input types (4 float features)
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    
    # Convert
    onnx_model = skl2onnx.convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)
    
    # Save
    with open("naive_bayes_eurusd.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print("Model successfully exported to 'naive_bayes_eurusd.onnx'")

if __name__ == "__main__":
    main()
