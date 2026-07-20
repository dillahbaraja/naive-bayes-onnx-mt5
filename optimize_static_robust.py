import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
import os
import warnings

warnings.filterwarnings('ignore')

def simulate_ea(df, pipeline, min_conf, min_spread, tz_offset, sl_mult, tp_mult, use_be, use_trend_filter):
    features = ['RSI', 'ATR', 'MACD_Main', 'Hour']
    X = df[features]
    probs = pipeline.predict_proba(X)
    labels = pipeline.predict(X)
    
    df_sim = df.copy()
    X_shifted = X.copy()
    X_shifted['Hour'] = (X_shifted['Hour'] + tz_offset) % 24
    probs_shifted = pipeline.predict_proba(X_shifted)
    labels_shifted = pipeline.predict(X_shifted)
    
    df_sim['Prob_Hold'] = probs_shifted[:, 0]
    df_sim['Prob_Buy'] = probs_shifted[:, 1]
    df_sim['Prob_Sell'] = probs_shifted[:, 2]
    df_sim['Pred_Label'] = labels_shifted
    
    df_sim['EMA_200'] = df_sim['Close'].ewm(span=200, adjust=False).mean()
    
    trades = []
    active_position = None
    holding_bars = 5
    
    close_prices = df_sim['Close'].values
    atr_vals = df_sim['ATR'].values
    ema_vals = df_sim['EMA_200'].values
    prob_buy = df_sim['Prob_Buy'].values
    prob_sell = df_sim['Prob_Sell'].values
    pred_labels = df_sim['Pred_Label'].values
    
    balance = 10000.0
    initial_balance = 10000.0
    lot_size = 0.1
    is_jpy = close_prices[0] > 50.0
    point_value = 100.0 if is_jpy else 10000.0
    
    for t in range(1, len(df_sim) - 5):
        if active_position is not None:
            bars_held = t - active_position['entry_bar']
            curr_close = close_prices[t]
            curr_high = df_sim['High'].values[t]
            curr_low = df_sim['Low'].values[t]
            
            if active_position['type'] == 'BUY':
                active_position['highest_price'] = max(active_position['highest_price'], curr_high)
            else:
                active_position['lowest_price'] = min(active_position['lowest_price'], curr_low)
            
            if use_be:
                atr_at_entry = atr_vals[active_position['entry_bar']]
                if active_position['type'] == 'BUY':
                    if (curr_high - active_position['entry_price']) > atr_at_entry * 1.0:
                        active_position['sl'] = max(active_position['sl'], active_position['entry_price'])
                else:
                    if (active_position['entry_price'] - curr_low) > atr_at_entry * 1.0:
                        active_position['sl'] = min(active_position['sl'], active_position['entry_price'])
            
            hit_sl = False
            hit_tp = False
            pips_profit = 0
            
            if active_position['type'] == 'BUY':
                if curr_low <= active_position['sl']:
                    hit_sl = True
                    pips_profit = (active_position['sl'] - active_position['entry_price']) * point_value
                elif curr_high >= active_position['tp']:
                    hit_tp = True
                    pips_profit = (active_position['tp'] - active_position['entry_price']) * point_value
            else:
                if curr_high >= active_position['sl']:
                    hit_sl = True
                    pips_profit = (active_position['entry_price'] - active_position['sl']) * point_value
                elif curr_low <= active_position['tp']:
                    hit_tp = True
                    pips_profit = (active_position['entry_price'] - active_position['tp']) * point_value
                    
            if hit_sl or hit_tp or bars_held >= holding_bars:
                if not hit_sl and not hit_tp:
                    if active_position['type'] == 'BUY':
                        pips_profit = (curr_close - active_position['entry_price']) * point_value
                    else:
                        pips_profit = (active_position['entry_price'] - curr_close) * point_value
                
                if is_jpy:
                    profit_usd = (curr_close - active_position['entry_price'] if active_position['type'] == 'BUY' else active_position['entry_price'] - curr_close) * 1000.0 * lot_size
                else:
                    profit_usd = (curr_close - active_position['entry_price'] if active_position['type'] == 'BUY' else active_position['entry_price'] - curr_close) * 100000.0 * lot_size
                
                if hit_sl:
                    if is_jpy:
                        profit_usd = (active_position['sl'] - active_position['entry_price'] if active_position['type'] == 'BUY' else active_position['entry_price'] - active_position['sl']) * 1000.0 * lot_size
                    else:
                        profit_usd = (active_position['sl'] - active_position['entry_price'] if active_position['type'] == 'BUY' else active_position['entry_price'] - active_position['sl']) * 100000.0 * lot_size
                elif hit_tp:
                    if is_jpy:
                        profit_usd = (active_position['tp'] - active_position['entry_price'] if active_position['type'] == 'BUY' else active_position['entry_price'] - active_position['tp']) * 1000.0 * lot_size
                    else:
                        profit_usd = (active_position['tp'] - active_position['entry_price'] if active_position['type'] == 'BUY' else active_position['entry_price'] - active_position['tp']) * 100000.0 * lot_size
                
                balance += profit_usd
                trades.append(profit_usd)
                active_position = None
                
        if active_position is None:
            sig = pred_labels[t]
            p_buy = prob_buy[t]
            p_sell = prob_sell[t]
            p_spread = abs(p_buy - p_sell)
            
            if sig == 1 and (p_buy < min_conf or p_spread < min_spread):
                sig = 0
            elif sig == 2 and (p_sell < min_conf or p_spread < min_spread):
                sig = 0
                
            if use_trend_filter and sig > 0:
                close_1 = close_prices[t-1]
                ema_1 = ema_vals[t-1]
                if sig == 1 and close_1 < ema_1:
                    sig = 0
                elif sig == 2 and close_1 > ema_1:
                    sig = 0
            
            if sig > 0:
                atr = atr_vals[t]
                entry_p = close_prices[t]
                
                if sig == 1:
                    sl = entry_p - (atr * sl_mult)
                    tp = entry_p + (atr * tp_mult)
                    active_position = {
                        'type': 'BUY', 'entry_price': entry_p, 'sl': sl, 'tp': tp, 'entry_bar': t, 'highest_price': entry_p, 'lowest_price': entry_p
                    }
                elif sig == 2:
                    sl = entry_p + (atr * sl_mult)
                    tp = entry_p - (atr * tp_mult)
                    active_position = {
                        'type': 'SELL', 'entry_price': entry_p, 'sl': sl, 'tp': tp, 'entry_bar': t, 'highest_price': entry_p, 'lowest_price': entry_p
                    }
                    
    net_profit = balance - initial_balance
    win_rate = np.sum(np.array(trades) > 0) / len(trades) if len(trades) > 0 else 0.0
    return net_profit, len(trades), win_rate

def optimize_pair(csv_file, pair_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_file)
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
    
    # --- ROBUST 5% / 95% WINSORIZATION FOR STATIC MODEL ---
    for col in ['ATR', 'MACD_Main']:
        lower_limit = df[col].quantile(0.05)
        upper_limit = df[col].quantile(0.95)
        df[col] = df[col].clip(lower_limit, upper_limit)
    
    X = df[features]
    y = df['Target']
    
    pipeline_b = Pipeline([
        ('discretizer', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform')),
        ('nb', BernoulliNB())
    ])
    pipeline_b.fit(X, y)
    
    print(f"\n================ OPTIMASI ROBUST STATIC MODEL FOR: {pair_name} ================")
    
    conf_options = [0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40]
    spread_options = [0.0, 0.01, 0.02, 0.05]
    tz_options = [0, 1]
    sl_options = [2.0, 2.5, 3.0, 3.5]
    tp_options = [2.5, 3.0, 4.0, 5.0]
    be_options = [True, False]
    
    best_profit = -999999.0
    best_params = {}
    best_trades = 0
    best_wr = 0
    
    df_test = df.tail(8000)
    
    for min_conf in conf_options:
        for min_spread in spread_options:
            for tz_offset in tz_options:
                for sl_mult in sl_options:
                    for tp_mult in tp_options:
                        for use_be in be_options:
                            if tp_mult < sl_mult:
                                continue
                            
                            profit, num_trades, wr = simulate_ea(
                                df_test, pipeline_b, min_conf, min_spread, tz_offset, 
                                sl_mult, tp_mult, use_be, use_trend_filter=True
                            )
                            
                            if num_trades >= 10:
                                if profit > best_profit:
                                    best_profit = profit
                                    best_trades = num_trades
                                    best_wr = wr
                                    best_params = {
                                        'min_conf': min_conf,
                                        'min_spread': min_spread,
                                        'tz_offset': tz_offset,
                                        'sl_mult': sl_mult,
                                        'tp_mult': tp_mult,
                                        'use_be': use_be
                                    }
                                    
    print(f"Hasil Optimasi Terbaik:")
    print(f"Net Profit: ${best_profit:.2f}")
    print(f"Jumlah Trades: {best_trades}")
    print(f"Win Rate: {best_wr*100:.2f}%")
    print(f"Parameter: {best_params}")

def main():
    optimize_pair("USDJPY_H1_Data.csv", "USDJPY")
    optimize_pair("EURJPY_H1_Data.csv", "EURJPY")

if __name__ == "__main__":
    main()
