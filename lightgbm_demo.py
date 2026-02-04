import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# 1. 下載數據
print("正在下載數據...")
df = yf.download("BTC-USD", period="730d", interval="1h")
df.columns = df.columns.get_level_values(0)
df = df[['Close', 'Volume']].copy()
# 2. 特徵工程
print("正在計算特徵...")
# Log return
df['return'] = np.log(df['Close'] / df['Close'].shift(1))
# SMA & distance
df['sma_20'] = df['Close'].rolling(20).mean()
df['dist_sma_20'] = df['Close'] / df['sma_20'] - 1
# Volatility
df['volatility'] = df['return'].rolling(20).std()
# RSI
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
rs = gain.rolling(14).mean() / loss.rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + rs))
# MACD
ema12 = df['Close'].ewm(span=12).mean()
ema26 = df['Close'].ewm(span=26).mean()
df['macd'] = ema12 - ema26
df['macd_signal'] = df['macd'].ewm(span=9).mean()
# Bollinger Band
bb_mid = df['Close'].rolling(20).mean()
bb_std = df['Close'].rolling(20).std()
df['bb_upper'] = bb_mid + 2 * bb_std
df['bb_lower'] = bb_mid - 2 * bb_std
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid
# 預測目標：下一小時報酬（回歸）
df['target'] = df['return'].shift(-1)
df = df.dropna()
features = [
    'return', 'dist_sma_20', 'volatility',
    'rsi', 'macd', 'macd_signal', 'bb_width', 'Volume'
]
# 3. Walk-forward Validation
print("開始 Walk-forward 訓練...")
train_size = int(len(df) * 0.7)
test_size = int(len(df) * 0.1)
equity_curve = []
capital = 1.0
params = {
    'objective': 'regression',
    'metric': 'l2',
    'learning_rate': 0.05,
    'num_leaves': 31
}
for start in range(train_size, len(df) - test_size, test_size):
    train = df.iloc[start - train_size:start]
    test = df.iloc[start:start + test_size]
    X_train = train[features]
    y_train = train['target']
    X_test = test[features]
    lgb_train = lgb.Dataset(X_train, y_train)
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100
    )
    preds = model.predict(X_test)
    # 4. 策略邏輯
    position = -np.sign(preds)  # +1 long / -1 short
    strategy_return = position * test['target'].values
    for r in strategy_return:
        capital *= (1 + r)
        equity_curve.append(capital)
# 5. 繪圖
plt.figure(figsize=(12, 5))
plt.plot(equity_curve)
plt.title("Equity Curve (Walk-forward)")
plt.xlabel("Time")
plt.ylabel("Capital")
plt.grid(True)
plt.show()
# 6. 特徵重要性
ax = lgb.plot_importance(model, importance_type='gain', max_num_features=10)
ax.set_title("Feature Importance")
plt.show()
