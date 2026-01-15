import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

# 1. 下載數據
df = yf.download("BTC-USD", period="730d", interval="1h")
df.columns = df.columns.get_level_values(0)
df = df[['Close', 'Volume']].copy()

# 2. Feature Engineering
df['return'] = np.log(df['Close'] / df['Close'].shift(1))
df['sma_20'] = df['Close'].rolling(20).mean()
df['dist_sma_20'] = df['Close'] / df['sma_20'] - 1
df['volatility'] = df['return'].rolling(20).std()
# RSI 相對強弱指數
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
rs = gain.rolling(14).mean() / loss.rolling(14).mean()
df['rsi'] = 100 - 100 / (1 + rs)
# MACD 指數平滑異同移動平均線
ema12 = df['Close'].ewm(span=12).mean()
ema26 = df['Close'].ewm(span=26).mean()
df['macd'] = ema12 - ema26
df['macd_signal'] = df['macd'].ewm(span=9).mean()
# Bollinger Band 布林通道
mid = df['Close'].rolling(20).mean()
std = df['Close'].rolling(20).std()
df['bb_width'] = (mid + 2 * std - (mid - 2 * std)) / mid
# Target
df['target'] = df['return'].shift(-1)
df = df.dropna()
features = [
    'return', 'dist_sma_20', 'volatility',
    'rsi', 'macd', 'macd_signal', 'bb_width', 'Volume'
]

# 3. Walk-forward Backtest
train_len = int(len(df) * 0.6)
test_len = int(len(df) * 0.1)
capital = 1.0
equity = []
positions = []
params = {
    'objective': 'regression',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'metric': 'l2'
}
transaction_cost = 0.0005  # 0.05%
for i in range(train_len, len(df) - test_len, test_len):
    train = df.iloc[i - train_len:i]
    test = df.iloc[i:i + test_len]
    model = lgb.train(
        params,
        lgb.Dataset(train[features], train['target']),
        num_boost_round=200
    )
    preds = model.predict(test[features])
    # Strategy Logic
    threshold = np.std(preds)
    signal = np.where(preds > threshold, 1,
             np.where(preds < -threshold, -1, 0))
    position_size = np.clip(np.abs(preds) / threshold, 0, 1)
    position = signal * position_size
    for pos, r in zip(position, test['target']):
        cost = transaction_cost * abs(pos)
        capital *= (1 + pos * r - cost)
        equity.append(capital)
        positions.append(pos)

# 4. Performance Metrics
equity = np.array(equity)
returns = np.diff(np.log(equity))
sharpe = np.mean(returns) / np.std(returns) * np.sqrt(24 * 365)
cummax = np.maximum.accumulate(equity)
drawdown = equity / cummax - 1
max_dd = drawdown.min()
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")

# 5. Plot
plt.figure(figsize=(12, 5))
plt.plot(equity)
plt.title("Equity Curve (Full Walk-forward)")
plt.grid()
plt.show()

ax = lgb.plot_importance(model, importance_type='gain', max_num_features=10)
ax.set_title("Feature Importance")
plt.show()

# Sharpe > 1
# Max DD < -30%
# Equity 不是只靠最後一段
# Feature importance 合理（不是全靠 Volume）

