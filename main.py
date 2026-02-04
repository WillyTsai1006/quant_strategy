import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 數據獲取 (Data Acquisition)
# 我們抓取 SPY (S&P 500 ETF) 的數據，從 2020 年至今
ticker = "SPY"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
# 只需要收盤價
df = data[['Close']].copy()
# 2. 特徵工程 (Feature Engineering)
# 計算短期 (20日) 與長期 (50日) 移動平均
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
# 3. 生成信號 (Signal Generation)
# 初始化信號為 0
df['Signal'] = 0.0
# 當 20日均線 > 50日均線時，標記為 1 (持有狀態)
# 只從第 50 天開始切片，因為前面數據是 NaN
df.loc[df.index[50:], 'Signal'] = \
    np.where(df['SMA_20'][50:] > df['SMA_50'][50:], 1.0, 0.0)
# 計算買賣點 (Position)：Signal 的差分
# 1 - 0 = 1 (買入), 0 - 1 = -1 (賣出)
df['Position'] = df['Signal'].diff()
# 4. 視覺化檢查
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price', alpha=0.5)
plt.plot(df['SMA_20'], label='SMA 20', alpha=0.9)
plt.plot(df['SMA_50'], label='SMA 50', alpha=0.9)
# 標記買入點 (向上三角形)
plt.plot(df[df['Position'] == 1].index, 
         df['SMA_20'][df['Position'] == 1], 
         '^', markersize=10, color='g', lw=0, label='Buy Signal')
# 標記賣出點 (向下三角形)
plt.plot(df[df['Position'] == -1].index, 
         df['SMA_20'][df['Position'] == -1], 
         'v', markersize=10, color='r', lw=0, label='Sell Signal')
plt.title(f'{ticker} Dual Moving Average Crossover Strategy')
plt.legend()
plt.show()
import numpy as np
# 5. 回測 (Backtesting)
# 計算每日市場報酬率 (Log returns are often preferred in quant finance, but simple returns are okay here)
# pct_change() 計算 (P_t - P_{t-1}) / P_{t-1}
df['Market_Returns'] = df['Close'].pct_change()
# 計算策略報酬率
# shift(1) 是關鍵！因為今天的信號是基於今天收盤產生的，只能在「明天」開盤執行
# 如果不 shift，會變成「預知未來」(Look-ahead Bias)，這是新手最常犯的錯
df['Strategy_Returns'] = df['Market_Returns'] * df['Signal'].shift(1)
# 6. 績效評估 (Performance Metrics)
# 計算累積報酬 (Cumulative Returns)
df['Cumulative_Market'] = (1 + df['Market_Returns']).cumprod()
df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
# 顯示最終報酬
print(f"市場累積報酬: {df['Cumulative_Market'].iloc[-1]:.2f}")
print(f"策略累積報酬: {df['Cumulative_Strategy'].iloc[-1]:.2f}")
# 繪製權益曲線 (Equity Curve)
plt.figure(figsize=(12, 6))
plt.plot(df['Cumulative_Market'], label='Market Returns (Buy & Hold)')
plt.plot(df['Cumulative_Strategy'], label='Strategy Returns')
plt.legend()
plt.show()
# 計算夏普比率 (Sharpe Ratio)
annualized_sharpe = df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * np.sqrt(252)
print(f"年化夏普比率: {annualized_sharpe:.2f}")