# LightGBM Algorithmic Trading Practice (量化交易策略練習)
這個專案記錄了我學習如何將機器學習模型 LightGBM 應用於金融交易策略的過程。專案內容包含從基礎的雙均線策略（Pandas 實作），到建立特徵工程、訓練 LightGBM 模型，並使用 Walk-forward Validation（滾動式回測） 來驗證策略在 BTC-USD 上的表現。
## 📂 檔案結構與功能
本專案包含三個主要腳本，代表了學習的三個階段：

1. main.py (基礎對照組：傳統策略)

    這是量化交易的入門練習，不涉及機器學習，主要用於熟悉 pandas 的操作與回測邏輯。

- 目標標的：SPY (S&P 500 ETF)

- 策略邏輯：雙均線策略 (Dual SMA Crossover)。當短期均線 (20 MA) 突破長期均線 (50 MA) 時買入。

- 學習重點：
    - 數據獲取 (yfinance) 與清洗。

    - 信號生成與向量化運算 (numpy.where)。

    - 關鍵概念：理解 shift(1) 的重要性，避免「預知未來」(Look-ahead Bias)。

2. lightgbm_demo.py (LightGBM 原型)

    這是將機器學習引入交易的第一步。

- 目標標的：BTC-USD (1小時線)

- 模型：LightGBM Regressor

- 特徵工程：包含 Log Return, SMA Distance, Volatility, RSI, MACD, Bollinger Bands。

- 策略邏輯：簡單的方向預測。若預測報酬 > 0 做多，< 0 做空。

- 驗證方法：實作了基礎的 Walk-forward Validation 迴圈。

3. lightgbm_walkforward_strategy.py (進階策略實作)

    這是 demo 的優化版本，加入了更貼近真實交易的限制與資金管理。

- 改進點：
    - 交易成本：考量 0.05% 的手續費與滑價。
    
    - 動態倉位 (Position Sizing)：根據預測值的強度與波動率閾值 (Threshold) 決定倉位大小，而非全倉進出。

    - 績效指標：計算夏普比率 (Sharpe Ratio) 與最大回落 (Max Drawdown)。
    
    - 特徵重要性：觀察哪些指標對模型預測最有效。
## 🛠️ 安裝需求
請確保安裝以下 Python 套件：
```Bash
pip install yfinance pandas numpy lightgbm matplotlib
```
## 📊 特徵工程 (Feature Engineering)
在 LightGBM 模型中，我使用了以下技術指標作為輸入特徵（Features）：
| 特徵名稱 | 描述 | 用途 |
| --------- | ---------- | ----------- |
| return | 對數收益率 | 捕捉價格動能 |
| dist_sma_20 | 價格與 20 MA 的距離 | 判斷乖離率 |
| volatility | 滾動波動率 | 衡量市場風險 |
| rsi | 相對強弱指標 (14日) | 判斷超買/超賣 |
| macd / signal | 指數平滑異同移動平均線 | 判斷趨勢變化 |
| bb_width | 布林通道寬度 | 衡量波動性壓縮與擴張 |
| Volume | 成交量 | 市場活躍度輔助 |
## 🚀 策略方法論： Walk-forward Validation
為了避免時間序列數據中的 Overfitting，本專案不使用隨機的 train_test_split。而是採用 滾動式回測 (Walk-forward)：

1. 選取前 60%~70% 數據作為初始訓練集。

2. 訓練模型並預測接下來一小段時間（如 10%）的測試集。

3. 記錄這段時間的策略績效。

4. 將視窗向後移動，將剛測試過的數據加入訓練集，重新訓練模型。

5. 重複上述步驟直到數據結束。

這種方法能模擬模型在真實時間推移下的適應能力。
## 📈 結果範例
執行 lightgbm_walkforward_strategy.py 後，將會產出：

1. Equity Curve：權益曲線圖，檢視資金增長情況。

2. Feature Importance：特徵重要性圖表，了解模型依賴哪些指標。

3. Terminal Output：
        - Sharpe Ratio (夏普比率)
        - Max Drawdown (最大回落)
## ⚠️ 免責聲明 (Disclaimer)
本專案代碼僅供學習與研究用途，不構成任何投資建議。加密貨幣與金融市場波動劇烈，實際交易存在本金損失風險。