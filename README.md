# [PYTORCH] 用深度 Q 學習玩俄羅斯方塊（Tetris）

## 介紹

這是用來訓練代理（agent）玩俄羅斯方塊的 Python 原始碼。它可以作為強化學習應用的一個非常基礎的範例。

<p align="center">
  <img src="demo/tetris.gif" width=600><br/>
  <i>俄羅斯方塊展示</i>
</p>

你也可以在這裡看到影片展示：[YouTube 示範](https://youtu.be/g96x6uATAR8)

## 環境需求

* **Python 3.6**
* **PIL**
* **cv2**
* **PyTorch**
* **NumPy**
* **matplotlib**
* **tensorboardX**

### 安裝方式

```bash
conda create -n tetris python=3.6
conda activate tetris
```

```bash
conda install pytorch=1.7.1 torchvision=0.8.2 torchaudio=0.7.2 cudatoolkit=11.0 -c pytorch
```

```bash
pip install -r requirements.txt
```

## 如何使用程式碼

* **從零開始訓練模型**：執行 `python train.py`
* **測試已訓練的模型**：執行 `python test.py`

### 已訓練模型

你可以在 **trained\_models/tetris** 資料夾中找到我訓練好的模型。

## train.py

此腳本實現了一個深度Q網絡（DQN），用於訓練一個AI代理來玩Tetris。它遵循標準的強化學習原則，代理從與Tetris環境的互動中學習並根據需要更新策略。

### 主要特點：
- **模型架構**：使用自定義的`DeepQNetwork`來預測Tetris遊戲中不同動作的Q值。
- **探索與利用**：使用epsilon-貪心算法來平衡訓練過程中的探索和利用。最初epsilon值較高（偏向探索），隨著訓練進行逐漸降低，更多偏向利用已學得的策略。
- **重放記憶**：將過去的經驗存儲在重放記憶緩衝區中，並從中隨機抽取樣本進行訓練。這有助於打破連續訓練樣本之間的相關性。
- **訓練循環**：代理與Tetris環境進行多個回合的互動，根據觀察到的獎勳和轉移更新Q值。模型會在定期的間隔儲存，以便日後使用或測試。

### 訓練過程：
1. **初始化**：初始化Tetris環境，創建`DeepQNetwork`模型，同時初始化優化器（Adam）和損失函數（均方誤差損失）。
2. **互動**：代理與環境進行互動，根據模型預測的Q值選擇動作。
3. **重放記憶**：將每次動作的狀態、獎勳和下個狀態存儲在重放記憶緩衝區中，當有足夠的互動發生後，使用這些經驗進行訓練。
4. **模型更新**：使用從重放記憶中抽取的樣本進行模型更新。根據貝爾曼方程，目標值是當前獎勳加上下個狀態的折扣未來Q值。
5. **日誌記錄**：使用TensorBoard記錄訓練進度，包括分數、放置的方塊數量和消除的行數。

### 超參數：
- **學習率**：控制模型更新的步長。
- **折扣因子（Gamma）**：決定未來獎勳的重要性。
- **Epsilon衰減**：控制訓練過程中的探索與利用之間的平衡。
- **重放記憶大小**：訓練用的過去經驗數量。

訓練完成的模型可以在後續測試中加載，檢查代理在Tetris遊戲中的表現。


## 詳細介紹

你可以參考這篇部落格文章：
[https://blog.csdn.net/qq128252/article/details/129145534](https://blog.csdn.net/qq128252/article/details/129145534)
