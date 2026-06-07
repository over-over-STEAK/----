# 🧠 Project: Pure NumPy GPT (Zero-Framework Autodiff Core)

本專案挑戰 **「從零打造 GPT」**。完全不依賴 TensorFlow、PyTorch 或 JAX 等現代深度學習框架，**僅使用 NumPy 進行基礎矩陣運算（Linear Algebra）**。由下至上完整手寫所有組件的前向傳播（Forward）與反向傳播（Backward）梯度推導，最終實現一個具備 Causal LM 預測能力的微型 GPT 生成模型。

---

## 1. 系統架構（Computational Graph Hierarchy）

整個模型的架構嚴格遵循自底向上的模組化設計，每個組件皆繼承或實現了明確的 `.forward(x)` 與 `.backward(dout)` 介面：
[ GPT Language Model ]
             │
  [ 3× Transformer Blocks ]
             │
┌──────────────┴──────────────┐
│ [ Causal Multi-Head Attn ]  │ [ FeedForward Network ]
│   ├── Linear (Q, K, V)      │   ├── Linear (d_model -> 4d_model)
│   ├── Split Heads           │   ├── ReLU / GeLU Activation
│   ├── Scaled Dot-Product    │   └── Linear (4d_model -> d_model)
│   └── Causal Mask & Softmax │
└──────────────┬──────────────┘
│
[ LayerNorm Layer ]
│
┌──────────────┴──────────────┐
│ [ Token Embedding (Wte) ]   │ [ Positional Encoding (Wpe) ]
└─────────────────────────────┴──────────────────────────────┘

## 2. 模型配置與訓練成果

本專案採用微型但結構完整的 GPT 配置，專門用於字元級（Character-level）或詞級（Token-level）的預測任務：

### 📋 核心超參數 (Hyperparameters)

* **Number of Layers ($L$):** 3 Layers
* **Attention Heads ($H$):** 4 Heads (每頭維度 $d_k = 96 / 4 = 24$)
* **Total Parameters:** **228,603** 個可學習參數

### 📈 訓練與收斂指標

* **優化器選擇:** SGD + Momentum（帶動量的隨機梯度下降，手寫實作）
* **30 Epochs 訓練表現:**
  * **Loss (交叉熵損失):** $2.96 \longrightarrow 1.99$
  * **Perplexity (困惑度 PPL):** $19 \longrightarrow 7$ (代表模型對下一個字的預測不確定性大幅降低)

---

## 3. 手寫反向傳播（Backward Pass）數學核心

由於不依賴自動微分引擎，專案針對以下四個核心算子進行了嚴格的微積分推導，並以 NumPy 矩陣乘法高效率實現：

### ① LayerNorm 完整推導

給定輸入 $x \in \mathbb{R}^{B \times T \times D}$，均值 $\mu$ 與方差 $\sigma^2$。反向傳播時，傳入的上游梯度為 $\partial L / \partial y$。我們必須同時求出對可學習參數 $\gamma, \beta$ 的梯度，以及對輸入 $x$ 的全域梯度：
$$\frac{\partial L}{\partial \gamma} = \sum y_{norm} \odot \frac{\partial L}{\partial y}, \quad \frac{\partial L}{\partial \beta} = \sum \frac{\partial L}{\partial y}$$
$$\frac{\partial L}{\partial x_i} = \frac{1}{D \cdot \sqrt{\sigma^2 + \epsilon}} \left[ D \cdot \gamma_i \cdot \frac{\partial L}{\partial y_i} - \sum_{j=1}^{D} \gamma_j \frac{\partial L}{\partial y_j} - y_{norm, i} \sum_{j=1}^{D} \gamma_j \frac{\partial L}{\partial y_j} y_{norm, j} \right]$$

### ② Causal Attention Mask 梯度

在自注意力機制中，注意力權重矩陣 $A = \text{Softmax}(\frac{QK^T}{\sqrt{d_k}} + M)$，其中 $M$ 為上三角禁行矩陣（因果遮罩，$-\infty$）。

* **Backward:** 在對 $A$ 求導並回傳至 $QK^T$ 時，**被遮罩區域 ($-\infty$) 的上游梯度必須嚴格歸零**，只允許歷史資訊的梯度回傳。

### ③ Softmax Cross-Entropy 聯合梯度

為了數值穩定性與運算效率，專案將最後輸出層的 Softmax 與 Cross-Entropy 損失函數合併求導。假設預測機率為 $p$，真實標籤的 One-hot 編碼為 $y$，則該層對 Logits ($z$) 的梯度可以極其優雅地簡化為：
$$\frac{\partial \mathcal{L}}{\partial z_i} = p_i - y_i$$

### ④ 殘差連接（Residual Connection）與梯度累加

GPT 大量使用 $x_{l+1} = x_l + \text{SubLayer}(x_l)$ 結構。

* 上游梯度會分流：一部分直接「無損傳回」前一層 $x_l$，另一部分流經 SubLayer。在分支匯合點，**各路徑傳回的 NumPy 矩陣梯度進行直接累加 (`+=`)**。

---

## 4. 自研文本生成策略 (Decoding Mechanics)

模型支援基於自迴歸（Autoregressive）的文本生成，並實作了兩種主流的機率採樣控制參數，以調整輸出品質：

$$\hat{P}(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

| 參數設定 | 生成行為風格 | 數學與表現原理 |
| :--- | :--- | :--- |
| **Temperature = 0.3** | 保守 / 確定性高 | 機率分佈被極化，高機率的字會被無限放大。模型傾向一直重複最常見的詞彙組合。 |
| **Temperature = 0.7** | 平衡 / 語意流暢 | 機率分佈逼近真實文本。AI 能在保持語意合理性的同時，產生富有變化且有意義的句子。 |
| **Temperature = 1.0** | 隨機 / 創意發散 | 機率分佈保持原樣。模型選擇低機率字的機率增加，驚喜感提升但伴隨胡言亂語（幻覺）的風險。 |
| **Top-k Sampling** | 截斷長尾雜訊 | 每次採樣前，強制只保留 Logits 前 $k$ 個最高的候選字，並將其餘機率歸零重新歸一化，防止生成出完全無關的火星文。 |

---

## 5. 如何運行此專案

### 依賴環境

* Python 3.10+

* NumPy >= 1.22

### 執行命令

1. **啟動底層架構訓練：**

   ```bash

   python train_gpt.py
