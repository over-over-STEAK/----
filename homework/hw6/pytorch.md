
# 非 Transformer 語言模型實作（PyTorch LSTM）

## 系統架構

本專案使用 **LSTM（Long Short-Term Memory）** 建立語言模型，透過學習文字序列中的上下文關係來預測下一個字元。與目前主流的 Transformer 架構不同，本系統完全不使用 Attention 機制，而是採用循環神經網路（RNN）系列中的 LSTM 進行序列建模。

```text
輸入文字
    │
Tokenizer
    │
文字轉換為 ID
    │
Embedding Layer
    │
LSTM Layer
    │
Fully Connected Layer
    │
Softmax
    │
預測下一個字元
```

---

## 專案模組設計

```text
lstm_pytorch.py
│
├── 模型定義
│   ├── LSTMLanguageModel
│   └── LSTMLanguageModelAdvanced
│
├── 資料處理
│   └── TextTokenizer
│
├── 訓練引擎
│   └── Trainer
│
└── CLI 介面
    ├── train
    ├── generate
    └── chat
```

---

# 1. 模型類別

## LSTMLanguageModel（基礎版）

模型架構：

```text
Embedding → LSTM → Linear
```

主要功能：

* Embedding：將字元 ID 轉換成向量表示
* LSTM：學習序列上下文資訊
* Linear：輸出各字元的機率分布

此版本結構簡單，適合驗證語言模型基本功能。

---

## LSTMLanguageModelAdvanced（進階版）

模型架構：

```text
Embedding
   ↓
Projection Layer
   ↓
Dropout
   ↓
Multi-layer LSTM
   ↓
Dropout
   ↓
Projection Layer
   ↓
Linear
```

新增功能：

### 多層 LSTM

```python
num_layers = 2
```

透過堆疊多層 LSTM 提升模型表達能力。

### Dropout

```python
dropout = 0.3
```

降低過擬合風險。

### Weight Tying

```python
fc.weight = embedding.weight
```

共享輸入與輸出權重：

* 減少參數數量
* 提升泛化能力

### 文字生成功能

模型內建 `generate()` 方法：

```python
generate(
    prompt_ids,
    max_new_tokens=100,
    temperature=1.0,
    top_k=20
)
```

生成流程：

1. 輸入 Prompt
2. 初始化 Hidden State
3. 計算下一字元機率
4. Temperature 調整隨機性
5. Top-k 篩選候選字元
6. Softmax 採樣
7. 重複生成直到結束

---

# 2. 資料處理模組

## TextTokenizer

本系統採用 Character-Level Tokenizer。

### 功能

```python
fit(text)
```

建立字元詞彙表。

```python
encode(text)
```

文字轉 ID。

範例：

```text
abc
↓
[0,1,2]
```

```python
decode(ids)
```

ID 轉回文字。

範例：

```text
[0,1,2]
↓
abc
```

### 模型保存

```python
save(path)
load(path)
```

保存與載入 Tokenizer，方便推論時重複使用。

---

# 3. 訓練引擎

## Trainer 類別

封裝完整訓練流程：

### 資料切割

建立訓練資料：

```text
x = 原始序列
y = 向右平移一個字元
```

例如：

```text
輸入:
Romeo

目標:
omeo!
```

模型學習每個位置預測下一個字元。

---

### 損失函數

```python
CrossEntropyLoss()
```

計算預測與真實答案差異。

---

### 優化器

```python
AdamW()
```

相比傳統 SGD：

* 收斂速度較快
* 訓練較穩定

---

### 梯度裁剪

```python
clip_grad_norm_
```

避免梯度爆炸問題。

---

### 學習率衰減

```python
StepLR()
```

隨訓練進行逐步降低學習率。

---

### Checkpoint

定期儲存：

```text
model.pt
tokenizer.tok
```

方便後續載入與推論。

---

# 4. CLI 操作模式

## 訓練模式

```bash
python lstm_pytorch.py train \
    --epochs 100 \
    --save model.pt
```

自訂資料集：

```bash
python lstm_pytorch.py train \
    --file 三國演義.txt \
    --epochs 200 \
    --hidden-size 256
```

---

## 文字生成模式

```bash
python lstm_pytorch.py generate \
    --checkpoint model.pt \
    --prompt "話說天下大勢" \
    --temperature 0.7 \
    --top-k 20
```

---

## 聊天模式

```bash
python lstm_pytorch.py chat \
    --checkpoint model.pt
```

使用者可持續輸入文字與模型互動。

---

# 5. 與 Transformer 的比較

| 項目       | 本專案 LSTM     | Transformer    |
| -------- | ------------ | -------------- |
| 架構       | RNN 系列       | Attention      |
| 記憶方式     | Hidden State | Self-Attention |
| 計算複雜度    | 較低           | 較高             |
| 長距離依賴    | 較弱           | 較強             |
| GPU 平行化  | 較差           | 較佳             |
| 是否符合題目要求 | ✓            | ✗              |

---

# 結論

本專案成功利用 PyTorch 建立一套非 Transformer 的語言模型系統，以 LSTM 作為核心架構完成文字學習與生成。系統包含 Tokenizer、LSTM 模型、訓練引擎及 CLI 操作介面，可執行訓練、文字生成與互動聊天等功能，符合「使用非 Transformer（Attention）方法實作語言模型」之要求。
