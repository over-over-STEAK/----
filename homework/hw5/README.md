# 從零打造 GPT

完全不依賴 TensorFlow、PyTorch 或 JAX 等現代深度學習框架，**僅使用 NumPy 進行基礎矩陣運算（Linear Algebra）**。由下至上完整手寫所有組件的前向傳播（Forward）與反向傳播（Backward）梯度推導，最終實現一個具備 Causal LM 預測能力的微型 GPT 生成模型。

## 專案結構

gpt-from-scratch/
├── gpt_model.py          # 完整 GPT 實作 (所有元件 forward/backward)
├── train_gpt.py          # 訓練腳本 + 文本生成展示
├── tests.py              # 23 個完整測試
└── test_gradients.py     # 數值梯度驗證腳本

## 測試結果 — 23/23 PASS

| 分類 | 測試項目 | 狀態 |
|:--|:--|:--|
| **基礎元件** | Linear/LayerNorm/Embedding 前向+反向梯度 | ✓ 7/7 |
| **記憶與狀態** | 長序列截斷、Batch 獨立性、記憶體限制 | ✓ 3/3 |
| **無效輸入防護** | Temperature=0、Top-k 邊界、NaN 防護、JSON 匯出 | ✓ 3/3 |
| **端到端模型** | 輸出形狀、梯度形狀、數值梯度檢驗、訓練收斂 | ✓ 4/4 |
| **生成策略** | Temperature 控制、Top-k 採樣、低溫確定性 | ✓ 3/3 |
| **參數與優化器** | 參數統計、梯度歸零、權重更新 | ✓ 3/3 |

## 訓練成果

- **總參數**: 357,024（3層 × 4頭 × 96維）
- **Loss 變化**: 3.89 → 3.36（持續下降中）
- **PPL 變化**: 49.09 → 28.68
- **優化器**: 自寫 SGD + Momentum（附學習率衰減排程）

## 核心實作

- **LayerNorm**: 完全手寫 forward + backward（含均值/方差鏈式求導）
- **Causal Attention**: Scaled Dot-Product + 上三角因果遮罩 + softmax 反向梯度
- **Cross-Entropy**: Softmax + CE 合併梯度 ($p_i - y_i$)
- **殘差連接**: 梯度分流累加
- **生成策略**: Temperature scaling + Top-k 截斷採樣

## 系統架構

[ GPT Language Model ]
             │
  [ 3× Transformer Blocks ]
             │
┌──────────────┴──────────────┐
│ [ Causal Multi-Head Attn ]  │ [ FeedForward Network ]
│   ├── Linear (Q, K, V)      │   ├── Linear (d_model -> 4d_model)
│   ├── Split Heads           │   ├── ReLU Activation
│   ├── Scaled Dot-Product    │   └── Linear (4d_model -> d_model)
│   └── Causal Mask & Softmax │
└──────────────┬──────────────┘
│
[ LayerNorm Layer ]
│
┌──────────────┴──────────────┐
│ [ Token Embedding (Wte) ]   │ [ Positional Encoding (Wpe) ]
└─────────────────────────────┴──────────────────────────────┘

## 執行方式

```bash
python train_gpt.py            # 訓練模型
python tests.py                # 執行完整測試
python test_gradients.py       # 梯度正確性驗證
```

## 依賴環境

- Python 3.10+
- NumPy >= 1.22
