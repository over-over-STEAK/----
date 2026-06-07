# 📡 Project: Sonar Mine Classifier (AI Subsea Detection Core)

本模組為 **DeepSonar OS** 的核心人工智慧引擎，專門用於處理多維海域聲納陣列訊號。系統採用經典的全連接前饋神經網路（MLP），精準辨識反射訊號來自「海底岩石（Rock）」或「潛在水雷（Mine）」。

## 1. 核心源碼 (`sonar_model.py`)

```python
# ==========================================
# sonar_model.py
# CREATED: 2026
# Project: Sonar Mine Classifier (New Version)
# ==========================================
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# SETUP: 固定隨機種子以確保實驗可重現性
np.random.seed(42)
tf.random.set_seed(42)

def train_new_model():
    print("\n--- NEW Sonar AI Training Sequence Initiated ---")

    # LOAD DATASET (Sonar: Rocks vs. Mines)
    try:
        url = "[https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data](https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data)"
        df = pd.read_csv(url, header=None)
        print("Data loaded successfully from UCI Archive.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # PREPROCESSING
    # X: 60組不同角度的聲納能量反彈值, y: 標籤 ('R' 或 'M')
    X = df.iloc[:, 0:60].values
    y = df.iloc[:, 60].values

    # 標籤二進位編碼 (R=1, M=0)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # 訓練/測試集分配 (75% / 25%)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

    # 特徵標準化 (將數據調整為平均值 0，標準差 1，加速 Adam 收斂)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # 導出標準化參數，確保前端 Web UI 輸入數據時基準一致
    np.save('scaler_mean.npy', sc.mean_)
    np.save('scaler_std.npy', sc.scale_)
    print("Preprocessors (Encoder & Scaler) finalized.")

    # BUILD THE ARCHITECTURE
    model = Sequential([
        # 輸入層 + 隱藏層 1 (60 inputs -> 24 neurons)
        Dense(units=24, activation='relu', input_dim=60),
        # 隨機失活層 (Dropout 20%) 防止參數過擬合
        Dropout(0.2),
        # 隱藏層 2 (12 neurons)
        Dense(units=12, activation='relu'),
        # 輸出層 (1 neuron, 輸出 0~1 的機率值)
        Dense(units=1, activation='sigmoid')
    ])

    # 編譯模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("New model architecture built and compiled.")

    # TRAIN
    print("Beginning Training (Epochs=35)...")
    history = model.fit(X_train, y_train, batch_size=8, epochs=35, verbose=0)
    print("Training complete.")

    # EVALUATE
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy*100:.2f}%")

    # SAVE EXPORTS
    model.save('sonar_classifier.h5')
    np.save('loss_history.npy', history.history['loss'])
    print("Saved model brain and loss history exported.")

if __name__ == "__main__":
    train_new_model()
2. 數學探討與超參數管道特徵 Z-Score 標準化聲納反射波在不同頻率下的振幅落差極大，若直接輸入會導致梯度震盪。本系統透過 StandardScaler 將所有特徵縮放：$$x_{scaled} = \frac{x - \mu}{\sigma}$$$\mu$ (Mean): scaler_mean.npy$\sigma$ (Standard Deviation): scaler_std.npy
3Web UI 戰情室對接指南
Input Grid Mapping: 前端 HTML 的 60 個「Array Signal Input」輸入框數值，必須在 JavaScript 或後端讀取 scaler_mean.npy 與 scaler_std.npy 進行縮放後，方可送入 sonar_classifier.h5 進行 model.predict()。

Loss Visualization: 右側的 ApexCharts 折線圖可直接加載 loss_history.npy 的 35 個浮點數數值，即時動態呈現模型的收斂軌跡。
## 檔案二：`cancer_model.md`

```markdown
#  Project: BioMatrix Cancer Classifier (Medical Diagnostics Core)

本模組為 **BioMatrix NeuroMatrix OS** 的核心生醫診斷引擊。透過深度學習技術，針對穿刺檢查提取的細胞核多維幾何特徵進行非線性擬合，精準評估腫瘤屬於「良性（Benign）」或「惡性（Malignant）」。

1. 核心源碼 (`cancer_model.py`)

```python
# ==========================================
# cancer_model.py
# CREATED: 2026
# Project: BioMatrix Cancer Classifier
# ==========================================
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# SETUP: 確保實驗可重現性
np.random.seed(2026)
tf.random.set_seed(2026)

def train_cancer_model():
    print("\n--- BioMatrix Cancer AI Training Sequence Initiated ---")

    # LOAD DATASET (威斯康辛大學乳腺癌資料集)
    try:
        url = "[https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)"
        df = pd.read_csv(url, header=None)
        print("Medical Bio-Data loaded successfully from UCI Archive.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # PREPROCESSING
    # 第 0 欄為 ID (捨棄); 第 1 欄為標籤 ('M'=惡性, 'B'=良性); 第 2~31 欄為 30 維細胞核物理特徵
    X = df.iloc[:, 2:32].values
    y = df.iloc[:, 1].values

    # 將文字標籤編碼 (M -> 1, B -> 0)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # 劃分訓練與測試集 (80% / 20% 醫療黃金切分比)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=2026)

    # 特徵標準化 (消除半徑、面積、周長等不同量綱帶來的巨大單位落差)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # 儲存預處理參數，供前端 UI 轉換使用者輸入的檢驗數據
    np.save('cancer_scaler_mean.npy', sc.mean_)
    np.save('cancer_scaler_std.npy', sc.scale_)
    print("Preprocessors (Medical Scaler) finalized.")

    # BUILD THE ARCHITECTURE (高度防過擬合微縮架構)
    model = Sequential([
        # 輸入層 + 隱藏層 1 (30 inputs -> 16 neurons)
        Dense(units=16, activation='relu', input_dim=30),
        # 提高隨機失活率至 25%，強制模型學習泛化特徵
        Dropout(0.25),
        # 隱藏層 2 (8 neurons)
        Dense(units=8, activation='relu'),
        # 輸出層 (1 neuron, 惡性機率預估)
        Dense(units=1, activation='sigmoid')
    ])

    # 編编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("BioNet architecture built and compiled.")

    # TRAIN
    print("Beginning Medical AI Training (Epochs=40)...")
    history = model.fit(X_train, y_train, batch_size=16, epochs=40, verbose=0)
    print("Training complete.")

    # EVALUATE
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Diagnostic Test Accuracy: {accuracy*100:.2f}%")

    # SAVE EXPORTS
    model.save('cancer_classifier.h5')
    np.save('cancer_loss_history.npy', history.history['loss'])
    print("Saved model brain and medical loss history exported.")

if __name__ == "__main__":
    train_cancer_model()
2. 醫療數據特化與防守型架構
多量綱特徵的挑戰
本資料集包含細胞核的 radius（半徑，十位數）、area（面積，千位數）與 smoothness（平滑度，零點幾）。如果未經 Z-score 標準化，神經網路的權重更新將完全被 area 主導，導致模型崩潰。

防過擬合機制（Regularization）
相較於聲納模型的 60 維輸入，癌症模型僅有 30 維。為了避免小樣本高特徵帶來的惡性「過擬合（Overfitting）」：

收縮隱藏層寬度： 將神經元縮減為 16 -> 8，限制模型的自由度。

拉高 Dropout 比率： 使用 Dropout(0.25)，在每次訓練迭代中隨機關閉 25% 的神經元，強迫網路提取最具魯棒性的生物特徵。

3. Web UI 戰情室對接指南
Input Grid Mapping: 配合前端 UI，原先 60 個聲納格請等比例縮減為 30 個生醫特徵輸入框（如病患報告中的細胞核半徑、周長等）。

## [https://gemini.google.com/app/6de837f53f7ccbd1?hl=zh-TW]
