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

# 1. SETUP: 確保實驗可重現性
np.random.seed(2026)
tf.random.set_seed(2026)

def train_cancer_model():
    print("\n--- BioMatrix Cancer AI Training Sequence Initiated ---")

    # 2. LOAD DATASET (威斯康辛大學乳腺癌資料集)
    # 包含 30 種細胞核的幾何特徵，用來預測腫瘤為良性(B)或惡性(M)
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
        # 由於此資料集沒有欄位名稱，我們直接讀取
        df = pd.read_csv(url, header=None)
        print("Medical Bio-Data loaded successfully from UCI Archive.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. PREPROCESSING (資料預處理)
    # 第 0 欄是 ID (丟棄)
    # 第 1 欄是標籤 (M = 惡性 Malignant, B = 良性 Benign)
    # 第 2 到 31 欄是 30 個生醫特徵
    X = df.iloc[:, 2:32].values
    y = df.iloc[:, 1].values

    # 將文字標籤編碼 (M malignant -> 1, B benign -> 0)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # 切分訓練集與測試集 (對稱的專業黃金比例：80/20 劃分)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=2026)

    # 生醫特徵（如半徑、質地、周長）單位差異極大，必須進行 Z-score 標準化
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # 儲存預處理參數，供前端 Web UI 轉換用戶輸入的 30 維數據
    np.save('cancer_scaler_mean.npy', sc.mean_)
    np.save('cancer_scaler_std.npy', sc.scale_)
    print("Preprocessors (Medical Scaler) finalized.")

    # 4. BUILD THE ARCHITECTURE (架構設計)
    # 設計一個對應 30 維輸入的高效前饋神經網路
    model = Sequential([
        # 輸入層 + 隱藏層 1 (30 個生醫特徵輸入 -> 16 個神經元)
        Dense(units=16, activation='relu', input_dim=30),
        # 隨機失活層，防止在小樣本醫療數據上過擬合
        Dropout(0.25),
        # 隱藏層 2 (8 個神經元)
        Dense(units=8, activation='relu'),
        # 輸出層 (1 個神經元，輸出 0~1 的機率值。越接近 1 代表惡性風險越高)
        Dense(units=1, activation='sigmoid')
    ])

    # 編譯模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("BioNet architecture built and compiled.")

    # 5. TRAIN (模型訓練)
    # 使用 40 個 Epoch 與 Batch Size = 16 進行權重優化
    print("Beginning Medical AI Training (Epochs=40)...")
    history = model.fit(X_train, y_train, batch_size=16, epochs=40, verbose=0)
    print("Training complete.")

    # 6. EVALUATE (盲測評估)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Diagnostic Test Accuracy: {accuracy*100:.2f}%")

    # 7. SAVE (導出 AI 大腦與戰情數據)
    model.save('cancer_classifier.h5')
    np.save('cancer_loss_history.npy', history.history['loss'])
    print("Saved model brain to 'cancer_classifier.h5'")
    print("Medical loss history exported.")

if __name__ == "__main__":
    train_cancer_model()