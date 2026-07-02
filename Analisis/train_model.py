"""
train_model.py
================
Script untuk melatih model Random Forest guna memprediksi status akademik
mahasiswa (Dropout / Enrolled / Graduate).

PENTING:
Script ini menggunakan dataset MENTAH (data.csv, sebelum di-scaling),
karena aplikasi Streamlit (app.py) membangun input prediksi dalam bentuk
nilai asli (misalnya usia 15-60, nilai masuk 0-200, dst), bukan nilai
yang sudah distandardisasi (z-score). Random Forest juga tidak
memerlukan feature scaling, jadi pendekatan ini sudah tepat.

Output:
- rf_model.pkl       : model Random Forest yang sudah dilatih
- feature_names.pkl  : daftar nama fitur (urutan kolom) yang digunakan model
                        (perhatikan: "feature_names" dengan huruf "s", sesuai
                        yang dipanggil oleh app.py)

Cara menjalankan:
    python train_model.py
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
# Dataset UCI "Predict Students' Dropout and Academic Success" biasanya
# menggunakan separator titik koma (;). Kita coba deteksi otomatis.
DATA_PATH = "data.csv"

try:
    df = pd.read_csv(DATA_PATH, sep=";")
    if df.shape[1] == 1:
        # Fallback kalau ternyata separatornya koma
        df = pd.read_csv(DATA_PATH, sep=",")
except Exception as e:
    raise SystemExit(f"Gagal membaca {DATA_PATH}: {e}")

print("=" * 60)
print("INFO DATASET")
print("=" * 60)
print(f"Jumlah baris  : {df.shape[0]}")
print(f"Jumlah kolom  : {df.shape[1]}")
print(f"Missing value : {df.isnull().sum().sum()}")
print(f"Duplikat      : {df.duplicated().sum()}")
print("\nDistribusi target:")
print(df["Target"].value_counts())

# ------------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------------
df = df.drop_duplicates()

for col in df.select_dtypes(include=["number"]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ------------------------------------------------------------------
# 3. PISAHKAN FEATURE (X) DAN TARGET (y)
# ------------------------------------------------------------------
TARGET_COL = "Target"

if TARGET_COL not in df.columns:
    raise SystemExit(f"Kolom '{TARGET_COL}' tidak ditemukan di dataset.")

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# Encoding kategorikal (tidak berpengaruh jika semua kolom sudah numerik,
# tapi tetap dijalankan untuk konsistensi dengan app.py yang juga memanggil
# pd.get_dummies() saat menyiapkan input prediksi).
X = pd.get_dummies(X, drop_first=True)

# Simpan urutan/nama fitur hasil encoding, dibutuhkan saat inference
feature_names = X.columns.tolist()

# ------------------------------------------------------------------
# 4. SPLIT DATA TRAIN / TEST
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("PEMBAGIAN DATA")
print("=" * 60)
print(f"Data latih : {X_train.shape[0]} baris")
print(f"Data uji   : {X_test.shape[0]} baris")

# ------------------------------------------------------------------
# 5. TRAINING MODEL RANDOM FOREST
# ------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

model.fit(X_train, y_train)

# ------------------------------------------------------------------
# 6. EVALUASI MODEL
# ------------------------------------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("HASIL EVALUASI MODEL")
print("=" * 60)
print(f"Akurasi: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (opsional, untuk insight)
# ------------------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)

print("\n" + "=" * 60)
print("TOP 10 FITUR PALING PENTING")
print("=" * 60)
print(importances.head(10))

# ------------------------------------------------------------------
# 8. SIMPAN MODEL DAN NAMA FITUR
# ------------------------------------------------------------------
with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)
print("Model disimpan sebagai       : rf_model.pkl")
print("Nama fitur disimpan sebagai  : feature_names.pkl")
