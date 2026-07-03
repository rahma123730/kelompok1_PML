import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
# Dataset UCI "Predict Students' Dropout and Academic Success" biasanya
# menggunakan separator titik koma (;). Kita coba deteksi otomatis.
try:
    df = pd.read_csv("data.csv", sep=";")
    if df.shape[1] == 1:
        # fallback kalau ternyata separatornya koma
        df = pd.read_csv("data.csv", sep=",")
except Exception as e:
    raise SystemExit(f"Gagal membaca data.csv: {e}")

print("Shape awal:", df.shape)

# 2. Data quality checking
print(df.info())
print("Missing values:\n", df.isnull().sum())
print("Jumlah duplikat:", df.duplicated().sum())

# 3. Cleaning sederhana
df = df.drop_duplicates()

for col in df.select_dtypes(include=["number"]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Pisahkan fitur dan target
if "Target" not in df.columns:
    raise SystemExit("Kolom 'Target' tidak ditemukan di dataset.")

y = df["Target"]
X = df.drop(columns=["Target"])

# Encoding kategorikal
X = pd.get_dummies(X, drop_first=True)

# Simpan urutan/nama fitur hasil encoding, dibutuhkan saat inference
feature_names = X.columns.tolist()

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Modeling
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Simpan model dan feature names
with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\nModel disimpan sebagai 'rf_model.pkl'")
print("Daftar fitur disimpan sebagai 'feature_names.pkl'")