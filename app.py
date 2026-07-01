import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="onTrack - StudyGuard AI", layout="wide", page_icon="🚀")

# =========================================================
# INFORMASI PENGEMBANG (GANTI SESUAI DATA ANDA)
# =========================================================
DEVELOPER_NAME = ""
DEVELOPER_NIM = ""
DEVELOPER_EMAIL = ""

# =========================================================
# CUSTOM CSS (Dark Mode, Glassmorphism, Neon)
# =========================================================
st.markdown("""
<style>
/* ================= FONT IMPORT ================= */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', 'Poppins', sans-serif !important;
}

/* ================= BACKGROUND HALAMAN ================= */
.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(80, 70, 230, 0.25), transparent 35%),
                radial-gradient(circle at 90% 15%, rgba(30, 180, 255, 0.22), transparent 40%),
                radial-gradient(circle at 15% 90%, rgba(140, 60, 255, 0.20), transparent 40%),
                radial-gradient(circle at 95% 95%, rgba(0, 200, 255, 0.18), transparent 35%),
                linear-gradient(160deg, #0a0a1a 0%, #12102b 40%, #150a2e 70%, #0a0a1a 100%);
    background-attachment: fixed;
    background-size: cover;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(2px 2px at 20% 30%, rgba(255,255,255,0.5) 0%, transparent 60%),
        radial-gradient(1.5px 1.5px at 70% 60%, rgba(120,180,255,0.6) 0%, transparent 60%),
        radial-gradient(2px 2px at 85% 20%, rgba(180,120,255,0.5) 0%, transparent 60%),
        radial-gradient(1px 1px at 40% 80%, rgba(255,255,255,0.4) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ================= JUDUL ================= */
.big-title, h1 {
    color: #ffffff !important;
    font-weight: 800 !important;
    text-shadow:
        0 0 10px rgba(80, 160, 255, 0.8),
        0 0 25px rgba(140, 90, 255, 0.6),
        0 0 45px rgba(80, 160, 255, 0.35);
    letter-spacing: 0.5px;
}

.sub-title {
    color: rgba(255, 255, 255, 0.65) !important;
    text-shadow: 0 0 12px rgba(120, 150, 255, 0.25);
}

.desc-text {
    color: rgba(255, 255, 255, 0.75) !important;
    font-size: 15px;
    line-height: 1.6;
}

h2, h3, h4, label, p, span, div {
    color: #e8e9f5;
}

/* ================= KARTU GLASSMORPHISM ================= */
.card, .result-card, .stForm, div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(90, 160, 255, 0.35) !important;
    box-shadow:
        0 0 20px rgba(80, 130, 255, 0.12),
        inset 0 0 20px rgba(255, 255, 255, 0.02) !important;
    padding: 24px !important;
}

.result-card {
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    box-shadow:
        0 0 30px rgba(120, 150, 255, 0.35),
        0 8px 32px rgba(0, 0, 0, 0.4) !important;
}

.card-dropout { background: linear-gradient(135deg, #e63946, #b3212f); }
.card-enrolled { background: linear-gradient(135deg, #f4a300, #c98400); }
.card-graduate { background: linear-gradient(135deg, #2a9d8f, #1e7268); }

.result-card h1 { font-size: 30px; margin-bottom: 4px; }
.result-card p { font-size: 14px; opacity: 0.9; }

/* ================= REKOMENDASI AKSI ================= */
.recommend-box {
    background: rgba(255, 255, 255, 0.06) !important;
    border-left: 4px solid #57e0ff !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    margin-top: 10px !important;
}

/* ================= SIDEBAR ================= */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 35, 0.55) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-right: 1px solid rgba(100, 150, 255, 0.2) !important;
}
section[data-testid="stSidebar"] * {
    color: #e8e9f5 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(100, 150, 255, 0.25) !important;
}

/* ================= INPUT TEXT / NUMBER ================= */
.stTextInput input, .stNumberInput input, textarea {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(90, 160, 255, 0.35) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border: 1px solid rgba(120, 190, 255, 0.9) !important;
    box-shadow: 0 0 12px rgba(80, 160, 255, 0.5) !important;
}

/* ================= SELECTBOX ================= */
div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(90, 160, 255, 0.35) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    box-shadow: none !important;
}
div[data-baseweb="select"] > div:hover {
    border: 1px solid rgba(120, 190, 255, 0.8) !important;
}
div[data-baseweb="popover"] ul {
    background: rgba(15, 12, 35, 0.95) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(90, 160, 255, 0.3) !important;
}
div[data-baseweb="popover"] li {
    color: #e8e9f5 !important;
}
div[data-baseweb="popover"] li:hover {
    background: rgba(80, 130, 255, 0.25) !important;
}

/* ================= SLIDER ================= */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #57e0ff !important;
    box-shadow: 0 0 10px rgba(87, 224, 255, 0.9), 0 0 20px rgba(120, 90, 255, 0.6) !important;
    border: 2px solid #ffffff !important;
}
div[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, #4facfe 0%, #8a5cff 100%) !important;
}
div[data-testid="stSlider"] > div > div {
    background: rgba(255, 255, 255, 0.12) !important;
}
div[data-testid="stSlider"] span {
    color: #a9c8ff !important;
    text-shadow: 0 0 6px rgba(80, 160, 255, 0.6);
}

/* ================= PROGRESS BAR ================= */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #4facfe 0%, #8a5cff 100%) !important;
    box-shadow: 0 0 10px rgba(120, 150, 255, 0.6) !important;
}
div[data-testid="stProgress"] {
    background: rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
}

/* ================= TOMBOL / BUTTON ================= */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 60%, #a855f7 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.6em 1.4em !important;
    box-shadow: 0 0 15px rgba(120, 90, 255, 0.4) !important;
    transition: all 0.25s ease-in-out !important;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
    box-shadow:
        0 0 25px rgba(90, 160, 255, 0.9),
        0 0 45px rgba(140, 90, 255, 0.7) !important;
    transform: translateY(-2px) scale(1.02) !important;
    filter: brightness(1.1) !important;
}
.stButton > button:active {
    transform: translateY(0px) scale(0.98) !important;
}

/* Tombol reset dibuat lebih netral (abu-abu) */
.reset-btn button {
    background: rgba(255, 255, 255, 0.08) !important;
    box-shadow: none !important;
}
.reset-btn button:hover {
    box-shadow: 0 0 12px rgba(255, 255, 255, 0.25) !important;
}

/* ================= RADIO (NAVIGASI SIDEBAR) ================= */
div[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(90, 160, 255, 0.2) !important;
    border-radius: 12px !important;
    padding: 8px 12px !important;
    margin-bottom: 6px !important;
    transition: all 0.2s ease-in-out !important;
}
div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    border: 1px solid rgba(120, 190, 255, 0.6) !important;
    box-shadow: 0 0 10px rgba(90, 160, 255, 0.3) !important;
}

/* ================= FILE UPLOADER ================= */
section[data-testid="stFileUploaderDropzone"] {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1.5px dashed rgba(90, 160, 255, 0.5) !important;
    border-radius: 16px !important;
}

/* ================= DATAFRAME / TABLE ================= */
div[data-testid="stDataFrame"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(90, 160, 255, 0.25) !important;
}

/* ================= METRIC ================= */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(90, 160, 255, 0.3) !important;
    border-radius: 16px !important;
    padding: 16px !important;
}

/* ================= ALERT (info / warning / success) ================= */
div[data-testid="stAlert"] {
    background: rgba(255, 255, 255, 0.06) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(90, 160, 255, 0.3) !important;
    color: #ffffff !important;
}

/* ================= EXPANDER ================= */
div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(90, 160, 255, 0.25) !important;
    border-radius: 16px !important;
}

/* ================= FOOTER ================= */
.footer-box {
    text-align: center;
    padding: 18px 0 8px 0;
    color: rgba(255,255,255,0.45) !important;
    font-size: 13px;
    border-top: 1px solid rgba(90, 160, 255, 0.15);
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL & FEATURE NAMES
# =========================================================
@st.cache_resource
def load_model():
    try:
        with open("rf_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        return model, feature_names
    except FileNotFoundError:
        st.error(
            "File model tidak ditemukan. Pastikan 'rf_model.pkl' dan "
            "'feature_names.pkl' ada di folder yang sama, atau jalankan "
            "train_model.py terlebih dahulu."
        )
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.stop()


model, feature_names = load_model()


def prepare_input(df_raw):
    """Encode dataframe input agar kolomnya sama persis dengan saat training."""
    df_encoded = pd.get_dummies(df_raw, drop_first=True)
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    return df_encoded


CARD_CLASS = {
    "Dropout": "card-dropout",
    "Enrolled": "card-enrolled",
    "Graduate": "card-graduate",
}
EMOJI = {
    "Dropout": "🔴",
    "Enrolled": "🟡",
    "Graduate": "🟢",
}
REKOMENDASI_AKSI = {
    "Dropout": "🚨 Segera hubungi dosen wali untuk sesi konseling akademik. "
               "Periksa juga status pembayaran UKT dan beban mata kuliah mahasiswa.",
    "Enrolled": "🟡 Pantau perkembangan akademik mahasiswa secara berkala. "
                "Disarankan ada check-in ringan dari dosen wali tiap akhir semester.",
    "Graduate": "🟢 Mahasiswa berada pada jalur yang baik. "
                "Tidak diperlukan intervensi khusus, cukup pemantauan rutin.",
}

# Default nilai form untuk keperluan Reset Form
DEFAULT_VALUES = {
    "in_age": 20,
    "in_gender": 1,
    "in_scholarship": 0,
    "in_debtor": 0,
    "in_tuition": 1,
    "in_admission_grade": 120.0,
    "in_cu1_enrolled": 6,
    "in_cu1_approved": 6,
    "in_cu1_grade": 13.0,
    "in_cu1_eval": 6,
    "in_cu2_enrolled": 6,
    "in_cu2_approved": 6,
    "in_cu2_grade": 13.0,
    "in_cu2_eval": 6,
}

for k, v in DEFAULT_VALUES.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_form():
    for k, v in DEFAULT_VALUES.items():
        st.session_state[k] = v
    st.session_state.pop("last_input_encoded", None)
    st.session_state.pop("last_prediction", None)


# =========================================================
# SIDEBAR NAVIGASI
# =========================================================
with st.sidebar:
    st.markdown("## 🚀 onTrack")
    st.caption("StudyGuard AI — Prediksi Risiko Dropout Mahasiswa")
    st.markdown("---")
    menu = st.radio(
        "Navigasi",
        ["🏠 Beranda", "📝 Input Manual", "📂 Upload Batch", "📊 Dashboard", "ℹ️ Tentang Aplikasi"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("© 2026 onTrack — StudyGuard AI")


# =========================================================
# HEADER (tampil di semua halaman)
# =========================================================
st.markdown('<p class="big-title">onTrack 🚀</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Prediksi Risiko Dropout Mahasiswa Berbasis Machine Learning</p>', unsafe_allow_html=True)


# =========================================================
# HALAMAN BERANDA (LANDING PAGE)
# =========================================================
if menu == "🏠 Beranda":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <p class="desc-text">
    <b>onTrack</b> adalah aplikasi berbasis <i>Machine Learning</i> yang membantu program studi,
    dosen wali, dan staf akademik untuk <b>mendeteksi risiko dropout mahasiswa lebih awal</b>.
    Dengan menganalisis data akademik, demografis, dan sosial-ekonomi mahasiswa, sistem ini
    memprediksi status mahasiswa ke dalam tiga kategori: <b>Dropout</b>, <b>Enrolled</b>, dan
    <b>Graduate</b>, sehingga pendampingan akademik dapat dilakukan secara <b>preventif</b>,
    bukan reaktif.
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📖 Cara Menggunakan Aplikasi"):
        st.markdown("""
**A. Input Manual (untuk 1 mahasiswa)**
1. Buka menu **📝 Input Manual** di sidebar.
2. Isi data mahasiswa pada kolom kiri: usia, jenis kelamin, status beasiswa, status hutang,
   status UKT, nilai masuk, serta data akademik semester 1 dan 2.
3. Klik tombol **🔍 Prediksi Sekarang**.
4. Hasil prediksi beserta tingkat keyakinan model dan rekomendasi aksi akan muncul di kolom kanan.
5. Klik **🔎 Analisis Faktor Risiko** untuk melihat 3 faktor paling berpengaruh terhadap prediksi tersebut.
6. Jika ingin mengulang dari awal, klik tombol **♻️ Reset Form**.

**B. Upload Batch (untuk banyak mahasiswa sekaligus)**
1. Buka menu **📂 Upload Batch** di sidebar.
2. Siapkan file CSV berisi data mahasiswa dengan format kolom yang sama seperti data latih model.
3. Upload file melalui tombol **Pilih file CSV**.
4. Sistem akan otomatis memproses dan menampilkan hasil prediksi untuk seluruh baris data.
5. Klik **⬇️ Download Laporan CSV** untuk menyimpan hasil, atau **⬇️ Download Summary PDF**
   untuk ringkasan singkat.

**C. Dashboard**
1. Buka menu **📊 Dashboard** untuk melihat ringkasan jumlah prediksi dan grafik interaktif
   faktor-faktor yang paling memengaruhi keputusan model.
        """)
    st.stop()


# =========================================================
# HALAMAN: INPUT MANUAL
# =========================================================
if menu == "📝 Input Manual":

    col_input, col_result = st.columns(2)

    # ---------------- KOLOM KIRI: INPUT ----------------
    with col_input:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Data Mahasiswa")

        # >>> GANTI nama fitur di sini jika berbeda dengan dataset Anda <<<
        st.slider("Usia saat masuk (Age at enrollment)", 15, 60, key="in_age")
        st.selectbox("Jenis Kelamin (Gender)", options=[1, 0],
                     format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan", key="in_gender")
        st.selectbox("Penerima Beasiswa (Scholarship holder)", options=[0, 1],
                     format_func=lambda x: "Ya" if x == 1 else "Tidak", key="in_scholarship")
        st.selectbox("Status Hutang (Debtor)", options=[0, 1],
                     format_func=lambda x: "Ya" if x == 1 else "Tidak", key="in_debtor")
        st.selectbox("UKT Lunas (Tuition fees up to date)", options=[1, 0],
                     format_func=lambda x: "Lunas" if x == 1 else "Belum Lunas", key="in_tuition")
        st.slider("Nilai Masuk (Admission grade)", 0.0, 200.0, key="in_admission_grade")

        st.markdown("**Semester 1**")
        st.slider("Mata kuliah diambil (1st sem enrolled)", 0, 12, key="in_cu1_enrolled")
        st.slider("Mata kuliah lulus (1st sem approved)", 0, 12, key="in_cu1_approved")
        st.slider("Rata-rata nilai (1st sem grade)", 0.0, 20.0, key="in_cu1_grade")
        st.slider("Jumlah evaluasi (1st sem evaluations)", 0, 20, key="in_cu1_eval")

        st.markdown("**Semester 2**")
        st.slider("Mata kuliah diambil (2nd sem enrolled)", 0, 12, key="in_cu2_enrolled")
        st.slider("Mata kuliah lulus (2nd sem approved)", 0, 12, key="in_cu2_approved")
        st.slider("Rata-rata nilai (2nd sem grade)", 0.0, 20.0, key="in_cu2_grade")
        st.slider("Jumlah evaluasi (2nd sem evaluations)", 0, 20, key="in_cu2_eval")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_clicked = st.button("🔍 Prediksi Sekarang", use_container_width=True)
        with col_btn2:
            st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
            st.button("♻️ Reset Form", use_container_width=True, on_click=reset_form)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- KOLOM KANAN: HASIL ----------------
    with col_result:
        if not predict_clicked and "last_prediction" not in st.session_state:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.info("👈 Isi data mahasiswa di sebelah kiri, lalu klik **Prediksi Sekarang** untuk melihat hasilnya di sini.")
            st.markdown('</div>', unsafe_allow_html=True)

        if predict_clicked:
            try:
                s = st.session_state
                # >>> Sesuaikan mapping ini dengan kolom asli dataset Anda <<<
                input_dict = {
                    "Marital status": 1,
                    "Application mode": 1,
                    "Application order": 1,
                    "Course": 171,
                    "Daytime/evening attendance\t": 1,
                    "Previous qualification": 1,
                    "Previous qualification (grade)": s["in_admission_grade"],
                    "Nacionality": 1,
                    "Mother's qualification": 1,
                    "Father's qualification": 1,
                    "Mother's occupation": 5,
                    "Father's occupation": 5,
                    "Admission grade": s["in_admission_grade"],
                    "Displaced": 0,
                    "Educational special needs": 0,
                    "Debtor": s["in_debtor"],
                    "Tuition fees up to date": s["in_tuition"],
                    "Gender": s["in_gender"],
                    "Scholarship holder": s["in_scholarship"],
                    "Age at enrollment": s["in_age"],
                    "International": 0,
                    "Curricular units 1st sem (credited)": 0,
                    "Curricular units 1st sem (enrolled)": s["in_cu1_enrolled"],
                    "Curricular units 1st sem (evaluations)": s["in_cu1_eval"],
                    "Curricular units 1st sem (approved)": s["in_cu1_approved"],
                    "Curricular units 1st sem (grade)": s["in_cu1_grade"],
                    "Curricular units 1st sem (without evaluations)": 0,
                    "Curricular units 2nd sem (credited)": 0,
                    "Curricular units 2nd sem (enrolled)": s["in_cu2_enrolled"],
                    "Curricular units 2nd sem (evaluations)": s["in_cu2_eval"],
                    "Curricular units 2nd sem (approved)": s["in_cu2_approved"],
                    "Curricular units 2nd sem (grade)": s["in_cu2_grade"],
                    "Curricular units 2nd sem (without evaluations)": 0,
                    "Unemployment rate": 10.8,
                    "Inflation rate": 1.4,
                    "GDP": 1.74,
                }

                input_df = pd.DataFrame([input_dict])
                input_encoded = prepare_input(input_df)

                prediction = model.predict(input_encoded)[0]
                proba = model.predict_proba(input_encoded)[0]
                proba_dict = dict(zip(model.classes_, proba))

                st.session_state["last_input_encoded"] = input_encoded
                st.session_state["last_prediction"] = prediction
                st.session_state["last_proba_dict"] = proba_dict

            except Exception as e:
                st.warning(f"Prediksi gagal dilakukan, silakan periksa kembali data yang diisi. Detail: {e}")

        if "last_prediction" in st.session_state:
            prediction = st.session_state["last_prediction"]
            proba_dict = st.session_state["last_proba_dict"]
            confidence = proba_dict.get(prediction, 0.0)

            card_class = CARD_CLASS.get(prediction, "card-enrolled")
            emoji = EMOJI.get(prediction, "⚪")

            st.markdown(f"""
            <div class="result-card {card_class}">
                <p>Hasil Prediksi Status Mahasiswa</p>
                <h1>{emoji} {prediction}</h1>
                <p>Tingkat Keyakinan Model: {confidence*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Probabilitas per Kategori**")
            for cls, p in sorted(proba_dict.items(), key=lambda x: -x[1]):
                st.write(f"{EMOJI.get(cls, '⚪')} {cls}")
                st.progress(float(p))
                st.caption(f"{p*100:.1f}%")

            st.markdown("**Rekomendasi Aksi**")
            st.markdown(
                f'<div class="recommend-box">{REKOMENDASI_AKSI.get(prediction, "Tidak ada rekomendasi khusus.")}</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            if st.button("🔎 Analisis Faktor Risiko", use_container_width=True):
                try:
                    importances = model.feature_importances_
                    imp_df = pd.DataFrame({
                        "Fitur": feature_names,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=False).head(3)

                    st.markdown("**3 Faktor Paling Berpengaruh untuk Mahasiswa Ini:**")
                    for _, row in imp_df.iterrows():
                        st.write(f"- **{row['Fitur']}** (bobot pengaruh: {row['Importance']*100:.1f}%)")
                except Exception as e:
                    st.warning(f"Analisis faktor risiko tidak dapat ditampilkan. Detail: {e}")
            st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# HALAMAN: UPLOAD BATCH
# =========================================================
elif menu == "📂 Upload Batch":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📂 Prediksi Massal dari File CSV")
    st.write("Upload file CSV berisi data mahasiswa (format kolom sama seperti dataset training).")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.warning("Silakan upload file CSV terlebih dahulu untuk melihat hasil prediksi.")
    else:
        try:
            try:
                df_upload = pd.read_csv(uploaded_file, sep=";")
                if df_upload.shape[1] == 1:
                    uploaded_file.seek(0)
                    df_upload = pd.read_csv(uploaded_file, sep=",")
            except Exception:
                uploaded_file.seek(0)
                df_upload = pd.read_csv(uploaded_file, sep=",")

            if df_upload.empty:
                st.warning("File CSV yang diupload kosong. Silakan upload file lain.")
            else:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("Preview data:")
                st.dataframe(df_upload.head(), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                df_features = df_upload.drop(columns=["Target"], errors="ignore")

                for col in df_features.select_dtypes(include=["number"]).columns:
                    df_features[col] = df_features[col].fillna(df_features[col].median())
                for col in df_features.select_dtypes(include=["object"]).columns:
                    df_features[col] = df_features[col].fillna(df_features[col].mode()[0])

                input_encoded = prepare_input(df_features)

                predictions = model.predict(input_encoded)
                proba = model.predict_proba(input_encoded)

                result_df = df_upload.copy()
                result_df["Predicted Status"] = predictions
                for i, cls in enumerate(model.classes_):
                    result_df[f"Prob_{cls}"] = (proba[:, i] * 100).round(2)

                st.session_state["last_batch_result"] = result_df

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.success(f"Berhasil memprediksi {len(result_df)} baris data.")
                st.dataframe(result_df, use_container_width=True)

                col_dl1, col_dl2 = st.columns(2)

                with col_dl1:
                    csv_result = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Laporan CSV",
                        data=csv_result,
                        file_name="hasil_prediksi_onTrack.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col_dl2:
                    counts = result_df["Predicted Status"].value_counts()
                    summary_lines = [
                        "RINGKASAN PREDIKSI - onTrack StudyGuard AI",
                        f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"Total data diproses: {len(result_df)}",
                        "",
                        "Distribusi Prediksi:",
                    ]
                    for status, count in counts.items():
                        pct = count / len(result_df) * 100
                        summary_lines.append(f"- {status}: {count} mahasiswa ({pct:.1f}%)")

                    summary_text = "\n".join(summary_lines)

                    st.download_button(
                        "⬇️ Download Summary PDF",
                        data=summary_text.encode("utf-8"),
                        file_name="summary_onTrack.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="File ringkasan berupa teks biasa (.txt), bukan PDF asli."
                    )
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"File tidak dapat diproses. Pastikan format CSV sudah sesuai. Detail: {e}")


# =========================================================
# HALAMAN: DASHBOARD
# =========================================================
elif menu == "📊 Dashboard":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Dashboard Analisis Model")
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        batch_result = st.session_state.get("last_batch_result")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            total_pred = len(batch_result) if batch_result is not None else 0
            st.metric("Total Prediksi Batch", total_pred)
        with m2:
            n_dropout = int((batch_result["Predicted Status"] == "Dropout").sum()) if batch_result is not None else 0
            st.metric("Prediksi Dropout", n_dropout)
        with m3:
            n_enrolled = int((batch_result["Predicted Status"] == "Enrolled").sum()) if batch_result is not None else 0
            st.metric("Prediksi Enrolled", n_enrolled)
        with m4:
            n_graduate = int((batch_result["Predicted Status"] == "Graduate").sum()) if batch_result is not None else 0
            st.metric("Prediksi Graduate", n_graduate)

        if batch_result is None:
            st.info("Belum ada data batch. Upload CSV di menu **Upload Batch** untuk melihat statistik prediksi di sini.")

        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Fitur": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Top Feature Importance**")
            top_n = st.slider("Jumlah fitur teratas", min_value=5, max_value=30, value=10)
            top_features = importance_df.head(top_n).sort_values(by="Importance", ascending=True)

            fig1 = px.bar(
                top_features,
                x="Importance",
                y="Fitur",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                title="Fitur Paling Berpengaruh terhadap Prediksi"
            )
            fig1.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Distribusi Status Prediksi**")

            if batch_result is not None:
                dist = batch_result["Predicted Status"].value_counts().reset_index()
                dist.columns = ["Status", "Jumlah"]
                title_chart = "Distribusi Hasil Prediksi Batch Terakhir"
            else:
                class_labels = list(model.classes_) if hasattr(model, "classes_") else []
                dist = pd.DataFrame({"Status": class_labels, "Jumlah": [1] * len(class_labels)})
                title_chart = "Distribusi Kelas (Placeholder — belum ada data batch)"

            fig2 = px.bar(
                dist,
                x="Status",
                y="Jumlah",
                color="Status",
                color_discrete_map={
                    "Dropout": "#e63946",
                    "Enrolled": "#f4a300",
                    "Graduate": "#2a9d8f"
                },
                title=title_chart
            )
            fig2.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Dashboard tidak dapat ditampilkan sepenuhnya. Detail: {e}")


# =========================================================
# HALAMAN: TENTANG APLIKASI
# =========================================================
elif menu == "ℹ️ Tentang Aplikasi":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ Tentang Aplikasi onTrack")

    try:
        n_features = len(feature_names)
    except Exception:
        n_features = "Tidak diketahui"

    st.markdown(f"""
    <p class="desc-text">
    <b>Nama Aplikasi:</b> onTrack — StudyGuard AI<br>
    <b>Model Machine Learning:</b> Random Forest Classifier<br>
    <b>Jumlah Fitur yang Digunakan:</b> {n_features} fitur<br>
    <b>Target Prediksi:</b> Dropout, Enrolled, Graduate<br><br>
    <b>Kontak Pengembang</b><br>
    Nama: {DEVELOPER_NAME}<br>
    NIM: {DEVELOPER_NIM}<br>
    Email: {DEVELOPER_EMAIL}
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer-box">Copyright © 2026 - onTrack</div>',
    unsafe_allow_html=True
)
