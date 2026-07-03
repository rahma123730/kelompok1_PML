from datetime import datetime
from html import escape
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except ModuleNotFoundError:
    px = None

try:
    from streamlit_echarts import st_echarts
except ModuleNotFoundError:
    st_echarts = None


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
MODEL_PATH = BASE_DIR / "rf_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_names.pkl"
TARGET_COL = "Target"

# Batas pengaman untuk pengisian manual. Nilai ini tidak mengubah data latih atau model.
MAX_COURSES_PER_SEMESTER = 12

APP_NAME = "onTrack"
APP_SUBTITLE = "Machine Learning untuk Prediksi Status Akademik Mahasiswa"

STATUS_COLOR = {
    "Dropout": "#ee6666",
    "Enrolled": "#fac858",
    "Graduate": "#91cc75",
}

ECHART_COLORS = ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de", "#3ba272"]

RECOMMENDATION = {
    "Dropout": (
        "Mahasiswa perlu mendapat pendampingan segera. Periksa performa akademik, "
        "status pembayaran, dan lakukan konseling akademik dengan dosen wali."
    ),
    "Enrolled": (
        "Mahasiswa masih aktif, tetapi perlu dipantau secara berkala. Cek perkembangan "
        "nilai semester dan kehadiran agar risiko tidak meningkat."
    ),
    "Graduate": (
        "Mahasiswa berada pada pola akademik yang baik. Tetap lakukan monitoring rutin "
        "dan dukung penyelesaian studi tepat waktu."
    ),
}


st.set_page_config(
    page_title=f"{APP_NAME} - Prediksi Mahasiswa",
    page_icon=":mortar_board:",
    layout="wide",
)


def load_static_css() -> None:
    """Memuat CSS utama dari assets/styles.css.

    Tema Light/Dark dinamis tetap ditambahkan oleh inject_theme_css()
    setelah pengguna memilih mode tampilan pada sidebar.
    """
    css_path = BASE_DIR / "assets" / "styles.css"

    if not css_path.exists():
        st.error(
            "File CSS tidak ditemukan. Pastikan folder assets dan file "
            "assets/styles.css berada di lokasi yang sama dengan app.py."
        )
        st.stop()

    try:
        css_content = css_path.read_text(encoding="utf-8")
    except OSError as error:
        st.error(f"Gagal membaca assets/styles.css: {error}")
        st.stop()

    st.markdown(
        f"<style>\n{css_content}\n</style>",
        unsafe_allow_html=True,
    )


load_static_css()


def clean_label(value: str) -> str:
    return str(value).replace("\t", " ").strip()


def read_csv_auto(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file, sep=";")

    if df.shape[1] == 1:
        if hasattr(path_or_file, "seek"):
            path_or_file.seek(0)
        df = pd.read_csv(path_or_file, sep=",")

    return df


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("File data.csv tidak ditemukan. Simpan data.csv di folder yang sama dengan app.py.")
        st.stop()

    df = read_csv_auto(DATA_PATH)
    return df


@st.cache_resource
def load_artifacts():
    missing_files = [
        path.name for path in [MODEL_PATH, FEATURE_PATH] if not path.exists()
    ]

    if missing_files:
        st.error(
            "File model belum lengkap: "
            + ", ".join(missing_files)
            + ". Jalankan train_model.py terlebih dahulu."
        )
        st.stop()

    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        with open(FEATURE_PATH, "rb") as file:
            feature_names = pickle.load(file)

    except ModuleNotFoundError as error:
        st.error(
            f"Library {error.name} belum terinstall. Jalankan: pip install scikit-learn"
        )
        st.stop()
    except Exception as error:
        st.error(f"Gagal memuat model: {error}")
        st.stop()

    return model, feature_names


def find_column(df: pd.DataFrame, feature_name: str) -> str | None:
    if feature_name in df.columns:
        return feature_name

    clean_feature = clean_label(feature_name)

    for column in df.columns:
        if clean_label(column) == clean_feature:
            return column

    return None


def dataset_default(df: pd.DataFrame, feature_names: list[str]) -> dict:
    defaults = {}

    for feature in feature_names:
        column = find_column(df, feature)

        if column is None:
            defaults[feature] = 0
            continue

        series = df[column].dropna()

        if series.empty:
            defaults[feature] = 0
        elif pd.api.types.is_numeric_dtype(series):
            value = series.median()
            defaults[feature] = int(value) if pd.api.types.is_integer_dtype(series) else float(value)
        else:
            mode_value = series.mode(dropna=True)
            defaults[feature] = mode_value.iloc[0] if not mode_value.empty else "Unknown"

    return defaults


def unique_options(df: pd.DataFrame, feature_name: str) -> list:
    column = find_column(df, feature_name)

    if column is None:
        return []

    values = df[column].dropna().unique().tolist()
    return sorted(values)


def clean_features(df_features: pd.DataFrame) -> pd.DataFrame:
    df_features = df_features.copy()

    numeric_cols = df_features.select_dtypes(include=["number"]).columns
    non_numeric_cols = df_features.columns.difference(numeric_cols)

    for column in numeric_cols:
        median_value = df_features[column].median()
        fill_value = 0 if pd.isna(median_value) else median_value
        df_features[column] = df_features[column].fillna(fill_value)

    for column in non_numeric_cols:
        mode_value = df_features[column].mode(dropna=True)
        fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
        df_features[column] = df_features[column].fillna(fill_value)

    return df_features


def prepare_input(df_raw: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    df_raw = clean_features(df_raw)
    df_encoded = pd.get_dummies(df_raw, drop_first=True)
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    return df_encoded


def set_value(input_dict: dict, key: str, value):
    matched_key = None

    for feature in input_dict:
        if clean_label(feature) == clean_label(key):
            matched_key = feature
            break

    if matched_key is not None:
        input_dict[matched_key] = value


def get_input_value(input_dict: dict, key: str, default=None):
    '''Ambil nilai input dengan pencocokan nama fitur yang tahan tab/spasi.'''
    for feature, value in input_dict.items():
        if clean_label(feature) == clean_label(key):
            return value
    return default


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_number(value, decimals: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"

    if number.is_integer():
        return str(int(number))

    return f"{number:.{decimals}f}".replace(".", ",")


def binary_label(value, yes: str = "Ya", no: str = "Tidak") -> str:
    return yes if to_float(value) == 1 else no


def predict_from_input(input_dict: dict, model, feature_names: list[str]):
    '''Menjalankan prediksi untuk input manual maupun skenario simulasi.'''
    raw_input = pd.DataFrame([input_dict])
    model_input = prepare_input(raw_input, feature_names)
    prediction = model.predict(model_input)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        probability_dict = dict(zip(model.classes_, probabilities))
    else:
        probability_dict = {prediction: 1.0}

    return prediction, probability_dict, model_input


def clear_prediction_state():
    for state_key in [
        "last_manual_prediction",
        "last_manual_probability",
        "last_manual_input",
        "last_manual_raw_input",
        "last_simulation_result",
    ]:
        st.session_state.pop(state_key, None)


def apply_demo_preset(preset_name: str):
    '''Mengisi hanya kontrol formulir yang terlihat untuk kebutuhan demonstrasi.'''
    presets = {
        "Risiko tinggi": {
            "age": 24,
            "admission_grade": 105.0,
            "scholarship": 0,
            "debtor": 1,
            "tuition": 0,
            "cu1_enrolled": 7,
            "cu1_approved": 2,
            "cu1_eval": 6,
            "cu1_grade": 7.5,
            "cu2_enrolled": 7,
            "cu2_approved": 1,
            "cu2_eval": 5,
            "cu2_grade": 5.5,
        },
        "Stabil": {
            "age": 20,
            "admission_grade": 125.0,
            "scholarship": 0,
            "debtor": 0,
            "tuition": 1,
            "cu1_enrolled": 6,
            "cu1_approved": 5,
            "cu1_eval": 8,
            "cu1_grade": 12.5,
            "cu2_enrolled": 6,
            "cu2_approved": 5,
            "cu2_eval": 8,
            "cu2_grade": 12.5,
        },
        "Berprestasi": {
            "age": 19,
            "admission_grade": 145.0,
            "scholarship": 1,
            "debtor": 0,
            "tuition": 1,
            "cu1_enrolled": 6,
            "cu1_approved": 6,
            "cu1_eval": 9,
            "cu1_grade": 15.5,
            "cu2_enrolled": 6,
            "cu2_approved": 6,
            "cu2_eval": 9,
            "cu2_grade": 15.5,
        },
    }

    if preset_name == "Reset ke nilai median":
        for widget_key in [
            "age", "admission_grade", "scholarship", "debtor", "tuition",
            "cu1_enrolled", "cu1_approved", "cu1_eval", "cu1_grade",
            "cu2_enrolled", "cu2_approved", "cu2_eval", "cu2_grade",
        ]:
            st.session_state.pop(widget_key, None)
        st.session_state.pop("active_demo_preset", None)
    else:
        for widget_key, value in presets[preset_name].items():
            st.session_state[widget_key] = value
        st.session_state["active_demo_preset"] = preset_name

    clear_prediction_state()


def show_input_quick_guide():
    """Panduan ringkas agar pengguna tidak perlu menebak arti setiap isian formulir."""
    with st.expander("Bantuan pengisian cepat", expanded=False):
        st.markdown(
            """
            **Urutan yang disarankan:**
            1. Pilih preset untuk mencoba aplikasi, atau isi data mahasiswa yang akan dianalisis.
            2. Lengkapi data pendaftaran dan status keuangan.
            3. Isi data akademik Semester 1 dan Semester 2, lalu tekan **Prediksi Sekarang**.
            4. Gunakan hasil, faktor risiko, dan simulasi sebagai bahan diskusi atau tindak lanjut; bukan keputusan akademik otomatis.
            """
        )
        st.info(
            "Pada formulir ini, istilah 'mata kuliah' dipakai sebagai penyederhanaan dari "
            "'curricular units' pada dataset model. Batas pengisian manual dibuat maksimal "
            f"{MAX_COURSES_PER_SEMESTER} mata kuliah per semester agar lebih masuk akal untuk konteks perkuliahan umum. "
            "Jumlah mata kuliah lulus tidak boleh lebih besar daripada jumlah yang diambil."
        )
        st.caption(
            "Nilai rata-rata menggunakan skala model 0–20, sedangkan jumlah evaluasi dapat mencakup "
            "ujian, tugas, kuis, atau bentuk penilaian lain; bukan jumlah mata kuliah."
        )


def validate_academic_input(input_dict: dict) -> list[str]:
    """Memastikan hubungan input akademik masuk akal sebelum model dijalankan."""
    errors = []

    for semester, enrolled_feature, approved_feature in [
        (
            "Semester 1",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (approved)",
        ),
        (
            "Semester 2",
            "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (approved)",
        ),
    ]:
        enrolled = to_float(get_input_value(input_dict, enrolled_feature))
        approved = to_float(get_input_value(input_dict, approved_feature))

        if enrolled > MAX_COURSES_PER_SEMESTER:
            errors.append(
                f"{semester}: jumlah mata kuliah diambil tidak boleh melebihi {MAX_COURSES_PER_SEMESTER}."
            )
        if approved > MAX_COURSES_PER_SEMESTER:
            errors.append(
                f"{semester}: jumlah mata kuliah lulus tidak boleh melebihi {MAX_COURSES_PER_SEMESTER}."
            )
        if approved > enrolled:
            errors.append(
                f"{semester}: jumlah mata kuliah lulus tidak boleh lebih besar daripada mata kuliah yang diambil."
            )

    return errors


def show_demo_presets():
    st.markdown("#### ✨ Quick Demo Presets")
    st.caption("Gunakan contoh siap pakai untuk mengeksplorasi perilaku model tanpa mengubah struktur formulir.")
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.button(
            "Risiko Tinggi",
            key="preset_high_risk",
            use_container_width=True,
            on_click=apply_demo_preset,
            args=("Risiko tinggi",),
        )
    with p2:
        st.button(
            "Stabil",
            key="preset_stable",
            use_container_width=True,
            on_click=apply_demo_preset,
            args=("Stabil",),
        )
    with p3:
        st.button(
            "Berprestasi",
            key="preset_achiever",
            use_container_width=True,
            on_click=apply_demo_preset,
            args=("Berprestasi",),
        )
    with p4:
        st.button(
            "Reset Input",
            key="preset_reset",
            use_container_width=True,
            on_click=apply_demo_preset,
            args=("Reset ke nilai median",),
        )

    active_preset = st.session_state.get("active_demo_preset")
    if active_preset:
        st.markdown(
            f'<div class="preset-note">Preset aktif: <strong>{escape(active_preset)}</strong>. '
            'Tinjau input, lalu jalankan prediksi.</div>',
            unsafe_allow_html=True,
        )


def risk_level(dropout_probability: float) -> tuple[str, str]:
    percentage = dropout_probability * 100
    if percentage >= 65:
        return "Tinggi", "#ee6666"
    if percentage >= 35:
        return "Sedang", "#fac858"
    return "Rendah", "#91cc75"


def risk_gauge_html(probability_dict: dict) -> str:
    dropout_probability = max(0.0, min(1.0, to_float(probability_dict.get("Dropout", 0))))
    percentage = dropout_probability * 100
    level, color = risk_level(dropout_probability)

    return f'''<div class="gauge-card">
        <div class="section-card-title">🚨 Risiko Dropout</div>
        <div class="gauge-ring" style="background: conic-gradient({color} 0 {percentage:.1f}%, #e7edf4 {percentage:.1f}% 100%);">
            <div class="gauge-center">
                <div class="gauge-value" style="color: {color} !important;">{percentage:.1f}%</div>
                <div class="gauge-caption">probabilitas model</div>
            </div>
        </div>
        <div class="gauge-status" style="color: {color} !important;">Risiko {level}</div>
    </div>'''


def profile_summary_html(input_dict: dict) -> str:
    semester_1 = (
        f"{format_number(get_input_value(input_dict, 'Curricular units 1st sem (approved)'))}/"
        f"{format_number(get_input_value(input_dict, 'Curricular units 1st sem (enrolled)'))} MK · "
        f"nilai {format_number(get_input_value(input_dict, 'Curricular units 1st sem (grade)'))}"
    )
    semester_2 = (
        f"{format_number(get_input_value(input_dict, 'Curricular units 2nd sem (approved)'))}/"
        f"{format_number(get_input_value(input_dict, 'Curricular units 2nd sem (enrolled)'))} MK · "
        f"nilai {format_number(get_input_value(input_dict, 'Curricular units 2nd sem (grade)'))}"
    )

    profile_items = [
        ("Usia saat masuk", f"{format_number(get_input_value(input_dict, 'Age at enrollment'))} tahun"),
        ("Nilai masuk", format_number(get_input_value(input_dict, "Admission grade"))),
        ("Program studi", f"Kode {get_input_value(input_dict, 'Course', '-')}"),
        ("UKT", binary_label(get_input_value(input_dict, "Tuition fees up to date"), "Lunas", "Belum lunas")),
        ("Tunggakan", binary_label(get_input_value(input_dict, "Debtor"), "Ada", "Tidak ada")),
        ("Beasiswa", binary_label(get_input_value(input_dict, "Scholarship holder"))),
        ("Semester 1", semester_1),
        ("Semester 2", semester_2),
    ]

    item_html = "".join(
        f'''<div class="profile-item">
            <div class="profile-label">{escape(label)}</div>
            <div class="profile-value">{escape(str(value))}</div>
        </div>'''
        for label, value in profile_items
    )

    return f'''<div class="profile-summary-card">
        <div class="section-card-title">👤 Ringkasan Profil Mahasiswa</div>
        <div class="profile-grid">{item_html}</div>
    </div>'''


def show_prediction_overview(input_dict: dict, probability_dict: dict):
    st.markdown(
        f'''<div class="result-overview-wrapper">
            <div class="result-overview-grid">
                {risk_gauge_html(probability_dict)}
                {profile_summary_html(input_dict)}
            </div>
        </div>''',
        unsafe_allow_html=True,
    )


def dataset_median(df_source: pd.DataFrame, feature_name: str):
    column = find_column(df_source, feature_name)
    if column is None:
        return None
    values = pd.to_numeric(df_source[column], errors="coerce").dropna()
    return None if values.empty else float(values.median())


def build_risk_insights(input_dict: dict, df_source: pd.DataFrame, probability_dict: dict):
    '''Memberi penjelasan berbasis indikator input; ini bukan interpretasi kausal internal model.'''
    risk_factors = []
    strengths = []
    actions = []
    financial_issue = False
    academic_issue = False

    tuition_status = to_float(get_input_value(input_dict, "Tuition fees up to date"))
    debtor_status = to_float(get_input_value(input_dict, "Debtor"))
    scholarship_status = to_float(get_input_value(input_dict, "Scholarship holder"))

    if tuition_status == 0:
        risk_factors.append("Status UKT terindikasi belum lunas.")
        financial_issue = True
    else:
        strengths.append("Status UKT tercatat lunas.")

    if debtor_status == 1:
        risk_factors.append("Terdapat tunggakan yang perlu diverifikasi.")
        financial_issue = True
    else:
        strengths.append("Tidak terdapat tunggakan pada data input.")

    if scholarship_status == 1:
        strengths.append("Mahasiswa tercatat sebagai penerima beasiswa.")

    for semester, enrolled_feature, approved_feature, grade_feature in [
        ("Semester 1", "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)"),
        ("Semester 2", "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)"),
    ]:
        enrolled = to_float(get_input_value(input_dict, enrolled_feature))
        approved = to_float(get_input_value(input_dict, approved_feature))
        grade = to_float(get_input_value(input_dict, grade_feature))
        median_grade = dataset_median(df_source, grade_feature)

        if enrolled > 0:
            completion_ratio = approved / enrolled
            if completion_ratio < 0.50:
                risk_factors.append(
                    f"Kelulusan mata kuliah {semester} rendah ({format_number(approved)}/{format_number(enrolled)} MK)."
                )
                academic_issue = True
            elif completion_ratio >= 0.80:
                strengths.append(
                    f"Kelulusan mata kuliah {semester} kuat ({format_number(approved)}/{format_number(enrolled)} MK)."
                )

        if median_grade is not None:
            if grade < median_grade:
                risk_factors.append(
                    f"Rata-rata nilai {semester} ({format_number(grade)}) berada di bawah median data ({format_number(median_grade)})."
                )
                academic_issue = True
            else:
                strengths.append(
                    f"Rata-rata nilai {semester} berada pada atau di atas median data." 
                )

    dropout_probability = to_float(probability_dict.get("Dropout", 0))
    if dropout_probability >= 0.65:
        actions.append("Lakukan pemantauan intensif dan jadwalkan konseling akademik dalam waktu dekat.")
    elif dropout_probability >= 0.35:
        actions.append("Tetapkan pemantauan berkala bersama dosen wali untuk mencegah peningkatan risiko.")
    else:
        actions.append("Pertahankan pemantauan rutin dan dukungan penyelesaian studi tepat waktu.")

    if financial_issue:
        actions.append("Verifikasi UKT dan tunggakan, kemudian arahkan ke layanan pembiayaan atau beasiswa yang relevan.")
    if academic_issue:
        actions.append("Evaluasi beban studi, mata kuliah yang belum lulus, serta rancang pendampingan belajar terarah.")
    if not financial_issue and not academic_issue:
        actions.append("Pertahankan faktor positif melalui evaluasi akademik dan administrasi secara periodik.")

    if not risk_factors:
        risk_factors.append("Tidak ada sinyal risiko utama dari indikator ringkas yang diperiksa pada input ini.")
    if not strengths:
        strengths.append("Belum ada faktor protektif kuat yang teridentifikasi dari indikator ringkas ini.")

    return risk_factors[:4], strengths[:3], actions[:3]


def insight_card(title: str, items: list[str], variant: str = "risk") -> str:
    list_items = "".join(f"<li>{escape(str(item))}</li>" for item in items)
    return f'''<div class="insight-card {variant}">
        <div class="insight-title">{escape(title)}</div>
        <ul class="insight-list">{list_items}</ul>
    </div>'''


def show_risk_explanation(input_dict: dict, df_source: pd.DataFrame, probability_dict: dict):
    risk_factors, strengths, actions = build_risk_insights(input_dict, df_source, probability_dict)
    st.markdown("#### 🚦 Faktor Risiko dan Tindakan")
    st.caption("Penjelasan ini menggunakan indikator input dan perbandingan ringkas dengan data latih; bukan alasan kausal individual dari model.")

    st.markdown(
        f'''<div class="insight-wrapper">
            <div class="insight-grid">
                {insight_card("Sinyal yang Perlu Diperhatikan", risk_factors, "risk")}
                {insight_card("Faktor Positif", strengths, "action")}
                {insight_card("Tindakan Prioritas", actions, "action")}
            </div>
        </div>''',
        unsafe_allow_html=True,
    )


def show_simulation_panel(base_input: dict, baseline_prediction: str, baseline_probability: dict, model, feature_names: list[str]):
    st.markdown("#### 🧪 Mode Simulasi: Bagaimana Jika?")
    st.caption("Uji skenario intervensi secara sementara. Simulasi tidak mengubah input utama atau data latih.")

    with st.expander("⚙️ Atur skenario simulasi", expanded=False):
        current_enrolled = int(max(0, min(
            MAX_COURSES_PER_SEMESTER,
            to_float(get_input_value(base_input, "Curricular units 2nd sem (enrolled)")),
        )))
        current_approved = int(max(0, min(
            current_enrolled,
            to_float(get_input_value(base_input, "Curricular units 2nd sem (approved)")),
        )))
        current_grade = max(0.0, min(20.0, to_float(get_input_value(base_input, "Curricular units 2nd sem (grade)"))))

        if "simulation_approved" in st.session_state:
            st.session_state["simulation_approved"] = int(max(
                0,
                min(current_enrolled, to_float(st.session_state["simulation_approved"])),
            ))

        s1, s2 = st.columns(2)
        with s1:
            simulation_tuition = st.selectbox(
                "Status UKT pada simulasi",
                ["Pertahankan kondisi saat ini", "Ubah menjadi lunas"],
                key="simulation_tuition",
            )
            simulation_debtor = st.selectbox(
                "Status tunggakan pada simulasi",
                ["Pertahankan kondisi saat ini", "Ubah menjadi tidak ada tunggakan"],
                key="simulation_debtor",
            )
        with s2:
            if current_enrolled == 0:
                simulation_approved = 0
                st.caption("Target MK lulus diset 0 karena jumlah MK Semester 2 yang diambil adalah 0.")
            else:
                simulation_approved = st.slider(
                    "Target MK lulus Semester 2",
                    min_value=0,
                    max_value=current_enrolled,
                    value=current_approved,
                    help="Target tidak dapat melebihi jumlah mata kuliah Semester 2 yang diambil.",
                    key="simulation_approved",
                )
            simulation_grade = st.slider(
                "Target rata-rata nilai Semester 2 (skala model 0–20)",
                min_value=0.0,
                max_value=20.0,
                value=float(current_grade),
                step=0.1,
                key="simulation_grade",
            )

        run_simulation = st.button("Bandingkan Hasil Simulasi", key="run_simulation", use_container_width=True)

    if run_simulation:
        simulation_input = base_input.copy()
        if simulation_tuition == "Ubah menjadi lunas":
            set_value(simulation_input, "Tuition fees up to date", 1)
        if simulation_debtor == "Ubah menjadi tidak ada tunggakan":
            set_value(simulation_input, "Debtor", 0)
        set_value(simulation_input, "Curricular units 2nd sem (approved)", simulation_approved)
        set_value(simulation_input, "Curricular units 2nd sem (grade)", simulation_grade)

        try:
            prediction, probability_dict, _ = predict_from_input(simulation_input, model, feature_names)
            st.session_state["last_simulation_result"] = {
                "prediction": prediction,
                "probability": probability_dict,
            }
        except Exception as error:
            st.error(f"Simulasi gagal: {error}")

    simulation_result = st.session_state.get("last_simulation_result")
    if simulation_result:
        baseline_risk = to_float(baseline_probability.get("Dropout", 0)) * 100
        simulated_risk = to_float(simulation_result["probability"].get("Dropout", 0)) * 100
        risk_change = simulated_risk - baseline_risk

        delta_class = "improved" if risk_change < 0 else "increased" if risk_change > 0 else ""
        delta_label = f"{risk_change:+.1f} poin"
        st.markdown(
            f'''<div class="simulation-comparison-wrapper">
                <div class="simulation-comparison-grid">
                    <div class="comparison-metric">
                        <div class="comparison-metric-label">Prediksi Awal</div>
                        <div class="comparison-metric-value">{escape(str(baseline_prediction))}</div>
                    </div>
                    <div class="comparison-metric">
                        <div class="comparison-metric-label">Prediksi Simulasi</div>
                        <div class="comparison-metric-value">{escape(str(simulation_result["prediction"]))}</div>
                    </div>
                    <div class="comparison-metric">
                        <div class="comparison-metric-label">Risiko Dropout</div>
                        <div class="comparison-metric-value">{simulated_risk:.1f}%</div>
                        <span class="comparison-metric-delta {delta_class}">{delta_label}</span>
                    </div>
                </div>
            </div>''',
            unsafe_allow_html=True,
        )


def number_widget(
    label: str,
    value,
    min_value,
    max_value,
    step,
    key: str,
    as_int: bool = False,
):
    if as_int:
        safe_value = int(max(int(min_value), min(int(max_value), int(to_float(value, min_value)))))
        if key in st.session_state:
            st.session_state[key] = int(
                max(int(min_value), min(int(max_value), int(to_float(st.session_state[key], safe_value))))
            )
        return st.number_input(
            label,
            min_value=int(min_value),
            max_value=int(max_value),
            value=safe_value,
            step=int(step),
            key=key,
        )

    safe_value = max(float(min_value), min(float(max_value), to_float(value, min_value)))
    if key in st.session_state:
        st.session_state[key] = max(
            float(min_value),
            min(float(max_value), to_float(st.session_state[key], safe_value)),
        )
    return st.number_input(
        label,
        min_value=float(min_value),
        max_value=float(max_value),
        value=float(safe_value),
        step=float(step),
        format="%.2f",
        key=key,
    )


def select_widget(
    label: str,
    options: list,
    value,
    key: str,
    labels: dict | None = None,
):
    if not options:
        return value

    index = options.index(value) if value in options else 0

    def format_option(item):
        display_value = labels.get(item, item) if labels else item
        return str(display_value)

    return st.selectbox(
        label,
        options=options,
        index=index,
        format_func=format_option,
        key=key,
    )


def show_header():
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-heading">
                <span class="chart-logo"><span></span><span></span><span></span></span>
                <p class="hero-title">{APP_NAME}</p>
                <span class="hero-status"><span></span>ML ready</span>
            </div>
            <p class="hero-subtitle">
                Dashboard ini membantu tim akademik memanfaatkan <strong>machine learning</strong>
                untuk memahami risiko status akademik mahasiswa melalui alur kerja yang ringkas,
                visual, dan mudah digunakan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str, color: str, variant: str = "line"):
    if variant == "bar":
        spark = f"""
        <svg class="kpi-spark" viewBox="0 0 220 70" preserveAspectRatio="none">
            <rect x="6" y="46" width="6" height="18" rx="3" fill="{color}"/>
            <rect x="28" y="42" width="6" height="22" rx="3" fill="{color}"/>
            <rect x="50" y="38" width="6" height="26" rx="3" fill="{color}"/>
            <rect x="72" y="34" width="6" height="30" rx="3" fill="{color}"/>
            <rect x="94" y="30" width="6" height="34" rx="3" fill="{color}"/>
            <rect x="116" y="24" width="6" height="40" rx="3" fill="{color}"/>
            <rect x="138" y="34" width="6" height="30" rx="3" fill="{color}"/>
            <rect x="160" y="22" width="6" height="42" rx="3" fill="{color}"/>
            <rect x="182" y="18" width="6" height="46" rx="3" fill="{color}"/>
            <rect x="204" y="16" width="6" height="48" rx="3" fill="{color}"/>
        </svg>
        """
    else:
        spark = f"""
        <svg class="kpi-spark" viewBox="0 0 220 70" preserveAspectRatio="none">
            <path d="M0,64 L0,52 L18,34 L36,40 L54,32 L72,34 L90,30 L108,20 L126,34 L144,18 L162,16 L180,20 L198,10 L220,16 L220,70 L0,70 Z"
                  fill="{color}" opacity="0.12"/>
            <polyline points="0,52 18,34 36,40 54,32 72,34 90,30 108,20 126,34 144,18 162,16 180,20 198,10 220,16"
                      fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        """

    badge_class = "good" if not delta.startswith("-") else "warn"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <span class="kpi-badge {badge_class}">{delta}</span>
            {spark}
        </div>
        """,
        unsafe_allow_html=True,
    )



def get_theme_palette() -> dict:
    """Menghasilkan palet UI dan chart berdasarkan mode yang dipilih pengguna."""
    dark_mode = st.session_state.get("dark_mode", False)

    if dark_mode:
        return {
            "is_dark": True,
            "app_bg": "#0b1220",
            "surface": "#121c2d",
            "surface_alt": "#18253a",
            "surface_soft": "#0f1828",
            "text": "#eef4ff",
            "muted": "#aab8cc",
            "border": "#2b3a52",
            "grid": "#293850",
            "input_bg": "#101a2a",
            "sidebar": "#0a1220",
            "sidebar_item": "#17243a",
            "sidebar_item_hover": "#21324d",
            "accent": "#73c0de",
            "accent_deep": "#5470c6",
            "table_head": "#18243a",
            "table_row": "#111b2b",
        }

    return {
        "is_dark": False,
        "app_bg": "#ffffff",
        "surface": "#ffffff",
        "surface_alt": "#f8fafc",
        "surface_soft": "#f1f5f9",
        "text": "#24283b",
        "muted": "#64748b",
        "border": "#d9dee7",
        "grid": "#eef2f7",
        "input_bg": "#ffffff",
        "sidebar": "#202b3d",
        "sidebar_item": "#35445a",
        "sidebar_item_hover": "#3b4b63",
        "accent": "#73c0de",
        "accent_deep": "#5470c6",
        "table_head": "#f8fafc",
        "table_row": "#ffffff",
    }


def inject_theme_css(dark_mode: bool):
    """Menimpa warna bawaan Streamlit agar light/dark mode selalu kontras dan terbaca."""
    theme = get_theme_palette()
    mode = "dark" if dark_mode else "light"

    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: {mode};
            --ot-app-bg: {theme["app_bg"]};
            --ot-surface: {theme["surface"]};
            --ot-surface-alt: {theme["surface_alt"]};
            --ot-surface-soft: {theme["surface_soft"]};
            --ot-text: {theme["text"]};
            --ot-muted: {theme["muted"]};
            --ot-border: {theme["border"]};
            --ot-grid: {theme["grid"]};
            --ot-input: {theme["input_bg"]};
            --ot-sidebar: {theme["sidebar"]};
            --ot-sidebar-item: {theme["sidebar_item"]};
            --ot-sidebar-hover: {theme["sidebar_item_hover"]};
            --ot-accent: {theme["accent"]};
            --ot-accent-deep: {theme["accent_deep"]};
            --ot-table-head: {theme["table_head"]};
            --ot-table-row: {theme["table_row"]};
        }}

        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
        .stApp, .main, .block-container {{
            background: var(--ot-app-bg) !important;
            color: var(--ot-text) !important;
            color-scheme: {mode} !important;
            transition: background-color 220ms ease, color 220ms ease;
        }}

        header[data-testid="stHeader"] {{
            background: transparent !important;
        }}

        h1, h2, h3, h4, h5, h6,
        p, label, div, span,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"],
        [data-testid="stText"] {{
            color: var(--ot-text);
        }}

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        .stCaption {{
            color: var(--ot-muted) !important;
        }}

        .top-actions, .hero-title, .hero-subtitle,
        .hero-subtitle strong, .kpi-label, .kpi-value,
        .section-card-title, .profile-value, .insight-title,
        .comparison-metric-value, .journey-title {{
            color: var(--ot-text) !important;
        }}

        .footer, .file-note, .profile-label, .gauge-caption,
        .preset-note, .journey-copy, .comparison-metric-label,
        .insight-list, .insight-list li {{
            color: var(--ot-muted) !important;
        }}

        .panel, .kpi-card,
        div[data-testid="stMetric"],
        div[data-testid="stExpander"],
        .gauge-card, .profile-summary-card,
        .insight-card.action, .comparison-metric,
        .journey-step {{
            background: var(--ot-surface) !important;
            border-color: var(--ot-border) !important;
            color: var(--ot-text) !important;
        }}

        .profile-item {{
            background: var(--ot-surface-alt) !important;
            border-color: var(--ot-grid) !important;
        }}

        .insight-card.risk {{
            background: {"#2a171b" if dark_mode else "#fff8f8"} !important;
            border-color: {"#7f303a" if dark_mode else "#fecaca"} !important;
        }}

        .soft-note {{
            background: var(--ot-surface-alt) !important;
            border-left-color: var(--ot-accent-deep) !important;
            color: var(--ot-text) !important;
        }}

        .soft-note, .soft-note * {{
            color: var(--ot-text) !important;
        }}

        .light-table {{
            background: var(--ot-surface) !important;
            border-color: var(--ot-border) !important;
            color: var(--ot-text) !important;
        }}

        .light-table th {{
            background: var(--ot-table-head) !important;
            color: var(--ot-text) !important;
            border-bottom-color: var(--ot-border) !important;
        }}

        .light-table td {{
            background: var(--ot-table-row) !important;
            color: var(--ot-text) !important;
            border-bottom-color: var(--ot-grid) !important;
        }}

        .file-pill, .comparison-metric-delta {{
            background: var(--ot-surface-soft) !important;
            color: var(--ot-text) !important;
            border-color: var(--ot-border) !important;
        }}

        .gauge-center {{
            background: var(--ot-surface) !important;
            border-color: var(--ot-grid) !important;
        }}

        .journey-step {{
            background: linear-gradient(145deg, var(--ot-surface), var(--ot-surface-alt)) !important;
        }}

        .journey-number {{
            background: {"#21365d" if dark_mode else "#eaf1ff"} !important;
            color: {"#a8d1ff" if dark_mode else "#5470c6"} !important;
        }}

        /* Accessibility patch: Streamlit expanders may keep a light summary surface in dark mode.
           These selectors force the whole trigger (including its nested text and icon) to inherit
           the active theme even when the cursor is not hovering over it. */
        div[data-testid="stExpander"],
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] [data-testid="stExpanderDetails"],
        div[data-testid="stExpander"] [data-testid="stExpanderDetails"] > div {{
            background: var(--ot-surface) !important;
            border-color: var(--ot-border) !important;
            color: var(--ot-text) !important;
        }}

        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary *,
        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary span,
        div[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"],
        div[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] * {{
            color: var(--ot-text) !important;
            fill: var(--ot-text) !important;
            opacity: 1 !important;
        }}

        div[data-testid="stExpander"] summary:hover,
        div[data-testid="stExpander"] summary:focus-visible {{
            background: var(--ot-surface-alt) !important;
            color: var(--ot-text) !important;
            outline: 2px solid var(--ot-accent) !important;
            outline-offset: -2px;
        }}

        /* The native uploader action is not a regular Streamlit button. Give it an explicit
           high-contrast treatment so the Upload / Browse label never disappears in dark mode. */
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"],
        [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {{
            background: linear-gradient(90deg, var(--ot-accent-deep), var(--ot-accent)) !important;
            border: 1px solid var(--ot-accent) !important;
            color: #ffffff !important;
            opacity: 1 !important;
            text-shadow: none !important;
        }}

        [data-testid="stFileUploaderDropzone"] button *,
        [data-testid="stFileUploader"] button *,
        [data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] *,
        [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] * {{
            color: #ffffff !important;
            fill: #ffffff !important;
            opacity: 1 !important;
        }}

        [data-testid="stFileUploaderDropzone"] button:hover,
        [data-testid="stFileUploader"] button:hover {{
            filter: brightness(1.08) !important;
            transform: translateY(-1px);
        }}

        /* Prevent legacy top-right helper labels from colliding on narrow screens. */
        .top-actions {{
            display: none !important;
        }}

        /* Form controls, select popovers, uploader, and data grid explicitly follow the active theme. */
        div[data-baseweb="select"] > div,
        .stNumberInput input,
        .stTextInput input,
        .stTextArea textarea,
        [data-testid="stDateInput"] input,
        [data-testid="stTimeInput"] input {{
            background: var(--ot-input) !important;
            border-color: var(--ot-border) !important;
            color: var(--ot-text) !important;
            caret-color: var(--ot-text) !important;
        }}

        .stNumberInput button,
        [data-testid="stNumberInputStepUp"],
        [data-testid="stNumberInputStepDown"] {{
            background: var(--ot-surface-alt) !important;
            color: var(--ot-text) !important;
            border-color: var(--ot-border) !important;
        }}

        div[data-baseweb="select"] svg,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {{
            color: var(--ot-text) !important;
            fill: var(--ot-text) !important;
        }}

        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [role="listbox"],
        [role="listbox"] li,
        [role="option"] {{
            background: var(--ot-surface) !important;
            color: var(--ot-text) !important;
            border-color: var(--ot-border) !important;
        }}

        [role="option"]:hover,
        [role="option"][aria-selected="true"] {{
            background: var(--ot-surface-alt) !important;
            color: var(--ot-text) !important;
        }}

        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploader"] section {{
            background: var(--ot-surface-alt) !important;
            border-color: var(--ot-border) !important;
            color: var(--ot-text) !important;
        }}

        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploader"] section * {{
            color: var(--ot-text) !important;
        }}

        /* Dataframe + toolbar: download, search, filter, fullscreen, dan menu. */
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] > div,
        [data-testid="stDataFrame"] [role="grid"],
        [data-testid="stDataFrame"] [role="grid"] > div {{
            background: var(--ot-surface) !important;
            color: var(--ot-text) !important;
            color-scheme: {mode} !important;
        }}

        /* Toolbar Streamlit dapat menjadi anak atau elemen terpisah dari DataFrame,
           sehingga kedua selector berikut dipertahankan agar semua ikon selalu terlihat. */
        [data-testid="stDataFrame"] [data-testid="stElementToolbar"],
        [data-testid="stDataFrame"] .stElementToolbar,
        [data-testid="stElementToolbar"],
        .stElementToolbar {{
            background: var(--ot-surface-alt) !important;
            color: var(--ot-text) !important;
            border: 1px solid var(--ot-border) !important;
            border-radius: 8px !important;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.16) !important;
            opacity: 1 !important;
        }}

        /* Tombol lihat data, download, search, filter, fullscreen, dan menu titik tiga. */
        [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button,
        [data-testid="stDataFrame"] .stElementToolbar button,
        [data-testid="stElementToolbar"] button,
        .stElementToolbar button,
        [data-testid="stDataFrame"] [data-testid="stElementToolbar"] [role="button"],
        [data-testid="stDataFrame"] .stElementToolbar [role="button"],
        [data-testid="stElementToolbar"] [role="button"],
        .stElementToolbar [role="button"] {{
            background: transparent !important;
            color: var(--ot-text) !important;
            border: 0 !important;
            opacity: 1 !important;
            filter: none !important;
        }}

        /* Pastikan ikon SVG memakai warna aktif pada kedua mode. */
        [data-testid="stDataFrame"] [data-testid="stElementToolbar"] svg,
        [data-testid="stDataFrame"] .stElementToolbar svg,
        [data-testid="stElementToolbar"] svg,
        .stElementToolbar svg {{
            color: var(--ot-text) !important;
            fill: currentColor !important;
            stroke: currentColor !important;
            opacity: 1 !important;
        }}

        [data-testid="stDataFrame"] [data-testid="stElementToolbar"] svg *,
        [data-testid="stDataFrame"] .stElementToolbar svg *,
        [data-testid="stElementToolbar"] svg *,
        .stElementToolbar svg * {{
            color: inherit !important;
            opacity: 1 !important;
        }}

        /* Hover yang terlihat jelas, tanpa menghilangkan ikon. */
        [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button:hover,
        [data-testid="stDataFrame"] .stElementToolbar button:hover,
        [data-testid="stElementToolbar"] button:hover,
        .stElementToolbar button:hover,
        [data-testid="stDataFrame"] [data-testid="stElementToolbar"] [role="button"]:hover,
        [data-testid="stDataFrame"] .stElementToolbar [role="button"]:hover,
        [data-testid="stElementToolbar"] [role="button"]:hover,
        .stElementToolbar [role="button"]:hover {{
            background: var(--ot-surface-soft) !important;
            color: var(--ot-accent) !important;
            border-radius: 6px !important;
        }}

        [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button:hover svg,
        [data-testid="stDataFrame"] .stElementToolbar button:hover svg,
        [data-testid="stElementToolbar"] button:hover svg,
        .stElementToolbar button:hover svg {{
            color: var(--ot-accent) !important;
            fill: currentColor !important;
            stroke: currentColor !important;
        }}

        /* Kondisi tombol yang tidak dapat digunakan. */
        [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button:disabled,
        [data-testid="stDataFrame"] .stElementToolbar button:disabled,
        [data-testid="stElementToolbar"] button:disabled,
        .stElementToolbar button:disabled {{
            color: var(--ot-muted) !important;
            opacity: 0.55 !important;
        }}

        /* Kotak pencarian yang muncul dari toolbar tabel. */
        [data-testid="stDataFrame"] input,
        [data-testid="stDataFrame"] input[type="text"] {{
            background: var(--ot-input) !important;
            color: var(--ot-text) !important;
            border: 1px solid var(--ot-border) !important;
            caret-color: var(--ot-text) !important;
        }}

        [data-testid="stDataFrame"] input::placeholder {{
            color: var(--ot-muted) !important;
            opacity: 1 !important;
        }}

        [data-testid="stAlert"],
        div[data-testid="stNotification"],
        [data-testid="stInfo"] {{
            background: {"#15263d" if dark_mode else "#dbeafe"} !important;
            border-color: {"#31577f" if dark_mode else "#bfdbfe"} !important;
            color: var(--ot-text) !important;
        }}

        [data-testid="stAlert"] *,
        div[data-testid="stNotification"] *,
        [data-testid="stInfo"] * {{
            color: var(--ot-text) !important;
        }}

        [data-testid="stTabs"] [role="tablist"] {{
            border-bottom-color: var(--ot-border) !important;
        }}

        [data-testid="stTabs"] [role="tab"] {{
            color: var(--ot-muted) !important;
        }}

        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
            color: var(--ot-accent) !important;
            border-bottom-color: var(--ot-accent) !important;
        }}

        section[data-testid="stSidebar"] {{
            background: var(--ot-sidebar) !important;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            background: var(--ot-sidebar-item) !important;
            border-color: transparent !important;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background: var(--ot-sidebar-hover) !important;
            border-color: {"#456182" if dark_mode else "#6f829c"} !important;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
            background: linear-gradient(90deg, {"#1d3556" if dark_mode else "#33465f"}, {"#29486d" if dark_mode else "#3c536e"}) !important;
            border-color: var(--ot-accent) !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stToggle"] label,
        section[data-testid="stSidebar"] [data-testid="stToggle"] span,
        section[data-testid="stSidebar"] [data-testid="stToggle"] p {{
            color: #f8fafc !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stToggle"] [data-checked="true"] {{
            background: var(--ot-accent-deep) !important;
        }}

        iframe[title="streamlit_echarts.st_echarts"] {{
            background: var(--ot-surface) !important;
        }}

        @media (hover: hover) {{
            .panel:hover, .gauge-card:hover, .profile-summary-card:hover,
            .insight-card:hover, .comparison-metric:hover {{
                border-color: {"#425a79" if dark_mode else "#c9d7ec"} !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_chart(options: dict, height: str = "360px"):
    if st_echarts is not None:
        st_echarts(options=options, height=height, theme="")
        return

    st.info("Install streamlit-echarts untuk tampilan chart seperti referensi: pip install streamlit-echarts")


def light_table(df_table: pd.DataFrame):
    html = ['<table class="light-table">']
    html.append("<thead><tr>")

    for column in df_table.columns:
        html.append(f"<th>{column}</th>")

    html.append("</tr></thead><tbody>")

    for _, row in df_table.iterrows():
        html.append("<tr>")

        for value in row:
            html.append(f"<td>{value}</td>")

        html.append("</tr>")

    html.append("</tbody></table>")
    st.markdown("".join(html), unsafe_allow_html=True)


def probability_chart(probability_df: pd.DataFrame):
    chart_data = probability_df.sort_values("Probabilitas", ascending=True)
    statuses = chart_data["Status"].tolist()
    values = (chart_data["Probabilitas"] * 100).round(2).tolist()
    colors = [STATUS_COLOR.get(status, ECHART_COLORS[0]) for status in statuses]
    theme = get_theme_palette()

    options = {
        "color": colors,
        "backgroundColor": theme["surface"],
        "textStyle": {"color": theme["text"], "fontFamily": "Inter"},
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "backgroundColor": theme["surface_alt"],
            "borderColor": theme["border"],
            "textStyle": {"color": theme["text"]},
        },
        "grid": {"left": "3%", "right": "4%", "bottom": "8%", "top": "8%", "containLabel": True},
        "xAxis": {
            "type": "value",
            "max": 100,
            "axisLabel": {"formatter": "{value}%", "color": theme["text"]},
            "splitLine": {"lineStyle": {"color": theme["grid"]}},
        },
        "yAxis": {
            "type": "category",
            "data": statuses,
            "axisLabel": {"color": theme["text"]},
            "axisLine": {"show": False},
            "axisTick": {"show": False},
        },
        "series": [
            {
                "name": "Probabilitas",
                "type": "bar",
                "data": [
                    {"value": value, "itemStyle": {"color": colors[index]}}
                    for index, value in enumerate(values)
                ],
                "barWidth": 18,
                "label": {"show": True, "position": "right", "formatter": "{c}%", "color": theme["text"]},
            }
        ],
    }

    if st_echarts is not None:
        render_chart(options, height="260px")
    elif px is not None:
        fig = px.bar(
            chart_data,
            x="Probabilitas",
            y="Status",
            orientation="h",
            color="Status",
            color_discrete_map=STATUS_COLOR,
            text=(chart_data["Probabilitas"] * 100).round(1).astype(str) + "%",
        )
        fig.update_layout(
            height=260,
            template="plotly_dark" if theme["is_dark"] else "plotly_white",
            paper_bgcolor=theme["surface"],
            plot_bgcolor=theme["surface"],
            font={"color": theme["text"], "family": "Inter"},
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        fig.update_xaxes(tickformat=".0%", gridcolor=theme["grid"], color=theme["text"])
        fig.update_yaxes(color=theme["text"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(probability_df.set_index("Status")["Probabilitas"])


def feature_importance_chart(importance_df: pd.DataFrame):
    chart_data = importance_df.sort_values("Importance", ascending=True)
    theme = get_theme_palette()

    options = {
        "color": [theme["accent_deep"]],
        "backgroundColor": theme["surface"],
        "textStyle": {"color": theme["text"], "fontFamily": "Inter"},
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "backgroundColor": theme["surface_alt"],
            "borderColor": theme["border"],
            "textStyle": {"color": theme["text"]},
        },
        "grid": {"left": "3%", "right": "4%", "bottom": "4%", "top": "5%", "containLabel": True},
        "xAxis": {
            "type": "value",
            "axisLabel": {"color": theme["text"]},
            "splitLine": {"lineStyle": {"color": theme["grid"]}},
        },
        "yAxis": {
            "type": "category",
            "data": chart_data["Fitur"].tolist(),
            "axisLabel": {"color": theme["text"]},
            "axisLine": {"show": False},
            "axisTick": {"show": False},
        },
        "series": [
            {
                "name": "Importance",
                "type": "bar",
                "data": chart_data["Importance"].round(4).tolist(),
                "barWidth": 14,
                "itemStyle": {"borderRadius": [0, 4, 4, 0]},
            }
        ],
    }

    if st_echarts is not None:
        render_chart(options, height="460px")
    elif px is not None:
        fig = px.bar(
            chart_data,
            x="Importance",
            y="Fitur",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#dbeafe", theme["accent_deep"]],
        )
        fig.update_layout(
            height=460,
            template="plotly_dark" if theme["is_dark"] else "plotly_white",
            paper_bgcolor=theme["surface"],
            plot_bgcolor=theme["surface"],
            font={"color": theme["text"], "family": "Inter"},
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.update_xaxes(gridcolor=theme["grid"], color=theme["text"])
        fig.update_yaxes(color=theme["text"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(importance_df.set_index("Fitur")["Importance"])


def status_distribution_chart(distribution: pd.DataFrame):
    statuses = distribution["Status"].tolist()
    values = distribution["Jumlah"].tolist()
    theme = get_theme_palette()

    options = {
        "color": [
            STATUS_COLOR.get(status, ECHART_COLORS[index % len(ECHART_COLORS)])
            for index, status in enumerate(statuses)
        ],
        "backgroundColor": theme["surface"],
        "textStyle": {"color": theme["text"], "fontFamily": "Inter"},
        "tooltip": {
            "trigger": "item",
            "formatter": "{b}: {c} ({d}%)",
            "backgroundColor": theme["surface_alt"],
            "borderColor": theme["border"],
            "textStyle": {"color": theme["text"]},
        },
        "legend": {
            "bottom": 0,
            "left": "center",
            "textStyle": {"color": theme["text"]},
        },
        "series": [
            {
                "name": "Status",
                "type": "pie",
                "radius": ["42%", "70%"],
                "center": ["50%", "45%"],
                "avoidLabelOverlap": True,
                "itemStyle": {"borderRadius": 6, "borderColor": theme["surface"], "borderWidth": 2},
                "label": {"formatter": "{b}\\n{d}%", "color": theme["text"]},
                "labelLine": {"lineStyle": {"color": theme["muted"]}},
                "data": [
                    {
                        "name": status,
                        "value": int(values[index]),
                        "itemStyle": {
                            "color": STATUS_COLOR.get(status, ECHART_COLORS[index % len(ECHART_COLORS)])
                        },
                    }
                    for index, status in enumerate(statuses)
                ],
            }
        ],
    }

    if st_echarts is not None:
        render_chart(options, height="460px")
    elif px is not None:
        fig = px.pie(
            distribution,
            names="Status",
            values="Jumlah",
            color="Status",
            color_discrete_map=STATUS_COLOR,
            hole=0.42,
        )
        fig.update_layout(
            height=460,
            template="plotly_dark" if theme["is_dark"] else "plotly_white",
            paper_bgcolor=theme["surface"],
            plot_bgcolor=theme["surface"],
            font={"color": theme["text"], "family": "Inter"},
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(distribution.set_index("Status")["Jumlah"])


def show_result(prediction: str, probability_dict: dict, input_dict: dict | None = None):
    confidence = probability_dict.get(prediction, 0)
    color = STATUS_COLOR.get(prediction, "#2563eb")
    recommendation = RECOMMENDATION.get(prediction, "Tidak ada rekomendasi khusus.")

    st.markdown(
        f"""
        <div class="result-card" style="background: linear-gradient(135deg, {color}, #2f4554);">
            <p>Hasil Prediksi Status Mahasiswa</p>
            <h2>{prediction}</h2>
            <p>Model confidence: {confidence * 100:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if input_dict is not None:
        show_prediction_overview(input_dict, probability_dict)

    st.markdown("#### Probabilitas")

    probability_df = pd.DataFrame(
        {
            "Status": list(probability_dict.keys()),
            "Probabilitas": list(probability_dict.values()),
        }
    ).sort_values("Probabilitas", ascending=False)

    st.dataframe(
        probability_df.assign(Probabilitas=lambda item: (item["Probabilitas"] * 100).round(2)),
        hide_index=True,
        use_container_width=True,
    )

    probability_chart(probability_df)

    st.markdown("#### Rekomendasi Utama")
    st.markdown(f'<div class="soft-note">{recommendation}</div>', unsafe_allow_html=True)

    if input_dict is not None:
        show_risk_explanation(input_dict, df, probability_dict)


df = load_dataset()
model, feature_names = load_artifacts()
default_values = dataset_default(df, feature_names)

if TARGET_COL not in df.columns:
    st.error(f"Kolom target '{TARGET_COL}' tidak ditemukan di data.csv.")
    st.stop()


with st.sidebar:
    st.markdown('<div class="sidebar-title"><span class="sidebar-brand-mark">⌁</span>onTrack</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="sidebar-caption">{APP_NAME} - Prediksi Status Akademik Mahasiswa</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-field-label">Tampilan</div>', unsafe_allow_html=True)
    dark_mode = st.toggle(
        "Dark mode",
        key="dark_mode",
        help="Aktifkan untuk tampilan gelap dengan kontras teks yang disesuaikan.",
    )
    st.caption("Pilih light mode atau dark mode sesuai kenyamanan Anda.")

    st.markdown('<div class="sidebar-field-label sidebar-menu-label">Menu Utama</div>', unsafe_allow_html=True)

    menu_map = {
        "Beranda": "Beranda",
        "Input Manual": "Input Manual",
        "Upload CSV": "Upload Batch",
        "Dashboard": "Dashboard",
        "Panduan": "Tentang Aplikasi",
    }

    selected_menu = st.radio(
        "Navigasi",
        list(menu_map.keys()),
        label_visibility="collapsed",
    )

    menu = menu_map[selected_menu]

    st.markdown('<div class="sidebar-field-label">Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-caption">Random Forest Classifier</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-field-label">Berkas</div>', unsafe_allow_html=True)
    st.markdown(
        '''
        <div class="sidebar-caption">
            data.csv<br>
            rf_model.pkl<br>
            feature_names.pkl
        </div>
        ''',
        unsafe_allow_html=True,
    )


inject_theme_css(dark_mode)
show_header()


if menu == "Beranda":
    st.markdown(
        "Ringkasan analisis: "
        f"{df.shape[0]} data mahasiswa, {len(feature_names)} fitur model"
    )
    st.markdown(
        """
        <div class="journey-strip">
            <div class="journey-step">
                <span class="journey-number">01</span>
                <div>
                    <div class="journey-title">🔎 Explore</div>
                    <div class="journey-copy">Mulai dari overview untuk membaca pola data dan status akademik.</div>
                </div>
            </div>
            <div class="journey-step">
                <span class="journey-number">02</span>
                <div>
                    <div class="journey-title">🧠 Predict</div>
                    <div class="journey-copy">Gunakan input manual atau quick demo preset untuk membuat prediksi.</div>
                </div>
            </div>
            <div class="journey-step">
                <span class="journey-number">03</span>
                <div>
                    <div class="journey-title">🧪 Simulate</div>
                    <div class="journey-copy">Bandingkan skenario untuk membantu menentukan prioritas tindak lanjut.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        kpi_card(
            "Total Mahasiswa",
            f"{len(df):,}".replace(",", "."),
            "+ data",
            "#91cc75",
        )

    with k2:
        dropout_rate = (df[TARGET_COL].eq("Dropout").mean() * 100)
        kpi_card(
            "Risiko Dropout",
            f"{dropout_rate:.1f}%",
            "- risiko",
            "#ee6666",
        )

    with k3:
        kpi_card(
            "Fitur Model",
            str(len(feature_names)),
            "+ sinyal",
            "#5470c6",
            variant="bar",
        )

    with k4:
        top_target = df[TARGET_COL].value_counts().idxmax()
        kpi_card(
            "Status Terbanyak",
            top_target,
            "+ mayoritas",
            "#73c0de",
        )

    st.markdown("## Bagaimana distribusi mahasiswa?")
    st.caption(
        "Lihat pembagian kelas target pada dataset sebelum menggunakan prediksi manual atau unggah batch."
    )

    distribution = df[TARGET_COL].value_counts().rename_axis("Status").reset_index(name="Jumlah")
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Distribusi Target")
        status_distribution_chart(distribution)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Ringkasan Data")
        light_table(distribution)
        st.markdown(
            '<div class="file-note">File model dimuat: '
            '<span class="file-pill">rf_model.pkl</span> dan '
            '<span class="file-pill">feature_names.pkl</span>.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


elif menu == "Input Manual":
    col_input, col_result = st.columns([1.05, 0.95], gap="large")

    input_dict = default_values.copy()

    with col_input:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("📝 Input Data Mahasiswa")
        show_input_quick_guide()
        show_demo_presets()

        with st.expander("Data Pendaftaran", expanded=True):
            c1, c2, c3 = st.columns(3)

            with c1:
                set_value(
                    input_dict,
                    "Age at enrollment",
                    number_widget(
                        "Usia saat masuk",
                        default_values.get("Age at enrollment", 20),
                        15,
                        70,
                        1,
                        "age",
                        as_int=True,
                    ),
                )
                set_value(
                    input_dict,
                    "Admission grade",
                    number_widget(
                        "Nilai masuk",
                        default_values.get("Admission grade", 126.1),
                        0,
                        200,
                        0.1,
                        "admission_grade",
                    ),
                )

            with c2:
                set_value(
                    input_dict,
                    "Gender",
                    select_widget(
                        "Jenis kelamin",
                        unique_options(df, "Gender"),
                        default_values.get("Gender", 0),
                        "gender",
                        {0: "Perempuan", 1: "Laki-laki"},
                    ),
                )
                set_value(
                    input_dict,
                    "Course",
                    select_widget(
                        "Program studi / kode course",
                        unique_options(df, "Course"),
                        default_values.get("Course", 9238),
                        "course",
                    ),
                )

            with c3:
                set_value(
                    input_dict,
                    "Application mode",
                    select_widget(
                        "Mode pendaftaran",
                        unique_options(df, "Application mode"),
                        default_values.get("Application mode", 17),
                        "application_mode",
                    ),
                )
                set_value(
                    input_dict,
                    "Daytime/evening attendance",
                    select_widget(
                        "Waktu kuliah",
                        unique_options(df, "Daytime/evening attendance"),
                        default_values.get("Daytime/evening attendance\t", 1),
                        "attendance",
                        {0: "Kelas malam", 1: "Kelas siang"},
                    ),
                )

        with st.expander("Status Mahasiswa", expanded=True):
            c1, c2, c3 = st.columns(3)

            with c1:
                set_value(
                    input_dict,
                    "Scholarship holder",
                    select_widget(
                        "Penerima beasiswa",
                        unique_options(df, "Scholarship holder"),
                        default_values.get("Scholarship holder", 0),
                        "scholarship",
                        {0: "Tidak", 1: "Ya"},
                    ),
                )

            with c2:
                set_value(
                    input_dict,
                    "Debtor",
                    select_widget(
                        "Memiliki tunggakan",
                        unique_options(df, "Debtor"),
                        default_values.get("Debtor", 0),
                        "debtor",
                        {0: "Tidak", 1: "Ya"},
                    ),
                )

            with c3:
                set_value(
                    input_dict,
                    "Tuition fees up to date",
                    select_widget(
                        "UKT lunas",
                        unique_options(df, "Tuition fees up to date"),
                        default_values.get("Tuition fees up to date", 1),
                        "tuition",
                        {0: "Belum", 1: "Lunas"},
                    ),
                )

        with st.expander("Akademik Semester 1 dan 2", expanded=True):
            st.caption(
                f"Isikan jumlah mata kuliah yang diambil dan lulus pada tiap semester (maks. {MAX_COURSES_PER_SEMESTER}). "
                "Jumlah mata kuliah lulus tidak boleh melebihi jumlah yang diambil. "
                "Jumlah evaluasi dapat mencakup ujian, tugas, dan kuis; bukan jumlah mata kuliah."
            )
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Semester 1**")
                set_value(
                    input_dict,
                    "Curricular units 1st sem (enrolled)",
                    number_widget(
                        "Mata kuliah diambil (maks. 12)",
                        default_values.get("Curricular units 1st sem (enrolled)", 6),
                        0,
                        MAX_COURSES_PER_SEMESTER,
                        1,
                        "cu1_enrolled",
                        as_int=True,
                    ),
                )
                set_value(
                    input_dict,
                    "Curricular units 1st sem (approved)",
                    number_widget(
                        "Mata kuliah lulus (≤ diambil)",
                        default_values.get("Curricular units 1st sem (approved)", 5),
                        0,
                        MAX_COURSES_PER_SEMESTER,
                        1,
                        "cu1_approved",
                        as_int=True,
                    ),
                )
                set_value(
                    input_dict,
                    "Curricular units 1st sem (evaluations)",
                    number_widget(
                        "Jumlah evaluasi (ujian/tugas/kuis)",
                        default_values.get("Curricular units 1st sem (evaluations)", 8),
                        0,
                        50,
                        1,
                        "cu1_eval",
                        as_int=True,
                    ),
                )
                set_value(
                    input_dict,
                    "Curricular units 1st sem (grade)",
                    number_widget(
                        "Rata-rata nilai (skala model 0–20)",
                        default_values.get("Curricular units 1st sem (grade)", 12.28),
                        0,
                        20,
                        0.1,
                        "cu1_grade",
                    ),
                )

            with c2:
                st.markdown("**Semester 2**")
                set_value(
                    input_dict,
                    "Curricular units 2nd sem (enrolled)",
                    number_widget(
                        "Mata kuliah diambil (maks. 12)",
                        default_values.get("Curricular units 2nd sem (enrolled)", 6),
                        0,
                        MAX_COURSES_PER_SEMESTER,
                        1,
                        "cu2_enrolled",
                        as_int=True,
                    ),
                )
                set_value(
                    input_dict,
                    "Curricular units 2nd sem (approved)",
                    number_widget(
                        "Mata kuliah lulus (≤ diambil)",
                        default_values.get("Curricular units 2nd sem (approved)", 5),
                        0,
                        MAX_COURSES_PER_SEMESTER,
                        1,
                        "cu2_approved",
                        as_int=True,
                    ),
                )
                set_value(
                    input_dict,
                    "Curricular units 2nd sem (evaluations)",
                    number_widget(
                        "Jumlah evaluasi (ujian/tugas/kuis)",
                        default_values.get("Curricular units 2nd sem (evaluations)", 8),
                        0,
                        50,
                        1,
                        "cu2_eval",
                        as_int=True,
                    ),
                )
                set_value(
                    input_dict,
                    "Curricular units 2nd sem (grade)",
                    number_widget(
                        "Rata-rata nilai (skala model 0–20)",
                        default_values.get("Curricular units 2nd sem (grade)", 12.20),
                        0,
                        20,
                        0.1,
                        "cu2_grade",
                    ),
                )

        with st.expander("Fitur Lain dari Dataset"):
            st.caption(
                "Bagian ini memakai nilai tengah dari data.csv agar input tetap lengkap "
                "sesuai fitur model."
            )
            preview_defaults = pd.DataFrame(
                {
                    "Fitur": [clean_label(item) for item in input_dict.keys()],
                    "Nilai": list(input_dict.values()),
                }
            )
            st.dataframe(preview_defaults, hide_index=True, use_container_width=True)

        predict_clicked = st.button("Prediksi Sekarang", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("🎯 Hasil Prediksi")

        if predict_clicked:
            validation_errors = validate_academic_input(input_dict)
            if validation_errors:
                for validation_error in validation_errors:
                    st.error(validation_error)
            else:
                try:
                    prediction, probability_dict, model_input = predict_from_input(
                        input_dict,
                        model,
                        feature_names,
                    )

                    st.session_state["last_manual_prediction"] = prediction
                    st.session_state["last_manual_probability"] = probability_dict
                    st.session_state["last_manual_input"] = model_input
                    st.session_state["last_manual_raw_input"] = input_dict.copy()
                    st.session_state.pop("last_simulation_result", None)

                except Exception as error:
                    st.error(f"Prediksi gagal: {error}")

        if "last_manual_prediction" in st.session_state:
            show_result(
                st.session_state["last_manual_prediction"],
                st.session_state["last_manual_probability"],
                st.session_state.get("last_manual_raw_input"),
            )

            show_simulation_panel(
                st.session_state.get("last_manual_raw_input", input_dict),
                st.session_state["last_manual_prediction"],
                st.session_state["last_manual_probability"],
                model,
                feature_names,
            )

            with st.expander("Lihat input final untuk model"):
                st.dataframe(st.session_state["last_manual_input"], use_container_width=True)
        else:
            st.info("Isi data di sebelah kiri, lalu klik Prediksi Sekarang.")

        st.markdown("</div>", unsafe_allow_html=True)


elif menu == "Upload Batch":
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📤 Prediksi Massal dari CSV")
    st.write("Unggah file CSV dengan format kolom yang sama atau mirip dengan data latih.")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            batch_df = read_csv_auto(uploaded_file)
            feature_df = batch_df.drop(columns=[TARGET_COL], errors="ignore")
            model_input = prepare_input(feature_df, feature_names)

            predictions = model.predict(model_input)
            result_df = batch_df.copy()
            result_df["Status Prediksi"] = predictions

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(model_input)

                for index, class_name in enumerate(model.classes_):
                    result_df[f"Prob_{class_name}"] = (probabilities[:, index] * 100).round(2)

            st.session_state["last_batch_result"] = result_df

            st.success(f"Berhasil memproses {len(result_df)} baris data.")
            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh Hasil CSV",
                data=csv_data,
                file_name="hasil_prediksi_ontrack.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as error:
            st.error(f"File tidak dapat diproses: {error}")
    else:
        st.info("Unggah file CSV terlebih dahulu untuk menjalankan prediksi massal.")


elif menu == "Dashboard":
    st.subheader("📊 Model Insights Dashboard")

    batch_result = st.session_state.get("last_batch_result")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Latih", f"{len(df):,}".replace(",", "."))
    c2.metric("Jumlah Fitur", len(feature_names))
    c3.metric("Target Terbanyak", df[TARGET_COL].value_counts().idxmax())
    c4.metric("Prediksi Massal", 0 if batch_result is None else len(batch_result))

    importance_df = pd.DataFrame(
        {
            "Fitur": [clean_label(item) for item in feature_names],
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Fitur Paling Berpengaruh")
        top_n = st.slider("Jumlah fitur", min_value=5, max_value=min(30, len(importance_df)), value=10)
        top_importance = importance_df.head(top_n)
        feature_importance_chart(top_importance)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Distribusi Target")

        if batch_result is not None:
            distribution = batch_result["Status Prediksi"].value_counts().rename_axis("Status").reset_index(name="Jumlah")
            title = "Distribusi Prediksi Massal Terakhir"
        else:
            distribution = df[TARGET_COL].value_counts().rename_axis("Status").reset_index(name="Jumlah")
            title = "Distribusi Target Data Latih"

        st.caption(title)
        status_distribution_chart(distribution)

        st.markdown("</div>", unsafe_allow_html=True)


elif menu == "Tentang Aplikasi":
    st.subheader("ℹ️ Panduan & Info Aplikasi")
    st.caption("Gunakan halaman ini sebagai panduan singkat sebelum menjalankan prediction, upload batch CSV, atau mengeksplorasi dashboard.")

    tab_mulai, tab_fitur, tab_data = st.tabs([
        "Mulai Cepat",
        "Panduan Fitur",
        "Catatan Data",
    ])

    with tab_mulai:
        st.markdown("#### Cara menggunakan onTrack dalam 4 langkah")
        st.markdown(
            """
            1. Buka **Input Manual** untuk satu mahasiswa atau **Upload CSV** untuk banyak mahasiswa.
            2. Gunakan preset bila sedang melakukan demo, atau isi data yang tersedia.
            3. Klik **Prediksi Sekarang**, lalu periksa ringkasan profil dan probabilitas hasil.
            4. Gunakan **Faktor Risiko dan Tindakan** serta **Mode Simulasi** untuk mendiskusikan prioritas intervensi.
            """
        )
        st.info(
            "Hasil prediksi adalah alat skrining dan pendukung keputusan. "
            "Hasil ini tidak boleh menjadi satu-satunya dasar untuk mengambil keputusan akademik terhadap mahasiswa."
        )

    with tab_fitur:
        st.markdown("#### Fungsi setiap halaman")

        with st.expander("🏠 Beranda", expanded=True):
            st.write(
                "Menampilkan gambaran awal dataset: total mahasiswa, tingkat Dropout, jumlah fitur model, "
                "serta distribusi status akademik."
            )

        with st.expander("📝 Input Manual", expanded=True):
            st.write(
                "Digunakan untuk memprediksi satu mahasiswa. Isi data pendaftaran, status mahasiswa, "
                "dan kondisi akademik Semester 1–2, kemudian tekan **Prediksi Sekarang**."
            )
            st.markdown(
                f"- **Mata kuliah diambil/lulus:** dibatasi maksimal {MAX_COURSES_PER_SEMESTER} per semester pada formulir manual. "
                "Jumlah lulus harus kurang dari atau sama dengan jumlah yang diambil.\n"
                "- **Jumlah evaluasi:** dapat mencakup ujian, tugas, dan kuis; bukan jumlah mata kuliah.\n"
                "- **Rata-rata nilai:** menggunakan skala model 0–20."
            )

        with st.expander("🧩 Preset, Hasil, dan Simulasi", expanded=False):
            st.write(
                "Preset Risiko Tinggi, Stabil, dan Berprestasi menyediakan contoh input untuk demo. "
                "Setelah prediksi, lihat gauge risiko Dropout, ringkasan profil, probabilitas, faktor yang perlu diperhatikan, "
                "serta tindakan prioritas."
            )
            st.write(
                "Mode Simulasi memungkinkan pengujian perubahan UKT, tunggakan, jumlah mata kuliah lulus, dan nilai Semester 2 "
                "tanpa mengubah input utama atau data latih."
            )

        with st.expander("📤 Upload CSV", expanded=False):
            st.write(
                "Digunakan untuk memproses banyak mahasiswa melalui CSV. Unggah file dengan kolom yang sama atau semirip mungkin "
                "dengan data latih, kemudian unduh hasil prediksi setelah pemrosesan selesai."
            )

        with st.expander("📊 Dashboard", expanded=False):
            st.write(
                "Menampilkan fitur model yang paling berpengaruh secara global dan distribusi status dari data latih atau hasil unggahan massal terakhir."
            )

    with tab_data:
        st.markdown("#### Memahami skala dan batasan data")
        st.warning(
            "Dataset model menggunakan istilah 'curricular units'. Pada aplikasi ini istilah tersebut disederhanakan menjadi "
            "'mata kuliah' agar mudah dibaca. Karena struktur kurikulum setiap kampus dapat berbeda, angka ini perlu ditafsirkan "
            "secara hati-hati dan divalidasi dengan data akademik lokal."
        )
        st.markdown(
            """
            **Pengaman yang digunakan pada input manual:**
            - Mata kuliah diambil dan lulus dibatasi **0–12** per semester.
            - Mata kuliah lulus tidak boleh lebih banyak daripada yang diambil.
            - Nilai rata-rata memakai rentang **0–20** sesuai skala model.
            - Input pada bagian *Fitur Lain dari Dataset* menggunakan nilai tengah data latih agar model tetap menerima struktur data lengkap.

            Batas 12 ini hanya diterapkan pada formulir manual untuk mencegah isian yang tidak realistis. Data latih, file unggahan massal, dan model Random Forest tidak diubah oleh pengaman tersebut.
            """
        )

        st.markdown("#### Informasi teknis")
        st.write(
            f"{APP_NAME} menggunakan model Random Forest untuk memprediksi status akademik mahasiswa "
            "berdasarkan fitur dari dataset UCI Student Dropout and Academic Success."
        )
        st.write(f"Jumlah fitur model: {len(feature_names)}")
        st.write(f"Lokasi file model: `{MODEL_PATH.name}`")
        st.write(f"Lokasi file fitur: `{FEATURE_PATH.name}`")
        st.write(f"Terakhir dibuka: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


st.markdown(
    f'<div class="footer">Copyright 2026 - {APP_NAME}</div>',
    unsafe_allow_html=True,
)
