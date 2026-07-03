# onTrack - Student Academic Status Prediction

Aplikasi web berbasis **Streamlit** dan **Machine Learning** untuk memprediksi status akademik mahasiswa: `Dropout`, `Enrolled`, atau `Graduate`.

## Fitur

- Prediksi satu mahasiswa melalui form input manual
- Quick Demo Presets untuk kebutuhan demonstrasi
- Probabilitas status dan indikator risiko dropout
- Ringkasan profil mahasiswa dan rekomendasi tindakan
- Mode simulasi "Bagaimana Jika?"
- Prediksi massal melalui upload CSV
- Dashboard eksploratif dengan filter
- Light Mode dan Dark Mode

## Teknologi

- Python
- Streamlit
- Pandas
- Scikit-learn
- Random Forest Classifier
- Plotly
- Streamlit ECharts

## Struktur Folder

```text
ontrack-student-success-ml/
|- app.py
|- data.csv
|- rf_model.pkl
|- feature_names.pkl
|- train_model.py
|- requirements.txt
|- assets/
|  `- styles.css
`- docs/
   `- SOP_onTrack_GitHub_Streamlit.docx
```

## Cara Menjalankan Lokal

```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Buka aplikasi pada `http://localhost:8501`.

## Catatan Data dan Etika

Hasil prediksi digunakan sebagai alat skrining dan pendukung keputusan. Hasil tidak boleh menjadi satu-satunya dasar keputusan akademik terhadap mahasiswa. Pastikan dataset yang diunggah ke repository tidak berisi data pribadi atau rahasia.

## Link

- GitHub Repository: **Tambahkan link setelah repository dibuat**
- Streamlit Cloud: **Tambahkan link setelah deployment berhasil**

## Pembuat

- Nama: **Isi nama kamu**
- NIM: **Isi NIM kamu**
- Mata Kuliah: Pemrograman Python (Machine Learning)
