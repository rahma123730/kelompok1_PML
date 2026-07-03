# onTrack — Student Academic Status Prediction

> Aplikasi *Machine Learning* berbasis Streamlit untuk membantu memprediksi status akademik mahasiswa sebagai **Dropout**, **Enrolled**, atau **Graduate**.

---

## 1. Deskripsi Proyek

**onTrack** adalah aplikasi web interaktif yang dibuat sebagai proyek EAS mata kuliah Pemrograman Python Machine Learning. Aplikasi ini memanfaatkan model **Random Forest Classifier** untuk melakukan prediksi awal terhadap status akademik mahasiswa berdasarkan karakteristik akademik, administrasi/keuangan, dan data pendaftaran yang tersedia pada dataset.

Hasil aplikasi bersifat **alat skrining dan pendukung keputusan**. Hasil prediksi tidak boleh dipakai sebagai satu-satunya dasar untuk menetapkan keputusan akademik terhadap mahasiswa.

---

## 2. Tujuan Proyek

1. Menerapkan proses *Machine Learning* dari data hingga aplikasi web.
2. Membuat model klasifikasi untuk prediksi status akademik mahasiswa.
3. Menyediakan prediksi individual dan prediksi massal melalui file CSV.
4. Menyajikan hasil prediksi secara visual melalui probabilitas kelas, indikator risiko, dan rekomendasi tindak lanjut.
5. Menyediakan dokumentasi penggunaan, pengujian, repository, dan deployment untuk kebutuhan EAS kelompok.

---

## 3. Fitur Utama

- Prediksi satu mahasiswa melalui formulir input manual.
- *Quick Demo Presets* untuk demonstrasi skenario risiko tinggi, stabil, dan berprestasi.
- Prediksi tiga kelas: `Dropout`, `Enrolled`, dan `Graduate`.
- Tabel dan grafik probabilitas setiap kelas prediksi.
- Indikator risiko *dropout*.
- Ringkasan profil mahasiswa dan rekomendasi tindak lanjut.
- Mode simulasi **“Bagaimana Jika?”** untuk membandingkan perubahan skenario.
- Prediksi massal melalui unggahan file CSV.
- Dashboard eksploratif dengan ringkasan dan visualisasi status akademik.
- Light mode dan dark mode.
- Panduan penggunaan aplikasi di dalam menu aplikasi.

---

## 4. Teknologi yang Digunakan

| Teknologi | Kegunaan |
|---|---|
| Python | Bahasa pemrograman utama |
| Streamlit | Framework aplikasi web interaktif |
| Pandas | Pemrosesan dan analisis data |
| Scikit-learn | Pelatihan dan penggunaan model *Machine Learning* |
| Random Forest Classifier | Algoritma klasifikasi status akademik |
| Plotly | Visualisasi data interaktif |
| Streamlit ECharts | Visualisasi dashboard tambahan |
| GitHub | *Version control* dan publikasi source code |
| Streamlit Community Cloud | Deployment aplikasi web |

---

## 5. Struktur Repository

```text
kelompok1_PML/
│
├── app.py                         # File utama aplikasi Streamlit
├── data.csv                       # Dataset yang dipakai aplikasi
├── rf_model.pkl                   # Model Random Forest terlatih
├── feature_names.pkl              # Daftar fitur yang dibutuhkan model
├── train_model.py                 # Script pelatihan model
├── requirements.txt               # Daftar dependensi Python
├── README.md                      # Dokumentasi utama proyek
├── .gitignore                     # File/folder yang tidak diunggah ke GitHub
│
├── assets/
│   └── styles.css                 # Styling antarmuka aplikasi
│
├── docs/
│   ├── SOP_onTrack_EAS_kelompok1_PML.md
│   ├── SOP_onTrack_EAS_kelompok1_PML.pdf
│   ├── test_cases.md
│   └── screenshots/
│
├── Analisis/                      # File analisis kelompok
├── Model/                         # File model/analisis pendukung kelompok
├── data/                          # Data pendukung kelompok
└── vscode/                        # Konfigurasi atau file pendukung VS Code
```

> Untuk deployment Streamlit, file inti yang digunakan adalah `app.py`, `data.csv`, `rf_model.pkl`, `feature_names.pkl`, `requirements.txt`, dan folder `assets/`.

---

## 6. Cara Menjalankan Aplikasi Secara Lokal

### 6.1 Clone repository

```bash
git clone https://github.com/rahma123730/kelompok1_PML.git
cd kelompok1_PML
```

### 6.2 Buat virtual environment

```bash
python -m venv .venv
```

### 6.3 Aktifkan virtual environment

**Git Bash**

```bash
source .venv/Scripts/activate
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 6.4 Install seluruh dependensi

```bash
python -m pip install -r requirements.txt
```

### 6.5 Jalankan aplikasi

```bash
streamlit run app.py
```

Buka alamat berikut pada browser:

```text
http://localhost:8501
```

---

## 7. Cara Menggunakan Aplikasi

### Beranda

Menu **Beranda** menampilkan ringkasan jumlah data, distribusi target, jumlah fitur model, serta status akademik yang paling dominan pada dataset.

### Input Manual

1. Buka menu **Input Manual**.
2. Gunakan salah satu *Quick Demo Presets* atau isi data mahasiswa secara manual.
3. Lengkapi data pendaftaran, status administrasi/keuangan, dan capaian akademik semester 1–2.
4. Tekan tombol prediksi.
5. Tinjau hasil kelas prediksi, probabilitas, indikator risiko, faktor yang perlu diperhatikan, dan rekomendasi tindakan.
6. Gunakan **Mode Simulasi: Bagaimana Jika?** untuk membandingkan skenario perubahan secara sementara.

### Upload CSV

1. Buka menu **Upload CSV**.
2. Upload file CSV dengan kolom yang sesuai dengan fitur model.
3. Sistem akan melakukan prediksi pada setiap baris data yang diunggah.
4. Tinjau hasil dan unduh keluaran prediksi apabila tersedia.

### Dashboard

1. Buka menu **Dashboard**.
2. Gunakan filter yang disediakan.
3. Tinjau visualisasi dan ringkasan pola status akademik pada data.

### Panduan

Menu **Panduan** berisi penjelasan singkat mengenai tujuan aplikasi, cara penggunaan, dan catatan penggunaan hasil prediksi secara bertanggung jawab.

---

## 8. Model Machine Learning

Aplikasi menggunakan algoritma:

```text
Random Forest Classifier
```

Model melakukan klasifikasi menjadi tiga kategori:

| Kelas Prediksi | Makna |
|---|---|
| `Dropout` | Mahasiswa diprediksi memiliki risiko tidak melanjutkan studi |
| `Enrolled` | Mahasiswa diprediksi masih aktif terdaftar |
| `Graduate` | Mahasiswa diprediksi memiliki pola yang mengarah pada penyelesaian studi |

File model yang dipakai oleh aplikasi:

- `rf_model.pkl`
- `feature_names.pkl`

---

## 9. Etika dan Batasan Penggunaan

- Hasil prediksi bersifat pendukung keputusan, bukan keputusan final.
- Hasil tidak boleh digunakan untuk diskriminasi terhadap mahasiswa.
- Verifikasi data dan pertimbangan akademik manusia tetap diperlukan.
- Jangan mengunggah data pribadi, data sensitif, atau data rahasia ke repository publik.
- Penjelasan faktor risiko pada aplikasi adalah indikator ringkas berbasis input dan bukan klaim hubungan sebab-akibat individual.

---

## 10. Dokumentasi

Dokumentasi lengkap ditempatkan pada folder `docs/`.

| Dokumen | Keterangan |
|---|---|
| `SOP_onTrack_EAS_kelompok1_PML.md` | SOP penggunaan, pengujian, dan deployment |
| `screenshots/` | Screenshot aplikasi, repository, dan deployment |

---

## 11. Anggota Kelompok

| No. | NIM | Nama |
|---:|---|---|
| 1 | 20124096 | SABILI PUTRA KIRANA |
| 2 | 20124021 | LAILANI ALIIFAH | 
| 3 | 20124033 | RAHMA AULIA DINI |
| 4 | 20124025 | MUHAFIDZ IKRAM RAMADHAN |
| 5 | 20124013 | EFLIN ATALYA MANUELA LUMBANTORUAN |
| 6 | 20124012 | BIMO SAPUTRA |

---

## 12. Link Proyek

- **GitHub Repository:** https://github.com/rahma123730/kelompok1_PML
- **Streamlit Deployment:** (https://ontrack.streamlit.app/)

---

## 13. Status Proyek

- [x] Aplikasi Streamlit tersedia.
- [x] Model Random Forest tersedia dalam format `.pkl`.
- [x] Kode sumber dan file inti tersedia di GitHub.
- [x] Dokumentasi README tersedia.
- [x] SOP diunggah ke folder `docs/`.
- [x] Aplikasi dideploy ke Streamlit Community Cloud.

---

## 14. Lisensi

Proyek ini dibuat untuk keperluan akademik pada tugas EAS mata kuliah Pemrograman Python Machine Learning.
