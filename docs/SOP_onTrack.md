# SOP dan Dokumentasi EAS
## onTrack — Student Academic Status Prediction

| Informasi | Keterangan |
|---|---|
| Mata Kuliah | Pemrograman Python Machine Learning |
| Nama Proyek | onTrack — Student Academic Status Prediction |
| Jenis Aplikasi | Aplikasi web Streamlit berbasis Machine Learning |
| Algoritma | Random Forest Classifier |
| Repository | https://github.com/rahma123730/kelompok1_PML |
| Dokumen | SOP penggunaan, pengujian, repository, dan deployment |
| Versi Dokumen | 1.0 |

---

## 1. Tujuan Dokumen

Dokumen ini menjadi panduan operasional untuk menjalankan, menggunakan, menguji, mendokumentasikan, dan melakukan deployment aplikasi **onTrack**. SOP ini disusun untuk kebutuhan EAS kelompok agar setiap anggota dapat menjalankan aplikasi melalui komputer lokal maupun melalui deployment Streamlit Community Cloud.

---

## 2. Deskripsi Aplikasi

**onTrack** adalah aplikasi web berbasis Streamlit untuk memprediksi status akademik mahasiswa dengan tiga keluaran kelas:

- `Dropout`
- `Enrolled`
- `Graduate`

Aplikasi memakai model **Random Forest Classifier** yang disimpan dalam `rf_model.pkl`. Daftar fitur input model disimpan dalam `feature_names.pkl`. Aplikasi menampilkan hasil prediksi, probabilitas tiap kelas, risiko dropout, indikator yang perlu diperhatikan, rekomendasi tindak lanjut, simulasi skenario, unggah CSV batch, dan dashboard eksploratif.

---

## 3. Anggota Kelompok

| No. | NIM | Nama |
|---:|---|---|
| 1 | 20124096 | SABILI PUTRA KIRANA |
| 2 | 20124021 | LAILANI ALIIFAH | 
| 3 | 20124033 | RAHMA AULIA DINI |
| 4 | 20124025 | MUHAFIDZ IKRAM RAMADHAN |
| 5 | 20124013 | EFLIN ATALYA MANUELA LUMBANTORUAN |
| 6 | 20124012 | BIMO SAPUTRA |
---

## 4. Prasyarat Sistem

Sebelum menjalankan aplikasi secara lokal, pastikan tersedia:

1. Sistem operasi Windows, macOS, atau Linux.
2. Python yang kompatibel dengan isi file `requirements.txt`.
3. Koneksi internet untuk instalasi dependensi dan deployment.
4. Browser modern seperti Google Chrome, Microsoft Edge, atau Mozilla Firefox.
5. Git/GitHub Desktop untuk sinkronisasi repository.

---

## 5. Struktur File yang Wajib Tersedia

Pastikan file berikut tersedia di folder utama repository:

```text
kelompok1_PML/
├── app.py
├── data.csv
├── rf_model.pkl
├── feature_names.pkl
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
└── assets/
    └── styles.css
```

Keterangan penting:

- `app.py`, `data.csv`, `rf_model.pkl`, dan `feature_names.pkl` harus berada pada level folder yang sama.
- Folder `assets/` harus berada pada level folder yang sama dengan `app.py`.
- File `.venv/` tidak perlu dan tidak boleh diunggah ke GitHub.

---

## 6. SOP Menjalankan Aplikasi Secara Lokal

### 6.1 Masuk ke folder proyek

Buka Terminal, PowerShell, Command Prompt, atau Git Bash dari folder repository `kelompok1_PML`.

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

Saat berhasil, terminal biasanya menunjukkan awalan `(.venv)`.

### 6.4 Install dependensi

```bash
python -m pip install -r requirements.txt
```

Jika terdapat error versi dependensi, periksa kembali kecocokan versi Python dan isi `requirements.txt`, kemudian ulangi instalasi pada virtual environment yang aktif.

### 6.5 Jalankan Streamlit

```bash
streamlit run app.py
```

Aplikasi dapat dibuka melalui browser pada alamat:

```text
http://localhost:8501
```

### 6.6 Menghentikan aplikasi

Pada jendela terminal yang menjalankan Streamlit, tekan:

```text
Ctrl + C
```

---

## 7. SOP Penggunaan Fitur Aplikasi

### 7.1 Beranda

1. Buka menu **Beranda**.
2. Tinjau ringkasan jumlah mahasiswa, risiko dropout pada dataset, jumlah fitur model, dan status terbanyak.
3. Tinjau grafik distribusi target untuk memahami komposisi data.

### 7.2 Input Manual

1. Buka menu **Input Manual**.
2. Gunakan preset **Risiko Tinggi**, **Stabil**, atau **Berprestasi** untuk demonstrasi cepat; atau isi data manual.
3. Isi data pendaftaran seperti usia, nilai masuk, jenis kelamin, program studi, mode pendaftaran, dan waktu kuliah.
4. Isi status mahasiswa seperti beasiswa, tunggakan, dan status UKT.
5. Isi data akademik semester 1 dan semester 2.
6. Pastikan jumlah mata kuliah lulus tidak lebih besar dari jumlah mata kuliah yang diambil.
7. Tekan tombol prediksi.
8. Tinjau hasil prediksi, probabilitas, risiko dropout, profil ringkas, faktor risiko, faktor positif, dan rekomendasi.

### 7.3 Mode Simulasi “Bagaimana Jika?”

1. Lakukan prediksi manual terlebih dahulu.
2. Buka bagian simulasi.
3. Ubah skenario yang diperlukan, misalnya status UKT, tunggakan, jumlah mata kuliah lulus semester 2, atau nilai semester 2.
4. Tekan tombol perbandingan simulasi.
5. Bandingkan risiko awal dan risiko hasil simulasi.
6. Gunakan hasil simulasi sebagai bahan diskusi/pendampingan, bukan keputusan otomatis.

### 7.4 Upload CSV

1. Buka menu **Upload CSV**.
2. Siapkan CSV dengan kolom yang relevan dengan fitur model.
3. Upload file CSV.
4. Periksa ringkasan dan hasil prediksi yang ditampilkan.
5. Unduh keluaran hasil prediksi apabila fitur unduh tersedia.

### 7.5 Dashboard

1. Buka menu **Dashboard**.
2. Atur filter sesuai kebutuhan analisis.
3. Tinjau visualisasi dan ringkasan pola data mahasiswa.

### 7.6 Panduan

1. Buka menu **Panduan**.
2. Baca informasi tujuan aplikasi dan catatan penggunaan hasil prediksi secara bertanggung jawab.

---

## 8. SOP Pengelolaan Repository GitHub

### 8.1 File yang perlu diunggah

File inti yang wajib berada di GitHub:

- `app.py`
- `data.csv`
- `rf_model.pkl`
- `feature_names.pkl`
- `train_model.py`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `assets/styles.css`
- Dokumentasi di folder `docs/`

### 8.2 File yang tidak perlu diunggah

Pastikan `.gitignore` mengecualikan:

```text
.venv/
venv/
__pycache__/
backup/
*.log
.streamlit/secrets.toml
.env
```

### 8.3 Prosedur commit dan push melalui GitHub Desktop

1. Buka GitHub Desktop.
2. Pilih repository `kelompok1_PML`.
3. Buka tab **Changes**.
4. Periksa file yang berubah sebelum melakukan commit.
5. Isi ringkasan commit yang jelas, misalnya:

```text
Add onTrack SOP and improve project README
```

6. Klik **Commit to main**.
7. Klik **Push origin**.
8. Buka repository melalui browser dan lakukan *refresh* untuk memastikan file sudah muncul.

---

## 9. SOP Deployment Streamlit Community Cloud

### 9.1 Persiapan

Sebelum deployment, pastikan:

- Repository GitHub bersifat publik atau dapat diakses oleh akun Streamlit.
- `requirements.txt` ada di folder utama repository.
- `app.py` ada di folder utama repository.
- Semua file model, data, dan folder `assets/` sudah di-push ke GitHub.

### 9.2 Langkah deployment

1. Buka Streamlit Community Cloud.
2. Login menggunakan akun yang terhubung dengan GitHub.
3. Pilih **Create app** atau **New app**.
4. Pilih repository:

```text
rahma123730/kelompok1_PML
```

5. Pilih branch:

```text
main
```

6. Isi path file utama:

```text
app.py
```

7. Klik **Deploy**.
8. Tunggu sampai proses instalasi dan build selesai.
9. Buka tautan aplikasi yang dihasilkan.
10. Uji menu Beranda, Input Manual, Upload CSV, Dashboard, dan Panduan.
11. Salin tautan deployment ke README pada bagian **Link Proyek**.

---

## 10. Test Cases

| ID | Komponen | Skenario Uji | Hasil yang Diharapkan | Status |
|---|---|---|---|---|
| TC-01 | Menjalankan aplikasi | Menjalankan `streamlit run app.py` | Aplikasi terbuka pada `localhost:8501` tanpa error | Belum diisi |
| TC-02 | Beranda | Membuka menu Beranda | Ringkasan data dan distribusi target tampil | Belum diisi |
| TC-03 | Input Manual | Mengisi input valid dan menjalankan prediksi | Kelas prediksi dan probabilitas tampil | Belum diisi |
| TC-04 | Validasi akademik | Jumlah MK lulus melebihi MK diambil | Aplikasi menampilkan pesan validasi dan mencegah prediksi tidak wajar | Belum diisi |
| TC-05 | Preset demo | Klik preset Risiko Tinggi/Stabil/Berprestasi | Form input terisi sesuai skenario preset | Belum diisi |
| TC-06 | Simulasi | Mengubah skenario UKT/tunggakan/nilai | Perbandingan prediksi dan risiko simulasi tampil | Belum diisi |
| TC-07 | Upload CSV | Mengunggah CSV yang sesuai | Hasil prediksi batch tampil tanpa error | Belum diisi |
| TC-08 | Dashboard | Membuka Dashboard dan mengubah filter | Visualisasi dan ringkasan berubah sesuai filter | Belum diisi |
| TC-09 | Mode tampilan | Mengaktifkan dark mode | Warna teks dan elemen UI tetap terbaca | Belum diisi |
| TC-10 | Deployment | Membuka aplikasi dari Streamlit Cloud | Aplikasi dapat diakses publik dan fitur utama berjalan | Belum diisi |

> Setelah pengujian dilakukan, ganti kolom **Status** menjadi `Pass`, `Fail`, atau `Perlu Perbaikan`, lalu tambahkan bukti screenshot pada folder `docs/screenshots/`.

---

## 11. Troubleshooting

| Masalah | Kemungkinan Penyebab | Tindakan |
|---|---|---|
| `File CSS tidak ditemukan` | Folder `assets/` atau `styles.css` tidak berada pada lokasi yang benar | Pastikan `assets/styles.css` sejajar dengan `app.py` |
| `data.csv tidak ditemukan` | File data tidak berada di folder utama | Pindahkan `data.csv` ke folder yang sama dengan `app.py` |
| `File model belum lengkap` | `rf_model.pkl` atau `feature_names.pkl` tidak tersedia | Pastikan kedua file berada di folder utama aplikasi |
| Library belum terpasang | Dependensi belum diinstal | Aktifkan `.venv`, lalu jalankan `python -m pip install -r requirements.txt` |
| Aplikasi gagal berjalan setelah deployment | File belum di-push atau path entrypoint salah | Pastikan branch `main`, path `app.py`, file inti, dan `requirements.txt` sudah tersedia |

---

## 12. Catatan Etika dan Keamanan Data

1. Prediksi aplikasi tidak menggantikan penilaian akademik oleh pihak kampus.
2. Hasil prediksi tidak boleh digunakan untuk diskriminasi.
3. Hindari unggah data identitas pribadi atau data rahasia ke repository publik.
4. Gunakan hasil prediksi untuk mendukung pemantauan, konseling, dan perencanaan intervensi yang manusiawi.
5. Faktor risiko yang ditampilkan adalah indikator ringkas, bukan bukti kausal individual.

---

## 13. Bukti yang Perlu Disiapkan untuk EAS

Simpan screenshot berikut di folder `docs/screenshots/`:

1. Halaman repository GitHub yang menampilkan file aplikasi.
2. Isi `README.md`.
3. Halaman aplikasi onTrack pada menu Beranda.
4. Prediksi manual beserta hasilnya.
5. Upload CSV atau dashboard.
6. Halaman deployment Streamlit Cloud beserta URL aplikasi.
7. Hasil pengujian test case.

---

## 14. Riwayat Revisi Dokumen

| Versi | Tanggal | Keterangan |
|---|---|---|
| 1.0 | 3 Juli 2026 | Penyusunan SOP awal untuk proyek onTrack EAS Kelompok 1 |
