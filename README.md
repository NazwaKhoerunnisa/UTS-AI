# UTS Intelegensi Buatan
## Klasifikasi Kultivar Anggur (Kasus 1)

### Nama : Nazwa Khoerunnisa
### Kelas : TIF RP 23B
### NIM: 23552011093

Repository ini berisi hasil pengerjaan Ujian Tengah Semester mata kuliah Intelegensi Buatan dengan topik klasifikasi kultivar anggur menggunakan Machine Learning.

### Penentuan Kasus Berdasarkan NIM

Kasus yang dikerjakan adalah **Kasus 1 (Wine Classification)**. Penentuan kasus dilakukan berdasarkan perhitungan jumlah seluruh digit NIM dengan ketentuan yang diberikan pada soal UTS.

Perhitungan:

- NIM: 23552011093
- Jumlah seluruh digit: 2 + 3 + 5 + 5 + 2 + 0 + 1 + 1 + 0 + 9 + 3 = 31
- Hasil modulo 10: 31 mod 10 = 1

Berdasarkan hasil perhitungan tersebut, kasus yang diperoleh adalah **Kasus 1**, yaitu klasifikasi kultivar anggur menggunakan dataset Wine.

### Dataset
Dataset yang digunakan adalah Wine Dataset dengan 178 data dan 13 fitur numerik. Target klasifikasi terdiri dari tiga kelas, yaitu Kultivar A, Kultivar B, dan Kultivar C.

### Metode dan Algoritma
Beberapa algoritma klasifikasi yang dibandingkan:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Pemilihan model dilakukan menggunakan Cross Validation. Pipeline mencakup rekayasa fitur dan StandardScaler.

### Hasil
Model terbaik berdasarkan Cross Validation adalah Support Vector Machine (RBF) dengan variasi fitur tambah_buang.

Hasil evaluasi pada data pengujian:
- Accuracy: 97,22%
- Precision Macro: 97,78%
- Recall Macro: 96,67%
- F1-Score Macro: 97,10%
- ROC-AUC OVR Macro: 100%

## Laporan UTS

Laporan Ujian Tengah Semester Intelegensi Buatan dapat dilihat melalui tautan berikut:

[📄 Lihat Laporan UTS Intelegensi Buatan (PDF)](./Laporan_UTS_Intelegensi%20Buatan_Nazwa%20Khoerunnisa.pdf)

### Aplikasi
Aplikasi dibuat menggunakan Streamlit untuk menerima input fitur anggur dan menampilkan hasil prediksi beserta probabilitas kelas.

### File Proyek
- `fitur.py` — Pengolahan dan rekayasa fitur.
- `latih.py` — Pelatihan dan perbandingan model.
- `app.py` — Aplikasi klasifikasi menggunakan Streamlit.
- `models/wine_model.joblib` — Model terbaik yang telah disimpan.
- `models/metadata.json` — Metadata model.
- `kasus01_wine_colab.ipynb` — Notebook pengerjaan di Google Colab.
- `hasil_uts_wine.zip` — Arsip file proyek.

