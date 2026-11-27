"""app.py - Aplikasi Streamlit: klasifikasi kultivar anggur (Kasus 1, UTS Intelegensi Buatan)."""
import os

import joblib
import pandas as pd
import streamlit as st

import fitur  # noqa: F401  (diperlukan agar joblib dapat memuat FunctionTransformer)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "wine_model.joblib")

LABEL = {
    "alcohol": "Alkohol",
    "malic_acid": "Asam malat",
    "ash": "Abu (ash)",
    "alcalinity_of_ash": "Alkalinitas abu",
    "magnesium": "Magnesium",
    "total_phenols": "Total fenol",
    "flavanoids": "Flavanoid",
    "nonflavanoid_phenols": "Fenol non-flavanoid",
    "proanthocyanins": "Proantosianin",
    "color_intensity": "Intensitas warna",
    "hue": "Hue (rona warna)",
    "od280_od315_of_diluted_wines": "OD280/OD315 anggur encer",
    "proline": "Prolin",
}

st.set_page_config(page_title="Klasifikasi Kultivar Anggur", page_icon="🍷", layout="wide")


@st.cache_resource
def muat_bundle():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("File model belum ada. Jalankan `python latih.py` (atau notebook) terlebih dahulu.")
    st.stop()

bundle = muat_bundle()
pipeline = bundle["pipeline"]
fitur_mentah = bundle["fitur_mentah"]
kelas = bundle["kelas"]
rentang = bundle["rentang_fitur"]
metrik = bundle["metrik_test"]

st.title("🍷 Klasifikasi Kultivar Anggur")
st.caption("Menentukan kultivar (A, B, atau C) dari 13 hasil analisis kimia. "
           f"Model: {bundle['nama_model']} | varian fitur: {bundle['varian_fitur']}")

tab_prediksi, tab_metrik = st.tabs(["Prediksi", "Metrik evaluasi"])

# ------------------------------------------------------------------ Tab prediksi
with tab_prediksi:
    preset = st.selectbox(
        "Nilai awal input",
        ["Median data train"] + [f"Rata-rata {k}" for k in kelas],
        help="Pilih preset untuk mengisi semua input, lalu ubah nilainya sesuai sampel Anda.",
    )

    def nilai_awal(kolom):
        if preset == "Median data train":
            return rentang[kolom]["median"]
        return bundle["rata_kelas"][preset.replace("Rata-rata ", "")][kolom]

    st.subheader("Masukkan hasil analisis kimia (nilai mentah)")
    input_user = {}
    kolom_ui = st.columns(3)
    for i, nama in enumerate(fitur_mentah):
        r = rentang[nama]
        langkah = max(round((r["max"] - r["min"]) / 100, 3), 0.01)
        with kolom_ui[i % 3]:
            input_user[nama] = st.number_input(
                LABEL.get(nama, nama), min_value=0.0, value=float(nilai_awal(nama)),
                step=float(langkah), format="%.3f", key=f"{preset}_{nama}",
                help=f"Rentang data latih: {r['min']:.2f} sampai {r['max']:.2f}",
            )

    di_luar = [LABEL[n] for n in fitur_mentah
               if not (rentang[n]["min"] <= input_user[n] <= rentang[n]["max"])]
    if di_luar:
        st.warning("Nilai di luar rentang data latih untuk: " + ", ".join(di_luar)
                   + ". Prediksi bisa kurang andal.")

    if st.button("Prediksi kultivar", type="primary"):
        X_baru = pd.DataFrame([input_user])[fitur_mentah]
        proba = pipeline.predict_proba(X_baru)[0]
        prediksi = pipeline.classes_[proba.argmax()]

        st.success(f"Prediksi: **{prediksi}** (keyakinan {proba.max() * 100:.1f}%)")
        tabel = pd.DataFrame({"Kultivar": pipeline.classes_, "Probabilitas": proba})
        kiri, kanan = st.columns(2)
        with kiri:
            st.bar_chart(tabel.set_index("Kultivar"))
        with kanan:
            tabel["Probabilitas"] = (tabel["Probabilitas"] * 100).round(2).astype(str) + " %"
            st.dataframe(tabel, hide_index=True)

# ------------------------------------------------------------------ Tab metrik
with tab_metrik:
    st.subheader("Hasil evaluasi pada data test")
    st.caption(f"Data latih: {bundle['jumlah_train']} sampel, data test: {bundle['jumlah_test']} sampel "
               "(evaluasi test dilakukan satu kali).")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Akurasi", f"{metrik['akurasi']:.3f}")
    m2.metric("Precision macro", f"{metrik['precision_macro']:.3f}")
    m3.metric("Recall macro", f"{metrik['recall_macro']:.3f}")
    m4.metric("F1 macro", f"{metrik['f1_macro']:.3f}")
    m5.metric("ROC-AUC (OvR)", f"{metrik['roc_auc_ovr_macro']:.3f}")

    st.markdown("**Confusion matrix** (baris = kelas asli, kolom = prediksi)")
    st.dataframe(pd.DataFrame(bundle["confusion_matrix"],
                              index=[f"asli: {k}" for k in kelas],
                              columns=[f"prediksi: {k}" for k in kelas]))

    st.markdown("**Laporan klasifikasi**")
    laporan = pd.DataFrame(bundle["laporan_klasifikasi"]).T
    st.dataframe(laporan.round(3))

    st.markdown("**Perbandingan model (cross-validation di data train)**")
    st.dataframe(pd.DataFrame(bundle["hasil_cv"]), hide_index=True)
