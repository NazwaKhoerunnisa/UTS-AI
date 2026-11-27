"""latih.py - pelatihan, evaluasi, dan penyimpanan model Kasus 1 (Wine).

Jalankan:  python latih.py [path_dataset]
Hasil   :  models/wine_model.joblib dan models/metadata.json
"""
import json
import os
import sys
import warnings

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from fitur import FITUR_ASLI, MODE_FITUR, TARGET, tambah_fitur

warnings.filterwarnings("ignore")
SEED = 42
DATASET_DEFAULT = os.path.join("dataset", "kasus01_wine.csv")
MODEL_PATH = os.path.join("models", "wine_model.joblib")


def buat_pipeline(model, mode):
    return Pipeline([
        ("fitur", FunctionTransformer(tambah_fitur, kw_args={"mode": mode})),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def daftar_model():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def main(path_dataset=DATASET_DEFAULT):
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(path_dataset).drop_duplicates().reset_index(drop=True)
    X, y = df[FITUR_ASLI], df[TARGET]

    # 1) Split dulu, sebelum preprocessing apa pun
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    # 2) Pemilihan model dengan cross-validation di data train
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    baris = []
    for mode in MODE_FITUR:
        for nama, model in daftar_model().items():
            hasil = cross_validate(buat_pipeline(model, mode), X_train, y_train, cv=cv,
                                   scoring={"acc": "accuracy", "f1": "f1_macro"}, n_jobs=-1)
            baris.append({"varian_fitur": mode, "model": nama,
                          "acc_mean": hasil["test_acc"].mean(), "acc_std": hasil["test_acc"].std(),
                          "f1_mean": hasil["test_f1"].mean(), "f1_std": hasil["test_f1"].std()})
    hasil_cv = pd.DataFrame(baris)
    hasil_cv["jumlah_fitur"] = hasil_cv["varian_fitur"].map(
        lambda m: tambah_fitur(X_train.head(1), mode=m).shape[1])
    # F1 tertinggi, lalu std terkecil, lalu jumlah fitur paling sedikit bila skor sama
    hasil_cv = (hasil_cv
                .sort_values(["f1_mean", "f1_std", "jumlah_fitur"], ascending=[False, True, True],
                             key=lambda k: k.round(4) if k.name in ("f1_mean", "f1_std") else k)
                .reset_index(drop=True))
    print(hasil_cv.round(4).to_string(index=False))
    terbaik = hasil_cv.iloc[0]
    mode, nama = terbaik["varian_fitur"], terbaik["model"]
    print(f"\nTerpilih: {nama} | varian fitur: {mode}")

    # 3) Latih di train, evaluasi di test satu kali
    pipeline = buat_pipeline(daftar_model()[nama], mode).fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    kelas = list(pipeline.classes_)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    metrik = {"akurasi": float(accuracy_score(y_test, y_pred)),
              "precision_macro": float(prec), "recall_macro": float(rec), "f1_macro": float(f1),
              "roc_auc_ovr_macro": float(roc_auc_score(y_test, y_proba, multi_class="ovr",
                                                       average="macro", labels=kelas))}
    print("\nMetrik data test:")
    for k, v in metrik.items():
        print(f"  {k:>18}: {v:.4f}")

    # 4) Simpan pipeline + metadata
    cm = confusion_matrix(y_test, y_pred, labels=kelas)
    bundle = {
        "pipeline": pipeline, "nama_model": nama, "varian_fitur": mode,
        "fitur_mentah": FITUR_ASLI, "kelas": kelas,
        "rentang_fitur": {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
                              "median": float(X_train[c].median())} for c in FITUR_ASLI},
        "rata_kelas": X_train.groupby(y_train).mean().round(4).to_dict(orient="index"),
        "metrik_test": metrik, "confusion_matrix": cm.tolist(),
        "laporan_klasifikasi": classification_report(y_test, y_pred, output_dict=True, digits=4),
        "hasil_cv": hasil_cv.round(4).to_dict(orient="records"),
        "jumlah_train": int(len(X_train)), "jumlah_test": int(len(X_test)),
        "seed": SEED, "versi_sklearn": sklearn.__version__,
    }
    joblib.dump(bundle, MODEL_PATH)
    with open(os.path.join("models", "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k != "pipeline"}, f, indent=2, ensure_ascii=False)
    print("\nModel tersimpan di", MODEL_PATH)
    return bundle


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATASET_DEFAULT)
