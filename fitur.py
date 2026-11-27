"""fitur.py - fungsi feature engineering untuk Kasus 1 (Wine, target kultivar)."""
import pandas as pd

TARGET = "kultivar"

FITUR_ASLI = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline",
]

# Fitur yang berkorelasi tinggi dan informasinya sudah terwakili oleh fitur rasio
FITUR_REDUNDAN = ["total_phenols", "od280_od315_of_diluted_wines"]

MODE_FITUR = ("asli", "tambah", "tambah_buang")
EPS = 1e-9  # mencegah pembagian dengan nol


def tambah_fitur(X, mode="tambah_buang"):
    """Menambah fitur rasio (dan opsional membuang fitur redundan).

    Operasi dilakukan per baris tanpa mempelajari statistik data,
    sehingga aman dipakai di dalam pipeline (tidak ada data leakage).
    """
    if mode not in MODE_FITUR:
        raise ValueError(f"mode harus salah satu dari {MODE_FITUR}")

    if isinstance(X, pd.DataFrame):
        X = X[FITUR_ASLI].copy()
    else:
        X = pd.DataFrame(X, columns=FITUR_ASLI)

    if mode == "asli":
        return X

    X["rasio_flavanoid_fenol"] = X["flavanoids"] / (X["total_phenols"] + EPS)
    X["rasio_od280_flavanoid"] = X["od280_od315_of_diluted_wines"] / (X["flavanoids"] + EPS)
    X["rasio_proline_alkohol"] = X["proline"] / (X["alcohol"] + EPS)

    if mode == "tambah_buang":
        X = X.drop(columns=FITUR_REDUNDAN)
    return X
