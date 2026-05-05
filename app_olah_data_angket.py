
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Olah Data Angket", layout="wide")

st.title("Aplikasi Olah Data Angket")
st.write("Upload file Excel/CSV, lalu hasil olah data akan muncul otomatis.")

# ---------------------------
# Fungsi bantuan
# ---------------------------
def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ubah kolom jawaban angket menjadi numerik jika memungkinkan."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).str.replace(",", ".", regex=False)

        converted = pd.to_numeric(out[col], errors="coerce")

        # kalau ada angka valid, pakai hasil konversi
        if converted.notna().sum() > 0:
            out[col] = converted

    return out

def cronbach_alpha(data: pd.DataFrame):
    """Menghitung Cronbach's Alpha untuk item numerik."""
    numeric = data.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if numeric.shape[1] < 2:
        return np.nan
    numeric = numeric.dropna()
    if numeric.shape[0] < 2:
        return np.nan
    item_var = numeric.var(axis=0, ddof=1)
    total_var = numeric.sum(axis=1).var(ddof=1)
    k = numeric.shape[1]
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_var.sum() / total_var)

def kategori_mean(nilai):
    if pd.isna(nilai):
        return "-"
    if nilai <= 1.80:
        return "Sangat Tidak Setuju / Sangat Rendah"
    elif nilai <= 2.60:
        return "Tidak Setuju / Rendah"
    elif nilai <= 3.40:
        return "Netral / Sedang"
    elif nilai <= 4.20:
        return "Setuju / Tinggi"
    else:
        return "Sangat Setuju / Sangat Tinggi"

def detect_variable_groups(columns):
    """
    Deteksi otomatis kelompok variabel dari nama kolom.
    Contoh kolom: X1.1, X1_1, X1-1, X2.3, Z1, Y5
    """
    groups = {}
    for col in columns:
        name = str(col).strip().upper()
        if name.startswith("X1"):
            groups.setdefault("X1 - Ketersediaan Ruang Parkir", []).append(col)
        elif name.startswith("X2"):
            groups.setdefault("X2 - Perilaku Parkir Ilegal", []).append(col)
        elif name.startswith("X3"):
            groups.setdefault("X3 - Kepadatan Aktivitas Pasar", []).append(col)
        elif name.startswith("X4"):
            groups.setdefault("X4 - Manajemen dan Pengelolaan Parkir", []).append(col)
        elif name.startswith("Z"):
            groups.setdefault("Z - Lamanya Waktu Pencarian Parkir", []).append(col)
        elif name.startswith("Y"):
            groups.setdefault("Y - Kemacetan Lalu Lintas", []).append(col)
    return groups

def make_download_excel(sheets: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, data in sheets.items():
            safe_name = name[:31]
            data.to_excel(writer, sheet_name=safe_name, index=False)
    return output.getvalue()

# ---------------------------
# Sidebar upload
# ---------------------------
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Pilih file Excel atau CSV",
    type=["xlsx", "xls", "csv"]
)

st.sidebar.info(
    "Saran nama kolom: X1.1-X1.5, X2.1-X2.5, X3.1-X3.5, X4.1-X4.5, Z1-Z5, Y1-Y5."
)

if uploaded_file is None:
    st.warning("Silakan upload file tabulasi angket terlebih dahulu.")
    st.stop()

# ---------------------------
# Baca file
# ---------------------------
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        excel = pd.ExcelFile(uploaded_file)
        sheet_name = st.sidebar.selectbox("Pilih sheet", excel.sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
except Exception as e:
    st.error(f"File gagal dibaca: {e}")
    st.stop()

df = clean_numeric(df)
st.subheader("1. Data yang Diupload")
st.dataframe(df, use_container_width=True)

# ---------------------------
# Pilih kolom numerik
# ---------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    st.error("Tidak ada kolom numerik yang bisa diolah. Pastikan jawaban angket berisi angka 1 sampai 5.")
    st.stop()

st.sidebar.header("Pengaturan Kolom")
selected_cols = st.sidebar.multiselect(
    "Pilih kolom item angket yang ingin diolah",
    options=numeric_cols,
    default=numeric_cols
)

if not selected_cols:
    st.warning("Pilih minimal satu kolom item angket.")
    st.stop()

data = df[selected_cols].copy()

# ---------------------------
# Ringkasan umum
# ---------------------------
st.subheader("2. Ringkasan Umum")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah Responden", len(df))
c2.metric("Jumlah Item Diolah", len(selected_cols))
c3.metric("Rata-rata Keseluruhan", f"{data.mean().mean():.2f}")
c4.metric("Cronbach's Alpha", "-" if pd.isna(cronbach_alpha(data)) else f"{cronbach_alpha(data):.3f}")

# ---------------------------
# Statistik deskriptif item
# ---------------------------
st.subheader("3. Statistik Deskriptif per Item")
desc = pd.DataFrame({
    "Item": selected_cols,
    "N": data.count().values,
    "Mean": data.mean().round(3).values,
    "Median": data.median().round(3).values,
    "Min": data.min().values,
    "Max": data.max().values,
    "Std Dev": data.std().round(3).values,
})
desc["Kategori"] = desc["Mean"].apply(kategori_mean)
st.dataframe(desc, use_container_width=True)

# ---------------------------
# Frekuensi jawaban Likert
# ---------------------------
st.subheader("4. Frekuensi Jawaban Skala Likert")
freq_rows = []
for col in selected_cols:
    counts = data[col].value_counts(dropna=False).sort_index()
    total = data[col].notna().sum()
    row = {"Item": col}
    for score in [1, 2, 3, 4, 5]:
        row[f"F{score}"] = int(counts.get(score, 0))
        row[f"%{score}"] = round((counts.get(score, 0) / total * 100), 2) if total > 0 else 0
    freq_rows.append(row)

freq = pd.DataFrame(freq_rows)
st.dataframe(freq, use_container_width=True)

# ---------------------------
# Rata-rata per variabel
# ---------------------------
st.subheader("5. Rata-rata per Variabel")
detected_groups = detect_variable_groups(selected_cols)

if detected_groups:
    variable_rows = []
    variable_scores = pd.DataFrame(index=df.index)

    for var_name, cols in detected_groups.items():
        valid_cols = [c for c in cols if c in data.columns]
        if valid_cols:
            score = data[valid_cols].mean(axis=1)
            variable_scores[var_name] = score
            variable_rows.append({
                "Variabel": var_name,
                "Jumlah Item": len(valid_cols),
                "Mean": round(score.mean(), 3),
                "Std Dev": round(score.std(), 3),
                "Cronbach's Alpha": round(cronbach_alpha(data[valid_cols]), 3) if not pd.isna(cronbach_alpha(data[valid_cols])) else np.nan,
                "Kategori": kategori_mean(score.mean())
            })

    variable_summary = pd.DataFrame(variable_rows)
    st.dataframe(variable_summary, use_container_width=True)

    st.write("Grafik rata-rata variabel:")
    fig, ax = plt.subplots()
    ax.bar(variable_summary["Variabel"], variable_summary["Mean"])
    ax.set_ylabel("Mean")
    ax.set_ylim(0, 5)
    ax.tick_params(axis="x", rotation=30)
    st.pyplot(fig)

    st.subheader("6. Korelasi Antar Variabel")
    if variable_scores.shape[1] >= 2:
        corr = variable_scores.corr().round(3)
        st.dataframe(corr, use_container_width=True)

        fig2, ax2 = plt.subplots()
        im = ax2.imshow(corr.values)
        ax2.set_xticks(range(len(corr.columns)))
        ax2.set_yticks(range(len(corr.columns)))
        ax2.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax2.set_yticklabels(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax2.text(j, i, corr.iloc[i, j], ha="center", va="center")
        st.pyplot(fig2)
    else:
        corr = pd.DataFrame()
        st.info("Korelasi antar variabel membutuhkan minimal 2 variabel.")

else:
    st.info("Kelompok variabel belum terdeteksi. Gunakan nama kolom seperti X1.1, X2.1, X3.1, X4.1, Z1, Y1.")
    variable_summary = pd.DataFrame()
    variable_scores = pd.DataFrame()
    corr = pd.DataFrame()

# ---------------------------
# Download hasil
# ---------------------------
st.subheader("7. Download Hasil Olah Data")

sheets = {
    "Statistik_Item": desc,
    "Frekuensi_Likert": freq,
}
if not variable_summary.empty:
    sheets["Rata_Rata_Variabel"] = variable_summary
if not corr.empty:
    sheets["Korelasi"] = corr.reset_index().rename(columns={"index": "Variabel"})

excel_bytes = make_download_excel(sheets)

st.download_button(
    label="Download Hasil Olah Data Excel",
    data=excel_bytes,
    file_name="hasil_olah_data_angket.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
