import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

st.set_page_config(page_title="Olah Data Angket", layout="wide")

st.title("Aplikasi Olah Data Angket")
st.write("Upload file Excel/CSV, lalu hasil olah data akan muncul otomatis.")


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str).str.strip().str.replace(",", ".", regex=False)

        converted = pd.to_numeric(out[col], errors="coerce")

        if converted.notna().sum() > 0:
            out[col] = converted

    return out


def cronbach_alpha(data: pd.DataFrame):
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
        return "Sangat Rendah"
    elif nilai <= 2.60:
        return "Rendah"
    elif nilai <= 3.40:
        return "Sedang"
    elif nilai <= 4.20:
        return "Tinggi"
    else:
        return "Sangat Tinggi"


def detect_variable_groups(columns):
    groups = {}

    for col in columns:
        name = str(col).strip().upper().replace(" ", "")

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
        has_sheet = False

        for name, data in sheets.items():
            if data is not None and not data.empty:
                safe_name = str(name)[:31]
                data.to_excel(writer, sheet_name=safe_name, index=False)
                has_sheet = True

        if not has_sheet:
            pd.DataFrame({"Keterangan": ["Belum ada data hasil olah."]}).to_excel(
                writer, sheet_name="Keterangan", index=False
            )

    return output.getvalue()


def find_y_column(variable_scores: pd.DataFrame):
    for col in variable_scores.columns:
        col_text = str(col).upper()
        if col_text.startswith("Y") or "KEMACETAN" in col_text:
            return col
    return None


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


try:
    if uploaded_file.name.lower().endswith(".csv"):
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


numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

exclude_keywords = ["RESPONDEN", "NAMA", "NO", "ID"]
default_cols = [
    col for col in numeric_cols
    if not any(keyword in str(col).upper() for keyword in exclude_keywords)
]

if len(default_cols) == 0:
    default_cols = numeric_cols

if len(numeric_cols) == 0:
    st.error("Tidak ada kolom numerik yang bisa diolah. Pastikan jawaban angket berisi angka 1 sampai 5.")
    st.stop()


st.sidebar.header("Pengaturan Kolom")
selected_cols = st.sidebar.multiselect(
    "Pilih kolom item angket yang ingin diolah",
    options=numeric_cols,
    default=default_cols
)

if not selected_cols:
    st.warning("Pilih minimal satu kolom item angket.")
    st.stop()

data = df[selected_cols].copy()


st.subheader("2. Ringkasan Umum")

alpha_all = cronbach_alpha(data)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah Responden", len(df))
c2.metric("Jumlah Item Diolah", len(selected_cols))
c3.metric("Rata-rata Keseluruhan", f"{data.mean().mean():.2f}")
c4.metric("Cronbach's Alpha", "-" if pd.isna(alpha_all) else f"{alpha_all:.3f}")


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


st.subheader("4. Frekuensi Jawaban Skala Likert")

freq_rows = []

for col in selected_cols:
    counts = data[col].value_counts(dropna=False).sort_index()
    total = data[col].notna().sum()

    row = {"Item": col}

    for score in [1, 2, 3, 4, 5]:
        jumlah = int(counts.get(score, 0))
        row[f"F{score}"] = jumlah
        row[f"%{score}"] = round((jumlah / total * 100), 2) if total > 0 else 0

    freq_rows.append(row)

freq = pd.DataFrame(freq_rows)
st.dataframe(freq, use_container_width=True)


st.subheader("5. Rata-rata per Variabel")

detected_groups = detect_variable_groups(selected_cols)

variable_summary = pd.DataFrame()
variable_scores = pd.DataFrame(index=df.index)
corr = pd.DataFrame()
coef_df = pd.DataFrame()

if detected_groups:
    variable_rows = []

    for var_name, cols in detected_groups.items():
        valid_cols = [c for c in cols if c in data.columns]

        if valid_cols:
            score = data[valid_cols].mean(axis=1)
            variable_scores[var_name] = score

            alpha_var = cronbach_alpha(data[valid_cols])

            variable_rows.append({
                "Variabel": var_name,
                "Jumlah Item": len(valid_cols),
                "Mean": round(score.mean(), 3),
                "Std Dev": round(score.std(), 3),
                "Cronbach's Alpha": round(alpha_var, 3) if not pd.isna(alpha_var) else np.nan,
                "Kategori": kategori_mean(score.mean())
            })

    variable_summary = pd.DataFrame(variable_rows)
    st.dataframe(variable_summary, use_container_width=True)

    if not variable_summary.empty:
        st.write("Grafik rata-rata variabel:")
        fig, ax = plt.subplots()
        ax.bar(variable_summary["Variabel"], variable_summary["Mean"])
        ax.set_ylabel("Mean")
        ax.set_ylim(0, 5)
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)

else:
    st.warning(
        "Kelompok variabel belum terdeteksi. Pastikan nama kolom seperti X1.1, X2.1, X3.1, X4.1, Z1, dan Y1."
    )


st.subheader("6. Korelasi Antar Variabel")

if not variable_scores.empty and variable_scores.shape[1] >= 2:
    corr = variable_scores.corr().round(3)
    st.dataframe(corr, use_container_width=True)

    fig2, ax2 = plt.subplots()
    ax2.imshow(corr.values)
    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax2.set_yticklabels(corr.columns)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax2.text(j, i, corr.iloc[i, j], ha="center", va="center")

    st.pyplot(fig2)

else:
    st.info("Korelasi antar variabel membutuhkan minimal 2 variabel yang terdeteksi.")


st.subheader("7. Analisis Regresi: Faktor Utama Kemacetan")

if not SKLEARN_AVAILABLE:
    st.error("Library scikit-learn belum terinstall. Jalankan: python -m pip install scikit-learn")

elif variable_scores.empty:
    st.warning("Regresi belum bisa dilakukan karena skor variabel belum terbentuk.")

else:
    y_col = find_y_column(variable_scores)

    if y_col is None:
        st.warning(
            "Variabel Y/Kemacetan belum ditemukan. Pastikan kolom item kemacetan diberi nama Y1, Y2, Y3, dan seterusnya."
        )

    else:
        x_cols = [col for col in variable_scores.columns if col != y_col]

        if len(x_cols) == 0:
            st.warning("Regresi membutuhkan minimal 1 variabel X untuk memprediksi Y.")

        else:
            reg_data = variable_scores[x_cols + [y_col]].dropna()

            if len(reg_data) < 3:
                st.warning("Data responden terlalu sedikit untuk analisis regresi.")

            else:
                X = reg_data[x_cols]
                Y = reg_data[y_col]

                model = LinearRegression()
                model.fit(X, Y)

                pred = model.predict(X)
                r2 = r2_score(Y, pred)

                coef_df = pd.DataFrame({
                    "Variabel": x_cols,
                    "Koefisien Regresi": model.coef_
                })

                coef_df["Arah Pengaruh"] = coef_df["Koefisien Regresi"].apply(
                    lambda x: "Positif" if x > 0 else ("Negatif" if x < 0 else "Tidak Ada")
                )

                coef_df["Nilai Absolut"] = coef_df["Koefisien Regresi"].abs()
                coef_df = coef_df.sort_values(by="Nilai Absolut", ascending=False)

                st.write("Tabel koefisien regresi:")
                st.dataframe(coef_df.drop(columns=["Nilai Absolut"]), use_container_width=True)

                st.metric("R Square", f"{r2:.3f}")

                faktor_utama = coef_df.iloc[0]

                st.success(
                    f"Faktor utama yang paling mempengaruhi kemacetan adalah "
                    f"{faktor_utama['Variabel']} dengan koefisien "
                    f"{faktor_utama['Koefisien Regresi']:.3f}."
                )

                st.info(
                    "Catatan: Faktor utama ditentukan berdasarkan nilai absolut koefisien regresi terbesar. "
                    "Jika koefisien positif, artinya semakin tinggi variabel tersebut maka kemacetan cenderung meningkat."
                )


st.subheader("8. Download Hasil Olah Data")

regresi_export = pd.DataFrame()
if not coef_df.empty:
    regresi_export = coef_df.copy()
    if "Nilai Absolut" in regresi_export.columns:
        regresi_export = regresi_export.drop(columns=["Nilai Absolut"])

sheets = {
    "Statistik_Item": desc,
    "Frekuensi_Likert": freq,
    "Rata_Rata_Variabel": variable_summary,
    "Korelasi": corr.reset_index().rename(columns={"index": "Variabel"}) if not corr.empty else pd.DataFrame(),
    "Regresi_Faktor_Utama": regresi_export,
}

excel_bytes = make_download_excel(sheets)

st.download_button(
    label="Download Hasil Olah Data Excel",
    data=excel_bytes,
    file_name="hasil_olah_data_angket.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
