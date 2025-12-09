# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# === ReportLab untuk PDF ===
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="SPK WASPAS", layout="wide")
st.title("🔎 Sistem Pendukung Keputusan Penentuan Bonus Akhir Tahun Outward Bound Indonesia - Metode WASPAS")

# ============================================================
# Fungsi WASPAS
# ============================================================
def waspas(df_values, weights, impacts, lamb=0.5):
    X = df_values.copy().astype(float)
    m, n = X.shape

    # Normalisasi bobot
    w = np.array(weights, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        w = w / w.sum()

    # Normalisasi
    R = np.zeros_like(X.values)
    for j in range(n):
        colvals = X.iloc[:, j].values.astype(float)

        if impacts[j] == "Benefit":
            denom = colvals.max()
            R[:, j] = colvals / denom if denom != 0 else 0
        else:
            denom = colvals.min()
            R[:, j] = denom / colvals
            R[:, j] = np.nan_to_num(R[:, j])

    # WSM
    Q1 = R.dot(w)

    # WPM
    Q2 = np.ones(m)
    for i in range(m):
        for j in range(n):
            Q2[i] *= R[i, j] ** w[j]

    # WASPAS
    Q = lamb * Q1 + (1 - lamb) * Q2

    result = pd.DataFrame({
        'Q1_WSM': Q1,
        'Q2_WPM': Q2,
        'Q_WASPAS': Q
    }, index=df_values.index)

    result["Rank"] = result["Q_WASPAS"].rank(ascending=False, method="min").astype(int)

    return result.sort_values("Rank")


# ============================================================
# Fungsi Generate PDF menggunakan ReportLab (platypus)
# ============================================================
def generate_pdf(df_result, winners):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    title = Paragraph("<b>Hasil Perhitungan SPK Metode WASPAS</b>", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Pemenang
    winner_text = "<b>Pemenang (Rank 1):</b><br/>" + "<br/>".join([f"- {w}" for w in winners])
    elements.append(Paragraph(winner_text, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Tabel hasil
    table_data = [ ["Alternatif"] + df_result.columns.tolist() ]
    for idx, row in df_result.iterrows():
        table_data.append([idx] + list(row.values))

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("GRID", (0,0), (-1,-1), 0.8, colors.black),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
    ]))

    elements.append(table)
    doc.build(elements)
    pdf_value = buffer.getvalue()
    buffer.close()
    return pdf_value


# ============================================================
# Input Jumlah Kriteria & Alternatif
# ============================================================
st.subheader("⚙️ Input Parameter SPK")

col1, col2 = st.columns(2)
with col1:
    jml_kriteria = st.number_input("Jumlah Kriteria", min_value=1, max_value=20, value=3)
with col2:
    jml_alternatif = st.number_input("Jumlah Alternatif", min_value=1, max_value=50, value=3)


# ============================================================
# Input Kriteria
# ============================================================
st.subheader("📌 Input Kriteria")

kriteria_names = []
weights = []
impacts = []

for i in range(jml_kriteria):
    c1, c2, c3 = st.columns(3)

    with c1:
        nama = st.text_input(f"Nama Kriteria {i+1}", value=f"C{i+1}")
    with c2:
        bobot = st.number_input(f"Bobot K{i+1}", min_value=0.0, value=1.0)
    with c3:
        tipe = st.selectbox(f"Tipe K{i+1}", ["Benefit", "Cost"])

    kriteria_names.append(nama)
    weights.append(bobot)
    impacts.append(tipe)


# ============================================================
# Input Alternatif & Nilai Matriks
# ============================================================
st.subheader("📋 Input Alternatif & Nilai Matriks Keputusan")

data = {}
alt_names = []

for a in range(jml_alternatif):
    st.markdown(f"### ➤ Alternatif {a+1}")
    alt_name = st.text_input(f"Nama Alternatif {a+1}", value=f"A{a+1}")
    alt_names.append(alt_name)

    nilai_alt = []
    c_cols = st.columns(jml_kriteria)

    for k in range(jml_kriteria):
        nilai = c_cols[k].number_input(f"{kriteria_names[k]} ({alt_name})", value=0.0)
        nilai_alt.append(nilai)

    data[alt_name] = nilai_alt

# Tampilkan matriks
df = pd.DataFrame(data, index=kriteria_names).T
st.write("### 📊 Matriks Keputusan")
st.dataframe(df)


# ============================================================
# Input λ
# ============================================================
lamb = st.slider("Nilai λ (Lambda)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)


# ============================================================
# Tombol Hitung WASPAS
# ============================================================
if st.button("🚀 Hitung WASPAS"):
    result = waspas(df, weights, impacts, lamb)

    st.subheader("🏆 Hasil Perhitungan WASPAS")
    st.dataframe(result)

    # Ambil pemenang (rank 1)
    winners = result[result["Rank"] == 1].index.tolist()

    if len(winners) == 1:
        st.success(f"Karyawan yang berhak mendapakatkan bonus adalah **{winners[0]}**.")
    else:
        st.success("🔥 Terdapat lebih dari satu alternatif terbaik (Rank 1):")
        for w in winners:
            st.write(f"- **{w}**")

    # === Generate PDF ===
    pdf_file = generate_pdf(result, winners)

    st.download_button(
        label="📄 Download Hasil dalam PDF",
        data=pdf_file,
        file_name="hasil_waspas.pdf",
        mime="application/pdf"
    )
