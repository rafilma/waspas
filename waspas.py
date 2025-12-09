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
st.subheader("Kelompok 3")

# ============================================================
# Fungsi Penghitungan WASPAS
# ============================================================
def waspas(df_values, weights, impacts, lamb=0.5):
    X = df_values.copy().astype(float)
    m, n = X.shape

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

    # WASPAS final
    Q = lamb * Q1 + (1 - lamb) * Q2

    result = pd.DataFrame({
        'Q1_WSM': Q1,
        'Q2_WPM': Q2,
        'Q_WASPAS': Q
    }, index=df_values.index)

    result["Rank"] = result["Q_WASPAS"].rank(ascending=False, method="min").astype(int)

    # Hitung Bonus
    bonus_list = []
    for val in result["Q_WASPAS"]:
        if val >= 0.90:
            bonus_list.append("100%")
        elif val >= 0.80:
            bonus_list.append("80%")
        elif val >= 0.60:
            bonus_list.append("60%")
        else:
            bonus_list.append("50%")

    result["Bonus"] = bonus_list

    return result.sort_values("Rank")

# ============================================================
# Fungsi Generate PDF
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
    winner_text = "<b>Karyawan dengan nilai terbaik (Bonus 100%):</b><br/>" + "<br/>".join([f"- {w}" for w in winners])
    elements.append(Paragraph(winner_text, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Tabel hasil
    table_data = [["Alternatif"] + df_result.columns.tolist()]
    for idx, row in df_result.iterrows():
        table_data.append([idx] + list(row.values))

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
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
# SIDEBAR INPUT
# ============================================================
st.sidebar.header("⚙️ Pengaturan SPK WASPAS")

# ─────────────────────────────────────────────
# Input jumlah kriteria & alternatif
# ─────────────────────────────────────────────
jml_kriteria = st.sidebar.number_input("Jumlah Kriteria", min_value=1, max_value=20, value=3)
jml_alternatif = st.sidebar.number_input("Jumlah Alternatif", min_value=1, max_value=50, value=3)

# ─────────────────────────────────────────────
# Input Kriteria
# ─────────────────────────────────────────────
st.sidebar.subheader("📌 Input Kriteria")

kriteria_names = []
weights = []
impacts = []

for i in range(jml_kriteria):
    st.sidebar.markdown(f"**Kriteria {i+1}**")
    nama = st.sidebar.text_input(f"Nama Kriteria {i+1}", value=f"C{i+1}")
    bobot = st.sidebar.number_input(f"Bobot K{i+1}", value=1.0)
    tipe = st.sidebar.selectbox(f"Tipe K{i+1}", ["Benefit", "Cost"])

    kriteria_names.append(nama)
    weights.append(bobot)
    impacts.append(tipe)

# ─────────────────────────────────────────────
# Input Alternatif & Nilai Matrix
# ─────────────────────────────────────────────
st.sidebar.subheader("📋 Input Alternatif")

data = {}
alt_names = []

for a in range(jml_alternatif):
    st.sidebar.markdown(f"**Alternatif {a+1}**")
    alt_name = st.sidebar.text_input(f"Nama Alternatif {a+1}", value=f"A{a+1}")
    alt_names.append(alt_name)

    nilai_alt = []
    for k in range(jml_kriteria):
        nilai = st.sidebar.number_input(f"{kriteria_names[k]} ({alt_name})", value=0.0)
        nilai_alt.append(nilai)

    data[alt_name] = nilai_alt

# Membuat DataFrame
df = pd.DataFrame(data, index=kriteria_names).T

st.write("### 📊 Matriks Keputusan")
st.dataframe(df)

# Lambda fixed
lamb = 0.5

# ============================================================
# TOMBOL HITUNG
# ============================================================
if st.button("🚀 Hitung WASPAS"):
    result = waspas(df, weights, impacts, lamb)

    st.subheader("🏆 Hasil Perhitungan WASPAS")
    st.dataframe(result)

    winners = result[result["Rank"] == 1].index.tolist()

    st.success("Perhitungan selesai! Bonus ditentukan berdasarkan nilai WASPAS.")

    # PDF
    pdf_file = generate_pdf(result, winners)

    st.download_button(
        label="📄 Download PDF",
        data=pdf_file,
        file_name="hasil_waspas.pdf",
        mime="application/pdf"
    )
