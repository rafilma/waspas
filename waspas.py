# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ReportLab
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------
# Helper: Inisialisasi session_state
# ---------------------------
def ensure_state():
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    # criteria: list of dicts: {'name': str, 'weight': float, 'impact': 'Benefit'/'Cost'}
    if "criteria" not in st.session_state:
        # default contoh
        st.session_state.criteria = [
            {"name": "C1", "weight": 1.0, "impact": "Benefit"},
            {"name": "C2", "weight": 1.0, "impact": "Benefit"},
            {"name": "C3", "weight": 1.0, "impact": "Benefit"},
        ]
    # alternatives: dict name -> list of nilai sesuai urutan criteria
    if "alternatives" not in st.session_state:
        st.session_state.alternatives = {
            "A1": [0.0 for _ in st.session_state.criteria],
            "A2": [0.0 for _ in st.session_state.criteria],
            "A3": [0.0 for _ in st.session_state.criteria],
        }

ensure_state()

# ---------------------------
# UI: Sidebar Navigation
# ---------------------------
st.set_page_config(page_title="SPK WASPAS - CRUD", layout="wide")
st.sidebar.title("Menu")
menu = st.sidebar.radio("Navigasi", ["Dashboard", "Kriteria", "Alternatif", "Bobot", "WASPAS"])

# fungsi untuk mengganti halaman
def go(page):
    st.session_state.page = page

# navigasi melalui tombol (opsional)
if menu != st.session_state.page:
    st.session_state.page = menu

# header
st.title("🔎 SPK WASPAS — Penentuan Bonus Akhir Tahun")
st.subheader("Kelompok 3 — Aplikasi CRUD Kriteria & Alternatif")

# ---------------------------
# Utility: Sync lengths when criteria changes
# ---------------------------
def sync_alternatives_length():
    """Pastikan setiap alternatif punya list nilai sesuai jumlah kriteria."""
    n = len(st.session_state.criteria)
    for name, vals in st.session_state.alternatives.items():
        if len(vals) < n:
            st.session_state.alternatives[name] = vals + [0.0] * (n - len(vals))
        elif len(vals) > n:
            st.session_state.alternatives[name] = vals[:n]

# ---------------------------
# CRUD: Kriteria
# ---------------------------
def page_kriteria():
    st.header("📌 Kelola Kriteria")

    cols = st.columns([2,1,1,1])
    cols[0].markdown("**Nama Kriteria**")
    cols[1].markdown("**Bobot (angka)**")
    cols[2].markdown("**Tipe**")
    cols[3].markdown("**Aksi**")

    # Tabel kriteria + edit inline
    for idx, crit in enumerate(st.session_state.criteria):
        c1, c2, c3, c4 = st.columns([2,1,1,1])
        name_input = c1.text_input(f"name_{idx}", value=crit["name"], key=f"name_{idx}")
        weight_input = c2.number_input(f"weight_{idx}", value=float(crit["weight"]), format="%.6f", key=f"weight_{idx}")
        impact_input = c3.selectbox(f"impact_{idx}", ["Benefit", "Cost"], index=0 if crit["impact"]=="Benefit" else 1, key=f"impact_{idx}")
        remove = c4.button("Hapus", key=f"delcrit_{idx}")
        # update
        st.session_state.criteria[idx]["name"] = name_input.strip() if name_input.strip()!="" else f"C{idx+1}"
        st.session_state.criteria[idx]["weight"] = float(weight_input)
        st.session_state.criteria[idx]["impact"] = impact_input

        if remove:
            del st.session_state.criteria[idx]
            sync_alternatives_length()
            st.experimental_rerun()

    st.markdown("---")
    st.subheader("Tambah Kriteria Baru")
    with st.form("form_add_crit", clear_on_submit=True):
        new_name = st.text_input("Nama Kriteria", value=f"C{len(st.session_state.criteria)+1}")
        new_weight = st.number_input("Bobot", value=1.0)
        new_impact = st.selectbox("Tipe", ["Benefit", "Cost"])
        submitted = st.form_submit_button("Tambah Kriteria")
        if submitted:
            st.session_state.criteria.append({"name": new_name.strip() if new_name.strip()!="" else f"C{len(st.session_state.criteria)+1}",
                                              "weight": float(new_weight),
                                              "impact": new_impact})
            sync_alternatives_length()
            st.success("Kriteria berhasil ditambahkan.")
            st.experimental_rerun()

    st.markdown("---")
    st.write("Preview kriteria saat ini:")
    dfc = pd.DataFrame(st.session_state.criteria)
    st.dataframe(dfc)

# ---------------------------
# CRUD: Alternatif
# ---------------------------
def page_alternatif():
    st.header("👥 Kelola Alternatif")
    # list alternatif
    alt_names = list(st.session_state.alternatives.keys())

    st.subheader("Alternatif Eksisting")
    if len(alt_names) == 0:
        st.info("Belum ada alternatif. Tambahkan di bagian 'Tambah Alternatif'.")
    else:
        for idx, name in enumerate(alt_names):
            st.markdown(f"**{name}**")
            cols = st.columns(len(st.session_state.criteria) + 2)
            # header
            for j, crit in enumerate(st.session_state.criteria):
                cols[j].markdown(f"**{crit['name']}**")
            cols[-2].markdown("**Aksi**")
            cols[-1].markdown("**Hapus**")
            # values row (editable)
            values = st.session_state.alternatives[name]
            # Use number_input for each criteria value
            updated_vals = []
            for j, crit in enumerate(st.session_state.criteria):
                v = cols[j].number_input(f"{name}_{j}", value=float(values[j]) if j < len(values) else 0.0, key=f"{name}_val_{j}")
                updated_vals.append(float(v))
            # action buttons
            if cols[-2].button("Edit Nama", key=f"editname_{name}"):
                new_name = st.text_input(f"Nama baru untuk {name}", value=name, key=f"newname_{name}")
                if st.button("Simpan nama", key=f"savenew_{name}"):
                    # rename
                    vals = st.session_state.alternatives.pop(name)
                    st.session_state.alternatives[new_name] = vals
                    st.success("Nama alternatif diubah.")
                    st.experimental_rerun()
            if cols[-1].button("Hapus Alternatif", key=f"delalt_{name}"):
                st.session_state.alternatives.pop(name, None)
                st.success(f"Alternatif {name} dihapus.")
                st.experimental_rerun()
            # simpan perubahan nilai otomatis
            st.session_state.alternatives[name] = updated_vals

    st.markdown("---")
    st.subheader("Tambah Alternatif Baru")
    with st.form("form_add_alt", clear_on_submit=True):
        alt_name = st.text_input("Nama Alternatif", value=f"A{len(st.session_state.alternatives)+1}")
        # initial values for new alternatif (based on criteria)
        initial = []
        for crit in st.session_state.criteria:
            initial.append(st.number_input(f"Nilai untuk {crit['name']}", value=0.0, key=f"newalt_val_{crit['name']}"))
        add = st.form_submit_button("Tambah Alternatif")
        if add:
            st.session_state.alternatives[alt_name] = initial
            st.success("Alternatif ditambahkan.")
            st.experimental_rerun()

    st.markdown("---")
    st.write("Preview matriks keputusan:")
    df = pd.DataFrame(st.session_state.alternatives, index=[c["name"] for c in st.session_state.criteria]).T
    st.dataframe(df)

# ---------------------------
# Halaman Bobot (lihat + normalisasi)
# ---------------------------
def page_bobot():
    st.header("⚖️ Pemeriksaan & Normalisasi Bobot")
    st.write("Daftar bobot kriteria saat ini. Anda dapat melakukan normalisasi agar jumlah bobot = 1.")

    df = pd.DataFrame(st.session_state.criteria)
    st.dataframe(df[["name","weight","impact"]].rename(columns={"name":"Kriteria","weight":"Bobot","impact":"Tipe"}))

    if st.button("Normalisasi Bobot (sum -> 1)"):
        weights = [c["weight"] for c in st.session_state.criteria]
        total = sum(weights)
        if total == 0:
            st.warning("Total bobot = 0, tidak bisa dinormalisasi.")
        else:
            for i in range(len(st.session_state.criteria)):
                st.session_state.criteria[i]["weight"] = st.session_state.criteria[i]["weight"] / total
            st.success("Bobot berhasil dinormalisasi (jumlah = 1).")
            st.experimental_rerun()

    st.markdown("---")
    st.write("Jika ingin mengubah bobot satu-per-satu, kembali ke halaman Kriteria dan edit nilai bobot di sana.")

# ---------------------------
# Fungsi Perhitungan WASPAS (sama seperti sebelumnya)
# ---------------------------
def compute_waspas_df():
    # konversi ke DataFrame keputusan:
    criteria = st.session_state.criteria
    alts = st.session_state.alternatives
    crit_names = [c["name"] for c in criteria]
    if len(crit_names) == 0 or len(alts) == 0:
        return None

    df = pd.DataFrame(alts, index=list(alts.keys()))
    df.columns = crit_names
    # bobot & impacts
    weights = [c["weight"] for c in criteria]
    impacts = [c["impact"] for c in criteria]
    # lamb fixed
    lamb = 0.5

    # pastikan numeric
    X = df.copy().astype(float).values
    m, n = X.shape
    w = np.array(weights, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        # normalisasi sementara untuk perhitungan (tidak menyimpan)
        w = w / w.sum() if w.sum() != 0 else np.array([1.0/n]*n)
    # R matrix
    R = np.zeros_like(X)
    for j in range(n):
        col = X[:, j]
        if impacts[j] == "Benefit":
            denom = col.max()
            R[:, j] = col / denom if denom != 0 else 0
        else:
            denom = col.min()
            with np.errstate(divide='ignore', invalid='ignore'):
                R[:, j] = denom / col
            R[:, j] = np.nan_to_num(R[:, j], nan=0.0, posinf=0.0, neginf=0.0)
    # Q1
    Q1 = R.dot(w)
    # Q2
    Q2 = np.ones(m)
    for i in range(m):
        prod = 1.0
        for j in range(n):
            r = R[i, j]
            # jika r == 0 dan w[j] > 0 -> produk 0
            if r == 0 and w[j] > 0:
                prod = 0.0
                break
            prod *= r ** w[j]
        Q2[i] = prod
    Q = lamb * Q1 + (1 - lamb) * Q2

    res = pd.DataFrame({
        "Q1_WSM": Q1,
        "Q2_WPM": Q2,
        "Q_WASPAS": Q
    }, index=df.index)
    res["Rank"] = res["Q_WASPAS"].rank(ascending=False, method="min").astype(int)

    # Bonus mapping
    def bonus_from_q(q):
        # Rules given: 90-100% -> 100%, 80-89% -> 80%, 60-79% -> 60%, <59% -> 50%
        # Note Q in [0,1], map accordingly
        if q >= 0.90:
            return "100%"
        elif q >= 0.80:
            return "80%"
        elif q >= 0.60:
            return "60%"
        else:
            return "50%"
    res["Bonus"] = res["Q_WASPAS"].apply(bonus_from_q)
    return df, res

# ---------------------------
# Fungsi Generate PDF (ReportLab)
# ---------------------------
def generate_pdf(df_decision, df_result):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20,leftMargin=20, topMargin=30,bottomMargin=18)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("<b>Laporan Hasil SPK - Metode WASPAS</b>", styles["Title"]))
    elements.append(Spacer(1,12))
    elements.append(Paragraph("Laporan ini berisi hasil perhitungan WASPAS dan penentuan bonus berdasarkan Q_WASPAS.", styles["Normal"]))
    elements.append(Spacer(1,12))

    # Matriks keputusan (df_decision)
    elements.append(Paragraph("<b>Matriks Keputusan (Alternatif x Kriteria)</b>", styles["Heading3"]))
    elements.append(Spacer(1,6))
    # build table data
    table_dec = [ ["Alternatif"] + list(df_decision.columns) ]
    for idx, row in df_decision.iterrows():
        table_dec.append([idx] + [str(v) for v in row.values])
    t = Table(table_dec, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                           ("GRID",(0,0),(-1,-1),0.5,colors.black)]))
    elements.append(t)
    elements.append(Spacer(1,12))

    # Hasil perhitungan
    elements.append(Paragraph("<b>Hasil Perhitungan WASPAS</b>", styles["Heading3"]))
    elements.append(Spacer(1,6))
    table_res = [ ["Alternatif"] + list(df_result.columns) ]
    for idx, row in df_result.iterrows():
        # round numeric values for readability
        row_vals = []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                row_vals.append(f"{v:.4f}")
            else:
                row_vals.append(str(v))
        table_res.append([idx] + row_vals)
    tr = Table(table_res, repeatRows=1)
    tr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
                            ("GRID",(0,0),(-1,-1),0.5,colors.black)]))
    elements.append(tr)
    elements.append(Spacer(1,12))

    # Pemenang (Rank 1)
    winners = df_result[df_result["Rank"]==1].index.tolist()
    winners_text = "<br/>".join([f"- {w} (Bonus: {df_result.loc[w,'Bonus']})" for w in winners])
    elements.append(Paragraph("<b>Pemenang (Rank 1)</b>", styles["Heading3"]))
    elements.append(Spacer(1,6))
    elements.append(Paragraph(winners_text or "Tidak ada pemenang.", styles["Normal"]))
    elements.append(Spacer(1,12))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ---------------------------
# Halaman WASPAS
# ---------------------------
def page_waspas():
    st.header("⚖️ Perhitungan WASPAS & Hasil")
    sync_alternatives_length()

    if len(st.session_state.criteria) == 0:
        st.warning("Belum ada kriteria. Tambahkan kriteria terlebih dahulu.")
        return
    if len(st.session_state.alternatives) == 0:
        st.warning("Belum ada alternatif. Tambahkan alternatif terlebih dahulu.")
        return

    df_decision, df_result = compute_waspas_df()

    st.subheader("Matriks Keputusan")
    st.dataframe(df_decision)

    st.subheader("Hasil Perhitungan")
    st.dataframe(df_result)

    st.info("Nilai λ (Lambda) digunakan = 0.50 (fixed).")

    # tombol download PDF
    pdf = generate_pdf(df_decision, df_result)
    st.download_button("📄 Unduh Laporan PDF", data=pdf, file_name="laporan_waspas.pdf", mime="application/pdf")

    # highlight winners
    winners = df_result[df_result["Rank"]==1].index.tolist()
    if len(winners) == 1:
        st.success(f"Pemenang (Rank 1): {winners[0]}  —  Bonus: {df_result.loc[winners[0],'Bonus']}")
    else:
        st.success("Pemenang (Rank 1) (lebih dari satu):")
        for w in winners:
            st.write(f"- {w}  —  Bonus: {df_result.loc[w,'Bonus']}")

# ---------------------------
# Dashboard (ringkasan)
# ---------------------------
def page_dashboard():
    st.header("Dashboard")
    st.markdown("Ringkasan data saat ini:")
    st.write(f"- Jumlah Kriteria: {len(st.session_state.criteria)}")
    st.write(f"- Jumlah Alternatif: {len(st.session_state.alternatives)}")
    st.markdown("Kriteria:")
    st.dataframe(pd.DataFrame(st.session_state.criteria))
    st.markdown("Contoh matriks keputusan:")
    if len(st.session_state.alternatives)>0:
        df = pd.DataFrame(st.session_state.alternatives, index=[c["name"] for c in st.session_state.criteria]).T
        st.dataframe(df)
    else:
        st.write("Belum ada alternatif.")

# ---------------------------
# Router
# ---------------------------
page = st.session_state.page

if page == "Dashboard":
    page_dashboard()
elif page == "Kriteria":
    page_kriteria()
elif page == "Alternatif":
    page_alternatif()
elif page == "Bobot":
    page_bobot()
elif page == "WASPAS":
    page_waspas()
else:
    st.write("Halaman tidak ditemukan.")
