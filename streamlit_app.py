import streamlit as st
import requests
import io
import fitz  # PyMuPDF
import pickle
import os

API_BASE = "http://127.0.0.1:8000"
VECTOR_PATH = "vector.index"
META_PATH = "metadata.pkl"

st.set_page_config(page_title="Invoice Reimbursement System", layout="centered")
st.title("🧾 Invoice Reimbursement RAG System")

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Upload & Analyze", "Chatbot Query"])

# --- Section 1: Upload and Analyze ---
if section == "Upload & Analyze":
    st.header("📤 Upload HR Policy and Invoices")
    policy_file = st.file_uploader("Upload HR Policy PDF", type=["pdf"])
    invoice_zip = st.file_uploader("Upload ZIP of Invoice PDFs", type=["zip"])

    if st.button("Analyze Invoices"):
        if policy_file and invoice_zip:
            try:
                policy_file.seek(0)
                policy_bytes = policy_file.read()

                with open("debug_policy_from_streamlit.pdf", "wb") as f:
                    f.write(policy_bytes)

                pdf_text = ""
                with fitz.open(stream=policy_bytes, filetype="pdf") as doc:
                    for page in doc:
                        pdf_text += page.get_text()

                files = {
                    "invoice_zip": (invoice_zip.name, invoice_zip, "application/zip")
                }
                data = {
                    "policy_text": pdf_text
                }

                with st.spinner("Analyzing invoices..."):
                    response = requests.post(f"{API_BASE}/analyze/", files=files, data=data)
                    if response.ok:
                        st.success("✅ Invoices analyzed successfully!")
                    else:
                        st.error(f"❌ Analysis failed: {response.text}")
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
        else:
            st.warning("Please upload both the HR policy and the invoice ZIP.")

# --- Section 2: Chatbot Query ---
elif section == "Chatbot Query":
    st.header("💬 Ask About Invoices")
    query = st.text_area("Your question", placeholder="e.g., Why was invoice IN-102 declined?")

    try:
        with open(META_PATH, "rb") as f:
            metas = pickle.load(f)
    except:
        metas = []

    # Extract unique filter values
    unique_employees = sorted({m.get("employee_name", "") for m in metas if m.get("employee_name")})
    unique_dates = sorted({m.get("date", "") for m in metas if m.get("date")})
    status_options = ["Fully Reimbursed", "Partially Reimbursed", "Declined"]

    with st.expander("🔎 Optional Filters"):
        employee_name = st.selectbox("Filter by Employee Name", [""] + unique_employees, key="filter_name")
        status = st.selectbox("Filter by Status", [""] + status_options)
        date = st.selectbox("Filter by Date (YYYY-MM-DD)", [""] + unique_dates)

    with st.expander("📋 Indexed Metadata (Debug View)"):
        st.json(metas)

    filters = {}
    if employee_name:
        filters["employee_name"] = employee_name
    if status:
        filters["status"] = status
    if date:
        filters["date"] = date

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Ask Now"):
            if query.strip():
                payload = {"query": query}
                if filters:
                    payload["filters"] = filters

                with st.spinner("Generating response..."):
                    res = requests.post(f"{API_BASE}/chatbot/", json=payload)
                    if res.ok:
                        st.markdown(res.json()["response"])
                    else:
                        st.error("⚠️ Error fetching response from backend.")
            else:
                st.warning("Query cannot be empty.")

    # ✅ Reset vector store via backend
    with col2:
        if st.button("🗑️ Reset Vector Store"):
            try:
                res = requests.post(f"{API_BASE}/reset/")
                if res.ok:
                    st.success("🧹 Vector store fully cleared via backend.")
                else:
                    st.error("❌ Failed to reset vector store from backend.")
            except Exception as e:
                st.error(f"❌ Exception: {e}")
