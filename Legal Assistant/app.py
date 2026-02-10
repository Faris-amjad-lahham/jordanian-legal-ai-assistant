import os
import streamlit as st

from Legal_QA import legal_qa_chatbot
from contract_backend import (
    ContractStore,
    generate_contract,
    update_section,
    export_to_word,
)

# =====================
# Page Config
# =====================
st.set_page_config(page_title="Jordanian Legal Assistant", layout="wide")

st.title("🇯🇴 Jordanian Legal Assistant")

# =====================
# Session State
# =====================
if "contract_store" not in st.session_state:
    st.session_state.contract_store = ContractStore()

if "last_export_path" not in st.session_state:
    st.session_state.last_export_path = None

if "contract_intro_shown" not in st.session_state:
    st.session_state.contract_intro_shown = False

# =====================
# Mode Selection
# =====================
mode = st.radio(
    "Choose mode:",
    ["Legal QA (RAG)", "Contract Drafting"],
    horizontal=True
)

# ============================================================
# LEGAL QA (RAG)
# ============================================================
if mode == "Legal QA (RAG)":
    st.subheader("Ask about the Jordanian Labour Law (RAG)")

    question = st.text_input("Your question (Arabic or English):")

    # 🔒 SAFE ROUTE: Top-K is hidden and fixed
    TOP_K = 3

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            res = legal_qa_chatbot(
                question,
                top_k=TOP_K,
                return_metadata=True
            )

            st.markdown("### Answer")
            st.write(res["answer"])

            with st.expander("Show Retrieval Details"):
                st.write("Dataset path:", res.get("dataset_path"))
                st.write("Retrieved article IDs:", res.get("retrieved_article_ids", []))
                st.text((res.get("context") or "")[:5000])

# ============================================================
# CONTRACT DRAFTING (SAFE CHAT-STYLE UI)
# ============================================================
else:
    st.subheader("Contract Drafting (Non-binding)")

    store: ContractStore = st.session_state.contract_store

    col1, col2 = st.columns(2, gap="large")

    # =====================
    # LEFT SIDE — CHAT UI
    # =====================
    with col1:
        st.markdown("### 💬 Contract Assistant")

        # Intro message (shown once)
        if not st.session_state.contract_intro_shown:
            st.info(
                """
👋 **Welcome! I am your Jordanian Employment Contract Assistant.**

I can help you generate a **non-binding employment contract** for informational purposes only.

To create a contract, please provide **ALL** of the following details in one message:

- Employer name  
- Employee name  
- Job title  
- Work location  
- Contract type (Limited / Unlimited)  
- Start date  
- End date (required for Limited contracts)  
- Salary (JOD)  
- Working hours per day  
- Working days per week  
- Termination terms  

⚠️ This system does **not** provide legal advice.
"""
            )
            st.session_state.contract_intro_shown = True

        # Chat-like input (SAFE: no parsing yet)
        user_message = st.text_area(
            "Your message:",
            height=220,
            placeholder="Type all contract details here..."
        )

        st.caption(
            "ℹ️ For now, please also fill the form on the right to generate the contract."
        )

    # =====================
    # RIGHT SIDE — EXISTING FORM + RESULTS
    # =====================
    with col2:
        st.markdown("### 📝 Create / Regenerate Contract")

        contract_type = st.selectbox("Contract type", ["Limited", "Unlimited"])
        employer = st.text_input("Employer name")
        employee = st.text_input("Employee name")
        job_title = st.text_input("Job title")
        location = st.text_input("Work location")
        start_date = st.text_input("Start date (e.g., 01/03/2026)")
        end_date = st.text_input("End date (for Limited)")
        salary = st.text_input("Salary (JOD)")
        hours = st.text_input("Working hours per day")
        days = st.text_input("Working days per week")
        termination = st.text_area("Termination terms", height=100)

        if st.button("Generate Contract"):
            data = {
                "date": "Today",
                "contract_type": contract_type,
                "employer": employer,
                "employee": employee,
                "job_title": job_title,
                "location": location,
                "start_date": start_date,
                "end_date": end_date,
                "salary": salary,
                "hours": hours,
                "days": days,
                "termination": termination,
            }
            generate_contract(store, data)
            st.success("Contract generated and saved.")

        st.divider()

        st.markdown("### 📄 Current Contract")
        if store.current:
            st.text_area("Contract Preview", store.current, height=300)
        else:
            st.info("No contract generated yet.")

        st.markdown("### ✏️ Update a Section")
        section = st.selectbox(
            "Select section to update",
            ["salary", "location", "job_title", "termination"]
        )
        new_val = st.text_area("New value", height=80)

        if st.button("Apply Update"):
            result = update_section(store, section, new_val)
            if result.startswith("❌"):
                st.error(result)
            else:
                st.success("Section updated and previous version saved.")

        st.markdown("### 🕒 Previous Versions (Last 5)")
        if store.versions:
            idx = st.selectbox(
                "Choose a previous version",
                range(len(store.versions)),
                format_func=lambda i: f"Version {i+1} — {store.versions[i].created_at}"
            )
            st.text_area(
                "Previous Contract",
                store.versions[idx].text,
                height=200
            )
        else:
            st.caption("No previous versions yet.")

        st.markdown("### ⬇️ Export to Word (.docx)")
        if st.button("Export Current Contract"):
            path = export_to_word(store.current)
            st.session_state.last_export_path = path
            if path.startswith("❌"):
                st.error(path)
            else:
                st.success(f"Saved: {path}")

        if (
            st.session_state.last_export_path
            and os.path.isfile(st.session_state.last_export_path)
        ):
            with open(st.session_state.last_export_path, "rb") as f:
                st.download_button(
                    "Download last exported contract",
                    data=f,
                    file_name=os.path.basename(st.session_state.last_export_path),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
