"""
app.py — Jordanian Legal AI Assistant
Full-featured Streamlit application with:
  • Hybrid semantic + BM25 Legal Q&A (Arabic & English)
  • Arabic RTL rendering
  • Empty-state onboarding with sample chips
  • Citation deep-dive panel (full article text)
  • "Know Your Rights" quick-reference sidebar cards
  • Conversation PDF export
  • Session-based rate limiting
  • Contract drafting with validation (Create / Edit / Preview / Analyze)
  • Contract document upload & compliance analysis
  • Load extracted fields into contract editor
  • st.secrets + env var API key support
"""

import io
import os

import streamlit as st

from Legal_QA import legal_qa_chatbot_stream, get_article_by_id
from contract_backend import (
    ContractStore,
    generate_contract_ai,
    chat_edit_contract,
    export_to_word,
    get_validation_summary,
    validate_contract_data,
    analyze_uploaded_document,
    extract_fields_from_text,
)
from utils import RateLimiter, export_conversation_pdf

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jordanian Legal Assistant",
    page_icon="🇯🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# API KEY — st.secrets (cloud) → env var (local)
# ─────────────────────────────────────────────────────────────────────────────
def _get_api_key() -> str:
    try:
        return st.secrets["OPENAI_API_KEY"]  # type: ignore
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


_api_key = _get_api_key()
if _api_key:
    os.environ["OPENAI_API_KEY"] = _api_key


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ──────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp { background-color: #0f1117; font-family: 'Inter', sans-serif; }

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #1e2533;
}

/* ── Hide Streamlit chrome ─────────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Confidence badges ─────────────────────────────────────────────────── */
.badge-high {
    background: #1a4731; color: #4ade80;
    padding: 3px 10px; border-radius: 12px; font-size: 12px;
}
.badge-low {
    background: #4a1f1f; color: #f87171;
    padding: 3px 10px; border-radius: 12px; font-size: 12px;
}

/* ── Section headers ───────────────────────────────────────────────────── */
.section-title {
    font-size: 11px; font-weight: 600; color: #718096;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;
}

/* ── Validation boxes ──────────────────────────────────────────────────── */
.val-error   { background:#2d1515; border-left:3px solid #e53e3e; padding:8px 12px; margin:4px 0; border-radius:4px; font-size:13px; color:#fc8181; }
.val-warning { background:#2d2415; border-left:3px solid #d69e2e; padding:8px 12px; margin:4px 0; border-radius:4px; font-size:13px; color:#f6c90e; }
.val-info    { background:#152d2d; border-left:3px solid #319795; padding:8px 12px; margin:4px 0; border-radius:4px; font-size:13px; color:#81e6d9; }

/* ── Score box (sidebar) ───────────────────────────────────────────────── */
.score-box   { text-align:center; padding:16px; background:#1a1f2e; border-radius:12px; border:1px solid #2d3748; }
.score-num   { font-size:42px; font-weight:800; }
.score-label { font-size:12px; color:#718096; margin-top:4px; }

/* ── Onboarding panel ──────────────────────────────────────────────────── */
.onboard-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1421 100%);
    border: 1px solid #2d3748; border-radius: 16px;
    padding: 32px; text-align: center; margin: 24px 0;
}
.onboard-title { font-size: 26px; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
.onboard-sub   { font-size: 14px; color: #718096; margin-bottom: 24px; }

/* ── Know Your Rights cards ────────────────────────────────────────────── */
.right-card {
    background: #1a1f2e; border: 1px solid #2d3748;
    border-radius: 8px; padding: 8px 12px; margin: 4px 0; font-size: 13px;
}
.right-key   { color: #718096; }
.right-value { color: #e2e8f0; font-weight: 600; }

/* ── Analysis results ──────────────────────────────────────────────────── */
.analysis-field-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.analysis-field-table th {
    background: #1a1f2e; color: #718096; padding: 6px 10px;
    text-align: left; font-weight: 600; border-bottom: 1px solid #2d3748;
}
.analysis-field-table td { padding: 6px 10px; color: #e2e8f0; border-bottom: 1px solid #1e2533; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "contract_store": ContractStore(),
        "last_export_path": None,
        "qa_messages": [],
        "qa_lang_override": None,
        "contract_messages": [],
        # Document upload
        "extracted_contract_fields": None,
        "analysis_result": None,
        # Pending sample chip click
        "_pending_question": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _extract_text_from_upload(uploaded_file) -> str:
    """Extract plain text from a PDF or DOCX uploaded file object."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            import pdfplumber  # type: ignore
            with pdfplumber.open(uploaded_file) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        elif name.endswith(".docx"):
            from docx import Document  # type: ignore
            doc = Document(io.BytesIO(uploaded_file.read()))
            return "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc:
        return f"Error reading file: {exc}"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/c/c0/Flag_of_Jordan.svg",
        width=56,
    )
    st.markdown("## 🇯🇴 Legal Assistant")
    st.markdown("*Jordanian Labour Law No. 8 / 1996*")
    st.divider()

    mode = st.radio(
        "Mode",
        ["⚖️ Legal Q&A", "📄 Contract Drafting"],
        label_visibility="collapsed",
    )
    st.divider()

    # API key status
    if _api_key:
        st.success("✅ API Key connected")
    else:
        st.error("❌ OPENAI_API_KEY not set")
        st.caption("Run: `setx OPENAI_API_KEY sk-...` then restart VS Code")

    st.divider()

    # ── Mode-specific sidebar content ────────────────────────────────────────
    if mode == "⚖️ Legal Q&A":
        st.markdown('<p class="section-title">Q&A Settings</p>', unsafe_allow_html=True)

        lang_choice = st.selectbox(
            "Response Language",
            ["Auto-detect", "English", "Arabic"],
            help="Auto-detect matches your question language",
        )
        st.session_state.qa_lang_override = (
            None if lang_choice == "Auto-detect"
            else "en" if lang_choice == "English"
            else "ar"
        )
        top_k = st.slider("Sources retrieved", 1, 6, 3)

        st.divider()

        # ── Know Your Rights quick cards ─────────────────────────────────────
        with st.expander("📋 Know Your Rights", expanded=False):
            rights = [
                ("💰 Min Wage", "260 JOD/month (2023)"),
                ("📅 Annual Leave", "14 days/yr → 21 after 5 years"),
                ("⏰ Max Hours", "8h/day · 48h/week"),
                ("📋 Notice Period", "30 days minimum"),
                ("🧪 Max Probation", "90 days (3 months)"),
                ("👶 Maternity Leave", "10 weeks (fully paid)"),
                ("👨 Paternity Leave", "3 days (paid)"),
                ("💼 Overtime Pay", "125% ordinary wage"),
                ("🗓️ Holiday Pay", "150% ordinary wage"),
                ("📊 End-of-Service", "1 month salary per year"),
            ]
            for key, val in rights:
                st.markdown(
                    f'<div class="right-card"><span class="right-key">{key}: </span>'
                    f'<span class="right-value">{val}</span></div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        RateLimiter.sidebar_meter("qa")

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.qa_messages = []
            st.rerun()

    else:  # Contract Drafting
        st.markdown('<p class="section-title">Contract Tools</p>', unsafe_allow_html=True)

        store: ContractStore = st.session_state.contract_store

        # Completeness score
        score = store.completeness_score()
        color = "#4ade80" if score >= 80 else "#f6c90e" if score >= 50 else "#f87171"
        st.markdown(
            f"""
            <div class="score-box">
                <div class="score-num" style="color:{color}">{score}%</div>
                <div class="score-label">Contract Completeness</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")

        if store.versions:
            st.markdown('<p class="section-title">Version History</p>', unsafe_allow_html=True)
            ver_idx = st.selectbox(
                "Restore a version",
                range(len(store.versions)),
                format_func=lambda i: (
                    f"v{i+1} — {store.versions[i].created_at[-8:]} "
                    f"({store.versions[i].meta.get('action', 'edit')})"
                ),
            )
            if st.button("↩️ Restore this version", use_container_width=True):
                store.restore_version(ver_idx)
                st.success("Version restored!")
                st.rerun()

        st.divider()
        RateLimiter.sidebar_meter("contract")

        if st.button("🗑️ Reset contract", use_container_width=True):
            st.session_state.contract_store = ContractStore()
            st.session_state.contract_messages = []
            st.session_state.extracted_contract_fields = None
            st.session_state.analysis_result = None
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# LEGAL Q&A MODE
# ─────────────────────────────────────────────────────────────────────────────
if mode == "⚖️ Legal Q&A":
    st.markdown("## ⚖️ Legal Q&A")
    st.caption("Ask about Jordanian Labour Law in Arabic or English")

    if not _api_key:
        st.error("⚠️ OPENAI_API_KEY is not set. Please set it and restart the app.")

    # ── Empty state / onboarding ──────────────────────────────────────────────
    if not st.session_state.qa_messages:
        st.markdown(
            """
            <div class="onboard-card">
                <div class="onboard-title">🇯🇴 Jordanian Labour Law Assistant</div>
                <div class="onboard-sub">
                    Ask questions in Arabic or English about your rights under<br>
                    <strong>Labour Law No. 8 of 1996</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_en, col_ar = st.columns(2)
        sample_en = [
            "What is the minimum wage in Jordan?",
            "How many annual leave days am I entitled to?",
            "Can I be fired without notice?",
        ]
        sample_ar = [
            "ما هو الحد الأدنى للأجور في الأردن؟",
            "كم عدد أيام الإجازة السنوية المستحقة؟",
            "متى يحق لي الاستقالة دون إشعار؟",
        ]
        with col_en:
            st.markdown("**Try asking (EN):**")
            for q in sample_en:
                if st.button(q, key=f"chip_en_{q[:20]}", use_container_width=True):
                    st.session_state._pending_question = q
                    st.rerun()
        with col_ar:
            st.markdown("**جرب أن تسأل (AR):**")
            for q in sample_ar:
                if st.button(q, key=f"chip_ar_{q[:20]}", use_container_width=True):
                    st.session_state._pending_question = q
                    st.rerun()

    # ── Chat history ──────────────────────────────────────────────────────────
    for mi, msg in enumerate(st.session_state.qa_messages):
        with st.chat_message(msg["role"]):
            is_ar = msg.get("lang") == "ar"
            if is_ar:
                st.markdown(
                    f'<div dir="rtl" style="text-align:right">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg["content"])

            if msg["role"] == "assistant":
                # Confidence badge
                if msg.get("low_confidence"):
                    st.markdown(
                        '<span class="badge-low">⚠️ Low confidence — verify with a lawyer</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span class="badge-high">✅ High confidence</span>',
                        unsafe_allow_html=True,
                    )

                # Citation deep-dive
                arts = msg.get("articles", [])
                if arts:
                    with st.expander(
                        f"📖 View cited articles: {', '.join(str(a) for a in arts)}"
                    ):
                        for art_id in arts:
                            col_e, col_a = st.columns(2)
                            with col_e:
                                st.markdown(f"**Article {art_id} (EN)**")
                                en_art = get_article_by_id(art_id, "en")
                                st.text_area(
                                    "", en_art["text"] if en_art else "Not found",
                                    height=180,
                                    key=f"hist_art_{mi}_{art_id}_en",
                                    label_visibility="collapsed",
                                )
                            with col_a:
                                st.markdown(f"**المادة {art_id} (AR)**")
                                ar_art = get_article_by_id(art_id, "ar")
                                st.text_area(
                                    "", ar_art["text"] if ar_art else "غير موجود",
                                    height=180,
                                    key=f"hist_art_{mi}_{art_id}_ar",
                                    label_visibility="collapsed",
                                )

                # Follow-up suggestions
                if msg.get("followup"):
                    with st.expander("💡 Follow-up questions"):
                        for qi, q in enumerate(msg["followup"]):
                            if st.button(q, key=f"fq_{mi}_{qi}"):
                                st.session_state._pending_question = q
                                st.rerun()

    # ── Export + Clear row ────────────────────────────────────────────────────
    if st.session_state.qa_messages:
        col_exp, col_clr = st.columns([2, 1])
        with col_exp:
            if st.button("📥 Export Chat (PDF)", use_container_width=True):
                pdf_bytes = export_conversation_pdf(
                    st.session_state.qa_messages, "Legal Q&A — Jordanian Labour Law"
                )
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name="legal_qa_session.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.warning("PDF export unavailable. Run: pip install fpdf2")
        with col_clr:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.qa_messages = []
                st.rerun()

    # ── Rate limit check ──────────────────────────────────────────────────────
    if not RateLimiter.check("qa"):
        RateLimiter.show_warning("qa")
    else:
        # ── Chat input ────────────────────────────────────────────────────────
        question = st.chat_input("Ask about Jordanian Labour Law...")

        # Handle sample chip click
        if st.session_state.get("_pending_question"):
            question = st.session_state._pending_question
            st.session_state._pending_question = None

        if question and question.strip():
            q = question.strip()
            st.session_state.qa_messages.append({"role": "user", "content": q})

            with st.chat_message("user"):
                st.markdown(q)

            # Build history context (last 20 messages)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.qa_messages[:-1]
            ]

            with st.chat_message("assistant"):
                with st.spinner("Searching Jordanian Labour Law…"):
                    stream, metadata = legal_qa_chatbot_stream(
                        q,
                        top_k=top_k,
                        lang_override=st.session_state.qa_lang_override,
                        chat_history=history,
                    )

                detected_lang = metadata.get("detected_lang", "en")
                is_ar = detected_lang == "ar"

                # Confidence badge (shown before streaming so user sees it immediately)
                if metadata.get("low_confidence"):
                    st.markdown(
                        '<span class="badge-low">⚠️ Low confidence — verify with a lawyer</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<span class="badge-high">✅ High confidence</span>',
                        unsafe_allow_html=True,
                    )

                # Streaming answer with RTL support
                placeholder = st.empty()
                full_answer = ""
                for chunk in stream:
                    full_answer += chunk
                    if is_ar:
                        placeholder.markdown(
                            f'<div dir="rtl" style="text-align:right">{full_answer}▌</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        placeholder.markdown(f"{full_answer}▌")

                # Final answer (no cursor)
                if is_ar:
                    placeholder.markdown(
                        f'<div dir="rtl" style="text-align:right">{full_answer}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    placeholder.markdown(full_answer)

                # Article citations + deep-dive
                articles = metadata.get("retrieved_article_ids", [])
                if articles:
                    st.caption(
                        f"📌 Based on Articles: {', '.join(str(a) for a in articles)}"
                    )
                    with st.expander("📖 View cited articles (EN + AR)"):
                        for art_id in articles:
                            col_e, col_a = st.columns(2)
                            with col_e:
                                st.markdown(f"**Article {art_id} (EN)**")
                                en_art = get_article_by_id(art_id, "en")
                                st.text_area(
                                    "",
                                    en_art["text"] if en_art else "Not found.",
                                    height=180,
                                    key=f"new_art_{art_id}_en",
                                    label_visibility="collapsed",
                                )
                            with col_a:
                                st.markdown(f"**المادة {art_id} (AR)**")
                                ar_art = get_article_by_id(art_id, "ar")
                                st.text_area(
                                    "",
                                    ar_art["text"] if ar_art else "غير موجود.",
                                    height=180,
                                    key=f"new_art_{art_id}_ar",
                                    label_visibility="collapsed",
                                )

                # Follow-up suggestions
                followup = metadata.get("followup_suggestions", [])
                if followup:
                    with st.expander("💡 You might also ask"):
                        for fq in followup:
                            st.markdown(f"→ *{fq}*")

                # Persist to history + track token usage (estimate)
                RateLimiter.increment("qa", max(len(full_answer) // 4, 500))

            st.session_state.qa_messages.append({
                "role": "assistant",
                "content": full_answer,
                "articles": articles,
                "low_confidence": metadata.get("low_confidence", False),
                "followup": followup,
                "lang": detected_lang,
            })


# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT DRAFTING MODE
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("## 📄 Contract Drafting")
    st.caption(
        "Generate non-binding employment contract drafts — for informational purposes only"
    )

    store: ContractStore = st.session_state.contract_store

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Create Contract", "💬 Edit via Chat", "📋 Preview & Export", "📂 Analyze Contract"]
    )

    # ── TAB 1: CREATE CONTRACT ───────────────────────────────────────────────
    with tab1:
        # Notification if fields were loaded from analysis
        pre = st.session_state.get("extracted_contract_fields") or {}
        if pre:
            st.success(
                "✅ Fields loaded from your uploaded contract. Review and edit below, then generate."
            )
            if st.button("❌ Clear loaded fields"):
                st.session_state.extracted_contract_fields = None
                # Clear form keys
                for fk in [
                    "cf_employer", "cf_employee", "cf_job_title", "cf_location",
                    "cf_start_date", "cf_end_date", "cf_salary", "cf_hours",
                    "cf_days", "cf_probation", "cf_termination", "cf_benefits",
                ]:
                    st.session_state.pop(fk, None)
                st.rerun()

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<p class="section-title">Parties</p>', unsafe_allow_html=True)
            employer  = st.text_input("Employer name *",  key="cf_employer",  value=pre.get("employer", ""),  placeholder="e.g. Jordan Tech Co.")
            employee  = st.text_input("Employee name *",  key="cf_employee",  value=pre.get("employee", ""),  placeholder="e.g. Ahmad Al-Hassan")
            job_title = st.text_input("Job title *",      key="cf_job_title", value=pre.get("job_title", ""), placeholder="e.g. Software Engineer")
            location  = st.text_input("Work location *",  key="cf_location",  value=pre.get("location", ""),  placeholder="e.g. Amman, Jordan")

            st.markdown('<p class="section-title">Contract Term</p>', unsafe_allow_html=True)

            # Determine default contract type index
            _ct = pre.get("contract_type", "Limited (Fixed-term)")
            _ct_opts = ["Limited (Fixed-term)", "Unlimited"]
            _ct_idx = 1 if "unlimited" in _ct.lower() else 0
            contract_type = st.selectbox("Contract type *", _ct_opts, index=_ct_idx)

            col_s, col_e = st.columns(2)
            with col_s:
                start_date = st.text_input("Start date *", key="cf_start_date", value=pre.get("start_date", ""), placeholder="DD/MM/YYYY")
            with col_e:
                end_date = st.text_input(
                    "End date", key="cf_end_date",
                    value=pre.get("end_date", ""),
                    placeholder="DD/MM/YYYY",
                    disabled=("Unlimited" in contract_type),
                )
            probation = st.text_input(
                "Probation period (days)", key="cf_probation",
                value=pre.get("probation_days", ""),
                placeholder="e.g. 90",
                help="Max 90 days under Jordanian law",
            )

        with col2:
            st.markdown('<p class="section-title">Compensation & Hours</p>', unsafe_allow_html=True)
            salary = st.text_input(
                "Monthly salary (JOD) *", key="cf_salary",
                value=pre.get("salary", ""),
                placeholder="Min: 260 JOD",
            )
            col_h, col_d = st.columns(2)
            with col_h:
                hours = st.text_input("Hours/day *", key="cf_hours", value=pre.get("hours", ""), placeholder="e.g. 8")
            with col_d:
                days  = st.text_input("Days/week *", key="cf_days",  value=pre.get("days",  ""), placeholder="e.g. 5")
            benefits = st.text_area(
                "Additional benefits", key="cf_benefits",
                value=pre.get("benefits", ""),
                placeholder="e.g. Health insurance, transport allowance",
                height=80,
            )

            st.markdown('<p class="section-title">Termination</p>', unsafe_allow_html=True)
            termination = st.text_area(
                "Termination terms", key="cf_termination",
                value=pre.get("termination", ""),
                placeholder="e.g. Either party may terminate with 30 days written notice.",
                height=100,
                help="Jordanian law requires minimum 30 days notice",
            )

        st.divider()

        # Live validation
        data = {
            "employer": employer, "employee": employee, "job_title": job_title,
            "location": location,
            "contract_type": contract_type.split(" ")[0],
            "start_date": start_date, "end_date": end_date,
            "salary": salary, "hours": hours, "days": days,
            "probation_days": probation,
            "termination": termination, "benefits": benefits,
        }

        issues = validate_contract_data(data)
        summary = get_validation_summary(issues)

        if issues:
            with st.expander(
                f"⚠️ {summary['error_count']} errors · {summary['warning_count']} warnings"
                f" · {summary['info_count']} info — click to review",
                expanded=summary["error_count"] > 0,
            ):
                for issue in summary["errors"]:
                    ref = f" *({issue.article_ref})*" if issue.article_ref else ""
                    st.markdown(
                        f'<div class="val-error">❌ {issue.message}{ref}</div>',
                        unsafe_allow_html=True,
                    )
                for issue in summary["warnings"]:
                    ref = f" *({issue.article_ref})*" if issue.article_ref else ""
                    st.markdown(
                        f'<div class="val-warning">⚠️ {issue.message}{ref}</div>',
                        unsafe_allow_html=True,
                    )
                for issue in summary["infos"]:
                    ref = f" *({issue.article_ref})*" if issue.article_ref else ""
                    st.markdown(
                        f'<div class="val-info">ℹ️ {issue.message}{ref}</div>',
                        unsafe_allow_html=True,
                    )

        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            gen_disabled = summary["error_count"] > 0 or not RateLimiter.check("contract")
            label = (
                "✨ Generate Contract"
                if not gen_disabled
                else f"❌ Fix {summary['error_count']} error(s) first"
                if summary["error_count"] > 0
                else "⚠️ Session limit reached"
            )
            if st.button(label, use_container_width=True, disabled=gen_disabled, type="primary"):
                with st.spinner("Generating contract with GPT-4o-mini…"):
                    contract_text, val_issues = generate_contract_ai(store, data)
                RateLimiter.increment("contract", 3000)
                st.success("✅ Contract generated! Go to **Preview & Export** tab.")
                st.session_state.extracted_contract_fields = None  # Clear pre-load after generation
                st.rerun()

        with col_b2:
            if summary["error_count"] > 0:
                st.info(f"🔒 {summary['error_count']} error(s) blocking generation")
            elif not RateLimiter.check("contract"):
                RateLimiter.show_warning("contract")

    # ── TAB 2: EDIT VIA CHAT ─────────────────────────────────────────────────
    with tab2:
        if not store.current:
            st.info("👈 Generate a contract first in the **Create Contract** tab.")
        else:
            st.markdown("Chat to edit your contract naturally:")
            st.caption(
                "e.g. *'Change salary to 800 JOD'*, "
                "*'Add a non-compete clause'*, "
                "*'Translate to Arabic'*"
            )

            for msg in st.session_state.contract_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            edit_msg = st.chat_input("What would you like to change?")
            if edit_msg and edit_msg.strip():
                if not RateLimiter.check("contract", 2000):
                    RateLimiter.show_warning("contract")
                else:
                    st.session_state.contract_messages.append(
                        {"role": "user", "content": edit_msg.strip()}
                    )
                    with st.spinner("Applying changes…"):
                        updated, reply, _ = chat_edit_contract(store, edit_msg.strip())
                    RateLimiter.increment("contract", 2000)
                    st.session_state.contract_messages.append(
                        {"role": "assistant", "content": reply}
                    )
                    st.rerun()

    # ── TAB 3: PREVIEW & EXPORT ──────────────────────────────────────────────
    with tab3:
        if not store.current:
            st.info("👈 Generate a contract first in the **Create Contract** tab.")
        else:
            col_prev, col_exp = st.columns([3, 1], gap="large")

            with col_prev:
                st.markdown('<p class="section-title">Contract Preview</p>', unsafe_allow_html=True)
                st.text_area(
                    "Contract text",
                    store.current,
                    height=520,
                    label_visibility="collapsed",
                )

            with col_exp:
                st.markdown('<p class="section-title">Export</p>', unsafe_allow_html=True)

                if st.button("📥 Export to Word (.docx)", use_container_width=True, type="primary"):
                    with st.spinner("Generating Word document…"):
                        path = export_to_word(store.current, store.lang)
                    if path.startswith("❌"):
                        st.error(path)
                    else:
                        st.session_state.last_export_path = path
                        st.success("Ready to download!")

                if (
                    st.session_state.last_export_path
                    and os.path.isfile(st.session_state.last_export_path)
                ):
                    with open(st.session_state.last_export_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Contract",
                            data=f,
                            file_name=os.path.basename(st.session_state.last_export_path),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )

                st.divider()
                st.markdown('<p class="section-title">Stats</p>', unsafe_allow_html=True)
                st.metric("Words", len(store.current.split()))
                st.metric("Versions saved", len(store.versions))
                st.metric("Completeness", f"{store.completeness_score()}%")

                st.divider()
                st.markdown('<p class="section-title">Legal Notice</p>', unsafe_allow_html=True)
                st.caption(
                    "This is a non-binding draft for informational purposes only. "
                    "Always consult a licensed Jordanian lawyer before signing any contract."
                )

    # ── TAB 4: ANALYZE EXISTING CONTRACT ─────────────────────────────────────
    with tab4:
        st.markdown("### 📂 Analyze Your Existing Contract")
        st.caption(
            "Upload a PDF or DOCX contract and we'll check it against "
            "Jordanian Labour Law No. 8 / 1996."
        )

        uploaded = st.file_uploader(
            "Upload your contract",
            type=["pdf", "docx"],
            help="Maximum file size: 200 MB",
        )

        if uploaded:
            col_btn, col_info = st.columns([1, 2])
            with col_btn:
                analyze_btn = st.button(
                    "🔍 Analyze Contract",
                    use_container_width=True,
                    type="primary",
                    disabled=not RateLimiter.check("contract", 2000),
                )
            with col_info:
                st.caption(f"📄 {uploaded.name} · {uploaded.size / 1024:.1f} KB")

            if analyze_btn:
                with st.spinner("Extracting text and analyzing against Jordanian Labour Law…"):
                    raw_text = _extract_text_from_upload(uploaded)
                    result = analyze_uploaded_document(raw_text)
                    st.session_state.analysis_result = result
                    RateLimiter.increment("contract", 2000)

        result = st.session_state.get("analysis_result")
        if result:
            if not result.get("success"):
                st.error(result.get("ai_analysis", "Analysis failed."))
            else:
                st.divider()

                # ── AI narrative summary ──────────────────────────────────────
                st.markdown("#### 🤖 AI Analysis Summary")
                st.info(result.get("ai_analysis", ""))

                st.divider()

                # ── Extracted fields table ────────────────────────────────────
                st.markdown("#### 🔍 Extracted Contract Fields")
                fields = result.get("fields_extracted", {})
                field_labels = {
                    "employer": "Employer", "employee": "Employee",
                    "job_title": "Job Title", "location": "Location",
                    "contract_type": "Contract Type",
                    "start_date": "Start Date", "end_date": "End Date",
                    "salary": "Salary (JOD)", "hours": "Hours/Day",
                    "days": "Days/Week", "probation_days": "Probation (days)",
                    "termination": "Termination", "benefits": "Benefits",
                }
                rows_html = "".join(
                    f"<tr><td><strong>{label}</strong></td>"
                    f"<td>{fields.get(key, '—') or '—'}</td></tr>"
                    for key, label in field_labels.items()
                )
                st.markdown(
                    f'<table class="analysis-field-table"><thead><tr>'
                    f'<th style="width:35%">Field</th><th>Extracted Value</th>'
                    f'</tr></thead><tbody>{rows_html}</tbody></table>',
                    unsafe_allow_html=True,
                )

                st.divider()

                # ── Validation results ────────────────────────────────────────
                vsum = result.get("validation_summary", {})
                st.markdown(
                    f"#### ⚖️ Compliance Check  "
                    f"— {vsum.get('error_count', 0)} errors · "
                    f"{vsum.get('warning_count', 0)} warnings · "
                    f"{vsum.get('info_count', 0)} info"
                )

                for issue in vsum.get("errors", []):
                    ref = f" *({issue.article_ref})*" if issue.article_ref else ""
                    st.markdown(
                        f'<div class="val-error">❌ {issue.message}{ref}</div>',
                        unsafe_allow_html=True,
                    )
                for issue in vsum.get("warnings", []):
                    ref = f" *({issue.article_ref})*" if issue.article_ref else ""
                    st.markdown(
                        f'<div class="val-warning">⚠️ {issue.message}{ref}</div>',
                        unsafe_allow_html=True,
                    )
                for issue in vsum.get("infos", []):
                    ref = f" *({issue.article_ref})*" if issue.article_ref else ""
                    st.markdown(
                        f'<div class="val-info">ℹ️ {issue.message}{ref}</div>',
                        unsafe_allow_html=True,
                    )

                st.divider()

                # ── Load into editor ──────────────────────────────────────────
                st.markdown("#### 🔄 Load into Contract Editor")
                st.caption(
                    "Load the extracted fields into the **Create Contract** tab "
                    "to regenerate a compliant version."
                )
                if st.button("📋 Load fields into contract editor", use_container_width=True, type="primary"):
                    st.session_state.extracted_contract_fields = fields
                    # Also write each field into the form's session state keys
                    key_map = {
                        "employer": "cf_employer", "employee": "cf_employee",
                        "job_title": "cf_job_title", "location": "cf_location",
                        "start_date": "cf_start_date", "end_date": "cf_end_date",
                        "salary": "cf_salary", "hours": "cf_hours", "days": "cf_days",
                        "probation_days": "cf_probation",
                        "termination": "cf_termination", "benefits": "cf_benefits",
                    }
                    for fld, sk in key_map.items():
                        val = fields.get(fld, "")
                        if val:
                            st.session_state[sk] = val
                    st.success("✅ Fields loaded! Switch to the **Create Contract** tab to review and generate.")
                    st.rerun()
