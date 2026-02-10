# 🇯🇴 Jordanian Legal AI Assistant

## Scenario
This project is a Jordanian Legal AI Assistant developed for educational purposes.
The system helps users understand the Jordanian Labour Law and generate non-binding employment contract drafts.
It is designed as an academic project and does not provide legal advice.

---

## User Flow
1. The user opens the Streamlit application.
2. The user selects one of two modes:
   - Legal Question Answering (RAG)
   - Contract Drafting (Non-binding)
3. In Legal QA mode:
   - The user asks a question in Arabic or English.
   - The system retrieves relevant legal articles.
   - The system generates an answer based only on legal text.
4. In Contract Drafting mode:
   - The user provides contract details.
   - The system generates a non-binding employment contract.
   - The user can update sections or export the contract.

---

## Architecture
The following image shows the system architecture:

![System Architecture](architecture/system_architecture.png)

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run src/app.py
