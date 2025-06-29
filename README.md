# 🧾 Invoice Reimbursement System

## 🚀 Project Overview
This is an intelligent Invoice Reimbursement System that automates the analysis of employee invoice PDFs against a provided HR reimbursement policy using Large Language Models (LLMs). It also supports natural language queries using a Retrieval-Augmented Generation (RAG) chatbot.

The system consists of two main components:
1. **Invoice Analysis Endpoint** – Analyzes invoices against an HR policy and stores results in a vector database.
2. **RAG Chatbot Endpoint** – Lets users query the processed invoice data using natural language.

## 🏗️ Tech Stack
- **Backend Framework:** FastAPI
- **Frontend:** Streamlit (optional UI)
- **LLMs:** OpenAI, Groq (LLaMA 3 or similar), or Gemini
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS
- **PDF Parsing:** PyMuPDF (`fitz`)
- **Prompt Engineering:** Structured prompts for both invoice analysis and chatbot

## 🛠️ Installation Instructions
```bash
# Clone the repository
git clone <your_repo_link>
cd invoice_rag_project

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Running the Application

### 1. Run FastAPI Backend
```bash
uvicorn app.main:app --reload
```

### 2. Run Streamlit Frontend (Optional)
```bash
streamlit run streamlit_app.py
```

## 📤 Usage Guide

### 1. Analyze Endpoint (POST `/analyze-invoices`)
**Input:** HR Policy PDF, ZIP file of invoice PDFs, Employee name  
**Output:** JSON success response, stores metadata + embeddings in vector DB.

### 2. Chatbot Endpoint (POST `/chat`)
**Input:** Natural language query with optional filters (employee name, date, status, etc.)  
**Output:** Markdown response answering based on stored invoices

## 🧠 Prompt Design

### Invoice Analysis Prompt
- Guides LLM to extract invoice data, match against HR policy
- Outputs: `status`, `reason`, `date`, `invoice_id`

### Chatbot Prompt
- Instructs LLM to extract filters and keywords
- Uses vector search and metadata filtering to answer user queries

## 📁 Project Structure
```
invoice_rag/
│
├── app/
│   ├── api/
│   │   ├── analyze.py         # Endpoint for analyzing invoices
│   │   └── chatbot.py         # Endpoint for chatbot RAG queries
│   ├── models/schemas.py      # Pydantic models
│   ├── services/
│   │   ├── llm_analyzer.py    # LLM prompt and invoice logic
│   │   ├── pdf_utils.py       # PDF reading utilities
│   │   └── vector_store.py    # FAISS vector DB wrapper
│
├── streamlit_app.py           # Optional UI for testing
├── requirements.txt
├── .env
└── README.md
```

## 📊 Dataset Link
Sample HR Policy and invoices are located inside the project:
- `debug_policy_from_streamlit.pdf`
- Test invoices in ZIP format (can upload via UI or API)

## ✅ Features Implemented
- [x] PDF Parsing
- [x] LLM-based invoice analysis
- [x] FAISS vector store integration
- [x] Metadata filtering (employee, date, status)
- [x] Natural language RAG chatbot
- [x] Optional UI via Streamlit

## 🔍 Optional Enhancements (Included)
- [x] Streamlit UI for demo
- [ ] Advanced error handling (basic implemented)
- [ ] Multiprocessing (can be added)

## 🎥 Demo Instructions
> You should create a 2-min Loom video showing:
> - Invoice Upload + Policy upload (analyze endpoint)
> - Asking chatbot queries like:
>   - "What invoices were fully reimbursed for Rekha?"
>   - "Which invoices were declined and why?"

## 📌 Author
Manohar — 2024 Year AI/ML Project