import os
from fastapi import FastAPI
from dotenv import load_dotenv
from app.api import analyze, chatbot
from app.services.vector_store import ensure_vector_files_exist
from app.services.llm_analyzer import reset_vector_store

# 🔄 Load environment variables
load_dotenv()

app = FastAPI(title="Invoice Reimbursement System")

# ✅ Ensure vector.index and metadata.pkl are initialized
ensure_vector_files_exist()

# ✅ Include API routes
app.include_router(analyze.router, prefix="/analyze", tags=["Invoice Analysis"])
app.include_router(chatbot.router, prefix="/chatbot", tags=["RAG Chatbot"])

@app.get("/")
def root():
    return {"message": "Invoice Reimbursement System is running with Groq + LLaMA3"}

@app.post("/reset/")
def reset_all():
    reset_vector_store()
    return {"message": "Vector store fully cleared."}
