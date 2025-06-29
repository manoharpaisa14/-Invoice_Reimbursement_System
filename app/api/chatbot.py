from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from dotenv import load_dotenv
import os
from groq import Groq
from app.services.vector_store import search_similar

load_dotenv()
router = APIRouter()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ChatQuery(BaseModel):
    query: str
    filters: Optional[Dict[str, str]] = None

@router.post("/")
def rag_chatbot(query: ChatQuery):
    try:
        print("💬 Received Query:", query.query)
        print("🔎 Received Filters:", query.filters)

        # Perform vector + metadata search
        results = search_similar(query.query, query.filters or {})

        print(f"📄 Results Found: {len(results)}")

        allowed_statuses = ["Fully Reimbursed", "Partially Reimbursed", "Declined"]
        filtered = [r for r in results if r.get("status") in allowed_statuses]

        if not filtered:
            return {
                "response": (
                    "Unfortunately, I couldn't find any relevant documents to provide a clear answer to this question.\n\n"
                    "Please check if the filters (employee name, status, date) are accurate."
                )
            }

        context_parts = []
        for r in filtered:
            context_parts.append(
                f"Invoice ID: {r.get('invoice_id', 'N/A')}\n"
                f"Employee: {r.get('employee_name', 'N/A')}\n"
                f"Date: {r.get('date', 'Unknown')}\n"
                f"Status: {r.get('status', 'Unknown')}\n"
                f"Reason: {r.get('reason', 'N/A')}\n"
            )
        context = "\n\n".join(context_parts)

        prompt = f"""
You are an intelligent invoice reimbursement assistant.
Based on the **context** below, answer the **question** clearly and helpfully in markdown format.

Context:
{context}

Question:
{query.query}
"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )

        return {"response": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
