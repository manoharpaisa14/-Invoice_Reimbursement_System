import os
import json
import requests
import re
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-8b-8192"

def clean_json(raw_text):
    json_start = raw_text.find('{')
    json_end = raw_text.rfind('}')
    if json_start != -1 and json_end != -1:
        return raw_text[json_start:json_end+1]
    return "{}"

def analyze_invoice_with_policy(invoice_text: str, policy_text: str) -> dict:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a strict and intelligent invoice reimbursement assistant.

Given an HR policy and an employee invoice, you must decide whether the invoice is:

- Fully Reimbursed → All expenses comply 100% with the policy.
- Partially Reimbursed → Some expenses comply, others exceed limits or are excluded.
- Declined → No expenses comply or invoice is invalid.

You MUST respond ONLY in this exact JSON format:
{{
  "status": "Fully Reimbursed / Partially Reimbursed / Declined",
  "reason": "..."
}}

🛑 Do NOT include extra text or explanation outside the JSON. Just return a JSON object.

### HR POLICY
{policy_text}

### EMPLOYEE INVOICE
{invoice_text}
"""

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an intelligent invoice reimbursement analyzer."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(GROQ_URL, headers=headers, json=data)

    if response.status_code == 200:
        try:
            raw_text = response.json()["choices"][0]["message"]["content"]
            print("🧠 LLM Raw Response:", raw_text)
            parsed = json.loads(clean_json(raw_text))
            status = parsed.get("status", "Undetermined").strip()
            if status not in ["Fully Reimbursed", "Partially Reimbursed", "Declined"]:
                status = "Undetermined"
            return {
                "status": status,
                "reason": parsed.get("reason", "No reason provided.")
            }
        except Exception:
            return {"status": "Undetermined", "reason": "Could not parse response"}
    else:
        return {"status": "Undetermined", "reason": "LLM failed"}

def extract_name_with_llm(invoice_text: str) -> str:
    prompt = f"""
Extract the customer's or employee's name from the invoice below.
Respond only with the name. If not found, say UNKNOWN.

Invoice:
{invoice_text}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}]
    }

    response = requests.post(GROQ_URL, headers=headers, json=data)
    if response.status_code == 200:
        try:
            result = response.json()["choices"][0]["message"]["content"].strip()
            if result.upper() == "UNKNOWN":
                return None

            if result.lower() not in invoice_text.lower():
                print(f"❌ LLM name rejected — not present in invoice: {result}")
                return None

            possible_names = set(re.findall(r'\b[A-Z][a-z]{2,}\b', invoice_text))
            if result not in possible_names:
                print(f"❌ LLM name '{result}' not in valid capitalized names: {possible_names}")
                return None

            return result
        except:
            return None
    return None

def reset_vector_store():
    from app.services.vector_store import VECTOR_PATH, META_PATH
    import shutil

    deleted = []
    for path in [VECTOR_PATH, META_PATH,
                 os.path.abspath(os.path.join(os.path.dirname(__file__), '../../vector.index')),
                 os.path.abspath(os.path.join(os.path.dirname(__file__), '../../metadata.pkl'))]:
        if os.path.exists(path):
            os.remove(path)
            deleted.append(path)
    print(f"🧹 Fully cleared vector store files: {deleted}")
