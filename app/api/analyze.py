from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import os
import shutil
import re
from datetime import datetime
from app.services.pdf_utils import extract_text_from_pdf, extract_invoices_from_zip
from app.services.llm_analyzer import analyze_invoice_with_policy, extract_name_with_llm
from app.services.vector_store import save_to_vector_store, ensure_vector_files_exist

router = APIRouter()


def extract_metadata(invoice_text: str):
    date_match = re.search(
        r"(?:Invoice Date|Date)\s*[:\-\u2013]?\s*([\d]{1,2}[ /-]?[A-Za-z]{3,9}[ /-]?[\d]{2,4}|\d{4}-\d{2}-\d{2})",
        invoice_text
    )
    name_match = re.search(
        r"Customer Name\s*[:\-\u2013]?\s*([A-Za-z ]{2,})",
        invoice_text
    )
    if not name_match:
        name_match = re.search(
            r"(?:Employee Name|Name)\s*[:\-\u2013]?\s*([A-Za-z ]{2,})",
            invoice_text
        )
    extracted_date = date_match.group(1).strip() if date_match else None
    extracted_name = name_match.group(1).strip() if name_match else None
    return {"date": extracted_date, "employee_name": extracted_name}


def normalize_date(date_str: str) -> str:
    try:
        for fmt in ("%d %b %Y", "%d %B %Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    except:
        pass
    return None


@router.post("/")
async def analyze_invoices(
    invoice_zip: UploadFile = File(...),
    policy_text: str = Form(...)
):
    try:
        ensure_vector_files_exist()
        temp_dir = "temp_invoices"
        os.makedirs(temp_dir, exist_ok=True)

        zip_path = os.path.join(temp_dir, invoice_zip.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(invoice_zip.file, buffer)

        invoice_paths = extract_invoices_from_zip(zip_path, temp_dir)
        results = []

        for path in invoice_paths:
            invoice_text = extract_text_from_pdf(path)

            try:
                analysis = analyze_invoice_with_policy(invoice_text, policy_text)
                status = analysis.get("status", "Undetermined")
                reason = analysis.get("reason", "No reason provided.")
                print("🧠 LLM Output:", analysis)

                if status not in ["Fully Reimbursed", "Partially Reimbursed", "Declined"]:
                    print(f"❌ Skipping invoice {path} — invalid status: {status}")
                    continue

            except Exception as e:
                print(f"❌ LLM failed on {path}: {e}")
                continue

            invoice_id = os.path.basename(path).replace(".pdf", "")
            extracted = extract_metadata(invoice_text)

            raw_name = extracted.get("employee_name")
            if not raw_name:
                raw_name = extract_name_with_llm(invoice_text)

            extracted_name = raw_name.strip() if raw_name else "Unknown"
            raw_date = extracted.get("date")
            iso_date = normalize_date(raw_date.strip()) if raw_date else None

            if not raw_name or not raw_date:
                print(f"⚠️ Missing metadata → Invoice: {invoice_id} | Name: {raw_name} | Date: {raw_date}")

            full_text = invoice_text + "\n\n" + policy_text + f"\n\nStatus: {status}\nReason: {reason}"

            print(f"📄 Invoice ID: {invoice_id}")
            print(f"🧑 Employee: {extracted_name}")
            print(f"📅 Date: {iso_date}")
            print(f"📊 Status: {status} — Reason: {reason}")

            try:
                save_to_vector_store(
                    full_text,
                    metadata={
                        "invoice_id": invoice_id,
                        "employee_name": extracted_name,
                        "status": status,
                        "reason": reason,
                        "full_text": invoice_text,
                        "date": iso_date
                    }
                )
            except Exception as e:
                print(f"❌ Failed to save to vector store: {e}")

            results.append({
                "invoice_id": invoice_id,
                "status": status,
                "reason": reason
            })

        shutil.rmtree(temp_dir)
        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
