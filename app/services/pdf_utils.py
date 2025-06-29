import zipfile
import fitz  # PyMuPDF
import os
from typing import List, Tuple

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_invoices_from_zip(zip_path: str, extract_dir: str) -> List[str]:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    pdf_files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith(".pdf")]
    return pdf_files
