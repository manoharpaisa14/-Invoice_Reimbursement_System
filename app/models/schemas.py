from pydantic import BaseModel
from typing import Optional, List

class InvoiceAnalysisRequest(BaseModel):
    employee_name: str

class InvoiceMetadata(BaseModel):
    employee_name: str
    invoice_id: str
    date: str
    status: str
    reason: str

class ChatQueryRequest(BaseModel):
    query: str
    filters: Optional[dict] = None  # e.g., {"employee_name": "John", "status": "Declined"}
