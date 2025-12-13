from typing import Dict, Optional, Any
from pydantic import BaseModel

class QueryRequest(BaseModel):
    text: str

class ClarifyRequest(BaseModel):
    answer: str
    context: Dict[str, Any]

class QueryResponse(BaseModel):
    status: str
    original_query: str
    reformulated_query: Optional[str] = None
    clarifying_question: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
