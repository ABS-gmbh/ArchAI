from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    doc_id: str
    page_id: Optional[str] = None
    query: Optional[str] = None
