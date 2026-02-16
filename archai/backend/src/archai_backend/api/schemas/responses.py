from typing import List

from pydantic import BaseModel


class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
