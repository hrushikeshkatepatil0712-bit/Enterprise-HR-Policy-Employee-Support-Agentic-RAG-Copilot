
from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


# ROUTE DECISION
class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        description="kb for HR and policy questions; direct for greetings and simple chat"
    )


# EVIDENCE GRADING
class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description="Whether the retrieved evidence is sufficient to answer"
    )


# LANGGRAPH AGENT STATE
class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int
    trace: List[str]
    citations: List[str]
