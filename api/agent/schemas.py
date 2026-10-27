"""Pydantic models for the conversational agent."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from api.schemas import Slide
AgentStatus = Literal["ok", "llm_unavailable", "error"]
ActionType = Literal["locate", "open_recognize", "open_task"]
class AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
class ChatMessage(AgentModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
class ChatRequest(AgentModel):
    sessionId: str | None = Field(default=None, max_length=100)
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)
class Citation(AgentModel):
    refType: Literal["source", "doc", "slide"]
    refId: str
    title: str
    url: str | None = None
class AgentAction(AgentModel):
    type: ActionType
    slideId: str | None = None
    taskId: str | None = None
class ToolCallRecord(AgentModel):
    name: str
    args: dict
    ok: bool
class ChatResponse(AgentModel):
    status: AgentStatus
    answer: str
    records: list[Slide] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    actions: list[AgentAction] = Field(default_factory=list)
    toolCalls: list[ToolCallRecord] = Field(default_factory=list)
    traceId: str | None = None
    version: str | None = None
