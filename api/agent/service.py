"""Conversational agent orchestration (function-calling loop).

``iter_agent_events`` is the streaming core: it calls the LLM with
``stream=True`` and yields JSON-serializable events while working:

- ``{"type": "delta", "content": ...}`` — answer text chunk
- ``{"type": "status", "tool": ..., "ok": ...}`` — a tool call finished
- ``{"type": "done", "response": ...}`` — final ``ChatResponse`` payload
- ``{"type": "error", "response": ...}`` — terminal failure payload

``run_agent`` drains the generator so the non-streaming endpoint keeps its
original request/response behavior.
"""
import json
from api.agent import llm, prompts
from api.agent.records import TraceRecorder
from api.agent.schemas import ChatResponse, ToolCallRecord
from api.agent.tools import TOOL_SPECS, execute_tool
MAX_TOOL_ROUNDS = 5
UNAVAILABLE_MESSAGE = "助手暂时无法连接语言模型，你可以先使用「探索」或「图鉴」浏览，或稍后再试。"
ERROR_MESSAGE = "助手处理这个问题时出现了问题，请稍后重试或换个问法。"
EXCEEDED_MESSAGE = "这个问题需要的步骤太多了，请换一个更具体的问法。"


def iter_agent_events(
    request,
    catalog,
    retriever,
    tracer: TraceRecorder,
    max_rounds: int = MAX_TOOL_ROUNDS,
):
    trace_id = tracer.new_id()
    if not llm.is_configured():
        tracer.record(trace_id, {"status": "llm_unavailable"})
        yield {
            "type": "done",
            "response": ChatResponse(
                status="llm_unavailable",
                answer=UNAVAILABLE_MESSAGE,
                traceId=trace_id,
                version=catalog.version,
            ).model_dump(),
        }
        return
    client = llm.get_client()
    messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT}]
    messages.extend(message.model_dump() for message in request.messages)
    records_map: dict = {}
    citations_map: dict = {}
    actions = []
    tool_log: list[ToolCallRecord] = []

    def build(status: str, answer: str) -> ChatResponse:
        return ChatResponse(
            status=status,
            answer=answer,
            records=list(records_map.values()),
            citations=list(citations_map.values()),
            actions=actions,
            toolCalls=tool_log,
            traceId=trace_id,
            version=catalog.version,
        )

    try:
        for _ in range(max_rounds):
            stream = client.chat.completions.create(
                model=llm.model_name(),
                messages=messages,
                tools=TOOL_SPECS,
                tool_choice="auto",
                stream=True,
            )
            content_parts: list[str] = []
            pending_calls: dict[int, dict] = {}
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    content_parts.append(delta.content)
                    yield {"type": "delta", "content": delta.content}
                for call in delta.tool_calls or []:
                    slot = pending_calls.setdefault(
                        call.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if call.id:
                        slot["id"] = call.id
                    if call.function:
                        if call.function.name:
                            slot["name"] += call.function.name
                        if call.function.arguments:
                            slot["arguments"] += call.function.arguments
            if not pending_calls:
                tracer.record(
                    trace_id,
                    {"status": "ok", "tools": [entry.name for entry in tool_log]},
                )
                yield {
                    "type": "done",
                    "response": build("ok", "".join(content_parts)).model_dump(),
                }
                return
            ordered_calls = [pending_calls[index] for index in sorted(pending_calls)]
            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(content_parts),
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for call in ordered_calls
                    ],
                }
            )
            for call in ordered_calls:
                try:
                    args = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = execute_tool(call["name"], args, catalog, retriever)
                tool_log.append(
                    ToolCallRecord(name=result.name, args=result.args, ok=result.ok)
                )
                yield {"type": "status", "tool": result.name, "ok": result.ok}
                for record in result.records:
                    records_map[record.id] = record
                for citation in result.citations:
                    citations_map[(citation.refType, citation.refId)] = citation
                actions.extend(result.actions)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result.payload, ensure_ascii=False),
                    }
                )
        tracer.record(trace_id, {"status": "exceeded_rounds"})
        yield {"type": "done", "response": build("ok", EXCEEDED_MESSAGE).model_dump()}
    except Exception as exc:  # network, timeout, upstream API error
        tracer.record(trace_id, {"status": "error", "error": str(exc)})
        yield {"type": "error", "response": build("error", ERROR_MESSAGE).model_dump()}


def run_agent(request, catalog, retriever, tracer: TraceRecorder, max_rounds: int = MAX_TOOL_ROUNDS):
    final: ChatResponse | None = None
    for event in iter_agent_events(request, catalog, retriever, tracer, max_rounds):
        if event["type"] in ("done", "error"):
            final = ChatResponse.model_validate(event["response"])
    if final is None:  # pragma: no cover - the generator always terminates
        final = ChatResponse(status="error", answer=ERROR_MESSAGE)
    return final
