"""Conversational agent orchestration (function-calling loop)."""
import json
from api.agent import llm, prompts
from api.agent.records import TraceRecorder
from api.agent.schemas import ChatResponse, ToolCallRecord
from api.agent.tools import TOOL_SPECS, execute_tool
MAX_TOOL_ROUNDS = 5
UNAVAILABLE_MESSAGE = "助手暂时无法连接语言模型，你可以先使用「探索」或「图鉴」浏览，或稍后再试。"
ERROR_MESSAGE = "助手处理这个问题时出现了问题，请稍后重试或换个问法。"
EXCEEDED_MESSAGE = "这个问题需要的步骤太多了，请换一个更具体的问法。"
def _assistant_message(message) -> dict:
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ],
    }
def run_agent(request, catalog, retriever, tracer: TraceRecorder, max_rounds: int = MAX_TOOL_ROUNDS):
    trace_id = tracer.new_id()
    if not llm.is_configured():
        tracer.record(trace_id, {"status": "llm_unavailable"})
        return ChatResponse(
            status="llm_unavailable",
            answer=UNAVAILABLE_MESSAGE,
            traceId=trace_id,
            version=catalog.version,
        )
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
            completion = client.chat.completions.create(
                model=llm.model_name(),
                messages=messages,
                tools=TOOL_SPECS,
                tool_choice="auto",
            )
            message = completion.choices[0].message
            if not message.tool_calls:
                tracer.record(
                    trace_id,
                    {"status": "ok", "tools": [entry.name for entry in tool_log]},
                )
                return build("ok", message.content or "")
            messages.append(_assistant_message(message))
            for call in message.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = execute_tool(call.function.name, args, catalog, retriever)
                tool_log.append(
                    ToolCallRecord(name=result.name, args=result.args, ok=result.ok)
                )
                for record in result.records:
                    records_map[record.id] = record
                for citation in result.citations:
                    citations_map[(citation.refType, citation.refId)] = citation
                actions.extend(result.actions)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result.payload, ensure_ascii=False),
                    }
                )
        tracer.record(trace_id, {"status": "exceeded_rounds"})
        return build("ok", EXCEEDED_MESSAGE)
    except Exception as exc:  # network, timeout, upstream API error
        tracer.record(trace_id, {"status": "error", "error": str(exc)})
        return build("error", ERROR_MESSAGE)
