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
import re
from unicodedata import normalize
from api.agent import llm, prompts
from api.agent.records import TraceRecorder
from api.agent.schemas import ChatResponse, Citation, ToolCallRecord
from api.agent.tools import TOOL_SPECS, execute_tool
from api.agent.recognition import task_lookup_events
from api.recognition_access import RecognitionTaskContext
from api.schemas import Catalog
MAX_TOOL_ROUNDS = 5
UNAVAILABLE_MESSAGE = "助手暂时无法连接语言模型，你可以先使用「探索」或「图鉴」浏览，或稍后再试。"
ERROR_MESSAGE = "助手处理这个问题时出现了问题，请稍后重试或换个问法。"
EXCEEDED_MESSAGE = "这个问题需要的步骤太多了，请换一个更具体的问法。"
SLIDE_ID_REQUEST = re.compile(
    r"(?:(?:请\s*)?(?:帮我\s*)?(?:定位(?:到)?|找(?:到|一下)?|查(?:找|一下|询|看)?|搜索)\s*)?"
    r"(?:滑坡\s*)?(?:编号\s*:?\s*)?#?\s*((?:ls_|demo-)?[0-9]+)"
    r"\s*号?\s*(?:滑坡)?\s*(?:在哪(?:里|儿)?)?",
    re.IGNORECASE,
)
CATALOGUE_COUNTS = re.compile(
    r"(?:现在|目前|当前)?(?:目录(?:里|中)?|火星(?:上)?|火星滑坡|滑坡)"
    r"(?:一共|总共)?(?:共有|有|收录了?)?多少(?:条|处|个)?(?:滑坡|记录)?"
    r"|(?:现在|目前|当前)?(?:一共|总共)?(?:共有|有)多少(?:条|处|个)?(?:火星)?滑坡"
)
TYPE_DIRECTORY_REQUEST = re.compile(
    r"(滑移型|滑塌型|流动型|复合型|SL|SP|FL|CL)\s*(?:滑坡)?\s*"
    r"(有哪些|(?:一共|总共)?(?:共有|有)多少(?:条|处|个)?|数量)",
    re.IGNORECASE,
)
TYPE_CODES = {name: code for code, name in prompts.CATALOG_TYPE_NAMES.items()}
FOUR_TYPE_OVERVIEW = {
    "火星滑坡分哪几类", "火星滑坡分为哪四类", "火星滑坡分为哪几类",
    "火星滑坡有哪些类型", "火星滑坡有哪几种类型", "本项目的滑坡分哪几类",
}


def _classification_reply(question: str, retriever, trace_id: str, version: str) -> ChatResponse | None:
    compare = (
        re.search(r"(?:四|4)\s*(?:类|分类|种(?:类型|滑坡))", question)
        and re.search(r"(?:六|6)\s*(?:类|分类|种(?:类型|滑坡))", question)
        and any(term in question for term in ("区别", "差别", "不同", "对比", "比较", "对照"))
    )
    if not compare and question not in FOUR_TYPE_OVERVIEW:
        return None

    ids = ("doc-overview", "doc-six-types") if compare else ("doc-overview",)
    sources = {doc.doc_id: doc for doc in getattr(retriever, "docs", ()) if doc.doc_id in ids}
    if any(doc_id not in sources for doc_id in ids):
        return None
    citations = [
        Citation(refType="doc", refId=doc_id, title=sources[doc_id].source_title,
                 url=sources[doc_id].source_url)
        for doc_id in ids
    ]
    items = [f"- **{code}** · {name}" for code, name in prompts.CATALOG_TYPE_NAMES.items()]
    if compare:
        answer = (
            "1. **本项目采用的四类**\n"
            + "\n".join(f"   {item}" for item in items)
            + "\n\n2. **跨天体研究的六类**\n"
            "   崩塌型、滑移型、流动型、扩散型、蠕变型、复合型。\n\n"
            "张乐飞等的体系用于火星编目。张晨阳等的体系用于比较太阳系多个固态天体上的滑坡。"
        )
    else:
        answer = "**本项目采用的四类**\n\n" + "\n".join(items)
    return ChatResponse(status="ok", answer=answer, citations=citations, traceId=trace_id, version=version)


def _catalog_short_reply(message: str, catalog: Catalog, retriever, trace_id: str) -> ChatResponse | None:
    """Answer greetings and routine catalogue questions without a model round trip."""
    question = normalize("NFKC", message).strip().rstrip("。?!").strip()
    if question.lower() in {"你好", "您好", "嗨", "哈喽", "hi", "hello"}:
        return ChatResponse(
            status="ok", answer="你好！想查哪个滑坡编号，或看看目录统计？",
            traceId=trace_id, version=catalog.version,
        )
    if reply := _classification_reply(question, retriever, trace_id, catalog.version):
        return reply
    if question in {"定位一个滑坡", "定位一处滑坡", "定位滑坡", "帮我定位一个滑坡"}:
        return ChatResponse(
            status="ok",
            answer="请输入要定位的滑坡编号。也可以问「有哪些」，我会列出几条目录记录。",
            traceId=trace_id,
            version=catalog.version,
        )
    slide_id = SLIDE_ID_REQUEST.fullmatch(question)
    type_query = TYPE_DIRECTORY_REQUEST.fullmatch(question)
    if slide_id:
        result = execute_tool("locate_slide", {"id": slide_id[1]}, catalog, retriever)
        if result.ok:
            record = result.records[0]
            longitude = "西经" if record.longitude < 0 else "东经"
            latitude = "南纬" if record.latitude < 0 else "北纬"
            kind = prompts.CATALOG_TYPE_NAMES[record.type]
            area = f"面积 {record.areaKm2:,.2f} km²。" if record.areaKm2 is not None else ""
            answer = (
                f"已找到「{record.name}」，类型为{kind}（{record.type}）。"
                f"位置在{longitude} {abs(record.longitude):.4f}°、{latitude} {abs(record.latitude):.4f}°。"
                f"{area}点击下方按钮可在火星球上定位。"
            )
        else:
            answer = f"{result.payload['error']} 可以问「有哪些」查看目录。"
    elif type_query or CATALOGUE_COUNTS.fullmatch(question) or question in {
        "滑坡数量统计", "统计滑坡数量", "帮我统计一下滑坡数量", "按类型统计滑坡数量",
    }:
        if type_query:
            code = TYPE_CODES.get(type_query[1], type_query[1].upper())
            kind = prompts.CATALOG_TYPE_NAMES[code]
            args = {"type": code}
        else:
            kind = "滑坡"
            args = {}
        if type_query and type_query[2] == "有哪些":
            result = execute_tool("search_catalog", {**args, "limit": 5}, catalog, retriever)
            total = result.payload["total"]
            answer = f"当前目录有 {total:,} 条{kind}滑坡。"
            if total:
                answer += f"下面列出前 {len(result.records)} 条。"
        else:
            result = execute_tool("get_stats", args, catalog, retriever)
            answer = f"当前目录收录 {result.payload['total']:,} 条{kind}{'滑坡' if type_query else ''}。"
            if not type_query:
                counts = result.payload["byType"]
                answer += "、".join(
                    f"{name} {counts[code]:,} 条" for code, name in prompts.CATALOG_TYPE_NAMES.items()
                ) + "。"
    elif question in {"有哪些", "有哪些滑坡", "滑坡有哪些", "火星滑坡有哪些", "看看滑坡目录", "列出一些滑坡"}:
        result = execute_tool("search_catalog", {"limit": 5}, catalog, retriever)
        total = result.payload["total"]
        if total:
            answer = f"当前目录收录了 {total:,} 条滑坡记录。下面列出前 {len(result.records)} 条，点击卡片可以查看详情。"
        else:
            answer = "当前加载的滑坡目录为空。"
    else:
        return None
    citations = {(item.title, item.url): item for item in result.citations}
    return ChatResponse(
        status="ok",
        answer=answer,
        records=result.records,
        citations=list(citations.values()),
        actions=result.actions,
        toolCalls=[ToolCallRecord(name=result.name, args=result.args, ok=result.ok)],
        traceId=trace_id,
        version=catalog.version,
    )


def iter_agent_events(
    request,
    catalog,
    retriever,
    tracer: TraceRecorder,
    max_rounds: int = MAX_TOOL_ROUNDS,
    *,
    recognition_context: RecognitionTaskContext | None = None,
):
    trace_id = tracer.new_id()
    if request.recognitionTaskId is not None:
        yield from task_lookup_events(request.recognitionTaskId, recognition_context, tracer, trace_id, catalog.version)
        return
    latest = request.messages[-1]
    if latest.role == "user":
        reply = _catalog_short_reply(latest.content, catalog, retriever, trace_id)
        if reply is not None:
            tracer.record(trace_id, {"status": reply.status, "tools": [entry.name for entry in reply.toolCalls]})
            for call in reply.toolCalls:
                yield {"type": "status", "tool": call.name, "ok": call.ok}
            yield {"type": "done", "response": reply.model_dump()}
            return
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
                        "content": json.dumps(result.model_payload, ensure_ascii=False),
                    }
                )
        tracer.record(trace_id, {"status": "exceeded_rounds"})
        yield {"type": "done", "response": build("ok", EXCEEDED_MESSAGE).model_dump()}
    except Exception:  # Do not retain upstream exception text, URLs or credentials.
        tracer.record(trace_id, {"status": "error", "code": "AGENT_UPSTREAM_ERROR"})
        yield {"type": "error", "response": build("error", ERROR_MESSAGE).model_dump()}


def run_agent(request, catalog, retriever, tracer: TraceRecorder, max_rounds: int = MAX_TOOL_ROUNDS,
              *, recognition_context: RecognitionTaskContext | None = None):
    final: ChatResponse | None = None
    for event in iter_agent_events(request, catalog, retriever, tracer, max_rounds,
                                   recognition_context=recognition_context):
        if event["type"] in ("done", "error"):
            final = ChatResponse.model_validate(event["response"])
    if final is None:  # pragma: no cover - the generator always terminates
        final = ChatResponse(status="error", answer=ERROR_MESSAGE)
    return final
