"""Tool schemas and execution for the Mars landslide assistant."""
from dataclasses import dataclass, field
from api.schemas import Catalog, Slide
from api.agent.schemas import AgentAction, Citation
@dataclass
class ToolResult:
    name: str
    args: dict
    ok: bool
    payload: dict
    records: list[Slide] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    actions: list[AgentAction] = field(default_factory=list)
def _error(name: str, args: dict, message: str) -> ToolResult:
    return ToolResult(name=name, args=args, ok=False, payload={"error": message})
def _record_haystack(record: Slide) -> str:
    return " ".join(
        (
            record.id,
            "" if record.sourceId is None else str(record.sourceId),
            record.name,
            record.summary,
        )
    ).casefold()
def _filter(catalog: Catalog, q: str | None, slide_type: str | None) -> list[Slide]:
    records = catalog.records
    if slide_type:
        records = [record for record in records if record.type == slide_type]
    term = (q or "").strip().casefold()
    if term:
        records = [record for record in records if term in _record_haystack(record)]
    return records
def _brief(record: Slide) -> dict:
    return {
        "id": record.id,
        "name": record.name,
        "type": record.type,
        "longitude": record.longitude,
        "latitude": record.latitude,
        "areaKm2": record.areaKm2,
        "summary": record.summary,
        "source": record.source.title,
    }
def _source_citation(record: Slide) -> Citation:
    return Citation(
        refType="source",
        refId=record.id,
        title=record.source.title,
        url=record.source.url,
    )
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "按关键词或类型检索火星滑坡目录，返回记录列表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "编号、名称或摘要关键词"},
                    "type": {
                        "type": "string",
                        "enum": ["SL", "SP", "FL", "CL"],
                        "description": "滑坡类型",
                    },
                    "limit": {"type": "integer", "description": "返回条数，默认10"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_slide",
            "description": "按编号获取单条滑坡记录的完整详情。",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "滑坡记录编号"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": "按可选条件统计滑坡总数与各类型数量。",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string"},
                    "type": {"type": "string", "enum": ["SL", "SP", "FL", "CL"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "检索图鉴、类型说明、对比与案例等资料文本。",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "检索词或问题"},
                    "doc_type": {
                        "type": "string",
                        "enum": ["type_intro", "compare", "case"],
                    },
                    "limit": {"type": "integer", "description": "返回条数，默认5"},
                },
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "locate_slide",
            "description": "在探索页地图（火星球）上定位并聚焦指定编号的滑坡。",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
]
def execute_tool(name: str, args: dict, catalog: Catalog, retriever) -> ToolResult:
    args = args or {}
    if name == "search_catalog":
        matched = _filter(catalog, args.get("q"), args.get("type"))
        limit = int(args.get("limit", 10) or 10)
        page = matched[:limit]
        return ToolResult(
            name=name,
            args=args,
            ok=True,
            payload={
                "total": len(matched),
                "records": [_brief(record) for record in page],
                "dataMode": catalog.mode,
            },
            records=page,
            citations=[_source_citation(record) for record in page],
        )
    if name == "get_slide":
        slide_id = str(args.get("id", ""))
        for record in catalog.records:
            if record.id == slide_id:
                return ToolResult(
                    name=name,
                    args=args,
                    ok=True,
                    payload=_brief(record),
                    records=[record],
                    citations=[_source_citation(record)],
                )
        return _error(name, args, f"未找到编号为 {slide_id} 的记录。")
    if name == "get_stats":
        matched = _filter(catalog, args.get("q"), args.get("type"))
        counts = {kind: 0 for kind in ("SL", "SP", "FL", "CL")}
        for record in matched:
            counts[record.type] += 1
        return ToolResult(
            name=name,
            args=args,
            ok=True,
            payload={"total": len(matched), "byType": counts, "dataMode": catalog.mode},
        )
    if name == "search_docs":
        query = str(args.get("q", ""))
        limit = int(args.get("limit", 5) or 5)
        docs = retriever.search(query, doc_type=args.get("doc_type"), limit=limit)
        if not docs:
            return ToolResult(
                name=name,
                args=args,
                ok=True,
                payload={"results": [], "message": "现有资料中没有相关内容。"},
            )
        return ToolResult(
            name=name,
            args=args,
            ok=True,
            payload={
                "results": [
                    {
                        "doc_id": doc.doc_id,
                        "doc_type": doc.doc_type,
                        "title": doc.title,
                        "text": doc.text,
                        "source": doc.source_title,
                    }
                    for doc in docs
                ]
            },
            citations=[
                Citation(
                    refType="doc",
                    refId=doc.doc_id,
                    title=doc.source_title or doc.title,
                    url=doc.source_url,
                )
                for doc in docs
            ],
        )
    if name == "locate_slide":
        slide_id = str(args.get("id", ""))
        if any(record.id == slide_id for record in catalog.records):
            return ToolResult(
                name=name,
                args=args,
                ok=True,
                payload={"located": slide_id},
                actions=[AgentAction(type="locate", slideId=slide_id)],
            )
        return _error(name, args, f"未找到编号为 {slide_id} 的记录，无法定位。")
    return _error(name, args, f"未知工具：{name}")
