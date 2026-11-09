"""Tool schemas and execution for the Mars landslide assistant."""
from dataclasses import dataclass, field
from api.schemas import Catalog, Slide
from api.catalog_query import filter_records, parse_bbox
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
                    "bbox": {"type": "string", "description": "west,south,east,north；单位度；west>east表示跨180度经线"},
                    "q": {"type": "string", "description": "编号、名称或摘要关键词"},
                    "type": {
                        "type": "string",
                        "enum": ["SL", "SP", "FL", "CL"],
                        "description": "滑坡类型",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "返回条数，默认10"},
                    "offset": {"type": "integer", "minimum": 0, "description": "已读取条数，默认0，用于继续查询"},
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
                    "bbox": {"type": "string", "description": "west,south,east,north，与目录查询使用同一范围"},
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
            "description": "检索图鉴资料文本：类型说明、类型对比、空间分布、规模与流动性、成因诱发、研究方法与限制、研究简史。",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "检索词或问题"},
                    "doc_type": {
                        "type": "string",
                        "enum": [
                            "type_intro",
                            "compare",
                            "distribution",
                            "mobility",
                            "method",
                            "history",
                            "overview",
                        ],
                        "description": "资料类型：type_intro 类型说明 / compare 类型对比 / distribution 空间分布 / mobility 规模与流动性 / method 方法与限制 / history 研究简史 / overview 编目总览",
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
    if not isinstance(args, dict):
        return _error(name, {}, "工具参数必须为对象。")
    if name in ("search_catalog", "get_stats"):
        if args.get("type") is not None and args["type"] not in ("SL", "SP", "FL", "CL"):
            return _error(name, args, "未知滑坡类型。")
        if args.get("q") is not None and (not isinstance(args["q"], str) or len(args["q"]) > 200):
            return _error(name, args, "查询词须为不超过200字符的文本。")
    if name == "search_catalog":
        try:
            matched = filter_records(catalog, args.get("q"), args.get("type"), parse_bbox(args.get("bbox")))
        except ValueError as error:
            return _error(name, args, str(error))
        limit = args.get("limit", 10)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            return _error(name, args, "limit 须为1到100的整数。")
        offset = args.get("offset", 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            return _error(name, args, "offset 须为非负整数。")
        page = matched[offset:offset + limit]
        return ToolResult(
            name=name,
            args=args,
            ok=True,
            payload={
                "total": len(matched),
                "records": [_brief(record) for record in page],
                "dataMode": catalog.mode,
                "version": catalog.version,
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
        try:
            matched = filter_records(catalog, args.get("q"), args.get("type"), parse_bbox(args.get("bbox")))
        except ValueError as error:
            return _error(name, args, str(error))
        counts = {kind: 0 for kind in ("SL", "SP", "FL", "CL")}
        for record in matched:
            counts[record.type] += 1
        return ToolResult(
            name=name,
            args=args,
            ok=True,
            payload={"total": len(matched), "byType": counts, "dataMode": catalog.mode, "version": catalog.version},
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
