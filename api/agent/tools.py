"""Tool schemas and execution for the Mars landslide assistant."""
import re
from dataclasses import dataclass, field
from api.schemas import Catalog, Slide
from api.catalog_query import filter_records, parse_bbox
from api.agent.schemas import AgentAction, Citation
# Standalone SL/SP/FL/CL: the neighbours must not be ASCII word chars, so
# "ls_0493" and "display" do not match, but "有哪些SL案例" does (中文不是 ASCII 词字符)。
_TYPE_CODE = re.compile(r"(?<![A-Za-z0-9_])(SL|SP|FL|CL)(?![A-Za-z0-9_])", re.IGNORECASE)
# 用户也常用中文类型名提问，映射到同一个类型码。
_TYPE_NAMES = {"滑移型": "SL", "滑塌型": "SP", "流动型": "FL", "复合型": "CL"}


def _wanted_case_type(query: str) -> str | None:
    """Return the single type a question asks about, or None when it is ambiguous.

    问题里只点名一个类型时（SL / 滑移型 任一写法），案例结果只保留该类型；
    同时点名多个类型（如“SL 和 SP 的区别”）则不过滤。
    """
    codes = {match.group(1).upper() for match in _TYPE_CODE.finditer(query)}
    codes.update(code for name, code in _TYPE_NAMES.items() if name in query)
    return codes.pop() if len(codes) == 1 else None
@dataclass
class ToolResult:
    name: str
    args: dict
    ok: bool
    payload: dict
    records: list[Slide] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    actions: list[AgentAction] = field(default_factory=list)

    @property
    def model_payload(self) -> dict:
        """Catalog facts for a reply, without storage and coordinate-validation metadata."""
        facts = {
            key: value for key, value in self.payload.items()
            if key not in {"dataMode", "coordinateSystem", "version"}
        }
        if self.payload.get("dataMode") == "demo":
            facts["isDemo"] = True
        return facts


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


def _find_record(catalog: Catalog, slide_id: str) -> Slide | None:
    record = next((item for item in catalog.records if item.id.casefold() == slide_id.casefold()), None)
    if record is not None:
        return record
    digits = slide_id.casefold().removeprefix("ls_")
    if not digits.isascii() or not digits.isdecimal():
        return None
    number = digits.lstrip("0") or "0"
    return next((item for item in catalog.records if str(item.sourceId) == number), None)


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
                "properties": {"id": {"type": "string", "description": "原始数字编号或目录 ID，如 493 或 ls_0493"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": "按可选条件统计滑坡总数与各类型数量。查询全目录时省略筛选参数。",
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
            "description": "检索图鉴资料文本：类型说明、类型对比、图鉴案例、空间分布、规模与流动性、成因诱发、研究方法与限制、研究简史。",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "检索词或问题"},
                    "doc_type": {
                        "type": "string",
                        "enum": [
                            "type_intro",
                            "compare",
                            "case",
                            "distribution",
                            "mobility",
                            "method",
                            "history",
                            "overview",
                        ],
                        "description": "资料类型：type_intro 类型说明 / compare 类型对比 / case 图鉴案例 / distribution 空间分布 / mobility 规模与流动性 / method 方法与限制 / history 研究简史 / overview 编目总览",
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
                "properties": {"id": {"type": "string", "description": "原始数字编号或目录 ID，如 493 或 ls_0493"}},
                "required": ["id"],
            },
        },
    },
]
def execute_tool(name: str, args: dict, catalog: Catalog, retriever) -> ToolResult:
    args = args or {}
    if not isinstance(args, dict):
        return _error(name, {}, "工具参数必须为对象。")
    if name == "get_stats":
        args = {key: value for key, value in args.items()
                if key not in ("bbox", "q", "type") or not isinstance(value, str) or value.strip()}
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
                "coordinateSystem": catalog.coordinateSystem.model_dump(),
                "version": catalog.version,
            },
            records=page,
            citations=[_source_citation(record) for record in page],
        )
    if name in ("get_slide", "locate_slide"):
        slide_id = str(args.get("id", "")).strip()
        record = _find_record(catalog, slide_id)
        if record is None:
            return _error(name, args, f"当前目录中未找到编号 {slide_id}。")
        metadata = {"dataMode": catalog.mode, "coordinateSystem": catalog.coordinateSystem.model_dump()}
        if name == "locate_slide":
            return ToolResult(
                name=name,
                args=args,
                ok=True,
                payload={"located": record.id, **metadata},
                records=[record],
                citations=[_source_citation(record)],
                actions=[AgentAction(type="locate", slideId=record.id)],
            )
        return ToolResult(
            name=name,
            args=args,
            ok=True,
            payload={**_brief(record), **metadata},
            records=[record],
            citations=[_source_citation(record)],
        )
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
        docs = retriever.search(query, doc_type=args.get("doc_type"), limit=max(limit * 4, 20))
        # 问题里只点名了一个类型时，只保留该类型的案例，否则问“有哪些 SL 案例”
        # 会把 SP/FL/CL 的案例一起带出来（它们的标题都含“案例”）。
        wanted = _wanted_case_type(query)
        if wanted:
            docs = [doc for doc in docs if doc.doc_type != "case" or doc.case_type == wanted]
        docs = docs[:limit]
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
    return _error(name, args, f"未知工具：{name}")
