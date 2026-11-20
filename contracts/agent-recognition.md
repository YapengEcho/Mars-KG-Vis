# 识别结果、图鉴资料与助手：四人对接约定

更新：2026-11-11。已实现设备授权、单任务查询、同步/流式入口和结果白名单。

当前选中识别任务时，后端直接返回保存的状态、分数及待复核说明，**不调用语言模型**。滕佳杰维护解释所依据的图鉴、案例和来源；周璇接自然语言解释与资料引用；梁先楷接页面选择及凭据读取；崔亚鹏维护模型、后端、内容入库和联调。普通目录问答保持原有流程。

## 四人如何衔接

识别结果回答“模型给了哪类、分数多少”，图鉴资料回答“这个类型是什么意思、依据哪里、有哪些案例”。这两部分分别核对，再由助手组织说明。

| 成员 | 提供什么 | 交给谁 |
| --- | --- | --- |
| 崔亚鹏 | 已授权的真实任务摘要、接口和运行环境；经审核的内容入库及资源服务 | 梁接请求/结果，周接任务查询；与滕核对内容版本和可发布范围 |
| 滕佳杰 | 同一版本的四类图鉴、案例关联、逐条引用、资源可用性与缺项清单 | 崔入库，梁展示，周检索与引用 |
| 周璇 | 任务解释、资料检索、可回查的引用和页面动作 | 梁接助手页面；与崔验证授权，与滕核对解释依据 |
| 梁先楷 | 任务选择、设备凭据传递、图鉴/案例/历史跳转、真机验证 | 三人共同核对页面与原始数据是否一致 |

## 滕佳杰：图鉴、案例与来源

继续维护已有的[图鉴正文](../mobile/src/content/atlas.json)，不另写一套助手文案。现有4类说明、12例文本及2组对比可以复用；本周补的是核对、缺项标记和同版本交付。

### 本周交付

| 内容 | 需要核对的字段/事项 | 接收人 |
| --- | --- | --- |
| 四类说明与对比 | SL/SP/FL/CL的名称、定义、可观察特征、易混点、判读限制；每项保留论文定位 | 梁展示；周解释类型 |
| 案例关联 | `cases[].id`、`landslideId`、`type`、标题、观察要点、`citationLocator`；类型内`caseIds`一致 | 崔校验入库；梁跳转；周引用案例 |
| 文献和引用 | `sourceDocument.id/title/url/note`，类型/对比的`documentId/locator`；说明页码是PDF物理页还是印刷页 | 崔存来源；周让每个知识结论能回查 |
| 资源核对与缺项 | 12例逐条登记目录是否可关联、影像是否可公开、图片/类型是否对应、尺度来源、引用是否待核对、缺什么 | 梁显示有图/无图状态；崔安排资源；周避免推荐不存在的影像 |
| 内容版本与改动说明 | 更新`atlas.json.version`，注明改了哪些条目；导出包、网页正文与检索资料使用同一版本 | 崔、梁、周共同确认 |

资源核对表可用Markdown，列为：`caseId | landslideId | 类型 | 目录关联状态 | 公开影像状态 | 尺度来源 | 引用状态 | 缺项/负责人`。它是本次补交的检查表，不是新增运行接口。只核对现有标签和资料对应关系；有地质判断疑点就标出原文与疑点交组长，不要求重新判定专家类型。

当前公开目录与影像覆盖其中8例，另外4例只有已发布的案例文本。缺图时保留“暂无公开影像”的状态，不为凑齐12例上传其余私有图片。保留原坐标及逐张裁片尺度，不批量换算纬度，也不把某张图的米/像素写成全批统一值。

### 用已有脚本交付同一版本

在仓库根目录执行，下面的导出只读已公开图鉴文本，不关联私有目录：

```sh
python scripts/export_atlas_content.py --check
python scripts/export_atlas_content.py --out output/week4_atlas_handoff
```

脚本实际生成六份文件，统一与源`atlas.json`、资源核对表和简短改动说明一起交接：

| 文件 | 用途 |
| --- | --- |
| `atlas-content.json` | 图鉴内容及可选目录关联；崔入库优先使用源`atlas.json`，按[内容接口](content.md)校验 |
| `atlas-cases.csv` | 案例编号、类型、来源和逐张尺度的关联检查；梁核对页面入口，周建立案例回链 |
| `atlas-citations.csv` | 类型/对比/案例到文献编号及定位的登记；周与崔共用 |
| `atlas-search.jsonl` | 当前18条检索文本：4类＋12例＋2组对比，供周适配检索 |
| `atlas-export-summary.json` | 内容版本及数量，三方核对交付是否齐全 |
| `atlas-limits.md` | 判读限制，供页面提示及助手保留不确定性 |

不传`--catalog`时，导出统计中的`casesInCatalog=0`只表示这次没有执行目录关联，不能判成12例都缺失。公开资源覆盖情况以核对表为准。若需要完整目录关联，私下交付；该脚本的`--catalog`当前接受原始记录数组，不能直接传带`records`外壳的API目录对象。

### 给周璇的格式注意事项

`atlas-search.jsonl`的每行是`id/kind/title/text`，现有[检索加载器](../api/agent/retrieval.py)读取的却是`{"docs": [...]}`，字段为`doc_id/doc_type/title/text/slide_ids/source_title/source_url`。**两者还没有自动接通，不能把JSONL路径直接填入`MARS_DOCS_PATH`。** 滕提供源资料，周负责适配，不让滕重写第二份正文。

适配时保留`id→doc_id`、`title/text`；`kind=type`映射`type_intro`、`comparison`映射`compare`。案例用新增的`case`类型，需周同步工具枚举、索引和测试，不能只改枚举。案例的`slide_ids`从`atlas.json`中的`landslideId`关联；标题、URL及页码从`sourceDocument`和引用表补齐，不能由模型猜测。现有`Doc`没有独立定位字段，`source_title`保留文献标题及页码/图号，`source_url`保留来源链接，正文中的定位也保留。限制文本可归入已有`method`类型。以上是接入要求，格式转换及新增案例检索尚未实现。

保留现有知识库里有效的其他资料，按稳定ID更新和去重；这18条图鉴文本不应直接覆盖其余研究方法、分布和历史资料。网页、内容API及检索索引均记录使用的图鉴版本；索引版本和转换日志由周维护。

### 内容怎样用于识别解释

允许解释“CL通常指复合型，图鉴中的判据是……”，并引用出处。任务摘要没有图片判读证据，不能因此说“模型看到了这张图中的某个结构”，也不能把推荐案例说成用户上传图片的真实编号。模型分数、专家标签和文献定义分别保留各自来源；低分、模糊和资料不足时维持待复核。

崔导入前核对案例编号、类别和引用，按实际目录及公开范围选择一致的内容子集。只有8条目录的库不能直接导入全部12例；缺失或未经审核的部分不发布，不能用`--publish`绕过目录/来源检查。

## 梁先楷：发送什么

沿用 `POST /api/agent/chat` 或 `POST /api/agent/chat/stream`。HTTP头带上传这条图片时使用的设备凭据：

```http
Authorization: Bearer <existing-device-token>
Content-Type: application/json
```

请求结构（任务编号仅为格式示例）：

```json
{
  "sessionId": null,
  "recognitionTaskId": "0123456789abcdef0123456789abcdef",
  "messages": [{"role": "user", "content": "请查看这条识别记录"}]
}
```

- `recognitionTaskId`：结果/历史页选中的任务，32位小写十六进制。其他格式返回422。
- `sessionId`：可选聊天编号，不授予访问权。修改它不能查看其他设备记录。
- 普通问答省略`recognitionTaskId`和识别授权头；不需要识别存储或模型权重。
- 识别凭据只从现有设备会话读取，不为聊天重新创建。失效时提示用户，不悄悄换新凭据再读旧任务。
- 请求不能携带owner、token、result或自定义内部上下文。服务端不接受客户端提供的预测分数。

响应沿用`ChatResponse`，增加可空字段`recognitionTask`。普通问答及查询失败时为`null`；查询成功时只含：

| 字段 | 内容 |
| --- | --- |
| `taskId`、`kind`、`status` | 任务标识、classify及queued/running/completed/failed |
| `createdAt`、`finishedAt` | 原任务时间；结束时间可为空 |
| `modelVersion`、`preprocessingVersion` | 该次任务的版本，不替换成当前加载版本 |
| `errorCode` | 任务失败原因，可为空 |
| `result` | 完成时的类型、四类分数、confidence、uncertain、复核原因/策略、校准状态及版本；其他状态为空 |

不返回图片、上传输入元数据、模型资源哈希、访问凭据或服务端路径。四类分数按存储原值返回，不能当正确率。页面以`recognitionTask`为原始结果来源，不从`answer`解析分数。

查询成功时附带已有的`open_task`动作；打开原记录继续调用原有带授权的任务接口。`ChatResponse.status=ok`表示读取成功，不表示识别一定完成：必须查看`recognitionTask.status`。当前完成任务的答复明确提示自动解释尚未接入。

流式入口返回`status`及`done/error`事件，终态事件的`response`与同步结构一致。本轮任务查询不生成`delta`文字流。两个端点及错误响应均使用`Cache-Control: no-store`。

## 错误处理

| 阶段/情况 | 返回 |
| --- | --- |
| 缺失、错误格式、伪造或过期凭据 | HTTP 401 / `AUTH_REQUIRED` |
| 任务不存在、被删或属于其他设备 | 相同HTTP 404 / `TASK_NOT_FOUND`，不透露归属 |
| 未配置任务库 | HTTP 503 / `RECOGNITION_NOT_CONFIGURED` |
| 存储故障或损坏的结果数据 | HTTP 503 / `RECOGNITION_UNAVAILABLE`，不返回异常原文 |
| 任务编号或请求字段非法 | HTTP 422 |
| 预检后、查询前任务被删除或存储不可用 | 同步返回`status=error`；SSE返回终态`error`；两者`recognitionTask=null`、无打开动作 |

入口在创建SSE、调用聊天编排前完成身份与归属预检，所以入口401/404/503仍是普通JSON错误，前端先检查HTTP状态再解析事件。若SSE已经建立，则不能再改HTTP状态，需检查终态事件。

先检查凭据存在与格式，再查询存储。缺少凭据且没有任务库时返回401；格式正确的凭据遇到存储故障返回503，不能在无法查库时判断它是否已过期。

## 周璇：内部调用入口

HTTP入口会生成只在当前请求有效的`RecognitionTaskContext`，通过`recognition_context`关键字传给`run_agent`与`iter_agent_events`。它只保存设备身份哈希和选中任务，不保存原始凭据，不可JSON/pickle序列化，repr不包含身份。不要把它塞入请求模型、工具参数、全局变量、会话缓存或消息历史。

可直接复用：

```python
from api.agent.recognition import query_recognition_task

# context由后端注入，task_id是本次用户选中的任务。
result = query_recognition_task(context, {"taskId": task_id})
if result.ok:
    summary = result.payload  # 已验证且经过字段白名单筛选
else:
    code = result.payload["error"]["code"]
```

内部`ToolResult`记录名为`get_recognition_task`，参数仅允许taskId。其他键、没有上下文或换一个任务编号都失败，即使另一个任务也属于同一设备。每次读取重新查库，不使用预检时缓存的结果。

这不是新增的公网查询URL，也尚未注册到语言模型的`TOOL_SPECS`。当前编排在任务分支调用上述函数后直接返回事实摘要。周接解释时：

1. 保留HTTP预检和本次请求上下文；先查询并检查`result.ok`，再使用结果。
2. 只向模型开放当前选中任务；排队、运行、失败时继续直接返回状态。
3. 完成任务才把白名单结果交给模型解释，继续附带原始`recognitionTask`供页面核对；类型知识使用已有资料工具和引用。
4. 服务端供应商密钥与设备凭据分开。不要转发入站请求头，或把身份哈希、上下文、图片送入语言模型。
5. 追踪记录只存工具名、成功状态和固定错误码。不要把原始异常、请求头和完整任务结果写入日志。

关键代码：[共用认证/查询](../api/recognition_access.py)、[Agent查询桥接](../api/agent/recognition.py)、[HTTP入口](../api/main.py)、[聊天模型](../api/agent/schemas.py)。现有任务详情接口也已复用同一认证与归属查询。

## 验证与待接部分

自动测试使用临时数据库、合成图片与分数，不读取真实权重，不发送供应商请求。覆盖设备A/B隔离、过期凭据、同设备不同任务、并发请求、同步/SSE、预检后删除、无权请求不进入Agent、存储故障、白名单、上下文不可序列化和错误日志脱敏。

运行：`python -m pytest api/tests/test_agent_recognition.py api/tests/test_recognition.py api/tests/test_agent.py -q`。

共同验收时留下一条完整记录：选择已完成任务 → 核对原分数与待复核状态 → 解释候选类型 → 查看来源 → 打开一个有公开影像的同类型案例。另验证缺图案例、没有相关资料及任务失败三种情况，不能补写不存在的图片或依据。

滕交内容版本、六份导出文件、资源核对表及待确认项；周交索引版本、检索/解释和引用记录；梁交页面及真机跳转结果；崔交服务/接口版本和内容导入记录。四人分别确认后，才把这条联合流程记为完成。

仍待滕佳杰完成同版本内容与资源核对交付；周璇完成格式适配、自然语言工具注册、解释和引用；梁先楷完成任务选择、请求头、原生凭据持久化和真机验证。本轮不改动手机页面，不上传任何权重或运行数据库。
