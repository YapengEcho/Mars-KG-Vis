# CTX 裁图滑坡检测与四分类接口

第 5 周（11 月 16—22 日）后端联调约定。输入是一张 CTX 裁图，服务先找候选框，再对保留的每个框运行现有 SL、SP、FL、CL 四分类。当前结果需要人工复核。

## 提交与读取

| 操作 | 请求 |
| --- | --- |
| 查看服务和限制 | `GET /api/capabilities` 的 `detection` 包含 `available`、`modelVersion`、`preprocessingVersion`、`maxImageSide`、`maxImagePixels`、`maxUploadBytes`、`maxCandidates` |
| 提交 | `POST /api/landslide/detect`，`multipart/form-data` 只带一个 `file`，返回 `202`、任务对象与 `Location` |
| 查询状态 | `GET /api/recognition/tasks/{taskId}`，每约 1 秒一次，到 `completed` 或 `failed` 即停止 |
| 读取预览 | `GET /api/recognition/tasks/{taskId}/image`，返回与任务输入宽高相同的 JPEG；使用带授权的请求加载为 Blob |
| 历史列表 | `GET /api/recognition/tasks?kind=detect&page=1&pageSize=20`；已有客户端省略 `kind` 时只看到单图分类 |
| 删除 | `DELETE /api/recognition/tasks/{taskId}`，已结束任务返回 `204`，处理中返回 `409 TASK_BUSY` |

提交、查询、预览、列表、删除均复用[识别设备凭据](recognition.md)，必须带 `Authorization: Bearer <device-token>`。用服务端原任务编号回看，直接知道别人的任务编号无法读取。桌面浏览器关闭标签后，当前会话凭据的保存方式仍由客户端负责。生产服务遵循 API 启动说明中的访问要求。

接受单帧 PNG/JPEG，文件最多 8 MiB，最长边 2048 像素，总像素不超过 2048×2048。先应用 EXIF 方向。不接受整幅 CTX 图幅、坐标、地理参考、ROI 或自定义推理参数。

## 结果与绘图

任务 `kind=detect`，`input` 含 `width`、`height`、`preprocessingVersion`。任务顶层 `modelVersion` 是检测器版本。完成后 `result` 含：

| 字段 | 含义 |
| --- | --- |
| `outcome` | `detections` 表示有候选，`no_detections` 表示运行完成但阈值下未找到保留框 |
| `coordinateSpace` | 固定为 `image_pixels` |
| `truncated` | 去重后的候选超过上限时为 `true` |
| `candidates` | 每个候选含 `candidateId`、`box`、`detectionScore` 和 `classification` |

框是 `[x0, y0, x1, y1]`，以完成方向修正的输入图左上角为原点，右边和下边为开区间，允许等于图宽和图高。梁先楷可用 `input.width/height` 作为 SVG `viewBox` 或绘图基准，并与预览应用同一缩放和位移。框编号在当前任务内稳定。`classification` 复用[分类结果字段](recognition.md)，包含分类模型版本、预处理版本、四类分数与复核标记。

`queued`、`running`、`failed` 均为 `result=null`。失败时显示 `errorCode`，完成但无框时显示「本次没有检出的候选」，有框时默认显示「待复核」。检测分数和类别分数分开显示。助手读取已授权检测任务时已按「有候选 / 未检出 / 失败」三种情况作答，有候选一律标「待复核」（口径见[任务查询约定](agent-recognition.md)）；手机端传递任务编号与设备凭据、以及「打开原记录」跳转到检测模式仍待联调。分类模型对框内影像给出候选类型，单从分数不能确认地质类别或自动解释影像特征。

`401 AUTH_REQUIRED` 表示凭据缺失或失效；访问别人或不存在的任务统一为 `404 TASK_NOT_FOUND`。`413` 为尺寸或上传上限，`415` 为格式问题，`422` 为无效图片或表单，`429 QUEUE_FULL` 为队列满。缺少存储、检测器或分类器返回 `503`，具体原因以响应错误码和能力接口为准。停止页面轮询后服务端任务继续运行。

## 首版运行范围

服务以 1024 像素窗口滑动，最大步长 768，贴合图像边缘，最多运行 9 个窗口。按当前版本的置信度 0.20 筛选各窗口候选，映射回输入后以 IoU 0.50 去重，最多分类 12 个框。每个框按现有四分类预处理处理，并保留少量周围影像。当前阈值和窗口是联调设置，模型训练和独立测试沿用各自的实验协议。

检测和分类共用一个顺序工作进程、最多 8 个排队或运行任务；上限约束工作量。重启后未完成的任务标为 `PROCESS_RESTARTED`，已完成结果可回看。配置及权重版本见 [API 运行说明](../api/README.md)。

滕佳杰提供的演示图按允许公开的范围选取，以 CTX 裁图为单位。没有公开许可的图只在组内联调，发布时仅替换经确认的演示资料。当前公开的 224 像素分类裁片用于单目标分类示例。
