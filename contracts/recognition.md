# 单图分类与识别记录接口

本接口接收**已经裁好的单滑坡图片**，输出 SL、SP、FL、CL 四类分数。它不判断整幅图片有没有滑坡，也不在 CTX 大图里寻找目标。检测接分类安排在下一阶段。

## 能力与运行条件

先请求 `GET /api/capabilities`。`classifier.available=true` 才允许提交；`historyAvailable=true` 表示可以读取历史记录，即使当前没有加载权重。未配置权重时，分类入口仍返回 `503 MODEL_NOT_CONFIGURED`，不会生成模拟预测。

服务使用 DINOv2-base、最后两块 q/v rank4 LoRA 和线性头。部署需同时提供 base 配置与权重、适配参数、训练时的特征标准化参数，以及包含 SHA256 的私有 manifest。启动时检查全部资源，校验失败则拒绝启动。加载只访问本地文件。

配置方法见 [API 说明](../api/README.md)。任务单独保存在 `MARS_RECOGNITION_DB`，不写入地质目录，也不修改专家标签。

## 设备凭据

`POST /api/recognition/sessions` 无请求体，返回 `201`：

```json
{"token":"<device-token>","tokenType":"Bearer","expiresAt":"<UTC-time>"}
```

后续请求必须带 `Authorization: Bearer <device-token>`。凭据有效期 90 天，服务端只保存其哈希。每次打开页面不要重新创建；用原凭据才能查到原记录。APP 应把它保存在设备的受保护存储中，调试网页可采用会话存储。凭据丢失或到期后不自动找回旧记录；本版没有账号、跨设备同步或密码找回。

凭据不能放进 URL、日志、聊天内容、LLM 提示词或 Git。`sessionId` 聊天编号不能代替它。所有识别响应使用 `Cache-Control: no-store`。

## 提交和查询

| 请求 | 行为 |
| --- | --- |
| `POST /api/landslide/classify` | multipart 上传，返回 `202` 和任务对象；`Location` 指向查询地址 |
| `GET /api/recognition/tasks/{taskId}` | 返回当前设备的任务和结果 |
| `GET /api/recognition/tasks?page=1&pageSize=20` | 当前设备的历史记录，按创建顺序倒序；每页最多 100 条 |
| `GET /api/recognition/tasks/{taskId}/image` | 读取去除元数据的 JPEG 预览，最长边 1024 |
| `DELETE /api/recognition/tasks/{taskId}` | 删除已完成或失败的记录和预览；成功 `204`，处理中返回 `409 TASK_BUSY` |

上传字段：

- `file`：一张单帧 PNG 或 JPEG；文件不超过 8 MiB，总像素不超过 4096×4096，最长边 8192。服务在解析 multipart 前限制流式请求体，另留 64 KiB 表单开销。
- `roi`：可选 JSON 字符串 `[x0,y0,x1,y1]`。坐标以**应用 EXIF 朝向后的原图**左上角为原点，单位像素，右／下边界允许等于图宽／高；必须在图内且有正面积。
- 不传 `roi` 时将整张裁图作为待分类区域，记录 `roiSource=whole_image_crop`；传入时记录 `provided_box`。ROI 不是自动检测框。
- 只接受 `file` 和 `roi` 两个字段。

返回对象含 `taskId`、`kind=classify`、`status`、UTC 时间、`modelVersion`、`preprocessingVersion`、`input`、`result`、`errorCode`、`imageUrl`。`input` 保存原图尺寸、ROI、缩放尺寸与补边，不保存上传文件名或磁盘路径。

任务状态为 `queued → running → completed/failed`。首次 `202` 响应也可能已经是 `running` 或 `completed`；客户端检查状态，每隔约 1 秒查询，完成或失败后停止。离开页面可以停止查询，之后从历史列表恢复，不必重复上传。HTTP 查询超时不代表推理被取消。

`result` 在完成前为 `null`。完成示例（数值仅用于说明结构）：

```json
{
  "prediction": "CL",
  "probabilities": {"SL": 0.1, "SP": 0.2, "FL": 0.3, "CL": 0.4},
  "confidence": 0.4,
  "uncertain": true,
  "reviewReasons": ["MODEL_REQUIRES_HUMAN_REVIEW"],
  "reviewPolicy": "mandatory-review-v1",
  "scoresCalibrated": false,
  "modelVersion": "<frozen-model-version>",
  "modelDigest": "<manifest-sha256>",
  "preprocessingVersion": "gray-letterbox224-roiarea-v1"
}
```

四类分数和为 1，但未经校准，不能解释为地质真实性的概率。`prediction` 是分数最高的类型。当前模型尚不足以自动核准，**所有结果都保留“待复核”**，并非按一个临时阈值宣布高分结果可靠。Agent 必须按原值解释，并保留待复核状态，不得据此改目录标签。

`imageUrl` 同样需要 Bearer 凭据；网页应使用带授权头的 `fetch` 获取 Blob 再展示，不能直接用无授权的 `<img src>` 请求。其他设备访问一个真实任务与访问不存在的任务都返回同样的 `404 TASK_NOT_FOUND`。

## 固定预处理

版本 `gray-letterbox224-roiarea-v1`：应用 EXIF 朝向 → 转灰度 → 按比例双线性缩放至最长边 224 → 居中补黑边为 224×224 → 复制为三通道 → ImageNet 均值／标准差归一化。ROI 使用实际缩放尺寸同步映射。

DINOv2 生成 16×16 个 patch 特征；按每个 14×14 patch 与 ROI 的交叠面积加权汇总，不使用 CLS token，不把补边纳入 ROI 汇总权重。随后使用训练时固定的 768 维均值和标准差，再送入线性头、softmax。补边仍可能通过编码器影响特征，本实现不宣称非方形上传与方形训练裁图完全等效。

已有 224 方形裁片传入原 ROI 时可复现原推理；未给 ROI 的整图汇总是明确的新输入约定，不冒充人工框汇总。大图缩小后细节会减少，因此上传上限不是精度保证。

## 错误与重启

| 情况 | HTTP／状态 |
| --- | --- |
| 未配置模型／存储／工作进程不可用 | `503 MODEL_NOT_CONFIGURED / RECOGNITION_NOT_CONFIGURED / WORKER_UNAVAILABLE` |
| 缺少、错误或到期的设备凭据 | `401 AUTH_REQUIRED` |
| 文件／请求体／尺寸超限 | `413 UPLOAD_TOO_LARGE / IMAGE_DIMENSIONS_EXCEEDED` |
| 非 PNG/JPEG、多帧图 | `415 UNSUPPORTED_IMAGE / ANIMATED_IMAGE` |
| 空图、坏图、缺图、ROI 非法 | `422 EMPTY_IMAGE / INVALID_IMAGE / IMAGE_REQUIRED / INVALID_ROI` |
| 表单字段或 multipart 非法 | `422 INVALID_FORM` 或 `400` 解析错误 |
| 待处理任务已满（全服务最多 8 个） | `429 QUEUE_FULL`，`Retry-After: 5` |
| 推理失败 | 任务 `failed`，`errorCode=INFERENCE_FAILED` |
| 重启前未完成 | 任务 `failed`，`errorCode=PROCESS_RESTARTED`；用户可重新提交 |

任务被接受后不依赖上传连接继续存活。完成时保存结果并清除临时模型输入，只保留元数据和预览；已完成结果在服务重启、更换模型后仍按原版本读取。

同一个任务数据库只允许一个 API 工作进程，使用操作系统文件锁防止重复处理；启动命令指定 `--workers 1`。正常退出等待当前推理最多 30 秒，超时不能强行中断线程；异常退出后由下次启动标记中断任务失败。本版不提供 GPU 强制取消或硬执行超时。

## 前端和 Agent 对接

梁先楷：能力检查 → 获取／恢复凭据 → 选择单滑坡图片 → 提交 → 轮询 → 展示四类分数、待复核及版本；“我的”读取同一凭据下的列表、详情与预览。不能继续把 `202` 任务对象当作同步分类结果。

周璇：结果查询工具在服务端携带当前设备的授权头读取任务，再把 `result` 的数值和版本交给 Agent 解释；凭据不进入模型上下文。不凭 `taskId` 绕过访问校验，也不自行推断缺失结果。

当前网页已完成上述任务与历史交互；凭据使用会话存储，刷新保留，关闭标签页后不恢复。原生设备安全存储、真机联调及Agent结果工具仍待验收。公开仓库另含8张允许公开的训练示例，不包含真实权重、完整目录、其余影像或运行数据库。
