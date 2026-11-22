"""Authorized task lookup bridge; natural-language tool integration is separate."""
import re

from api.agent.schemas import AgentAction, ChatResponse, ToolCallRecord
from api.agent.tools import ToolResult
from api.agent import prompts
from api.recognition_access import (
    DetectTaskSummary,
    RecognitionAccessError,
    RecognitionTaskContext,
    TaskSummary,
    auth_required,
)


LOOKUP_NAME = 'get_recognition_task'
ERROR_MESSAGES = {
    'AUTH_REQUIRED': '请从本设备的识别记录中选择任务，再让助手查看。',
    'TASK_NOT_FOUND': '这条记录不存在，或不属于当前设备及选中的任务。',
    'INVALID_TOOL_ARGS': '任务查询参数不正确，请重新选择记录。',
    'UNSUPPORTED_TASK_KIND': '这条记录的类别暂不支持在助手中查看，请打开原记录。',
    'RECOGNITION_UNAVAILABLE': '识别记录暂时无法读取，请稍后重试。',
}
# queued / running 对单图分类和检测是同一种说法。
PENDING_MESSAGES = {
    'queued': '任务正在排队，尚无结果。',
    'running': '任务正在运行，尚无结果。',
}
SCREENING_MESSAGES = {
    'no_landslide': '筛查未发现明显滑坡特征，这张裁片没有进入四分类。',
    'unsupported_image': '这张图不符合灰度 CTX 裁片的输入要求，未进行滑坡分类。',
    'inconclusive': '筛查未能给出结果，这张裁片没有进入四分类。',
}


def query_recognition_task(context: RecognitionTaskContext | None, args: dict) -> ToolResult:
    """The tool adapter receives internal context and a taskId-only argument."""
    if (not isinstance(args, dict) or set(args) != {'taskId'} or
            not isinstance(args.get('taskId'), str) or not re.fullmatch(r'[a-f0-9]{32}', args['taskId'])):
        return ToolResult(name=LOOKUP_NAME, args={}, ok=False,
                          payload={'error': {'code': 'INVALID_TOOL_ARGS', 'message': 'Invalid task query.'}})
    safe_args = {'taskId': args['taskId']}
    try:
        if context is None:
            raise auth_required()
        summary = context.read(args['taskId'])
    except RecognitionAccessError as error:
        return ToolResult(name=LOOKUP_NAME, args=safe_args, ok=False,
                          payload={'error': {'code': error.code, 'message': error.message}})
    return ToolResult(name=LOOKUP_NAME, args=safe_args, ok=True, payload=summary.model_dump(mode='json'),
                      actions=[AgentAction(type='open_task', taskId=summary.taskId)])


def _score_text(scores: dict) -> str:
    return '、'.join(f'{kind} {value:.2f}' for kind, value in scores.items())


def _type_label(code: str) -> str:
    name = prompts.CATALOG_TYPE_NAMES.get(code)
    return f'{code} {name}' if name else code


def _classification_answer(summary: TaskSummary) -> str:
    if summary.result is not None:
        prefix = '该裁片已通过滑坡筛查。' if summary.screening else '这是一条仅做四分类的历史记录。'
        return (f'{prefix}候选类型：{_type_label(summary.result.prediction)}。'
                f'四类分数：{_score_text(summary.result.probabilities)}。'
                f'模型版本：{summary.modelVersion}。结果待复核。')
    if summary.status == 'completed' and summary.screening:
        return SCREENING_MESSAGES.get(summary.screening.decision, '这条记录没有类型结果。')
    if summary.status in PENDING_MESSAGES:
        return PENDING_MESSAGES[summary.status]
    return '这次识别未完成，没有分类结果。请打开原记录查看失败状态。'


def _detection_answer(summary: DetectTaskSummary) -> str:
    """检测任务：候选、未检出、失败三种情况分开说，且不把候选说成已确认类型。"""
    if summary.status in PENDING_MESSAGES:
        return PENDING_MESSAGES[summary.status]
    if summary.status == 'failed':
        code = summary.errorCode or '未知错误'
        return f'这次检测未完成（错误码 {code}），没有候选结果。请打开原记录查看失败原因。'
    result = summary.result
    if result is None:
        return '这次检测没有返回结果。请打开原记录查看状态。'
    if result.outcome == 'no_detections':
        return ('本次检测运行完成，但在当前阈值下没有找到候选框（未检出）。'
                '未检出不代表图上一定没有滑坡，可以换一张分辨率更高的裁片再试。')
    lines = [f'本次检测找到 {len(result.candidates)} 个候选框，均需人工复核：']
    for candidate in result.candidates:
        lines.append(
            f'· 候选 {candidate.candidateId}：分类为 {_type_label(candidate.classification.prediction)}，'
            f'四类分数 {_score_text(candidate.classification.probabilities)}，'
            f'检测分 {candidate.detectionScore:.2f}，待复核。'
        )
    if result.truncated:
        lines.append('候选数超过本次上限，这里只列出返回的部分。')
    lines.append(f'检测模型版本：{summary.modelVersion}；框坐标以输入图像素为单位，可在原记录中查看。')
    return '\n'.join(lines)


def _task_answer(summary: TaskSummary | DetectTaskSummary) -> str:
    return _detection_answer(summary) if summary.kind == 'detect' else _classification_answer(summary)


def _parse_summary(payload: dict) -> TaskSummary | DetectTaskSummary:
    model = DetectTaskSummary if payload.get('kind') == 'detect' else TaskSummary
    return model.model_validate(payload)


def task_lookup_events(task_id, context, tracer, trace_id, version):
    """Return original data deterministically; natural-language explanation is separate.

    No provider call is made here. This gives the APP a working, authenticated
    handshake without falsely advertising the still-pending language feature.
    """
    result = query_recognition_task(context, {'taskId': task_id})
    response = ChatResponse(status='ok' if result.ok else 'error', answer='',
                            actions=result.actions, traceId=trace_id, version=version,
                            toolCalls=[ToolCallRecord(name=result.name, args=result.args, ok=result.ok)])
    if result.ok:
        summary = _parse_summary(result.payload)
        response.recognitionTask = summary
        response.answer = _task_answer(summary)
        tracer.record(trace_id, {'status': 'task_lookup', 'tools': [LOOKUP_NAME]})
    else:
        code = result.payload['error']['code']
        response.answer = ERROR_MESSAGES.get(code, '识别记录暂时无法读取，请稍后重试。')
        tracer.record(trace_id, {'status': 'error', 'tools': [LOOKUP_NAME], 'code': code})
    yield {'type': 'status', 'tool': result.name, 'ok': result.ok}
    yield {'type': 'done' if result.ok else 'error', 'response': response.model_dump()}
