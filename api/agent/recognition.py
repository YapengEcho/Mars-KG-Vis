"""Authorized task lookup bridge; natural-language tool integration is separate."""
import re

from api.agent.schemas import AgentAction, ChatResponse, ToolCallRecord
from api.agent.tools import ToolResult
from api.recognition_access import RecognitionAccessError, RecognitionTaskContext, TaskSummary, auth_required


LOOKUP_NAME = 'get_recognition_task'
ERROR_MESSAGES = {
    'AUTH_REQUIRED': '请从本设备的识别记录中选择任务，再让助手查看。',
    'TASK_NOT_FOUND': '这条记录不存在，或不属于当前设备及选中的任务。',
    'INVALID_TOOL_ARGS': '任务查询参数不正确，请重新选择记录。',
    'UNSUPPORTED_TASK_KIND': '请从单图分类记录中选择任务。多候选检测结果可在原记录中查看。',
    'RECOGNITION_UNAVAILABLE': '识别记录暂时无法读取，请稍后重试。',
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


def task_lookup_events(task_id, context, tracer, trace_id, version):
    """Return original data deterministically until the explanation adapter lands.

    No provider call is made here. This gives the APP a working, authenticated
    handshake without falsely advertising the still-pending language feature.
    """
    result = query_recognition_task(context, {'taskId': task_id})
    response = ChatResponse(status='ok' if result.ok else 'error', answer='',
                            actions=result.actions, traceId=trace_id, version=version,
                            toolCalls=[ToolCallRecord(name=result.name, args=result.args, ok=result.ok)])
    if result.ok:
        summary = TaskSummary.model_validate(result.payload)
        response.recognitionTask = summary
        if summary.result is not None:
            scores = '，'.join(f'{kind} {score}' for kind, score in summary.result.probabilities.items())
            prefix = '该裁片已通过滑坡筛查。' if summary.screening else '这是一条仅做四分类的历史记录。'
            response.answer = (f'{prefix}候选类型：{summary.result.prediction}。四类分数：{scores}。'
                               f'模型版本：{summary.modelVersion}。结果待复核。')
        elif summary.status == 'completed' and summary.screening:
            response.answer = {
                'no_landslide': '筛查未发现明显滑坡特征，这张裁片没有进入四分类。',
                'unsupported_image': '这张图不符合灰度 CTX 裁片的输入要求，未进行滑坡分类。',
                'inconclusive': '筛查未能给出结果，这张裁片没有进入四分类。',
            }.get(summary.screening.decision, '这条记录没有类型结果。')
        elif summary.status in ('queued', 'running'):
            response.answer = '任务正在排队，尚无分类结果。' if summary.status == 'queued' else '任务正在识别，尚无分类结果。'
        else:
            response.answer = '这次识别未完成，没有分类结果。请打开原记录查看失败状态。'
        tracer.record(trace_id, {'status': 'task_lookup', 'tools': [LOOKUP_NAME]})
    else:
        code = result.payload['error']['code']
        response.answer = ERROR_MESSAGES.get(code, '识别记录暂时无法读取，请稍后重试。')
        tracer.record(trace_id, {'status': 'error', 'tools': [LOOKUP_NAME], 'code': code})
    yield {'type': 'status', 'tool': result.name, 'ok': result.ok}
    yield {'type': 'done' if result.ok else 'error', 'response': response.model_dump()}
