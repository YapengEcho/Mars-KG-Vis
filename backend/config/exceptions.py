from rest_framework.views import exception_handler

_TRANSLATE = {
    "Not found.": "未找到该滑坡",
    "Invalid page.": "页码无效",
    "No Landslide matches the given query.": "未找到该滑坡",
}


def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)
    if response is None:
        return None

    data = response.data
    detail = data.get("detail", data) if isinstance(data, dict) else data
    if isinstance(detail, list):
        detail = "; ".join(str(item) for item in detail)
    elif isinstance(detail, dict):
        detail = "; ".join(f"{key}: {value}" for key, value in detail.items())
    else:
        detail = str(detail)

    response.data = {"detail": _TRANSLATE.get(detail, detail)}
    return response
