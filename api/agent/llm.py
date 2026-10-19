"""DeepSeek (OpenAI-compatible) client wrapper.
Credentials are read from server-side environment variables only.
"""
import os
try:
    from openai import OpenAI
    SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - openai is an optional runtime dependency
    OpenAI = None
    SDK_AVAILABLE = False
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
def settings() -> dict:
    return {
        "base_url": os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        "api_key": (
            os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ),
        "timeout": float(os.environ.get("LLM_TIMEOUT", "30")),
    }
def is_configured() -> bool:
    return SDK_AVAILABLE and bool(settings()["api_key"])
def model_name() -> str:
    return settings()["model"]
def get_client():
    if not is_configured():
        return None
    config = settings()
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=config["timeout"],
    )
