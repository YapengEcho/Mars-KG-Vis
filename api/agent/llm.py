"""DeepSeek (OpenAI-compatible) client wrapper.
Credentials are read from server-side environment variables only.
An API key is bound to the provider behind the configured address: the default
DeepSeek endpoint only accepts DEEPSEEK_API_KEY, so a machine that has only
OPENAI_API_KEY configured stays unconfigured instead of sending that key to a
different vendor.
"""
import os
from urllib.parse import urlparse
try:
    from openai import OpenAI
    SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - openai is an optional runtime dependency
    OpenAI = None
    SDK_AVAILABLE = False
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEEPSEEK_HOST = "api.deepseek.com"
def _host(base_url: str) -> str:
    try:
        return (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return ""
def api_key_for(base_url: str) -> str:
    """Return the key belonging to the provider behind ``base_url``.

    Keys are never reused across providers. An unknown or empty host yields an
    empty key, so the agent fails closed instead of guessing.
    """
    host = _host(base_url)
    if not host:
        return ""
    if host == DEEPSEEK_HOST or host.endswith("." + DEEPSEEK_HOST):
        return os.environ.get("DEEPSEEK_API_KEY", "").strip()
    return os.environ.get("OPENAI_API_KEY", "").strip()
def settings() -> dict:
    base_url = (os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).strip()
    return {
        "base_url": base_url,
        "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        "api_key": api_key_for(base_url),
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
