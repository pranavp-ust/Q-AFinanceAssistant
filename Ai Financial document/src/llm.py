"""LLM access with retries, model fallback and JSON parsing.

Agents call `chat()` / `chat_json()`. If no provider is configured, or every retry and
the fallback model fail, `LLMUnavailable` is raised and the calling agent switches to
its deterministic rule-based fallback - so the workflow never hard-fails on the LLM.
"""
import json
import re
from functools import lru_cache

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.monitoring import logger, metrics


class LLMUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _model(name: str | None = None):
    p = settings.llm_provider
    if p == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name or settings.openai_model, temperature=0, timeout=60, max_retries=0)
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=name or settings.gemini_model, temperature=0, max_retries=0)
    return None


def llm_enabled() -> bool:
    try:
        return _model() is not None
    except Exception:  # missing package / key
        return False


def _invoke(model, system: str, user: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    resp = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = resp.content
    if isinstance(content, list):  # Gemini can return content parts
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content.strip()


def chat(system: str, user: str, agent: str = "") -> str:
    try:
        model = _model()
    except Exception as exc:
        raise LLMUnavailable(f"LLM init failed: {exc}") from exc
    if model is None:
        raise LLMUnavailable("LLM_PROVIDER=none")

    metrics.inc("llm_calls")
    candidates = [model]
    if settings.llm_provider == "openai" and settings.openai_fallback_model:
        candidates.append(_model(settings.openai_fallback_model))

    last_exc = None
    for idx, m in enumerate(candidates):
        try:
            for attempt in Retrying(stop=stop_after_attempt(settings.llm_max_retries),
                                    wait=wait_exponential(multiplier=1, min=1, max=8),
                                    retry=retry_if_exception_type(Exception), reraise=True):
                with attempt:
                    if attempt.retry_state.attempt_number > 1:
                        metrics.inc("llm_retries")
                        logger.warning("%s: LLM retry #%d", agent, attempt.retry_state.attempt_number - 1)
                    return _invoke(m, system, user)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if idx + 1 < len(candidates):
                metrics.inc("llm_model_fallbacks")
                logger.warning("%s: primary model failed (%s), switching to fallback model", agent, exc)
    metrics.inc("llm_failures")
    raise LLMUnavailable(f"LLM failed after retries: {last_exc}")


def _extract_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise


def chat_json(system: str, user: str, agent: str = ""):
    """Ask for JSON; if the reply does not parse, re-ask once with the parse error."""
    raw = chat(system, user, agent)
    try:
        return _extract_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        metrics.inc("llm_json_repairs")
        raw = chat(system, f"{user}\n\nYour previous reply was not valid JSON ({exc}). "
                           "Reply again with ONLY valid JSON, no prose.", agent)
        try:
            return _extract_json(raw)
        except (json.JSONDecodeError, ValueError) as exc2:
            raise LLMUnavailable(f"LLM returned invalid JSON twice: {exc2}") from exc2
