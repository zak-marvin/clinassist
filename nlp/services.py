"""
nlp/services.py
Preprocess -> LLM call (retry/backoff, model fallback) -> schema validation
-> one repair retry. The pipeline is a generator that yields progress events so
the view can stream them; run_pipeline() consumes it for non-streaming use.

Events:  {"type": "status", "message": str}
         {"type": "result", "data": {...}}
         {"type": "error",  "message": str}
"""

import re
import time
import random
import logging

from django.conf import settings
from pydantic import ValidationError

from .prompts import load_system_prompt, build_user_message, ACTIVE_VERSION
from .schemas import ModelRefusal, parse_output

logger = logging.getLogger(__name__)

# ── Resilience settings ───────────────────────────────────────────────────────
MAX_ATTEMPTS        = 3      # tries per model for transient errors
BASE_DELAY_S        = 1.0    # backoff: 1s, 2s, 4s ... (with jitter)
MAX_DELAY_S         = 8.0    # never wait longer than this between tries
REQUEST_TIMEOUT_S   = 25.0   # per-call timeout so a hung call can't freeze the page
TOTAL_BUDGET_S      = 45.0   # stop retrying after this much total time
MAX_REPAIR_ATTEMPTS = 1      # re-ask once if the output fails schema validation

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
FATAL_STATUS     = {401, 403}   # bad key / no access: retrying or switching model won't help


class LLMUnavailable(Exception):
    """No provider could return a response."""


# ── Clinical abbreviation map ─────────────────────────────────────────────────
ABBREV_MAP = {
    "hx": "history", "sx": "symptoms", "dx": "diagnosis", "rx": "treatment",
    "yo": "year old", "y/o": "year old", "m": "male", "f": "female",
    "c/o": "complains of", "h/o": "history of", "k/o": "known case of",
    "sob": "shortness of breath", "cp": "chest pain", "ha": "headache",
    "n/v": "nausea and vomiting", "n/v/d": "nausea vomiting diarrhea",
    "abd": "abdominal", "bp": "blood pressure", "hr": "heart rate",
    "rr": "respiratory rate", "temp": "temperature", "o2": "oxygen",
    "sats": "saturation", "hb": "haemoglobin", "wbc": "white blood cell",
    "rbc": "red blood cell", "fbc": "full blood count", "lft": "liver function test",
    "rft": "renal function test", "lfts": "liver function tests",
    "ua": "urinalysis", "uss": "ultrasound scan", "cxr": "chest x-ray",
    "rdt": "rapid diagnostic test", "pcr": "polymerase chain reaction",
    "cd4": "CD4 count", "vl": "viral load", "arv": "antiretroviral",
    "arvs": "antiretrovirals", "art": "antiretroviral therapy",
    "tb": "tuberculosis", "hiv": "HIV", "sti": "sexually transmitted infection",
    "act": "artemisinin combination therapy", "ipt": "isoniazid preventive therapy",
    "pmtct": "prevention of mother to child transmission",
    "muac": "mid upper arm circumference", "anc": "antenatal care",
    "dots": "directly observed treatment short course",
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_clinical_note(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\,\;\:\(\)\-\/\+°%]", " ", text)
    for abbr, expansion in ABBREV_MAP.items():
        pattern = r"(?<!\w)" + re.escape(abbr) + r"(?!\w)"
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    sentences = text.split(". ")
    text = ". ".join(s.capitalize() for s in sentences)
    return text.strip()


# ── Provider calls (return raw text; validation happens later) ───────────────
def _groq(messages: list, model: str) -> str:
    from groq import Groq
    # max_retries=0: we do our own backoff so we can log it and show it to the user.
    client = Groq(api_key=settings.GROQ_API_KEY, timeout=REQUEST_TIMEOUT_S, max_retries=0)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=3000,
        temperature=0.2,
        reasoning_effort="low",
        response_format={"type": "json_object"},  # remove this line if Groq returns a 400
    )
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        logger.warning("Groq output truncated on %s (hit max_tokens)", model)
    return choice.message.content or ""


def _gemini(messages: list) -> str:
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")  # likely retired: update before relying on it
    prompt = "\n\n".join(m["content"] for m in messages)
    return model.generate_content(prompt).text or ""


# ── Error classification + backoff ────────────────────────────────────────────
def _status_of(exc: Exception):
    return getattr(exc, "status_code", None)


def _is_retryable(exc: Exception) -> bool:
    status = _status_of(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    # No HTTP status: network-level problems are worth retrying.
    return type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "TimeoutError", "ConnectionError"}


def _retry_after(exc: Exception):
    try:
        return float(exc.response.headers.get("retry-after"))
    except Exception:
        return None


def _backoff_delay(attempt: int, exc: Exception) -> float:
    hinted = _retry_after(exc)
    if hinted is not None:
        return min(hinted, MAX_DELAY_S)
    exp = min(MAX_DELAY_S, BASE_DELAY_S * (2 ** (attempt - 1)))
    return exp * random.uniform(0.5, 1.0)  # jitter avoids synchronised retries


def _complete_events(messages: list, deadline: float):
    """Generator: yields status events, returns (raw_text, provider_label).
    Order: primary model (with backoff) -> fallback model -> Gemini."""
    if settings.GROQ_API_KEY:
        models = [settings.GROQ_MODEL]
        fallback = getattr(settings, "GROQ_FALLBACK_MODEL", "")
        if fallback and fallback != settings.GROQ_MODEL:
            models.append(fallback)

        fatal = False
        for model in models:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    return _groq(messages, model), f"groq:{model}"
                except Exception as e:
                    status = _status_of(e)
                    logger.warning("Groq %s attempt %d failed (status=%s): %s", model, attempt, status, e)
                    if status in FATAL_STATUS:
                        fatal = True
                        break
                    if not _is_retryable(e):
                        break  # e.g. 404 model gone / 400 bad request: try the next model
                    if attempt == MAX_ATTEMPTS:
                        break
                    delay = _backoff_delay(attempt, e)
                    if time.monotonic() + delay > deadline:
                        break
                    yield {"type": "status",
                           "message": f"Service busy, retrying in {delay:.0f}s ({attempt}/{MAX_ATTEMPTS})…"}
                    time.sleep(delay)
            if fatal:
                break
            if model != models[-1]:
                yield {"type": "status", "message": "Switching to backup model…"}

    if settings.GEMINI_API_KEY:
        try:
            yield {"type": "status", "message": "Trying backup provider…"}
            return _gemini(messages), "gemini"
        except Exception as e:
            logger.error("Gemini failed: %s", e)

    raise LLMUnavailable("The analysis service is temporarily unavailable. Please try again shortly.")


# ── Validated call with one repair retry ──────────────────────────────────────
def _summarise_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8]
        )
    return str(exc)


def get_valid_analysis_events(note: str):
    """Generator: yields status events, returns (Analysis, meta).
    Raises ModelRefusal, LLMUnavailable or ValueError."""
    messages = [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user",   "content": build_user_message(note)},
    ]
    started = time.monotonic()
    deadline = started + TOTAL_BUDGET_S
    last_error = "unknown"

    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        raw, provider = yield from _complete_events(messages, deadline)
        try:
            analysis = parse_output(raw)
            meta = {
                "prompt_version": ACTIVE_VERSION,
                "provider": provider,
                "repaired": attempt > 0,
                "latency_ms": int((time.monotonic() - started) * 1000),
            }
            return analysis, meta
        except ModelRefusal:
            raise
        except (ValueError, ValidationError) as e:
            last_error = _summarise_error(e)
            logger.warning("Invalid model output (attempt %d): %s", attempt + 1, last_error)
            if attempt < MAX_REPAIR_ATTEMPTS:
                yield {"type": "status", "message": "Checking output format…"}
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    "Your previous reply was invalid: " + last_error + ". "
                    "Reply again with ONLY the corrected JSON object, exactly matching "
                    "the required format."
                )},
            ]

    raise ValueError(f"Model output failed validation after repair: {last_error}")


# ── Input validation ──────────────────────────────────────────────────────────
INJECTION_PHRASES = [
    "ignore previous", "forget instructions", "you are now",
    "new system prompt", "disregard your", "jailbreak",
]

def validate_input(text: str) -> tuple:
    if not text or not text.strip():
        return False, "Input is empty — please enter symptoms or a clinical note."
    if len(text.strip()) < 15:
        return False, "Input too short — please provide more clinical detail."
    if len(text) > 5000:
        return False, "Input exceeds 5000 characters — please shorten the note."
    for phrase in INJECTION_PHRASES:
        if phrase in text.lower():
            return False, "Input contains invalid content."
    return True, ""


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline_events(clinical_note: str):
    valid, msg = validate_input(clinical_note)
    if not valid:
        yield {"type": "error", "message": msg}
        return

    yield {"type": "status", "message": "Analysing…"}
    cleaned = preprocess_clinical_note(clinical_note)

    try:
        analysis, meta = yield from get_valid_analysis_events(cleaned)
    except (ModelRefusal, LLMUnavailable) as e:
        yield {"type": "error", "message": str(e)}
        return
    except ValueError:
        yield {"type": "error", "message": "The model returned an unreadable response. Please try again."}
        return

    data = analysis.model_dump()
    entities = data.pop("entities")
    yield {"type": "result", "data": {"entities": entities, "llm_result": data, "meta": meta}}


def run_pipeline(clinical_note: str) -> dict:
    """Non-streaming wrapper: consume the events, return the final dict."""
    final = {"error": "No result produced."}
    for ev in run_pipeline_events(clinical_note):
        if ev["type"] == "result":
            final = ev["data"]
        elif ev["type"] == "error":
            final = {"error": ev["message"]}
    return final
