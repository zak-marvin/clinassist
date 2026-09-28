"""
nlp/services.py
Preprocess -> LLM (Groq, Gemini fallback) -> schema validation -> one repair retry.
"""

import re
import time
import logging

from django.conf import settings
from pydantic import ValidationError

from .prompts import load_system_prompt, build_user_message, ACTIVE_VERSION
from .schemas import Analysis, ModelRefusal, parse_output

logger = logging.getLogger(__name__)

MAX_REPAIR_ATTEMPTS = 1


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


# ── LLM providers (return raw text; validation happens later) ────────────────
def _groq(messages: list) -> str:
    from groq import Groq
    client = Groq(api_key=settings.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        max_tokens=3000,
        temperature=0.2,
        reasoning_effort="low",
        response_format={"type": "json_object"},  # remove this line if Groq returns a 400
    )
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        logger.warning("Groq output truncated (hit max_tokens)")
    return choice.message.content or ""


def _gemini(messages: list) -> str:
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")  # likely retired: update before relying on it
    prompt = "\n\n".join(m["content"] for m in messages)
    return model.generate_content(prompt).text or ""


def _complete(messages: list) -> tuple:
    """Try providers in order. Returns (raw_text, provider_label)."""
    if settings.GROQ_API_KEY:
        try:
            return _groq(messages), f"groq:{settings.GROQ_MODEL}"
        except Exception as e:
            logger.warning("Groq failed: %s", e)
    if settings.GEMINI_API_KEY:
        try:
            return _gemini(messages), "gemini"
        except Exception as e:
            logger.error("Gemini failed: %s", e)
    raise LLMUnavailable("The analysis service is temporarily unavailable. Please try again.")


# ── Validated call with one repair retry ──────────────────────────────────────
def _summarise_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:8]
        )
    return str(exc)


def get_valid_analysis(note: str) -> tuple:
    """Returns (Analysis, meta). Raises ModelRefusal, LLMUnavailable or ValueError."""
    messages = [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user",   "content": build_user_message(note)},
    ]
    started = time.perf_counter()
    last_error = "unknown"

    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        raw, provider = _complete(messages)
        try:
            analysis = parse_output(raw)
            meta = {
                "prompt_version": ACTIVE_VERSION,
                "provider": provider,
                "repaired": attempt > 0,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
            return analysis, meta
        except ModelRefusal:
            raise
        except (ValueError, ValidationError) as e:
            last_error = _summarise_error(e)
            logger.warning("Invalid model output (attempt %d): %s", attempt + 1, last_error)
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
def run_pipeline(clinical_note: str) -> dict:
    valid, msg = validate_input(clinical_note)
    if not valid:
        return {"error": msg}

    cleaned = preprocess_clinical_note(clinical_note)

    try:
        analysis, meta = get_valid_analysis(cleaned)
    except ModelRefusal as e:
        return {"error": str(e)}
    except LLMUnavailable as e:
        return {"error": str(e)}
    except ValueError:
        return {"error": "The model returned an unreadable response. Please try again."}

    data = analysis.model_dump()
    entities = data.pop("entities")
    return {"entities": entities, "llm_result": data, "meta": meta}
