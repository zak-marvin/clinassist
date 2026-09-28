"""
nlp/services.py
───────────────
Groq LLM-only pipeline. BioBERT removed — Groq handles both
entity extraction and clinical reasoning in one call.
"""

import re
import json
import logging
from django.conf import settings
from .prompts import load_system_prompt, build_user_message, ACTIVE_VERSION

logger = logging.getLogger(__name__)

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


# ── LLM call ─────────────────────────────────────────────────────────────────
def _call_groq(note: str) -> dict:
    from groq import Groq
    client = Groq(api_key=settings.GROQ_API_KEY)
    response = client.chat.completions.create(
    model=settings.GROQ_MODEL,
    messages=[
        {"role": "system", "content": load_system_prompt()},
        {"role": "user",   "content": build_user_message(note)},
    ],
    max_tokens=3000,
    temperature=0.2,
    reasoning_effort="low",
)
    return _safe_json_parse(response.choices[0].message.content)


def _call_gemini(note: str) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = load_system_prompt() + "\n\n" + build_user_message(note)
    return _safe_json_parse(model.generate_content(prompt).text)


def call_llm(note: str) -> dict:
    if settings.GROQ_API_KEY:
        try:
            return _call_groq(note)
        except Exception as e:
            logger.warning("Groq failed: %s — trying Gemini", e)
    if settings.GEMINI_API_KEY:
        try:
            return _call_gemini(note)
        except Exception as e:
            logger.error("Gemini also failed: %s", e)
    return {"error": "No LLM provider available. Set GROQ_API_KEY in .env"}


def _safe_json_parse(raw: str) -> dict:
    if not raw:
        return {"error": "Empty LLM response"}
    for cleaner in [
        lambda t: t,
        lambda t: re.sub(r"```(?:json)?", "", t).strip("` \n"),
        lambda t: (re.search(r"\{.*\}", t, re.DOTALL) or type("", (), {"group": lambda s: None})()).group(),
    ]:
        try:
            cleaned = cleaner(raw)
            if cleaned:
                return json.loads(cleaned)
        except (json.JSONDecodeError, AttributeError):
            continue
    return {
        "error": "Could not parse LLM response — please retry",
        "entities": {"symptoms": [], "diseases": [], "drugs": [], "lab_values": [], "anatomy": []},
        "possible_conditions": [],
        "red_flags": [],
        "investigations": ["Manual clinical assessment required"],
        "management_considerations": [],
        "clinical_note": "Response could not be parsed.",
        "disclaimer": "Always defer to a qualified clinician.",
    }


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
    result  = call_llm(cleaned)
    if "error" in result and not result.get("possible_conditions"):
        return {"error": result["error"]}

    # Normalise: pull entities out to top level for the view
    entities   = result.pop("entities", {
        "symptoms": [], "diseases": [], "drugs": [], "lab_values": [], "anatomy": []
    })
    llm_result = result

    return {"entities": entities, "llm_result": llm_result}
