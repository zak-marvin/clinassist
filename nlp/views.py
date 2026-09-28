import json
import logging
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_POST

from .services import run_pipeline, run_pipeline_events

logger = logging.getLogger(__name__)

DEMO_CASES = {
    "malaria": {
        "label": "Severe Malaria — Tororo District",
        "text": (
            "35-year-old female farmer from Tororo presenting with 3-day history "
            "of high-grade fever (39.8°C), rigors, chills, severe headache, and "
            "projectile vomiting. She is drowsy and confused on arrival. RDT "
            "positive for Plasmodium falciparum. Haemoglobin 7.2 g/dL. Spleen "
            "palpable 4cm below costal margin. Jaundice noted. Unable to sit up. "
            "No ACT taken prior to arrival."
        ),
    },
    "tb_hiv": {
        "label": "TB/HIV Co-infection — Kampala",
        "text": (
            "28-year-old male taxi driver, known HIV positive (CD4: 120), not on "
            "ARVs. 8-week history of productive cough with blood-stained sputum, "
            "drenching night sweats, and 10kg weight loss. Temperature 37.9°C. "
            "Right upper lobe cavitation on CXR. GeneXpert ordered. MUAC 19cm. "
            "Oral thrush present. Lives in congested room with 4 family members."
        ),
    },
    "pmtct": {
        "label": "HIV/PMTCT in Pregnancy — Mulago",
        "text": (
            "22-year-old primigravida, 28 weeks gestation, attending ANC at Mulago "
            "Hospital. HIV positive on Option B+ (TDF/3TC/EFV). Reports fatigue, "
            "oral thrush, and recurrent vaginal candidiasis. CD4: 185. Viral load "
            "2,400 copies/mL. Haemoglobin 9.1 g/dL. Partner HIV status unknown. "
            "Concerned about vertical transmission."
        ),
    },
}


def index(request):
    return render(request, "nlp/index.html", {"demo_cases": DEMO_CASES})


def _get_note(request) -> str:
    try:
        return json.loads(request.body).get("note", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return request.POST.get("note", "").strip()


@require_POST
def analyse(request):
    """Non-streaming endpoint: returns the final JSON in one response."""
    note = _get_note(request)
    if not note:
        return JsonResponse({"error": "No clinical note provided."}, status=400)

    logger.info("Analysing note (%d chars)", len(note))
    result = run_pipeline(note)
    if "error" in result:
        return JsonResponse(result, status=422)
    return JsonResponse(result)


@require_POST
def analyse_stream(request):
    """Streaming endpoint: newline-delimited JSON events (status... then result/error)."""
    note = _get_note(request)
    if not note:
        return JsonResponse({"error": "No clinical note provided."}, status=400)

    logger.info("Analysing note, streaming (%d chars)", len(note))

    def event_stream():
        try:
            for event in run_pipeline_events(note):
                yield json.dumps(event) + "\n"
        except Exception:
            logger.exception("Unhandled error in analysis stream")
            yield json.dumps({"type": "error", "message": "Unexpected server error."}) + "\n"

    response = StreamingHttpResponse(event_stream(), content_type="application/x-ndjson")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"   # stop proxies (nginx) buffering the stream
    return response
