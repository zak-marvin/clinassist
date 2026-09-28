from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).parent
ACTIVE_VERSION = "v2"  # bump when you change the prompt; log it in eval results


@lru_cache(maxsize=None)
def load_system_prompt(version: str = ACTIVE_VERSION) -> str:
    return (PROMPT_DIR / f"system_{version}.txt").read_text(encoding="utf-8")


def build_user_message(note: str) -> str:
    # Delimiters mark the note as data; strip any attempt to close the tag early.
    safe = note.replace("<clinical_note>", "").replace("</clinical_note>", "")
    return f"<clinical_note>\n{safe}\n</clinical_note>"