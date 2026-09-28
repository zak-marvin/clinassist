"""
nlp/schemas.py
Output contract for the LLM. Mirrors the JSON shape in prompts/system_v*.txt.
Change one, change the other.
"""
import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Level = Literal["High", "Moderate", "Low"]


class ModelRefusal(Exception):
    """The model deliberately returned {"error": "..."} (non-clinical / too vague)."""


def _norm_level(v):
    # Tolerate "high", "HIGH", "Medium" -> canonical values.
    if isinstance(v, str):
        s = v.strip().capitalize()
        return "Moderate" if s == "Medium" else s
    return v


class Entities(BaseModel):
    symptoms:   list[str] = Field(default_factory=list)
    diseases:   list[str] = Field(default_factory=list)
    drugs:      list[str] = Field(default_factory=list)
    lab_values: list[str] = Field(default_factory=list)
    anatomy:    list[str] = Field(default_factory=list)


class Condition(BaseModel):
    condition:  str = Field(min_length=1)
    likelihood: Level
    reasoning:  str = ""

    _norm = field_validator("likelihood", mode="before")(_norm_level)


class Analysis(BaseModel):
    entities:                  Entities
    possible_conditions:       list[Condition]
    red_flags:                 list[str] = Field(default_factory=list)
    investigations:            list[str] = Field(default_factory=list)
    management_considerations: list[str] = Field(default_factory=list)
    confidence:                Level
    clinical_note:             str = Field(min_length=1)
    disclaimer:                str = (
        "This output is for educational and clinical decision-support purposes only. "
        "It does not constitute a medical diagnosis. Always defer to a qualified clinician."
    )

    _norm = field_validator("confidence", mode="before")(_norm_level)


def extract_json(raw: str) -> dict:
    """Pull one JSON object out of raw model text (handles ``` fences and chatter)."""
    if not raw or not raw.strip():
        raise ValueError("Empty response")
    text = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start : end + 1])  # JSONDecodeError is a ValueError


def parse_output(raw: str) -> Analysis:
    """Raw text -> validated Analysis.
    Raises ModelRefusal, ValueError (bad JSON) or pydantic.ValidationError (bad shape)."""
    data = extract_json(raw)
    if "error" in data and "entities" not in data:
        raise ModelRefusal(str(data["error"]))
    return Analysis.model_validate(data)
