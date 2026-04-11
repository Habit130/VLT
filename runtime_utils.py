from __future__ import annotations

import os

import spacy


SPACY_MODEL_ALIASES = {
    "en_vectors_web_lg": "en_core_web_lg",
}


def resolve_spacy_model_name(model_name: str) -> str:
    return SPACY_MODEL_ALIASES.get(model_name, model_name)


def load_spacy_model(model_name: str):
    resolved_name = resolve_spacy_model_name(model_name)
    try:
        return spacy.load(resolved_name)
    except OSError:
        if resolved_name != model_name:
            return spacy.load(model_name)
        raise


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
