"""Casos fictícios para desenvolver os revisores sem dados do pré-processamento."""

import json
from pathlib import Path

from agents.case_context import CaseContext, build_case_context

CASE_NAMES = (
    "01_movimentos_concentrados",
    "02_pagamentos_regulares",
    "03_informacao_insuficiente",
)


def load_reference_case(name: str) -> dict:
    """Lê os dados e os critérios de avaliação de um dos três casos."""
    if name not in CASE_NAMES:
        raise ValueError(f"Caso de referência desconhecido: {name!r}")
    path = Path(__file__).with_name(f"{name}.json")
    return json.loads(path.read_text(encoding="utf-8"))


def build_reference_context(name: str) -> CaseContext:
    """Envia apenas factos e resultados simulados ao construtor existente."""
    # Os critérios são exclusivos da avaliadora. Enviá-los ao LLM revelaria
    # a resposta esperada e impediria uma avaliação independente dos relatórios.
    data = load_reference_case(name)["input"]
    return build_case_context(
        alert_id=data["alert_id"],
        alert=data["alert"],
        transactions=data["transactions"],
        model_output=data["model_output"],
    )
