from __future__ import annotations

from typing import Any

from anthropic import Anthropic
from openai import OpenAI

OPENAI_MODELS = {
    "GPT-6 Astra": {"id": "gpt-6-astra", "input_per_1m": 10.00, "output_per_1m": 50.00, "reasoning": ["low", "medium", "high", "xhigh", "max"], "description": "Höchste Qualität für anspruchsvollstes Reasoning und Coding."},
    "GPT-6.1 Sol": {"id": "gpt-6.1-sol", "input_per_1m": 2.00, "output_per_1m": 10.00, "reasoning": ["low", "medium", "high", "xhigh", "max"], "description": "Near-Astra-Leistung für komplexe Arbeit bei geringeren Kosten."},
    "GPT-6 Luna": {"id": "gpt-6-luna", "input_per_1m": 0.10, "output_per_1m": 0.50, "reasoning": ["none", "low", "medium", "high", "xhigh", "max"], "description": "Kostengünstig und schnell für fokussierte Aufgaben."},
    "GPT-5.6 Sol": {"id": "gpt-5.6-sol", "input_per_1m": 4.00, "output_per_1m": 20.00, "reasoning": ["none", "low", "medium", "high", "xhigh", "max"], "description": "Flaggschiff der GPT-5.6-Familie für komplexe professionelle Arbeit."},
    "GPT-5.6 Terra": {"id": "gpt-5.6-terra", "input_per_1m": 2.00, "output_per_1m": 12.00, "reasoning": ["none", "low", "medium", "high", "xhigh", "max"], "description": "Ausgewogener Mix aus Intelligenz und Kosten."},
    "GPT-5.6 Luna": {"id": "gpt-5.6-luna", "input_per_1m": 0.20, "output_per_1m": 1.20, "reasoning": ["none", "low", "medium", "high", "xhigh", "max"], "description": "Für kostenbewusste, schnelle und volumenreiche Aufgaben."},
}

CLAUDE_MODELS = {
    "Claude Opus 5.5": {"id": "claude-opus-5-5", "input_per_1m": 4.00, "output_per_1m": 20.00, "description": "Aktuelles Claude-Flaggschiff für komplexes Reasoning und Coding."},
    "Claude Sonnet 5": {"id": "claude-sonnet-5", "input_per_1m": 2.00, "output_per_1m": 10.00, "description": "Starke Coding- und Analyseleistung bei geringeren Kosten."},
    "Claude Haiku 4.5": {"id": "claude-haiku-4-5", "input_per_1m": 1.00, "output_per_1m": 5.00, "description": "Schnell und kosteneffizient."},
}

GROK_MODELS = {
    "Grok 4.7": {"id": "grok-4.7", "input_per_1m": 2.00, "output_per_1m": 6.00, "description": "xAI-Flaggschiff für Coding, Agentic Tasks und Knowledge Work."},
    "Grok 4.6": {"id": "grok-4.6", "input_per_1m": 2.00, "output_per_1m": 6.00, "description": "Leistungsstarkes Grok-Modell für Analyse und Coding."},
}

PROVIDERS = {"OpenAI": OPENAI_MODELS, "Anthropic Claude": CLAUDE_MODELS, "xAI Grok": GROK_MODELS}


def call_model(provider: str, model_id: str, api_key: str, history: list[dict[str, str]], prompt: str, reasoning_effort: str | None = None) -> Any:
    messages = [{"role": item["role"], "content": item["raw"]} for item in history]
    messages.append({"role": "user", "content": prompt})

    if provider == "OpenAI":
        client = OpenAI(api_key=api_key)
        kwargs = {"model": model_id, "input": messages}
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        return client.responses.create(**kwargs)

    if provider == "xAI Grok":
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        return client.responses.create(model=model_id, input=messages)

    if provider == "Anthropic Claude":
        client = Anthropic(api_key=api_key)
        return client.messages.create(model=model_id, max_tokens=8192, messages=messages)

    raise ValueError(f"Unbekannter Provider: {provider}")


def extract_output_text(provider: str, response: Any) -> str:
    if provider in {"OpenAI", "xAI Grok"}:
        return response.output_text or "(Keine Textausgabe.)"
    if provider == "Anthropic Claude":
        texts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "\n\n".join(texts) or "(Keine Textausgabe.)"
    raise ValueError(f"Unbekannter Provider: {provider}")


def extract_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)


def response_model(response: Any, fallback: str) -> str:
    return str(getattr(response, "model", None) or fallback)
