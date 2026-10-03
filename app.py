
import io
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI

MODELS = {
    "GPT-6 Astra": {
        "id": "gpt-6-astra",
        "input_per_1m": 5.00,
        "output_per_1m": 25.00,
        "reasoning": ["low", "medium", "high", "xhigh", "max"],
        "description": "Höchste Qualität für anspruchsvollstes Reasoning und Coding.",
    },
    "GPT-6.1 Sol": {
        "id": "gpt-6.1-sol",
        "input_per_1m": 2.00,
        "output_per_1m": 10.00,
        "reasoning": ["low", "medium", "high", "xhigh", "max"],
        "description": "Near-Astra-Leistung für komplexe Arbeit bei geringeren Kosten.",
    },
    "GPT-6 Luna": {
        "id": "gpt-6-luna",
        "input_per_1m": 0.10,
        "output_per_1m": 0.50,
        "reasoning": ["none", "low", "medium", "high", "xhigh", "max"],
        "description": "Kostengünstig und schnell für fokussierte Aufgaben.",
    },
    "GPT-5.6 Sol": {
        "id": "gpt-5.6-sol",
        "input_per_1m": 2.00,
        "output_per_1m": 10.00,
        "reasoning": ["none", "low", "medium", "high", "xhigh", "max"],
        "description": "Flaggschiff der GPT-5.6-Familie für komplexe professionelle Arbeit.",
    },
    "GPT-5.6 Terra": {
        "id": "gpt-5.6-terra",
        "input_per_1m": 1.00,
        "output_per_1m": 6.00,
        "reasoning": ["none", "low", "medium", "high", "xhigh", "max"],
        "description": "Ausgewogener Mix aus Intelligenz und Kosten.",
    },
    "GPT-5.6 Luna": {
        "id": "gpt-5.6-luna",
        "input_per_1m": 0.10,
        "output_per_1m": 0.60,
        "reasoning": ["none", "low", "medium", "high", "xhigh", "max"],
        "description": "Für kostenbewusste, schnelle und volumenreiche Aufgaben.",
    },
}

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_FILE_BYTES = 15 * 1024 * 1024
MAX_EXTRACTED_CHARS = 500_000

TEXT_SUFFIXES = {
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".sql", ".html", ".css", ".csv",
    ".xml", ".sh", ".bash", ".zsh", ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".php", ".rb", ".swift", ".kt", ".kts", ".log",
}

st.set_page_config(
    page_title="SOL API Chat",
    page_icon="◈",
    layout="wide",
)

st.title("◈ OpenAI API Chat")
st.caption("OpenAI API · GPT-5.6 Sol · API-Key nur zur Laufzeit")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "spent_usd" not in st.session_state:
    st.session_state.spent_usd = 0.0
if "last_usage" not in st.session_state:
    st.session_state.last_usage = None
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

with st.sidebar:
    st.subheader("API")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-…",
        help="Der Key wird von der App nicht auf Festplatte gespeichert.",
    )

    model_label = st.selectbox(
        "Modell",
        options=list(MODELS.keys()),
        index=list(MODELS.keys()).index("GPT-5.6 Sol"),
    )
    model_config = MODELS[model_label]
    model_id = model_config["id"]

    reasoning_effort = st.selectbox(
        "Reasoning",
        options=model_config["reasoning"],
        index=model_config["reasoning"].index("medium"),
        help="Nur die für das ausgewählte Modell unterstützten Stufen werden angeboten.",
    )

    st.caption(model_config["description"])
    st.caption(
        f"`{model_id}` · ${model_config['input_per_1m']:.2f}/1M Input · "
        f"${model_config['output_per_1m']:.2f}/1M Output"
    )

    budget = st.number_input(
        "Restbudget / Session-Budget (USD)",
        min_value=0.0,
        value=5.00,
        step=1.00,
        format="%.2f",
        help="Manueller Budgetwert. Das echte API-Guthaben des Kontos wird hier nicht automatisch abgefragt.",
    )

    st.divider()
    st.subheader("Kosten")
    st.metric("Session-Verbrauch", f"${st.session_state.spent_usd:.4f}")
    st.metric(
        "Verbleibendes Budget",
        f"${max(0.0, budget - st.session_state.spent_usd):.4f}",
    )

    if st.session_state.last_usage:
        usage = st.session_state.last_usage
        st.caption(
            f"Letzte Anfrage: {usage['input_tokens']:,} Input · "
            f"{usage['output_tokens']:,} Output · "
            f"~${usage['cost']:.4f}"
        )

    st.divider()
    if st.button("Chat leeren", use_container_width=True):
        st.session_state.messages = []
        st.session_state.spent_usd = 0.0
        st.session_state.last_usage = None
        st.session_state.last_response = ""
        st.rerun()

    st.caption(
        f"Kostenbasis aktuell: ${model_config['input_per_1m']:.2f}/1M Input · "
        f"${model_config['output_per_1m']:.2f}/1M Output"
    )


def extract_file_text(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(
            f"{uploaded_file.name}: Datei ist größer als 5 MB."
        )

    if suffix in TEXT_SUFFIXES:
        return raw.decode("utf-8", errors="replace")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise ValueError(
                f"{uploaded_file.name}: PDF konnte nicht gelesen werden: {exc}"
            ) from exc

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{uploaded_file.name}: Dateityp wird im MVP nicht unterstützt. "
            "Unterstützt sind Text/Code/JSON/CSV/SQL/LOG/MD und PDF."
        ) from exc


def build_request(user_text: str, uploaded_files) -> tuple[list, list[str]]:
    content = []
    display_files = []

    if user_text.strip():
        content.append({"type": "input_text", "text": user_text.strip()})

    total_bytes = 0

    for uploaded_file in uploaded_files or []:
        total_bytes += uploaded_file.size or 0
        if total_bytes > MAX_TOTAL_FILE_BYTES:
            raise ValueError("Gesamtgröße der Anhänge darf 15 MB nicht überschreiten.")

        extracted = extract_file_text(uploaded_file)

        if len(extracted) > MAX_EXTRACTED_CHARS:
            extracted = (
                extracted[:MAX_EXTRACTED_CHARS]
                + "\n\n[Datei hier gekürzt: 500.000 Zeichen Limit]"
            )

        content.append(
            {
                "type": "input_text",
                "text": (
                    f"\n\n--- DATEI: {uploaded_file.name} ---\n"
                    f"{extracted}\n"
                    f"--- ENDE DATEI: {uploaded_file.name} ---"
                ),
            }
        )
        display_files.append(uploaded_file.name)

    if not content:
        raise ValueError("Bitte Text eingeben oder mindestens eine Datei anhängen.")

    return [{"role": "user", "content": content}], display_files


def estimate_cost(input_tokens: int, output_tokens: int, model_config: dict) -> float:
    return (
        input_tokens * model_config["input_per_1m"] / 1_000_000
        + output_tokens * model_config["output_per_1m"] / 1_000_000
    )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["display"])


st.subheader("Neue Anfrage")

uploaded_files = st.file_uploader(
    "Dateien anhängen",
    accept_multiple_files=True,
    type=[
        "txt", "md", "py", "js", "ts", "tsx", "jsx", "json", "yaml", "yml",
        "toml", "ini", "cfg", "sql", "html", "css", "csv", "xml", "sh", "bash",
        "zsh", "java", "c", "cpp", "h", "hpp", "go", "rs", "php", "rb", "swift",
        "kt", "kts", "log", "pdf",
    ],
    help="Mehrere Dateien sind möglich. Maximal 5 MB je Datei / 15 MB gesamt.",
)

prompt = st.text_area(
    "Text",
    height=180,
    placeholder=(
        "Beispiel: Analysiere die angehängten Dateien. "
        "Finde die Root Cause und nenne nur die konkret notwendigen Änderungen."
    ),
)

send = st.button("An SOL senden", type="primary", use_container_width=True)

if send:
    if not api_key.strip():
        st.error("Bitte zuerst den OpenAI API Key eingeben.")
        st.stop()

    if st.session_state.spent_usd >= budget:
        st.error("Das eingegebene Session-Budget ist bereits aufgebraucht.")
        st.stop()

    try:
        current_input, display_files = build_request(prompt, uploaded_files)

        # Re-send the existing chat context so follow-up questions remain a chat.
        api_input = []
        for msg in st.session_state.messages:
            if msg["role"] == "assistant":
                api_input.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": msg["raw"]}],
                    }
                )
            else:
                api_input.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": msg["raw"]}],
                    }
                )
        api_input.extend(current_input)

        client = OpenAI(api_key=api_key.strip())

        with st.spinner("SOL arbeitet …"):
            response = client.responses.create(
                model=model_id,
                input=api_input,
                reasoning={"effort": reasoning_effort},
            )

        output_text = response.output_text or "(Keine Textausgabe.)"
        usage = response.usage

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cost = estimate_cost(input_tokens, output_tokens, model_config)

        st.session_state.spent_usd += cost
        st.session_state.last_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        }

        user_display = prompt.strip() or "(Dateianhang ohne Text)"
        if display_files:
            user_display += "\n\n**Dateien:** " + ", ".join(
                f"`{name}`" for name in display_files
            )

        # Store the exact current request text for conversational follow-ups.
        raw_parts = []
        if prompt.strip():
            raw_parts.append(prompt.strip())
        for uploaded_file in uploaded_files or []:
            raw_parts.append(
                f"\n--- DATEI: {uploaded_file.name} ---\n"
                f"{extract_file_text(uploaded_file)}"
                f"\n--- ENDE DATEI: {uploaded_file.name} ---"
            )
        raw_request = "\n".join(raw_parts)

        st.session_state.messages.append(
            {
                "role": "user",
                "display": user_display,
                "raw": raw_request,
            }
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "display": output_text,
                "raw": output_text,
            }
        )
        st.session_state.last_response = output_text

        # Show the successful response immediately in the current GUI run.
        st.chat_message("assistant").markdown(output_text)

        # The API response is the authoritative source for the actual model ID.
        model_returned = getattr(response, "model", None)
        if model_returned:
            st.caption(f"API-Modell: \`{model_returned}\`")

    except Exception as exc:
        st.error(f"API-Fehler: {exc}")


st.divider()
st.subheader("Export")

if st.session_state.messages:
    transcript_parts = [
        "# OpenAI API Chat Export",
        f"Model: {model_id}",
        f"Reasoning: {reasoning_effort}",
        f"Export: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    for msg in st.session_state.messages:
        role = "USER" if msg["role"] == "user" else "SOL"
        transcript_parts.append(f"## {role}\n\n{msg['display']}\n")

    transcript = "\n".join(transcript_parts)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Gesamten Chat herunterladen",
            data=transcript.encode("utf-8"),
            file_name="sol_chat_export.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        if st.session_state.last_response:
            st.download_button(
                "Letzte Antwort herunterladen",
                data=st.session_state.last_response.encode("utf-8"),
                file_name="sol_response.txt",
                mime="text/plain",
                use_container_width=True,
            )
else:
    st.caption("Noch kein Chat vorhanden.")
