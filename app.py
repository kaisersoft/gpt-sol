
import io
from datetime import datetime
from pathlib import Path

import streamlit as st
from providers import PROVIDERS, call_model, extract_output_text, extract_usage, response_model


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
    page_title="GPT-SOL - AI Model Chat V1.0",
    page_icon="◈",
    layout="wide",
)

logo_col, title_col = st.columns([1, 4], vertical_alignment="center")
with logo_col:
    st.image("assets/gpt_sol_logo.svg", width=120)
with title_col:
    st.markdown("## GPT-SOL - AI Model Chat V1.0")
    st.markdown(
        "*We make Models, that are good at coding and good at AI research.*"
    )
    st.caption("OpenAI · Anthropic Claude · xAI Grok · API-Keys nur zur Laufzeit")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "spent_usd" not in st.session_state:
    st.session_state.spent_usd = 0.0
if "last_usage" not in st.session_state:
    st.session_state.last_usage = None
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

with st.sidebar:
    st.subheader("Provider")
    provider = st.selectbox("Anbieter", options=list(PROVIDERS.keys()), index=0)
    provider_models = PROVIDERS[provider]
    default_model_index = list(provider_models.keys()).index("GPT-5.6 Sol") if provider == "OpenAI" else 0

    model_label = st.selectbox("Modell", options=list(provider_models.keys()), index=default_model_index)
    model_config = provider_models[model_label]
    model_id = model_config["id"]

    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-…", help="Optional – nur nötig für OpenAI.")
    claude_api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-…", help="Optional – nur nötig für Anthropic.")
    xai_api_key = st.text_input("xAI API Key", type="password", placeholder="xai-…", help="Optional – nur nötig für xAI.")

    reasoning_effort = None
    if provider == "OpenAI":
        reasoning_effort = st.selectbox("Reasoning", options=model_config["reasoning"], index=model_config["reasoning"].index("medium"))

    st.caption(model_config["description"])
    st.caption(f"`{model_id}` · ${model_config['input_per_1m']:.2f}/1M Input · ${model_config['output_per_1m']:.2f}/1M Output")

    budget = st.number_input("Restbudget / Session-Budget (USD)", min_value=0.0, value=5.00, step=1.00, format="%.2f", help="Manueller Budgetwert; das echte Provider-Guthaben wird nicht automatisch abgefragt.")

    st.divider()
    st.subheader("Kosten")
    st.metric("Session-Verbrauch", f"${st.session_state.spent_usd:.4f}")
    st.metric("Verbleibendes Budget", f"${max(0.0, budget - st.session_state.spent_usd):.4f}")

    if st.session_state.last_usage:
        usage = st.session_state.last_usage
        st.caption(f"Letzte Anfrage: {usage['input_tokens']:,} Input · {usage['output_tokens']:,} Output · ~${usage['cost']:.4f}")

    st.divider()
    if st.button("Chat leeren", use_container_width=True):
        st.session_state.messages = []
        st.session_state.spent_usd = 0.0
        st.session_state.last_usage = None
        st.session_state.last_response = ""
        st.rerun()

    st.caption(f"Kostenbasis aktuell: ${model_config['input_per_1m']:.2f}/1M Input · ${model_config['output_per_1m']:.2f}/1M Output")

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

send = st.button("An Modell senden", type="primary", use_container_width=True)

if send:
    selected_api_key = {
        "OpenAI": openai_api_key,
        "Anthropic Claude": claude_api_key,
        "xAI Grok": xai_api_key,
    }[provider]

    if not selected_api_key.strip():
        st.error(f"Bitte zuerst den {provider}-API-Key eingeben.")
        st.stop()

    if st.session_state.spent_usd >= budget:
        st.error("Das eingegebene Session-Budget ist bereits aufgebraucht.")
        st.stop()

    try:
        _, display_files = build_request(prompt, uploaded_files)
        raw_parts = []
        if prompt.strip():
            raw_parts.append(prompt.strip())
        for uploaded_file in uploaded_files or []:
            raw_parts.append(
                f"\n--- DATEI: {uploaded_file.name} ---\n"
                f"{extract_file_text(uploaded_file)}"
                f"\n--- ENDE DATEI: {uploaded_file.name} ---"
            )
        current_prompt = "\n".join(raw_parts)

        response = call_model(
            provider=provider,
            model_id=model_id,
            api_key=selected_api_key.strip(),
            history=st.session_state.messages,
            prompt=current_prompt,
            reasoning_effort=reasoning_effort,
        )

        output_text = extract_output_text(provider, response)
        input_tokens, output_tokens = extract_usage(response)
        cost = estimate_cost(input_tokens, output_tokens, model_config)

        st.session_state.spent_usd += cost
        st.session_state.last_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost": cost}

        user_display = prompt.strip() or "(Dateianhang ohne Text)"
        if display_files:
            user_display += "\n\n**Dateien:** " + ", ".join(f"`{name}`" for name in display_files)

        st.session_state.messages.append({"role": "user", "display": user_display, "raw": current_prompt})
        st.session_state.messages.append({"role": "assistant", "display": output_text, "raw": output_text})
        st.session_state.last_response = output_text

        st.chat_message("assistant").markdown(output_text)
        st.caption(f"{provider} · API-Modell: `{response_model(response, model_id)}`")

    except Exception as exc:
        st.error(f"API-Fehler: {exc}")

st.divider()
st.subheader("Export")

if st.session_state.messages:
    transcript_parts = [
        "# AI Model Chat Export",
        f"Provider: {provider}",
        f"Model: {model_id}",
        f"Reasoning: {reasoning_effort or 'default'}",
        f"Export: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for msg in st.session_state.messages:
        role = "USER" if msg["role"] == "user" else "ASSISTANT"
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
