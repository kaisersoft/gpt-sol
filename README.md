# SOL API Chat MVP

Kleine Streamlit-WebGUI für **GPT-5.6 Sol** über die OpenAI API.

## Funktionen

- API-Key zur Laufzeit
- keine Speicherung des API-Keys durch die App
- Chat mit Gesprächskontext
- Text-Eingabe und Text-Ausgabe
- mehrere Datei-Anhänge pro Anfrage
- Analyse von TXT/MD/Code/JSON/CSV/SQL/LOG und PDF
- Dateilimit: 5 MB je Datei, 15 MB gesamt
- Export des gesamten Chats als Markdown
- Export der letzten SOL-Antwort als TXT
- Session-Verbrauch und manuell gesetztes Restbudget

## Start lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

Repository auf GitHub anlegen und `app.py` als Streamlit-App deployen.

Da der API-Key zur Laufzeit eingegeben wird, ist für den MVP keine Secret-Datei notwendig.

## Kosten

Die App rechnet die Session-Kosten anhand der von der API gemeldeten Input-/Output-Tokens mit der Standardpreisannahme für GPT-5.6 Sol:
- $4 / 1 Mio. Input-Tokens
- $20 / 1 Mio. Output-Tokens

Der Budgetwert in der GUI ist ein eigener Session-Grenzwert. Das tatsächliche API-Guthaben des OpenAI-Kontos wird nicht automatisch ausgelesen.
