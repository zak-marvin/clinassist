# ClinAssist Uganda

A clinical decision-support web app for students and clinicians in Uganda. Paste a
clinical note and get structured, hedged output: extracted entities, possible
conditions with likelihoods, red flags, suggested investigations and management
considerations. Built with Django and an LLM served through Groq.

> **Educational use only. Not a substitute for professional clinical judgment.**
> Do not enter patient names, phone numbers or other identifying details.

---

## How a request flows

```
note → input validation → preprocessing → LLM call (retry / backoff / fallback)
     → schema validation → (one repair retry if invalid) → streamed result
```

Every stage emits a progress event, so the UI shows what is happening
("Service busy, retrying in 2s…") instead of a frozen spinner.

---

## LLM interaction layer

### Provider, model and parameters

| Setting | Value | Why |
|---|---|---|
| Provider | Groq | Free tier, low latency, OpenAI-compatible API |
| Primary model | `openai/gpt-oss-20b` | Current Groq replacement for the retired `llama-3.1-8b-instant`; fast and cheap enough for interactive use |
| Fallback model | `openai/gpt-oss-120b` | Stronger model used when the primary is rate-limited or unavailable |
| `temperature` | `0.2` | Clinical output should be consistent between runs, not creative |
| `max_tokens` | `3000` | gpt-oss is a reasoning model: hidden reasoning tokens count against the limit, and a small limit truncated the JSON |
| `reasoning_effort` | `low` | The task is extraction plus structured reasoning; higher effort adds latency without clear benefit here |
| `response_format` | `json_object` | Asks the API for JSON; the schema layer below is the actual guarantee |
| Request timeout | 25 s | A hung call must not freeze the page |
| Gemini | optional last-resort fallback | Only used if configured; model name must be kept current |

Models are configured through environment variables (`GROQ_MODEL`,
`GROQ_FALLBACK_MODEL`), not hard-coded, because providers retire models: the
original default, `llama-3.1-8b-instant`, was retired and caused every request to
fail until it was replaced.

### Versioned prompt

The system prompt lives in `nlp/prompts/system_v2.txt`, not in Python code.
`ACTIVE_VERSION` in `nlp/prompts/__init__.py` selects the version, and the version
is returned with every result (`meta.prompt_version`) so outputs can be traced to
the prompt that produced them. Change the prompt by adding a new file
(`system_v3.txt`) and bumping the version; old versions stay in git history.

The prompt defines the role and setting (Ugandan facilities, limited resources),
a fixed task list, explicit rules (hedged language, no drug doses, no invented
statistics, missing information lowers confidence), an output contract, and an
injection defence: the note is wrapped in `<clinical_note>` tags and declared
untrusted data.

### Schema-validated output

`nlp/schemas.py` defines the response contract with Pydantic (`Analysis`,
`Condition`, `Entities`). Every model response is parsed and validated:

1. Valid → returned.
2. Invalid or truncated → **one repair request** that sends the model its own reply
   plus the exact validation error.
3. Still invalid → the user gets a clear error. Malformed output is never rendered.

The model may also return `{"error": "..."}` for non-clinical or too-vague input;
this is surfaced to the user as a message.

### Resilience

| Failure | Behaviour |
|---|---|
| 429 / 5xx / timeout / connection error | Retry up to 3 times with exponential backoff (1 s, 2 s, 4 s, capped at 8 s) plus jitter; honours the `retry-after` header |
| Primary model still failing | Switch to the fallback model, same retry policy |
| 401 / 403 (bad key, no access) | Fail immediately: retrying cannot help |
| 404 / 400 (model gone, bad request) | No retry; move to the fallback model |
| Slow overall | No retries start after a 45 s total budget |
| Invalid model output | One repair request (above) |

Groq's SDK retries are disabled (`max_retries=0`) so all backoff is visible in our
logs and in the UI stream.

### Streaming

`POST /analyse/stream/` returns newline-delimited JSON (`application/x-ndjson`)
through Django's `StreamingHttpResponse`:

```
{"type": "status", "message": "Analysing…"}
{"type": "status", "message": "Service busy, retrying in 2s (1/3)…"}
{"type": "result", "data": { "entities": ..., "llm_result": ..., "meta": ... }}
```

Progress events are streamed rather than raw model tokens because the result must
be fully validated before it can be shown; half a JSON document is not renderable.
`POST /analyse/` remains available and returns the final JSON in one response.

---

## Tech stack

- **Backend**: Django 4.2, Pydantic 2
- **LLM**: Groq (`openai/gpt-oss-20b`, fallback `openai/gpt-oss-120b`)
- **Frontend**: Tailwind CSS (CDN), vanilla JS
- **Hosting**: Render

---

## Local setup

```bash
git clone https://github.com/zak-marvin/clinassist.git
cd clinassist
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env           # Windows: copy .env.example .env
python manage.py runserver
```

Open http://127.0.0.1:8000. Get a free Groq key at https://console.groq.com.

To see which models your key can use:

```bash
python -c "import os,requests;from dotenv import load_dotenv;load_dotenv();r=requests.get('https://api.groq.com/openai/v1/models',headers={'Authorization':'Bearer '+os.environ['GROQ_API_KEY']});print([m['id'] for m in r.json()['data']])"
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `DJANGO_SECRET_KEY` | Yes | Long random string |
| `DEBUG` | Yes | `True` locally, `False` in production |
| `ALLOWED_HOSTS` | Yes | Comma-separated hostnames |
| `GROQ_API_KEY` | Yes | From https://console.groq.com |
| `GROQ_MODEL` | No | Default `openai/gpt-oss-20b` |
| `GROQ_FALLBACK_MODEL` | No | Default `openai/gpt-oss-120b` |
| `GEMINI_API_KEY` | No | Optional last-resort fallback |

## Deploying to Render

Build command: `pip install -r requirements.txt`
Start command: `gunicorn clinassist.wsgi:application --timeout 90`

The longer worker timeout matters: retries and backoff can legitimately take longer
than gunicorn's 30 s default. Set the environment variables above in the dashboard.

## Project structure

```
clinassist/          Django project (settings, urls, wsgi)
nlp/
├── prompts/         Versioned system prompts + loader
├── schemas.py       Pydantic output contract
├── services.py      Pipeline: preprocessing, retries, fallback, validation, repair
├── views.py         Page, /analyse/ and /analyse/stream/
└── urls.py
templates/nlp/index.html
static/
```

## Known limitations

- Retrieval over guideline text (RAG) and an evaluation harness are not built yet.
- The preprocessing step (abbreviation expansion) can alter clinical meaning and
  should be simplified.
- Model output is inserted into the page as HTML and needs escaping.
- Output is unverified LLM text: differentials and management points must be checked
  by a clinician.

## License

MIT. Built for clinical education in Uganda.
