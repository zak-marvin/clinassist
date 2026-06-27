# ClinAssist Uganda

A clinical decision-support web app for students and clinicians in Uganda. Built with Django + Groq LLM.

> **Educational use only. Not a substitute for professional clinical judgment.**

---

## What it does

Paste a clinical note → get back:
- Extracted medical entities (symptoms, diseases, drugs, lab values, anatomy)
- Possible differential conditions with likelihood and reasoning
- Red flags requiring urgent attention
- Suggested investigations
- Management considerations

All powered by **Groq's free LLM API** (Llama 3.1). No GPU, no local model, no paid services required.

---

## Tech stack

- **Backend** — Django 4.2
- **LLM** — Groq API (free tier, llama-3.1-8b-instant)
- **Frontend** — Tailwind CSS, vanilla JS
- **Hosting** — Render (free tier)

---

## Local setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/clinassist-uganda.git
cd clinassist-uganda
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
DJANGO_SECRET_KEY=any-long-random-string
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
GROQ_API_KEY=gsk_...
```

Get a free Groq API key at https://console.groq.com

### 5. Run

```bash
python manage.py runserver
```

Open http://127.0.0.1:8000

---

## Deploying to Render

1. Push this repo to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set the following:

| Field | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build command** | `pip install -r requirements.txt` |
| **Start command** | `gunicorn clinassist.wsgi:application` |

5. Add environment variables in Render dashboard:

| Key | Value |
|---|---|
| `DJANGO_SECRET_KEY` | a long random string |
| `DEBUG` | `False` |
| `ALLOWED_HOSTS` | `your-app-name.onrender.com` |
| `GROQ_API_KEY` | your Groq key |

6. Click **Deploy** — Render will build and host it for free.

---

## Project structure

```
clinassist/
├── clinassist/         Django project (settings, urls, wsgi)
├── nlp/
│   ├── services.py     Groq LLM pipeline + entity extraction
│   ├── views.py        Index page + /analyse/ AJAX endpoint
│   └── urls.py
├── templates/nlp/
│   └── index.html      Full UI (Tailwind, vanilla JS)
├── static/
├── .env.example
├── requirements.txt
└── manage.py
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `DJANGO_SECRET_KEY` | Yes | Any long random string |
| `DEBUG` | Yes | `True` locally, `False` in production |
| `ALLOWED_HOSTS` | Yes | Comma-separated hostnames |
| `GROQ_API_KEY` | Yes | From https://console.groq.com |
| `GEMINI_API_KEY` | No | Optional Gemini fallback |

---

## License

MIT. Built for clinical education in Uganda.
