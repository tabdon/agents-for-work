## `skillmix-langgraph-starter` (LangGraph + OpenAI)

A production‑ready minimal LangGraph app that calls OpenAI with a single node, streams a final message over WebSocket, and serves a tiny test UI.

> ⚡ Quick start
>
> ```bash
> git clone <your-url>/skillmix-langgraph-starter
> cd skillmix-langgraph-starter
> cp .env.example .env   # set OPENAI_API_KEY
> python -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
> # open http://localhost:8000/ui/
> ```

### Layout

```
skillmix-langgraph-starter/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ .gitignore
└─ app/
   ├─ __init__.py
   ├─ main.py                 # FastAPI + WebSocket
   ├─ agents/
   │  ├─ __init__.py
   │  └─ langgraph_agent.py   # compiled LangGraph + adapter
   └─ static/
      └─ index.html           # tiny chat UI
```

### README.md

````md
# Skillmix LangGraph Starter

A lightweight starter to build, run, and test LangGraph agents with FastAPI + WebSockets and a minimal web UI.

## Features
- **One‑node LangGraph** that calls OpenAI via `langchain-openai`.
- **FastAPI + WS** with simple events: `started`, `token`, `log`, `error`, `done`.
- **Zero‑config UI** at `/ui`.
- **Env secrets** via `.env` + `python-dotenv`.

## Setup
```bash
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
````

## Env Vars

* `OPENAI_API_KEY` (required)
* `OPENAI_MODEL` (default `gpt-4o-mini`)
* `OPENAI_TEMPERATURE` (default `0`)

## Protocol

Client → WS `/ws`:

```json
{"provider":"langgraph","input":{"prompt":"Hello"}}
```

Server events: `started` → `token`(s) → `done` (or `error`).

````

---

## Security & Ops (both repos)

* Do **not** commit `.env` (already ignored).
* In production, restrict CORS and place behind TLS + reverse proxy (nginx).
* Prefer **instance roles** over static keys (especially for Bedrock).
* Add per‑user rate limits and request budgets upstream in Skillmix.

## Next Steps You Might Want

* Dockerfile + compose for labs or local.
* Poetry/uv packaging instead of `requirements.txt`.
* True incremental **token streaming** from LangGraph with `stream_mode="updates"`.
* Structured tool events (`type: "tool"`) if you add tools or a planner.
