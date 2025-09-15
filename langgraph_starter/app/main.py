from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger("skillmix.langgraph"); logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(h)

from app.agents.langgraph_agent import LangGraphAdapter

app = FastAPI(title="Skillmix LangGraph Runner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="ui")

class RunPayload(BaseModel):
    provider: str = "langgraph"
    session_id: Optional[str] = None
    input: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None

ADAPTER = LangGraphAdapter()

@app.get("/health")
async def health():
    return {"ok": True}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sid = id(ws)
    try:
        while True:
            raw = await ws.receive_text()
            data = RunPayload.model_validate_json(raw)
            await ws.send_json({"type": "started", "session_id": data.session_id or str(sid)})
            async for event in ADAPTER.astream(data.input, data.config or {}):
                await ws.send_json(event)
            await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        return
    except Exception as e:
        logger.exception("ws error")
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
