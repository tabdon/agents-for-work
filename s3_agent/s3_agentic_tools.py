# s3_agentic_tools_min.py
from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, TypedDict, Annotated

import boto3
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ---------- logging (simple) ----------
logger = logging.getLogger("s3_agentic_min")
if not logger.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    logger.setLevel(level)

# ---------- tiny S3 helper ----------
def s3_client():
    # Uses default AWS credentials (env vars, shared config, IAM role, etc.)
    return boto3.client("s3")

# ---------- Basic S3 operations (no encryption, no fancy options) ----------
def _s3_list(bucket: str, prefix: str = "") -> Dict[str, Any]:
    try:
        s3 = s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        items: List[Dict[str, Any]] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix or ""):
            for obj in page.get("Contents", []) or []:
                items.append(
                    {
                        "key": obj["Key"],
                        "size": obj.get("Size"),
                        "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else None,
                    }
                )
        return {"ok": True, "operation": "list", "bucket": bucket, "prefix": prefix or "", "count": len(items), "objects": items}
    except Exception as e:
        logger.exception("list error")
        return {"ok": False, "error": str(e)}

def _s3_read(bucket: str, key: str) -> Dict[str, Any]:
    try:
        s3 = s3_client()
        obj = s3.get_object(Bucket=bucket, Key=key)
        data: bytes = obj["Body"].read()
        # Keep response small but binary-safe with a short base64 preview
        preview_len = min(len(data), 4096)
        preview_b64 = base64.b64encode(data[:preview_len]).decode("ascii")
        return {
            "ok": True,
            "operation": "read",
            "bucket": bucket,
            "key": key,
            "content_length": obj.get("ContentLength"),
            "content_type": obj.get("ContentType"),
            "preview_b64": preview_b64,
            "preview_bytes": preview_len,
        }
    except Exception as e:
        logger.exception("read error")
        return {"ok": False, "error": str(e)}

def _s3_upload(bucket: str, key: str, text: str, content_type: str | None = None) -> Dict[str, Any]:
    try:
        if text is None:
            return {"ok": False, "error": "Provide 'text' content to upload."}
        s3 = s3_client()
        extra = {"ContentType": content_type} if content_type else {}
        s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), **extra)
        return {"ok": True, "operation": "upload", "bucket": bucket, "key": key}
    except Exception as e:
        logger.exception("upload error")
        return {"ok": False, "error": str(e)}

def _s3_delete(bucket: str, key: str) -> Dict[str, Any]:
    try:
        if not key:
            return {"ok": False, "error": "Provide 'key' to delete."}
        s3 = s3_client()
        s3.delete_object(Bucket=bucket, Key=key)
        return {"ok": True, "operation": "delete", "bucket": bucket, "key": key}
    except Exception as e:
        logger.exception("delete error")
        return {"ok": False, "error": str(e)}

# ---------- LangChain Tools (simple signatures) ----------
@tool
def s3_list(bucket: str, prefix: str = "") -> dict:
    """List objects in an S3 bucket (optionally under a prefix)."""
    return _s3_list(bucket=bucket, prefix=prefix)

@tool
def s3_read(bucket: str, key: str) -> dict:
    """Read an S3 object and return a short base64 preview + metadata."""
    return _s3_read(bucket=bucket, key=key)

@tool
def s3_upload(bucket: str, key: str, text: str, content_type: str | None = None) -> dict:
    """Upload plain text to S3 at the given key."""
    return _s3_upload(bucket=bucket, key=key, text=text, content_type=content_type)

@tool
def s3_delete(bucket: str, key: str) -> dict:
    """Delete a single S3 object by key."""
    return _s3_delete(bucket=bucket, key=key)

TOOLS = [s3_list, s3_read, s3_upload, s3_delete]

# ---------- Minimal LangGraph agent loop ----------
class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

SYSTEM = SystemMessage(content=(
    "You are a simple S3 helper. Use the tools to list, upload text, read, or delete objects. "
    "Be concise and ask for missing details (like bucket/key) if needed."
))

_llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
).bind_tools(TOOLS)

def agent_node(state: MessagesState) -> MessagesState:
    # Provide system + conversation history to the model
    msgs = [SYSTEM] + state["messages"]
    ai = _llm.invoke(msgs)
    # Return only the new message; LangGraph appends it for us
    return {"messages": [ai]}

tools_node = ToolNode(TOOLS)

def build_agentic_s3_graph():
    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()

# ---------- Public entrypoint ----------
def run_s3_agentic(prompt: str) -> str:
    """
    Give the agent a natural-language request (e.g., 'List files in bucket my-bucket').
    The model picks a tool, runs it, and replies with a summary.
    """
    graph = build_agentic_s3_graph()
    init_state = {"messages": [HumanMessage(content=prompt)]}
    out = graph.invoke(init_state)
    final_msg = out["messages"][-1]
    return getattr(final_msg, "content", str(final_msg))

# ---------- Demo ----------
if __name__ == "__main__":
    # Example (requires OPENAI_API_KEY + AWS creds)
     print(run_s3_agentic("Upload the text 'hello' to s3://93dik9isk2oiaf/notes/nuts.txt as text/plain."))
    # print(run_s3_agentic("can you read s3://93kdi3ke93kekd/notes/nuts.txt."))
    # print(run_s3_agentic("list my bucket 93kdi3ke93kekd recursively "))
    # print(run_s3_agentic("can you delete s3://93kdi3ke93kekd/notes/hi.txt "))
