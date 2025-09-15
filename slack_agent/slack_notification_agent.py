# s3_notification_agent.py
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, TypedDict, Annotated

import requests
import boto3
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# -------------------- logging --------------------
logger = logging.getLogger("s3_notification_agent")
if not logger.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    logger.setLevel(level)

# -------------------- AWS helpers --------------------
def dynamodb_client():
    return boto3.resource("dynamodb")


# -------------------- utils --------------------
def _truncate_json(obj: Any, max_len: int = 2800) -> str:
    try:
        s = json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return s if len(s) <= max_len else s[:max_len] + "\n…(truncated)…"


def _derive_slack_channel(severity: str) -> str:
    s = (severity or "info").lower()
    if s == "critical":
        return "#incidents"
    if s == "warning":
        return "#alerts"
    return "#notifications"


# -------------------- core operations --------------------
def _process_webhook(data: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Normalize incoming webhook data."""
    try:
        if not data:
            return {"ok": False, "operation": "process_webhook", "error": "Empty data received"}

        if source == "monitoring":
            return {
                "ok": True,
                "operation": "process_webhook",
                "source": source,
                "alert_type": data.get("alert_type", "unknown"),
                "severity": data.get("severity", "info"),
                "resource": data.get("resource", "unknown"),
                "message": data.get("message", "No message provided"),
                "raw_data": data,
                "timestamp": data.get("timestamp"),
                "metrics": data.get("metrics"),
            }
        else:
            return {
                "ok": True,
                "operation": "process_webhook",
                "source": source,
                "raw_data": data,
                "message": data.get("message") or "New event received",
            }
    except Exception as e:
        logger.exception("Webhook processing error")
        return {"ok": False, "operation": "process_webhook", "error": str(e)}


def _store_notification(notification_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if not notification_data:
            return {"ok": False, "operation": "store_notification", "error": "No notification data provided"}

        table_name = os.getenv("DYNAMODB_TABLE", "NotificationsTable")
        table = dynamodb_client().Table(table_name)

        notification_id = notification_data.get("id", f"notif_{os.urandom(4).hex()}")
        item = {
            "notification_id": notification_id,
            "timestamp": notification_data.get("timestamp", int(time.time())),
            "source": notification_data.get("source", "unknown"),
            "status": notification_data.get("status", "new"),
            "data": json.dumps(notification_data, ensure_ascii=False),
        }

        table.put_item(Item=item)
        return {"ok": True, "operation": "store_notification", "notification_id": notification_id}
    except Exception as e:
        logger.exception("DynamoDB storage error")
        return {"ok": False, "operation": "store_notification", "error": str(e)}


def _send_slack(message: str, channel: str, attachments: Optional[List[Dict[str, Any]]], severity: str, dry_run: bool):
    """Send via Slack webhook, or fake it in DRY_RUN."""
    try:
        if dry_run:
            return {
                "ok": True,
                "operation": "send_slack_notification",
                "dry_run": True,
                "channel": channel,
                "message_preview": message[:120],
            }

        slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not slack_webhook_url:
            return {
                "ok": False,
                "operation": "send_slack_notification",
                "error": "SLACK_WEBHOOK_URL not configured (set DRY_RUN=1 for local tests)",
            }

        attachments_list = attachments or []
        if attachments_list:
            for a in attachments_list:
                if severity == "critical":
                    a["color"] = "#FF0000"
                elif severity == "warning":
                    a["color"] = "#FFA500"
                else:
                    a["color"] = "#36C5F0"
        else:
            color = "#FF0000" if severity == "critical" else "#FFA500" if severity == "warning" else "#36C5F0"
            attachments_list = [{"color": color}]

        payload = {"channel": channel, "text": message, "attachments": attachments_list}
        resp = requests.post(slack_webhook_url, data=json.dumps(payload), headers={"Content-Type": "application/json"})
        if resp.status_code != 200:
            return {
                "ok": False,
                "operation": "send_slack_notification",
                "error": f"Slack API error: {resp.status_code} - {resp.text}",
            }
        return {"ok": True, "operation": "send_slack_notification", "status_code": resp.status_code, "channel": channel}
    except Exception as e:
        logger.exception("Slack notification error")
        return {"ok": False, "operation": "send_slack_notification", "error": str(e)}


def _update_notification_status(notification_id: str, status: str) -> Dict[str, Any]:
    try:
        if not notification_id:
            return {"ok": False, "operation": "update_notification_status", "error": "No notification ID provided"}

        table_name = os.getenv("DYNAMODB_TABLE", "NotificationsTable")
        table = dynamodb_client().Table(table_name)
        table.update_item(
            Key={"notification_id": notification_id},
            UpdateExpression="set #status = :s",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":s": status},
            ReturnValues="UPDATED_NEW",
        )
        return {
            "ok": True,
            "operation": "update_notification_status",
            "notification_id": notification_id,
            "new_status": status,
        }
    except Exception as e:
        logger.exception("Status update error")
        return {"ok": False, "operation": "update_notification_status", "error": str(e)}


# -------------------- ONE composite LangChain tool --------------------
@tool
def handle_notification(
    payload: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    dry_run: Optional[bool] = None,
) -> dict:
    """
    End-to-end handler to avoid LangGraph loops.

    Accepts EITHER:
      - payload: { "data": <dict>, "source": <str>, "dry_run": <bool?> }
    OR:
      - data: <dict>, source: <str>, dry_run: <bool?>

    In local tests, DRY_RUN defaults to true if SLACK_WEBHOOK_URL is missing or DRY_RUN env is truthy.
    """
    try:
        # Normalize inputs so downstream logic only deals with `payload`
        if payload is None:
            payload = {"data": data or {}, "source": source or "unknown"}
            if dry_run is not None:
                payload["dry_run"] = bool(dry_run)

        data = payload.get("data") or {}
        source = payload.get("source") or "unknown"

        # DRY_RUN defaulting
        dry_env = str(os.getenv("DRY_RUN", "")).lower() in ("1", "true", "yes", "on")
        effective_dry_run = bool(payload.get("dry_run", dry_env or not os.getenv("SLACK_WEBHOOK_URL")))

        processed = _process_webhook(data, source)
        if not processed.get("ok"):
            return {"ok": False, "stage": "process", "result": processed}

        stored = _store_notification(processed)
        notification_id = stored.get("notification_id") if stored.get("ok") else None

        # Build Slack message + attachments deterministically
        severity = (processed.get("severity") or "info").lower()
        channel = _derive_slack_channel(severity)
        bits = []
        if processed.get("alert_type"):
            bits.append(f"[{processed['alert_type']}]")
        if processed.get("resource"):
            bits.append(processed["resource"] + ":")
        bits.append(processed.get("message") or "New event received")
        message = " ".join(bits)

        raw_payload = processed.get("raw_data", processed)
        payload_str = _truncate_json(raw_payload, max_len=2800)
        attachments = [
            {
                "mrkdwn": True,
                "text": (
                    f"*Source:* `{processed.get('source','unknown')}` | "
                    f"*Severity:* `{severity}`"
                    + (f" | *Alert:* `{processed.get('alert_type')}`" if processed.get("alert_type") else "")
                    + (f" | *Resource:* `{processed.get('resource')}`" if processed.get("resource") else "")
                ),
            },
            {"mrkdwn": True, "text": f"*Raw payload*:\n```json\n{payload_str}\n```"},
        ]

        sent = _send_slack(message, channel, attachments, severity, dry_run=effective_dry_run)

        if notification_id and sent.get("ok"):
            _update_notification_status(notification_id, "sent")

        return {
            "ok": processed.get("ok") and stored.get("ok") and sent.get("ok"),
            "dry_run": effective_dry_run,
            "processed": processed,
            "stored": stored,
            "sent": sent,
        }
    except Exception as e:
        logger.exception("handle_notification error")
        return {"ok": False, "stage": "handle_notification", "error": str(e)}


TOOLS = [handle_notification]


# -------------------- LangGraph (single-shot) --------------------
class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


SYSTEM = SystemMessage(
    content=(
        "You are a Notification Agent.\n"
        "Call the SINGLE tool `handle_notification` exactly once with:\n"
        '{ "data": <full webhook dict>, "source": "<string>" }\n'
        "Do not call any other tools. After the tool result, summarize what happened and STOP."
    )
)

_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0).bind_tools(TOOLS)


def agent_node(state: MessagesState) -> MessagesState:
    msgs = [SYSTEM] + state["messages"]
    ai = _llm.invoke(msgs)
    return {"messages": [ai]}


tools_node = ToolNode(TOOLS)


def build_notification_agent_graph():
    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)

    g.add_edge(START, "agent")
    # If AI calls a tool -> go to tools; else END
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    # IMPORTANT: single-shot -> finish after tools
    g.add_edge("tools", END)
    return g.compile()


# -------------------- Public entrypoint --------------------
def process_notification(webhook_data: Dict[str, Any], source: str) -> str:
    """
    Process a webhook and (optionally) send Slack. Returns a short summary.
    Local runs default to DRY_RUN unless SLACK_WEBHOOK_URL is set or you pass dry_run=False in the tool payload.
    """
    graph = build_notification_agent_graph()
    prompt = (
        f"I received a webhook from {source} with the following data:\n"
        f"{json.dumps(webhook_data, indent=2, ensure_ascii=False)}\n\n"
        f"Please handle it."
    )
    # The model will call the single composite tool once
    init_state = {"messages": [HumanMessage(content=prompt)]}
    out = graph.invoke(init_state, config={"recursion_limit": 3})
    final_msg = out["messages"][-1]
    return getattr(final_msg, "content", str(final_msg))


# -------------------- Demo --------------------
if __name__ == "__main__":
    # Tip: set DRY_RUN=1 for local testing, or set SLACK_WEBHOOK_URL to actually send.
    example_monitoring_alert = {
        "alert_type": "high_cpu",
        "severity": "warning",
        "resource": "web-server-12",
        "message": "CPU usage exceeded 85% for 5 minutes",
        "timestamp": "2023-05-15T14:22:10Z",
        "metrics": {"cpu_usage": 87.5, "memory_usage": 65.2},
    }
    print(process_notification(example_monitoring_alert, "monitoring"))
