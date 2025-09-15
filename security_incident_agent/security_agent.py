from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, TypedDict, Annotated

import boto3
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ---------- Logging Setup ----------
logger = logging.getLogger("security_agent")
if not logger.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    logger.setLevel(level)

# ---------- AWS Client Helpers ----------
def cloudtrail_client():
    return boto3.client("cloudtrail")

def iam_client():
    return boto3.client("iam")

def sns_client():
    return boto3.client("sns")

# ---------- CloudTrail Investigation Operations ----------
def _investigate_user_activity(user_name: str, hours_back: int = 24) -> Dict[str, Any]:
    """Look up recent activity for a specific user."""
    try:
        ct = cloudtrail_client()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        events = []
        response = ct.lookup_events(
            LookupAttributes=[
                {'AttributeKey': 'Username', 'AttributeValue': user_name}
            ],
            StartTime=start_time,
            EndTime=end_time,
            MaxResults=50
        )

        for event in response.get('Events', []):
            events.append({
                'event_name': event.get('EventName'),
                'event_time': event.get('EventTime').isoformat() if event.get('EventTime') else None,
                'source': event.get('EventSource'),
                'error_code': event.get('ErrorCode'),
                'read_only': event.get('ReadOnly', True)
            })

        return {
            "ok": True,
            "operation": "investigate_user",
            "user": user_name,
            "hours_back": hours_back,
            "event_count": len(events),
            "events": events[:10]  # Limit for response size
        }
    except Exception as e:
        logger.exception("user investigation error")
        return {"ok": False, "error": str(e)}

def _analyze_failed_logins(hours_back: int = 12) -> Dict[str, Any]:
    """Check for failed login attempts across the account."""
    try:
        ct = cloudtrail_client()
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)

        failed_attempts = []
        response = ct.lookup_events(
            LookupAttributes=[
                {'AttributeKey': 'EventName', 'AttributeValue': 'ConsoleLogin'}
            ],
            StartTime=start_time,
            EndTime=end_time,
            MaxResults=100
        )

        for event in response.get('Events', []):
            if event.get('ErrorCode'):
                failed_attempts.append({
                    'user': event.get('Username', 'Unknown'),
                    'time': event.get('EventTime').isoformat() if event.get('EventTime') else None,
                    'error': event.get('ErrorCode'),
                    'source_ip': event.get('SourceIPAddress')
                })

        return {
            "ok": True,
            "operation": "failed_logins",
            "hours_back": hours_back,
            "failed_count": len(failed_attempts),
            "attempts": failed_attempts[:20]  # Limit for response size
        }
    except Exception as e:
        logger.exception("failed login analysis error")
        return {"ok": False, "error": str(e)}

def _get_user_info(user_name: str) -> Dict[str, Any]:
    """Get IAM user details and attached policies."""
    try:
        iam = iam_client()
        user = iam.get_user(UserName=user_name)['User']

        # Get attached policies
        policies = []
        attached = iam.list_attached_user_policies(UserName=user_name)
        for policy in attached.get('AttachedPolicies', []):
            policies.append(policy['PolicyName'])

        # Get groups
        groups = []
        group_response = iam.list_groups_for_user(UserName=user_name)
        for group in group_response.get('Groups', []):
            groups.append(group['GroupName'])

        return {
            "ok": True,
            "operation": "user_info",
            "user": user_name,
            "created": user.get('CreateDate').isoformat() if user.get('CreateDate') else None,
            "last_used": user.get('PasswordLastUsed').isoformat() if user.get('PasswordLastUsed') else None,
            "policies": policies,
            "groups": groups,
            "has_mfa": len(iam.list_mfa_devices(UserName=user_name).get('MFADevices', [])) > 0
        }
    except Exception as e:
        logger.exception("user info error")
        return {"ok": False, "error": str(e)}

def _send_security_alert(severity: str, title: str, details: str, recipient: str = None) -> Dict[str, Any]:
    """Send security alert email to the security team."""
    try:
        # For demo purposes, we'll just log the alert
        # In production, this would use SES or SNS to send actual emails

        recipient = recipient or os.getenv("SECURITY_EMAIL", "security@example.com")

        alert = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity.upper(),
            "title": title,
            "details": details,
            "recipient": recipient
        }

        # Log the alert (in production, send via SES/SNS)
        logger.warning(f"SECURITY ALERT: {json.dumps(alert, indent=2)}")

        return {
            "ok": True,
            "operation": "send_alert",
            "severity": severity,
            "title": title,
            "recipient": recipient,
            "sent_at": alert["timestamp"]
        }
    except Exception as e:
        logger.exception("alert sending error")
        return {"ok": False, "error": str(e)}

# ---------- LangChain Tools ----------
@tool
def investigate_user(user_name: str, hours_back: int = 24) -> dict:
    """Investigate recent CloudTrail activity for a specific user."""
    return _investigate_user_activity(user_name=user_name, hours_back=hours_back)

@tool
def check_failed_logins(hours_back: int = 12) -> dict:
    """Analyze failed login attempts across the AWS account."""
    return _analyze_failed_logins(hours_back=hours_back)

@tool
def lookup_user(user_name: str) -> dict:
    """Get IAM user details including policies, groups, and MFA status."""
    return _get_user_info(user_name=user_name)

@tool
def send_alert(severity: str, title: str, details: str) -> dict:
    """Send a security alert to the security team. Severity: LOW, MEDIUM, HIGH, CRITICAL."""
    return _send_security_alert(severity=severity, title=title, details=details)

TOOLS = [investigate_user, check_failed_logins, lookup_user, send_alert]

# ---------- LangGraph Agent Setup ----------
class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

SYSTEM = SystemMessage(content=(
    "You are a security incident response agent for AWS environments. "
    "When you receive security events, you should:\n"
    "1. Analyze the severity and potential impact\n"
    "2. Investigate the user or resource involved\n"
    "3. Check for patterns of suspicious activity\n"
    "4. Send appropriate alerts based on severity\n"
    "Be thorough but concise. Prioritize based on risk level."
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

def build_security_graph():
    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()

# ---------- Public Interface ----------
def process_security_event(event_data: Dict[str, Any]) -> str:
    """
    Process a security event from CloudTrail/EventBridge.
    The agent analyzes the event and takes appropriate action.
    """
    # Format the event for the agent
    event_summary = f"""
    Security Event Detected:
    - Event Name: {event_data.get('eventName', 'Unknown')}
    - User: {event_data.get('userIdentity', {}).get('userName', 'Unknown')}
    - Source IP: {event_data.get('sourceIPAddress', 'Unknown')}
    - Time: {event_data.get('eventTime', 'Unknown')}
    - Error Code: {event_data.get('errorCode', 'None')}

    Please investigate this event and take appropriate action.
    """

    graph = build_security_graph()
    init_state = {"messages": [HumanMessage(content=event_summary)]}
    out = graph.invoke(init_state)
    final_msg = out["messages"][-1]
    return getattr(final_msg, "content", str(final_msg))

def run_security_agent(prompt: str) -> str:
    """
    Direct interface for testing the security agent with natural language.
    """
    graph = build_security_graph()
    init_state = {"messages": [HumanMessage(content=prompt)]}
    out = graph.invoke(init_state)
    final_msg = out["messages"][-1]
    return getattr(final_msg, "content", str(final_msg))

# ---------- Demo ----------
if __name__ == "__main__":
    # Example security events for testing
    print("=" * 60)
    print("Security Incident Response Agent Demo")
    print("=" * 60)

    # Test with a suspicious event
    suspicious_event = {
        "eventName": "DeleteBucket",
        "userIdentity": {"userName": "suspicious-user"},
        "sourceIPAddress": "198.51.100.42",
        "eventTime": datetime.now().isoformat(),
        "errorCode": None
    }

    print("\nProcessing suspicious S3 deletion event...")
    print(process_security_event(suspicious_event))

    # Uncomment to test other scenarios:
    # print(run_security_agent("Check for failed login attempts in the last 6 hours"))
    # print(run_security_agent("Investigate user 'admin-user' activity for the last 48 hours"))
    # print(run_security_agent("Look up IAM details for user 'suspicious-user'"))
