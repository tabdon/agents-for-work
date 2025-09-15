# Building a Security Incident Response Agent with LangGraph

## Introduction to Your Project

In this lesson, you'll build an intelligent security incident response agent that monitors CloudTrail events through EventBridge. The agent will be able to:

- Analyze incoming security events from CloudTrail
- Investigate suspicious activities by querying additional CloudTrail data
- Look up user and IAM information for context
- Send prioritized email notifications to the appropriate security teams

This project combines the power of Large Language Models (LLMs) with AWS security services through a structured LangGraph workflow.

## Prerequisites

Before starting, ensure you have:

1. **Python environment**: Python 3.10+ with pip

2. **AWS credentials**:
   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_REGION
   - *Note: If using Skillmix, these credentials are available in the lab details section*

3. **OpenAI API key**:
   - OPENAI_API_KEY
   - *Note: You will need to bring your own OpenAI key*

4. **Email Configuration**:
   - SECURITY_EMAIL (where to send notifications)
   - *Note: For testing, we'll simulate email sending*

## Building the Security Incident Response Agent Step by Step

### Step 1: Set Up Basic Project Structure

Let's start by creating the essential components for our agent. Create a file called `security_agent.py`, and start building the code as follows.

```python
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

# Set up basic logging
logger = logging.getLogger("security_agent")
if not logger.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    logger.setLevel(level)

# Simple AWS client functions
def cloudtrail_client():
    return boto3.client("cloudtrail")

def iam_client():
    return boto3.client("iam")

def sns_client():
    return boto3.client("sns")
```

- Imported necessary libraries and modules
- Set up logging for debugging
- Created helper functions to get AWS service clients

### Step 2: Define the Security Investigation Functions

Now, let's implement the core security investigation operations our agent will use:

```python
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
```

- Created investigation functions for user activity analysis
- Added failed login detection capability
- Included IAM user information lookup
- Set up alert notification system (simulated for testing)

### Step 3: Convert Operations to LangChain Tools

Now, let's wrap our security operations as LangChain tools so they can be used by the LLM:

```python
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
```

- Wrapped each security operation with the `@tool` decorator
- Added clear docstrings to help the LLM understand each tool's purpose
- Created a TOOLS list to hold all available tools

### Step 4: Define the Agent State and Core Components

Next, let's set up the agent's state and core components:

```python
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
```

- Defined the agent's state using TypedDict
- Created a system message with clear security response instructions
- Set up the LLM with tools
- Created the agent_node function for processing messages
- Set up the tools_node for executing chosen tools

### Step 5: Build the LangGraph State Machine

Now, let's create the LangGraph state machine that will orchestrate the agent's workflow:

```python
def build_security_graph():
    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()
```

- Created a StateGraph with our MessagesState
- Added the agent and tools nodes
- Set up the flow: START → agent → [tools or END] → agent
- Added conditional logic for tool usage
- Compiled the graph for execution

### Step 6: Create the Main Interface and Event Handler

Finally, let's create functions to handle security events and run the agent:

```python
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
```

- Created process_security_event function to handle CloudTrail events
- Added run_security_agent for direct natural language interaction
- Included demo code with example security events

## Testing Your Security Agent

### Step 1: Setup and Installation

Run these commands at a CLI to setup a virtual environment and install packages:

```bash
# Create a virtual env folder
python -m venv venv

# Activate the virtual env
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -U langgraph langchain-openai langchain-core pydantic boto3 awscli
```

### Step 2: Configure Environment Variables

Set up your environment variables:

```bash
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export OPENAI_API_KEY=your_openai_key
export OPENAI_MODEL=gpt-4o-mini
export SECURITY_EMAIL=security-team@example.com
```

### Step 3: Generate Test Data

Create a file called `generate_test_events.py` to simulate security events:

```python
import json
import random
from datetime import datetime, timedelta

def generate_test_events():
    """Generate sample CloudTrail events for testing."""

    events = []

    # Suspicious deletion event
    events.append({
        "eventName": "DeleteBucket",
        "userIdentity": {"userName": "compromised-user"},
        "sourceIPAddress": "192.0.2.100",
        "eventTime": datetime.now().isoformat(),
        "errorCode": None,
        "responseElements": {"bucketName": "critical-data-bucket"}
    })

    # Failed login attempts
    for i in range(5):
        events.append({
            "eventName": "ConsoleLogin",
            "userIdentity": {"userName": f"user-{i}"},
            "sourceIPAddress": f"203.0.113.{random.randint(1,255)}",
            "eventTime": (datetime.now() - timedelta(hours=i)).isoformat(),
            "errorCode": "Failed authentication",
            "responseElements": {"ConsoleLogin": "Failure"}
        })

    # Unauthorized API calls
    events.append({
        "eventName": "CreateAccessKey",
        "userIdentity": {"userName": "junior-dev"},
        "sourceIPAddress": "198.51.100.1",
        "eventTime": datetime.now().isoformat(),
        "errorCode": "UnauthorizedAccess",
        "responseElements": None
    })

    # Root account usage
    events.append({
        "eventName": "ConsoleLogin",
        "userIdentity": {"type": "Root", "userName": "root"},
        "sourceIPAddress": "172.16.0.1",
        "eventTime": datetime.now().isoformat(),
        "errorCode": None,
        "responseElements": {"ConsoleLogin": "Success"}
    })

    return events

if __name__ == "__main__":
    test_events = generate_test_events()

    # Save events to file
    with open("test_events.json", "w") as f:
        json.dump(test_events, f, indent=2)

    print(f"Generated {len(test_events)} test events")
    print("\nSample event:")
    print(json.dumps(test_events[0], indent=2))
```

### Step 4: Test the Agent

Run your security agent with the test events:

```python
# In your Python interpreter or a test script:
import json
from security_agent import process_security_event, run_security_agent

# Load test events
with open("test_events.json", "r") as f:
    test_events = json.load(f)

# Process a critical event
critical_event = test_events[0]  # The DeleteBucket event
print("Processing critical event:")
print(process_security_event(critical_event))

# Test natural language queries
print("\nTesting investigation commands:")
print(run_security_agent("Check for any failed login attempts in the last 12 hours"))
print(run_security_agent("Investigate the user 'compromised-user' for the last 24 hours"))
```

### Step 5: Verify Agent Actions

Check the logs to see the security alerts that would be sent:

```bash
# View the agent logs
python security_agent.py 2>&1 | grep "SECURITY ALERT"
```

## Summary

You've successfully built an intelligent security incident response agent that can:
- Process and analyze CloudTrail security events
- Investigate user activities and patterns
- Detect suspicious behaviors like failed logins
- Send prioritized alerts to security teams

The architecture follows a clean pattern:
1. Core security investigation functions are defined as Python functions
2. Operations are wrapped as LangChain tools
3. A LangGraph state machine orchestrates the agent's workflow
4. The agent can process both structured events and natural language queries

This pattern can be extended to:
- Integrate with additional AWS security services (GuardDuty, Security Hub)
- Implement automated remediation actions
- Create more sophisticated threat detection patterns
- Build a complete Security Operations Center (SOC) automation system
