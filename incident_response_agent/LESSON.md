# Building an Incident Response Agent with LangGraph

## Introduction to Your Project

In this lesson, you'll build an intelligent incident response agent that automatically investigates CloudWatch alerts, diagnoses issues from logs, and sends notifications to your team. The agent will:

1. Process EventBridge alerts containing incident information
2. Analyze CloudWatch logs to diagnose problems
3. Determine root causes and recommend solutions
4. Send notification emails via SNS

This project combines the power of Large Language Models (LLMs) with AWS monitoring services through a structured LangGraph workflow to automate your incident response process.

## Prerequisites

Before starting, ensure you have:

1. **Python environment**: Python 3.8+ with pip
2. **AWS credentials**:
   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_REGION
   - *Note: If using Skillmix, these credentials are available in the lab details section*

3. **OpenAI API key**:
   - OPENAI_API_KEY
   - *Note: You will need to bring your own OpenAI key*

4. **Required packages**:
   ```
   pip install -U langgraph langchain-openai langchain-core pydantic boto3 awscli
   ```

## Building the Incident Response Agent Step by Step

### Step 1: Set Up Basic Project Structure

Let's start by creating the essential components for our agent:

```python
from __future__ import annotations

import base64
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, TypedDict, Annotated, Optional

import boto3
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Set up basic logging
logger = logging.getLogger("incident_response_agent")
if not logger.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    logger.setLevel(level)

# Helper functions for AWS service clients
def cloudwatch_logs_client():
    return boto3.client("logs")

def sns_client():
    return boto3.client("sns")
```

- Imported necessary libraries for AWS interactions, LangGraph, and LangChain
- Set up logging for troubleshooting
- Created helper functions to get the CloudWatch Logs and SNS clients

### Step 2: Define the Core Operations

Now, let's implement the core CloudWatch and SNS operations our agent will use:

```python
# ---------- Basic CloudWatch and SNS operations ----------
def _get_log_events(log_group: str, log_stream: Optional[str] = None, start_time: Optional[int] = None,
                   end_time: Optional[int] = None, limit: int = 100) -> Dict[str, Any]:
    """Retrieves events from CloudWatch Logs."""
    try:
        logs = cloudwatch_logs_client()

        # If no timeframe specified, default to last 30 minutes
        if not start_time:
            start_time = int((datetime.now() - timedelta(minutes=30)).timestamp() * 1000)
        if not end_time:
            end_time = int(datetime.now().timestamp() * 1000)

        # Different API calls depending on whether we have a specific stream
        events = []

        if log_stream:
            # Get events from a specific log stream
            response = logs.get_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                startTime=start_time,
                endTime=end_time,
                limit=limit
            )

            for event in response.get('events', []):
                events.append({
                    'timestamp': event.get('timestamp'),
                    'message': event.get('message')
                })
        else:
            # Use filter_log_events to search across all streams in the group
            paginator = logs.get_paginator('filter_log_events')

            # Collect events across pages up to our limit
            remaining = limit
            for page in paginator.paginate(
                logGroupName=log_group,
                startTime=start_time,
                endTime=end_time,
                limit=min(remaining, 50)
            ):
                for event in page.get('events', []):
                    events.append({
                        'timestamp': event.get('timestamp'),
                        'message': event.get('message'),
                        'logStream': event.get('logStreamName')
                    })
                    remaining -= 1
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break

        return {
            "ok": True,
            "operation": "get_log_events",
            "log_group": log_group,
            "log_stream": log_stream,
            "start_time": start_time,
            "end_time": end_time,
            "count": len(events),
            "events": events
        }
    except Exception as e:
        logger.exception("get_log_events error")
        return {"ok": False, "error": str(e)}

def _describe_log_group(log_group: str) -> Dict[str, Any]:
    """Get information about a specific log group."""
    try:
        logs = cloudwatch_logs_client()
        response = logs.describe_log_groups(logGroupNamePrefix=log_group, limit=1)

        log_groups = response.get('logGroups', [])
        if not log_groups or log_groups[0]['logGroupName'] != log_group:
            return {"ok": False, "error": f"Log group {log_group} not found"}

        # Get the most recent log streams
        streams_response = logs.describe_log_streams(
            logGroupName=log_group,
            orderBy='LastEventTime',
            descending=True,
            limit=5
        )

        return {
            "ok": True,
            "operation": "describe_log_group",
            "log_group": log_group,
            "retention_in_days": log_groups[0].get('retentionInDays'),
            "stored_bytes": log_groups[0].get('storedBytes'),
            "recent_streams": [
                {
                    "name": stream.get('logStreamName'),
                    "last_event": stream.get('lastEventTimestamp'),
                    "size": stream.get('storedBytes')
                }
                for stream in streams_response.get('logStreams', [])
            ]
        }
    except Exception as e:
        logger.exception("describe_log_group error")
        return {"ok": False, "error": str(e)}

def _send_sns_notification(topic_arn: str, subject: str, message: str) -> Dict[str, Any]:
    """Send an SNS notification."""
    try:
        sns = sns_client()
        response = sns.publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=message
        )

        return {
            "ok": True,
            "operation": "send_sns_notification",
            "topic_arn": topic_arn,
            "message_id": response.get('MessageId'),
        }
    except Exception as e:
        logger.exception("send_sns_notification error")
        return {"ok": False, "error": str(e)}

def _query_log_insights(log_group: str, query: str, start_time: Optional[int] = None,
                       end_time: Optional[int] = None) -> Dict[str, Any]:
    """Run a CloudWatch Logs Insights query."""
    try:
        logs = cloudwatch_logs_client()

        # If no timeframe specified, default to last 30 minutes
        if not start_time:
            start_time = int((datetime.now() - timedelta(minutes=30)).timestamp() * 1000)
        if not end_time:
            end_time = int(datetime.now().timestamp() * 1000)

        # Start the query
        start_query_response = logs.start_query(
            logGroupName=log_group,
            startTime=start_time,
            endTime=end_time,
            queryString=query,
            limit=100
        )

        query_id = start_query_response['queryId']

        # Poll for results (with timeout)
        max_attempts = 20
        attempts = 0
        while attempts < max_attempts:
            query_results = logs.get_query_results(queryId=query_id)
            status = query_results['status']

            if status in ['Complete', 'Failed', 'Cancelled']:
                break

            time.sleep(0.5)
            attempts += 1

        if status != 'Complete':
            return {
                "ok": False,
                "error": f"Query did not complete in time. Status: {status}"
            }

        # Process and return the results
        results = []
        for result in query_results.get('results', []):
            item = {}
            for field in result:
                item[field['field']] = field['value']
            results.append(item)

        return {
            "ok": True,
            "operation": "query_log_insights",
            "log_group": log_group,
            "query": query,
            "status": status,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.exception("query_log_insights error")
        return {"ok": False, "error": str(e)}
```

- Implemented four core operations:
  - `get_log_events`: Retrieves raw log events from CloudWatch Logs
  - `describe_log_group`: Gets information about a log group and its streams
  - `send_sns_notification`: Sends email notifications via SNS
  - `query_log_insights`: Runs CloudWatch Logs Insights queries for advanced analysis
- Each function handles its own error cases and returns structured responses
- Added flexible time range options with sensible defaults

### Step 3: Convert Operations to LangChain Tools

Now, let's wrap our operations as LangChain tools so they can be used by the LLM:

```python
# ---------- LangChain Tools ----------
@tool
def get_log_events(log_group: str, log_stream: Optional[str] = None, start_time: Optional[int] = None,
                  end_time: Optional[int] = None, limit: int = 100) -> dict:
    """
    Retrieve recent log events from CloudWatch Logs.

    Args:
        log_group: The CloudWatch log group name (required)
        log_stream: Optional specific log stream to query
        start_time: Optional start time in milliseconds since epoch
        end_time: Optional end time in milliseconds since epoch
        limit: Maximum number of events to return (default 100)
    """
    return _get_log_events(log_group=log_group, log_stream=log_stream,
                          start_time=start_time, end_time=end_time, limit=limit)

@tool
def describe_log_group(log_group: str) -> dict:
    """
    Get information about a CloudWatch log group including its recent log streams.

    Args:
        log_group: The CloudWatch log group name (required)
    """
    return _describe_log_group(log_group=log_group)

@tool
def send_sns_notification(topic_arn: str, subject: str, message: str) -> dict:
    """
    Send a notification email via SNS.

    Args:
        topic_arn: The ARN of the SNS topic (required)
        subject: Email subject line (required)
        message: Email body content (required)
    """
    return _send_sns_notification(topic_arn=topic_arn, subject=subject, message=message)

@tool
def query_log_insights(log_group: str, query: str, start_time: Optional[int] = None,
                      end_time: Optional[int] = None) -> dict:
    """
    Run a CloudWatch Logs Insights query.

    Args:
        log_group: The CloudWatch log group name (required)
        query: The Logs Insights query string (required)
        start_time: Optional start time in milliseconds since epoch
        end_time: Optional end time in milliseconds since epoch
    """
    return _query_log_insights(log_group=log_group, query=query,
                              start_time=start_time, end_time=end_time)

TOOLS = [get_log_events, describe_log_group, send_sns_notification, query_log_insights]
```

- Wrapped each operation with the `@tool` decorator
- Added detailed docstrings to help the LLM understand each tool's purpose and parameters
- Created a TOOLS list to hold all available tools

### Step 4: Define the Agent State and Core Components

Next, let's set up the agent's state and core components:

```python
# ---------- LangGraph agent loop ----------
class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

SYSTEM = SystemMessage(content=(
    "You are an Incident Response Agent that investigates AWS CloudWatch alerts. "
    "When you receive an incident alert from EventBridge, your job is to:"
    "\n1. Analyze the alert details to understand the problem"
    "\n2. Examine the relevant CloudWatch logs to diagnose the issue"
    "\n3. Identify the root cause of the problem"
    "\n4. Recommend solutions to fix the issue"
    "\n5. Send a notification to the responsible team with your findings"
    "\nBe thorough in your analysis but concise in your recommendations."
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
- Created a detailed system message that gives the LLM clear instructions on its incident response role
- Set up the LLM with our tools
- Created the agent_node function that processes messages and generates responses
- Set up the tools_node that will execute the chosen tools

### Step 5: Build the LangGraph State Machine

Now, let's create the LangGraph state machine that will orchestrate the agent's workflow:

def build_incident_response_graph():
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
- Added conditional logic to determine when to use tools or end the conversation
- Compiled the graph for execution

### Step 6: Create Alert Processing and Interface Functions

Finally, let's create functions to process alerts and provide a user-friendly interface:

```python
# ---------- Public entrypoint ----------
def handle_incident_alert(alert_data: Dict[str, Any]) -> str:
    """
    Process an incident alert from EventBridge.

    Args:
        alert_data: The alert data from EventBridge

    Returns:
        The final response from the agent
    """
    # Format the alert data into a human-readable message
    alert_message = format_alert_message(alert_data)

    # Run the agent with the alert message
    graph = build_incident_response_graph()
    init_state = {"messages": [HumanMessage(content=alert_message)]}
    out = graph.invoke(init_state)
    final_msg = out["messages"][-1]

    return getattr(final_msg, "content", str(final_msg))

def format_alert_message(alert_data: Dict[str, Any]) -> str:
    """Format EventBridge alert data into a readable message for the agent."""
    try:
        # Extract common fields
        alert_time = alert_data.get("time", datetime.now().isoformat())
        source = alert_data.get("source", "Unknown")
        detail_type = alert_data.get("detail-type", "Unknown Alert")
        details = alert_data.get("detail", {})

        # Extract resources if available
        resources = alert_data.get("resources", [])
        resource_str = "\n".join([f"- {r}" for r in resources]) if resources else "None specified"

        # Format the message
        message = f"""
INCIDENT ALERT
Time: {alert_time}
Source: {source}
Type: {detail_type}

Resources:
{resource_str}

Alert Details:
{json.dumps(details, indent=2)}

Please investigate this incident by:
1. Analyzing the relevant CloudWatch logs
2. Determining the root cause
3. Recommending solutions
4. Sending a notification with your findings to {details.get('notificationEmail', 'the responsible team')}
"""
        return message

    except Exception as e:
        logger.exception("Error formatting alert message")
        # Fallback to raw JSON if formatting fails
        return f"Please investigate this incident alert:\n{json.dumps(alert_data, indent=2)}"
```

- Created a `handle_incident_alert` function to process EventBridge alerts
- Added a `format_alert_message` function that converts raw alert data into a readable format
- Set up the workflow to run the agent with the formatted alert
- Added error handling to ensure the agent can still process alerts even if formatting fails

## Testing Your Incident Response Agent

### Step 1: Set Up Your Environment

First, ensure your environment variables are set:

```bash
export AWS_DEFAULT_REGION=us-west-2
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export OPENAI_API_KEY=your_openai_key
export OPENAI_MODEL=gpt-4o-mini
```

### Step 2: Create Test Resources

Let's create the AWS resources needed for testing:

1. Create a CloudWatch Log Group:

```bash
aws logs create-log-group --log-group-name /test/incident-response
```

2. Create an SNS Topic for notifications:

```bash
aws sns create-topic --name IncidentResponseNotifications

# Subscribe your email to the topic (you'll need to confirm subscription via email)
aws sns subscribe \
    --topic-arn arn:aws:sns:us-west-2:YOUR_ACCOUNT_ID:IncidentResponseNotifications \
    --protocol email \
    --notification-endpoint your-email@example.com
```

### Step 3: Generate Test Logs

Create a Python script called `generate_test_logs.py` to populate your CloudWatch log group with test data:

```python
import boto3
import time
import random
import json
from datetime import datetime

# Initialize CloudWatch Logs client
logs = boto3.client('logs')
LOG_GROUP = '/test/incident-response'
LOG_STREAM = f'test-stream-{int(time.time())}'

# Create a log stream
logs.create_log_stream(
    logGroupName=LOG_GROUP,
    logStreamName=LOG_STREAM
)

# Generate some normal logs
normal_logs = [
    {"timestamp": int(time.time() * 1000), "message": json.dumps({
        "level": "INFO",
        "message": f"Application started successfully",
        "timestamp": datetime.now().isoformat(),
        "component": "api-server"
    })},
    {"timestamp": int(time.time() * 1000) + 100, "message": json.dumps({
        "level": "INFO",
        "message": f"Processing request id=req-{random.randint(1000, 9999)}",
        "timestamp": datetime.now().isoformat(),
        "component": "request-handler"
    })},
    {"timestamp": int(time.time() * 1000) + 200, "message": json.dumps({
        "level": "INFO",
        "message": f"Database connection established",
        "timestamp": datetime.now().isoformat(),
        "component": "database"
    })}
]

# Generate error logs (to simulate an incident)
error_logs = []
for i in range(10):
    timestamp = int(time.time() * 1000) + 300 + (i * 100)

    if i < 3:
        # Database connection issues
        error_logs.append({
            "timestamp": timestamp,
            "message": json.dumps({
                "level": "WARN",
                "message": f"Database connection timeout after 5000ms",
                "timestamp": datetime.now().isoformat(),
                "component": "database",
                "connectionId": f"conn-{random.randint(1000, 9999)}"
            })
        })
    elif i < 6:
        # Database errors
        error_logs.append({
            "timestamp": timestamp,
            "message": json.dumps({
                "level": "ERROR",
                "message": f"Query execution failed: connection reset by peer",
                "timestamp": datetime.now().isoformat(),
                "component": "database",
                "queryId": f"q-{random.randint(1000, 9999)}",
                "errorCode": "CONNECTION_RESET"
            })
        })
    else:
        # Application errors due to database issues
        error_logs.append({
            "timestamp": timestamp,
            "message": json.dumps({
                "level": "ERROR",
                "message": f"API request failed due to database error",
                "timestamp": datetime.now().isoformat(),
                "component": "api-server",
                "requestId": f"req-{random.randint(1000, 9999)}",
                "path": f"/api/users/{random.randint(1, 100)}",
                "method": "GET",
                "statusCode": 500
            })
        })

# Put the logs
logs.put_log_events(
    logGroupName=LOG_GROUP,
    logStreamName=LOG_STREAM,
    logEvents=normal_logs + error_logs
)

print(f"Generated {len(normal_logs)} normal logs and {len(error_logs)} error logs in {LOG_GROUP}/{LOG_STREAM}")
print("Use this log stream in your test alert.")
```

Run the script to generate test logs:

```bash
python generate_test_logs.py
```

### Step 4: Create a Test Alert

Now, let's create a sample test alert to pass to our agent.

Save this as `test_incident.py`.

```python
import json
from datetime import datetime
from incident_response_agent import handle_incident_alert

# Replace these values with your actual resources
LOG_GROUP = '/test/incident-response'
LOG_STREAM = 'test-stream-XXXX'  # Use the stream name from the generate_test_logs.py output
SNS_TOPIC_ARN = 'arn:aws:sns:us-west-2:YOUR_ACCOUNT_ID:IncidentResponseNotifications'
YOUR_EMAIL = 'your-email@example.com'

# Create a test alert (simulating EventBridge)
test_alert = {
    "time": datetime.now().isoformat(),
    "source": "aws.cloudwatch",
    "detail-type": "CloudWatch Alarm State Change",
    "resources": [
        "arn:aws:cloudwatch:us-west-2:123456789012:alarm:DatabaseErrorsAlarm"
    ],
    "detail": {
        "alarmName": "DatabaseErrorsAlarm",
        "state": {
            "value": "ALARM",
            "reason": "Database error rate exceeded threshold of 5 errors per minute"
        },
        "previousState": {
            "value": "OK"
        },
        "configuration": {
            "metrics": [
                {
                    "id": "m1",
                    "metricStat": {
                        "metric": {
                            "namespace": "Custom/Database",
                            "name": "ErrorCount",
                            "dimensions": {
                                "ServiceName": "user-api"
                            }
                        }
                    }
                }
            ]
        },
        "logGroupName": LOG_GROUP,
        "logStreamName": LOG_STREAM,
        "notificationEmail": YOUR_EMAIL,
        "topicArn": SNS_TOPIC_ARN
    }
}

# Run the incident response agent
print("Running incident response agent...")
response = handle_incident_alert(test_alert)
print("\n=== AGENT RESPONSE ===\n")
print(response)
```

Save this as `test_incident.py` and run it:

```bash
python test_incident.py
```

The agent will:
1. Analyze the alert details
2. Query the CloudWatch logs
3. Determine the root cause (database connection issues)
4. Send a notification email with its findings

## Summary

You've successfully built an intelligent incident response agent that can:

1. Process EventBridge alerts containing information about infrastructure issues
2. Analyze CloudWatch logs to diagnose problems
3. Use advanced Logs Insights queries for deeper analysis
4. Send notification emails via SNS with findings and recommendations

This agent demonstrates a practical application of AI for DevOps, helping teams:
- Reduce mean time to detection (MTTD) for incidents
- Improve incident response quality with consistent analysis
- Free up engineers from repetitive investigation tasks
- Create better documentation of incidents and their resolutions

The architecture follows a clean pattern:
1. Core AWS operations are defined as Python functions
2. Operations are wrapped as LangChain tools
3. A LangGraph state machine orchestrates the agent's workflow
4. EventBridge alerts trigger the agent's investigation process

You can extend this agent to handle additional AWS services, support more complex analysis patterns, or integrate with your incident management system.
