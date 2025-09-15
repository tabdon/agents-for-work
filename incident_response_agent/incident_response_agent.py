# incident_response_agent.py
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

# ---------- logging (simple) ----------
logger = logging.getLogger("incident_response_agent")
if not logger.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    logger.setLevel(level)

# ---------- AWS Service Clients ----------
def cloudwatch_logs_client():
    # Uses default AWS credentials (env vars, shared config, IAM role, etc.)
    return boto3.client("logs")

def sns_client():
    # Uses default AWS credentials (env vars, shared config, IAM role, etc.)
    return boto3.client("sns")

# ---------- Basic CloudWatch and SNS operations ----------
def _get_log_events(log_group: str, log_stream: Optional[str] = None, start_time: Optional[int] = None,
                   end_time: Optional[int] = None, limit: int = 100) -> Dict[str, Any]:
    """
    Retrieves events from CloudWatch Logs.

    Args:
        log_group: The CloudWatch log group name
        log_stream: Optional specific log stream to query
        start_time: Optional start time in milliseconds since epoch
        end_time: Optional end time in milliseconds since epoch
        limit: Maximum number of events to return
    """
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
                limit=min(remaining, 50)  # AWS limits to 50 per page
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
    """
    Send an SNS notification.

    Args:
        topic_arn: The ARN of the SNS topic
        subject: Email subject line
        message: Email body content
    """
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
    """
    Run a CloudWatch Logs Insights query.

    Args:
        log_group: The CloudWatch log group name
        query: The Logs Insights query string
        start_time: Optional start time in milliseconds since epoch
        end_time: Optional end time in milliseconds since epoch
    """
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

def build_incident_response_graph():
    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()

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

# ---------- Demo ----------
if __name__ == "__main__":
    # Example test alert (simulating EventBridge)
    test_alert = {
        "time": datetime.now().isoformat(),
        "source": "aws.cloudwatch",
        "detail-type": "CloudWatch Alarm State Change",
        "resources": [
            "arn:aws:cloudwatch:us-west-2:123456789012:alarm:HighCPUAlarm"
        ],
        "detail": {
            "alarmName": "HighCPUAlarm",
            "state": {
                "value": "ALARM",
                "reason": "CPU utilization exceeded 80% threshold"
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
                                "namespace": "AWS/EC2",
                                "name": "CPUUtilization",
                                "dimensions": {
                                    "InstanceId": "i-1234567890abcdef0"
                                }
                            }
                        }
                    }
                ]
            },
            "logGroupName": "/aws/lambda/my-service-production",
            "notificationEmail": "ops-team@example.com",
            "topicArn": "arn:aws:sns:us-west-2:123456789012:AlertNotifications"
        }
    }

    # Run the agent with the test alert
    response = handle_incident_alert(test_alert)
    print("\n=== AGENT RESPONSE ===\n")
    print(response)
