## LESSON: Building an EC2 Operations Agent with LangGraph

## Introduction to Your Project

In this lesson, you'll build an intelligent agent that interacts with Amazon EC2 instances using natural language. The agent will be able to:

- List EC2 instances with optional filtering
- Start, stop, and reboot EC2 instances
- Retrieve detailed network information about instances, including IP addresses, security groups, and VPC details

This project combines the power of Large Language Models (LLMs) with AWS EC2 operations through a structured LangGraph workflow.

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

## Building the EC2 Agent Step by Step

### Step 1: Set Up Basic Project Structure

Let's start by creating the essential components for our agent:

```python
from __future__ import annotations

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

# Set up basic logging
logger = logging.getLogger("ec2_agentic")
if not logger.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    logger.setLevel(level)

# Simple EC2 client function
def ec2_client():
    # Uses default AWS credentials (env vars, shared config, IAM role, etc.)
    return boto3.client("ec2")
```

- Imported necessary libraries and modules
- Set up logging for debugging
- Created a simple helper function to get the EC2 client

### Step 2: Define the EC2 Operation Functions

Now, let's implement the core EC2 operations our agent will use:

```python
# ---------- Basic EC2 operations ----------
def _ec2_list_instances(filters: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        ec2 = ec2_client()
        response = ec2.describe_instances(Filters=filters or [])
        instances = []

        for reservation in response.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                instances.append({
                    "instance_id": instance.get("InstanceId"),
                    "state": instance.get("State", {}).get("Name"),
                    "instance_type": instance.get("InstanceType"),
                    "public_ip": instance.get("PublicIpAddress"),
                    "private_ip": instance.get("PrivateIpAddress"),
                    "vpc_id": instance.get("VpcId"),
                    "subnet_id": instance.get("SubnetId"),
                    "tags": instance.get("Tags", [])
                })

        return {"ok": True, "operation": "list_instances", "count": len(instances), "instances": instances}
    except Exception as e:
        logger.exception("list instances error")
        return {"ok": False, "error": str(e)}

def _ec2_instance_info(instance_id: str) -> Dict[str, Any]:
    try:
        ec2 = ec2_client()
        response = ec2.describe_instances(InstanceIds=[instance_id])

        if not response.get("Reservations") or not response["Reservations"][0].get("Instances"):
            return {"ok": False, "error": f"Instance {instance_id} not found"}

        instance = response["Reservations"][0]["Instances"][0]

        # Get security groups info
        security_groups = []
        for sg in instance.get("SecurityGroups", []):
            security_groups.append({
                "id": sg.get("GroupId"),
                "name": sg.get("GroupName")
            })

        # Extract network interfaces info
        network_interfaces = []
        for ni in instance.get("NetworkInterfaces", []):
            network_interfaces.append({
                "id": ni.get("NetworkInterfaceId"),
                "subnet_id": ni.get("SubnetId"),
                "vpc_id": ni.get("VpcId"),
                "private_ip": ni.get("PrivateIpAddress"),
                "public_ip": ni.get("Association", {}).get("PublicIp")
            })

        info = {
            "ok": True,
            "operation": "instance_info",
            "instance_id": instance_id,
            "state": instance.get("State", {}).get("Name"),
            "instance_type": instance.get("InstanceType"),
            "vpc_id": instance.get("VpcId"),
            "subnet_id": instance.get("SubnetId"),
            "public_ip": instance.get("PublicIpAddress"),
            "private_ip": instance.get("PrivateIpAddress"),
            "key_name": instance.get("KeyName"),
            "security_groups": security_groups,
            "network_interfaces": network_interfaces,
            "tags": instance.get("Tags", [])
        }

        return info
    except Exception as e:
        logger.exception("instance info error")
        return {"ok": False, "error": str(e)}

def _ec2_start_instance(instance_id: str) -> Dict[str, Any]:
    try:
        ec2 = ec2_client()
        response = ec2.start_instances(InstanceIds=[instance_id])

        state_changes = response.get("StartingInstances", [])
        if not state_changes:
            return {"ok": False, "error": f"No state change reported for instance {instance_id}"}

        return {
            "ok": True,
            "operation": "start_instance",
            "instance_id": instance_id,
            "previous_state": state_changes[0].get("PreviousState", {}).get("Name"),
            "current_state": state_changes[0].get("CurrentState", {}).get("Name")
        }
    except Exception as e:
        logger.exception("start instance error")
        return {"ok": False, "error": str(e)}

def _ec2_stop_instance(instance_id: str) -> Dict[str, Any]:
    try:
        ec2 = ec2_client()
        response = ec2.stop_instances(InstanceIds=[instance_id])

        state_changes = response.get("StoppingInstances", [])
        if not state_changes:
            return {"ok": False, "error": f"No state change reported for instance {instance_id}"}

        return {
            "ok": True,
            "operation": "stop_instance",
            "instance_id": instance_id,
            "previous_state": state_changes[0].get("PreviousState", {}).get("Name"),
            "current_state": state_changes[0].get("CurrentState", {}).get("Name")
        }
    except Exception as e:
        logger.exception("stop instance error")
        return {"ok": False, "error": str(e)}

def _ec2_reboot_instance(instance_id: str) -> Dict[str, Any]:
    try:
        ec2 = ec2_client()
        ec2.reboot_instances(InstanceIds=[instance_id])

        # Reboot doesn't return state changes, so we need to check the status separately
        return {
            "ok": True,
            "operation": "reboot_instance",
            "instance_id": instance_id,
            "message": f"Reboot request for instance {instance_id} has been sent"
        }
    except Exception as e:
        logger.exception("reboot instance error")
        return {"ok": False, "error": str(e)}
```

- Created five core EC2 operations: list instances, get instance info, start, stop, and reboot instances
- Each function handles its own error cases and returns structured responses
- Included detailed metadata in responses to make agent responses more informative, especially for networking details

### Step 3: Convert Operations to LangChain Tools

Now, let's wrap our EC2 operations as LangChain tools so they can be used by the LLM:

```python
# ---------- LangChain Tools (simple signatures) ----------
@tool
def ec2_list_instances(state: str = None) -> dict:
    """
    List EC2 instances, optionally filtering by state (running, stopped, etc.).
    If state is provided, only instances in that state will be returned.
    """
    filters = []
    if state:
        filters.append({"Name": "instance-state-name", "Values": [state]})
    return _ec2_list_instances(filters=filters)

@tool
def ec2_instance_info(instance_id: str) -> dict:
    """
    Get detailed information about an EC2 instance, including networking details,
    security groups, and VPC information.
    """
    return _ec2_instance_info(instance_id=instance_id)

@tool
def ec2_start_instance(instance_id: str) -> dict:
    """Start an EC2 instance that is currently stopped."""
    return _ec2_start_instance(instance_id=instance_id)

@tool
def ec2_stop_instance(instance_id: str) -> dict:
    """Stop a running EC2 instance."""
    return _ec2_stop_instance(instance_id=instance_id)

@tool
def ec2_reboot_instance(instance_id: str) -> dict:
    """Reboot an EC2 instance (must be in running state)."""
    return _ec2_reboot_instance(instance_id=instance_id)

TOOLS = [ec2_list_instances, ec2_instance_info, ec2_start_instance, ec2_stop_instance, ec2_reboot_instance]
```

- Wrapped each EC2 operation with the `@tool` decorator
- Added clear docstrings to help the LLM understand each tool's purpose
- Created a TOOLS list to hold all available tools

### Step 4: Define the Agent State and Core Components

Next, let's set up the agent's state and core components:

```python
# ---------- Minimal LangGraph agent loop ----------
class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

SYSTEM = SystemMessage(content=(
    "You are an EC2 operations assistant. You can help manage EC2 instances by starting, stopping, "
    "or rebooting them, as well as retrieving information about their network configuration, "
    "security groups, and VPC details. Be concise and ask for missing details (like instance IDs) if needed."
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
- Created a system message to give the LLM its identity and instructions for EC2 operations
- Set up the LLM with tools
- Created the agent_node function that processes messages and generates responses
- Set up the tools_node that will execute the chosen tools

### Step 5: Build the LangGraph State Machine

Now, let's create the LangGraph state machine that will orchestrate the agent's workflow:

```python
def build_agentic_ec2_graph():
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

### Step 6: Create a User-Friendly Interface

Finally, let's create a simple function to run the agent:

```python
# ---------- Public entrypoint ----------
def run_ec2_agentic(prompt: str) -> str:
    """
    Give the agent a natural-language request (e.g., 'List all running EC2 instances').
    The model picks a tool, runs it, and replies with a summary.
    """
    graph = build_agentic_ec2_graph()
    init_state = {"messages": [HumanMessage(content=prompt)]}
    out = graph.invoke(init_state)
    final_msg = out["messages"][-1]
    return getattr(final_msg, "content", str(final_msg))

# ---------- Demo ----------
if __name__ == "__main__":
    # Example (requires OPENAI_API_KEY + AWS creds)
    print(run_ec2_agentic("List all my running EC2 instances"))
    # print(run_ec2_agentic("Get network information for instance i-0abc123def456"))
    # print(run_ec2_agentic("Stop the instance i-0abc123def456"))
    # print(run_ec2_agentic("Start the instance i-0abc123def456"))
```

- Created a run_ec2_agentic function that provides a clean interface for user interaction
- Set up example usage to demonstrate the agent's capabilities
- Prepared the code to be run as a script

## Testing Your EC2 Agent

To test your EC2 agent:

1. Ensure your environment variables are set:
   ```bash
   export AWS_DEFAULT_REGION=us-west-2
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export OPENAI_API_KEY=your_openai_key
   export OPENAI_MODEL=gpt-4o-mini
   ```

2. Run your agent:
   ```bash
   python ec2_agentic_tools.py
   ```

3. Try different commands:
   - "List all my EC2 instances"
   - "How many running instances do I have?"
   - "Show me the networking details for instance i-1234abcd"
   - "Stop instance i-1234abcd"
   - "Start my instance i-1234abcd"
   - "Reboot the instance i-1234abcd"

## Example Interactions

Here are some examples of what your agent can do:

1. **Listing instances:**
   ```
   User: What EC2 instances are currently running?
   Agent: I'll check your running EC2 instances for you.

   You have 3 running EC2 instances:
   1. i-0abc123def456 (t2.micro) - Private IP: 10.0.1.45, Public IP: 54.123.45.67
   2. i-0def456abc789 (t2.small) - Private IP: 10.0.2.67, Public IP: 54.123.45.89
   3. i-0ghi789jkl012 (t3.medium) - Private IP: 10.0.3.89, Public IP: 54.123.45.90

   All instances are in VPC vpc-12345abcdef.
   ```

2. **Getting network details:**
   ```
   User: Show me the network details for instance i-0abc123def456
   Agent: Here are the network details for instance i-0abc123def456:

   State: running
   VPC: vpc-12345abcdef
   Subnet: subnet-abcdef123456
   Private IP: 10.0.1.45
   Public IP: 54.123.45.67

   Security Groups:
   - sg-123abc456def (default)
   - sg-789ghi012jkl (web-servers)

   Network Interfaces:
   - eni-1a2b3c4d (Primary, attached to subnet-abcdef123456)
   ```

3. **Starting an instance:**
   ```
   User: Start the instance i-0def456abc789
   Agent: I've initiated the start operation for instance i-0def456abc789.

   Previous state: stopped
   Current state: pending

   The instance is now starting up. It may take a few minutes to be fully running.
   ```

## Summary

You've successfully built an intelligent EC2 operations agent that can:
- Understand natural language requests for EC2 instance management
- Execute instance control operations (start, stop, reboot)
- Retrieve detailed network information about instances
- Generate helpful responses based on operation results

The architecture follows a clean pattern:
1. Core EC2 operations are defined as Python functions
2. Operations are wrapped as LangChain tools
3. A LangGraph state machine orchestrates the agent's workflow
4. The agent can be easily invoked with simple natural language queries

This pattern can be extended to support additional EC2 operations or integrated with other AWS services for a more comprehensive cloud management assistant.
