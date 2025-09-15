# ec2_agentic_tools.py
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

# ---------- logging (simple) ----------
logger = logging.getLogger("ec2_agentic")
if not logger.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    logger.setLevel(level)

# ---------- tiny EC2 helper ----------
def ec2_client():
    # Uses default AWS credentials (env vars, shared config, IAM role, etc.)
    return boto3.client("ec2")

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

def build_agentic_ec2_graph():
    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()

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
