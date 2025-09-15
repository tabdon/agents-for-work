# Building an S3 Management Agent with LangGraph

## Introduction to Your Project

In this lesson, you'll build an intelligent agent that interacts with Amazon S3 storage using natural language. The agent will be able to:

- List files in S3 buckets
- Read file contents from S3
- Upload text files to S3
- Delete files from S3

This project combines the power of Large Language Models (LLMs) with AWS S3 operations through a structured LangGraph workflow.

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


## Building the S3 Agent Step by Step

### Step 1: Set Up Basic Project Structure

Let's start by creating the essential components for our agent. Create a file called s3_agent.py, and start buiding the code as follows.

```python
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

# Set up basic logging
logger = logging.getLogger("s3_agentic_min")
if not logger.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    logger.setLevel(level)

# Simple S3 client function
def s3_client():
    # Uses default AWS credentials (env vars, shared config, IAM role, etc.)
    return boto3.client("s3")
```

- Imported necessary libraries and modules
- Set up logging for debugging
- Created a simple helper function to get the S3 client

### Step 2: Define the S3 Operation Functions

Now, let's implement the core S3 operations our agent will use:

```python
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
```

- Created four core S3 operations: list, read, upload, and delete
- Each function handles its own error cases and returns structured responses
- Included metadata in responses to make agent responses more informative

### Step 3: Convert Operations to LangChain Tools

Now, let's wrap our S3 operations as LangChain tools so they can be used by the LLM:

```python
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
```

- Wrapped each S3 operation with the `@tool` decorator
- Added clear docstrings to help the LLM understand each tool's purpose
- Created a TOOLS list to hold all available tools

### Step 4: Define the Agent State and Core Components

Next, let's set up the agent's state and core components:

```python
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
```

- Defined the agent's state using TypedDict
- Created a system message to give the LLM its identity and instructions
- Set up the LLM with tools
- Created the agent_node function that processes messages and generates responses
- Set up the tools_node that will execute the chosen tools

### Step 5: Build the LangGraph State Machine

Now, let's create the LangGraph state machine that will orchestrate the agent's workflow:

```python
def build_agentic_s3_graph():
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
    # Uncomment lines to run them during testing
    print(run_s3_agentic("Upload the text 'hello' to s3://<bucket>/notes/nuts.txt as text/plain."))
    # print(run_s3_agentic("can you read s3://93kdi3ke93kekd/notes/nuts.txt."))
    # print(run_s3_agentic("list my bucket 93kdi3ke93kekd recursively "))
    # print(run_s3_agentic("can you delete s3://93kdi3ke93kekd/notes/hi.txt "))
```

- Created a run_s3_agentic function that provides a clean interface for user interaction
- Set up example usage to demonstrate the agent's capabilities
- Prepared the code to be run as a script

## Testing Your S3 Agent

To test your S3 agent:

1. Setup virtualenv and install packages

  Run these commands at a CLI to setup a virtual environment and install packages. This assumes your computer has Pythong 3.10+ installed.

  ```
  # create a virtual env folder
  python -m venv venv

  # activate the virtual env
  source venv/bin/activate

  # install packages
  pip install -U langgraph langchain-openai langchain-core pydantic boto3 awscli
  ```

1. Ensure your environment variables are set:
   ```bash
   export AWS_DEFAULT_REGION=us-west-2
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export OPENAI_API_KEY=your_openai_key
   export OPENAI_MODEL=gpt-4o-mini
   ```

2. Create a test bucket (ensure the name is globally unique):
   ```bash
   aws s3api create-bucket --create-bucket-configuration LocationConstraint=us-west-2 --bucket your-unique-bucket-name
   ```

3. Run your agent:
   ```bash
   python s3_agentic_tools.py
   ```

4. Verify the actions in AWS:
   ```bash
   aws s3 ls s3://your-unique-bucket-name --recursive
   ```

## Summary

You've successfully built an intelligent S3 management agent that can:
- Understand natural language requests for S3 operations
- Execute operations on S3 buckets using AWS credentials
- Generate helpful responses based on operation results

The architecture follows a clean pattern:
1. Core S3 operations are defined as Python functions
2. Operations are wrapped as LangChain tools
3. A LangGraph state machine orchestrates the agent's workflow
4. The agent can be easily invoked with simple natural language queries

This pattern can be extended to support additional S3 operations or integrated with other AWS services for a more comprehensive cloud management assistant.
