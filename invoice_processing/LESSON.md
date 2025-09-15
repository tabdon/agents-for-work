# Building an Invoice Processing Agent with LangGraph

## Introduction to Your Project

In this lesson, you'll build an intelligent agent that processes invoice images using natural language. The agent will be able to:

- Receive invoice images (base64 encoded)
- Save invoice images to S3 storage
- Extract key information from invoices using OpenAI's vision model
- Store extracted invoice data in DynamoDB
- Provide summaries of processed invoices

This project combines the power of OpenAI's vision capabilities with AWS services (S3 and DynamoDB) through a structured LangGraph workflow.

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

## Building the Invoice Agent Step by Step

### Step 1: Set Up Basic Project Structure

Let's start by creating the essential components for our agent. Create a file called invoice_agent.py, and start building the code as follows.

```python
from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, TypedDict, Annotated

import boto3
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Set up basic logging
logger = logging.getLogger("invoice_agent")
if not logger.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    logger.setLevel(level)

# AWS client helper functions
def s3_client():
    return boto3.client("s3")

def dynamodb_client():
    return boto3.resource("dynamodb")
```

- Imported necessary libraries and modules
- Set up logging for debugging
- Created helper functions for AWS clients

### Step 2: Define the Core Processing Functions

Now, let's implement the core operations our agent will use:

```python
# ---------- Core invoice processing operations ----------
def _save_invoice_to_s3(bucket: str, key: str, image_base64: str) -> Dict[str, Any]:
    """Save a base64 encoded invoice image to S3."""
    try:
        s3 = s3_client()
        # Decode base64 image
        image_data = base64.b64decode(image_base64)

        # Upload to S3
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=image_data,
            ContentType="image/png"  # Adjust as needed
        )

        return {
            "ok": True,
            "operation": "save_to_s3",
            "bucket": bucket,
            "key": key,
            "size": len(image_data)
        }
    except Exception as e:
        logger.exception("S3 save error")
        return {"ok": False, "error": str(e)}

def _extract_invoice_data(image_base64: str) -> Dict[str, Any]:
    """Extract invoice information using OpenAI's vision model."""
    try:
        # Initialize OpenAI client with vision model
        client = ChatOpenAI(
            model="gpt-4o-mini",  # or gpt-4-vision-preview
            temperature=0
        )

        # Create message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the following information from this invoice: invoice number, date, vendor name, total amount, line items (description and amount). Return as JSON."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

        # Get extraction from OpenAI
        response = client.invoke(messages)

        # Parse the response
        try:
            invoice_data = json.loads(response.content)
        except:
            # If not valid JSON, return raw content
            invoice_data = {"raw_extraction": response.content}

        return {
            "ok": True,
            "operation": "extract_data",
            "invoice_data": invoice_data
        }
    except Exception as e:
        logger.exception("Extraction error")
        return {"ok": False, "error": str(e)}

def _save_to_dynamodb(table_name: str, invoice_data: dict) -> Dict[str, Any]:
    """Save extracted invoice data to DynamoDB."""
    try:
        dynamodb = dynamodb_client()
        table = dynamodb.Table(table_name)

        # Add a unique ID and timestamp if not present
        if "invoice_id" not in invoice_data:
            invoice_data["invoice_id"] = f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        invoice_data["processed_at"] = datetime.now().isoformat()

        # Put item in DynamoDB
        table.put_item(Item=invoice_data)

        return {
            "ok": True,
            "operation": "save_to_dynamodb",
            "table": table_name,
            "invoice_id": invoice_data.get("invoice_id")
        }
    except Exception as e:
        logger.exception("DynamoDB save error")
        return {"ok": False, "error": str(e)}

def _create_invoice_summary(invoice_data: dict) -> Dict[str, Any]:
    """Create a nice summary of the processed invoice."""
    try:
        summary = f"""
Invoice Processing Summary:
--------------------------
Invoice Number: {invoice_data.get('invoice_number', 'N/A')}
Date: {invoice_data.get('date', 'N/A')}
Vendor: {invoice_data.get('vendor_name', 'N/A')}
Total Amount: {invoice_data.get('total_amount', 'N/A')}

Line Items:
"""
        line_items = invoice_data.get('line_items', [])
        if line_items:
            for item in line_items:
                summary += f"  - {item.get('description', 'N/A')}: {item.get('amount', 'N/A')}\n"
        else:
            summary += "  No line items found\n"

        return {
            "ok": True,
            "operation": "create_summary",
            "summary": summary
        }
    except Exception as e:
        logger.exception("Summary creation error")
        return {"ok": False, "error": str(e)}
```

- Created functions for S3 storage, data extraction, DynamoDB storage, and summary generation
- Each function handles errors gracefully and returns structured responses
- Included metadata to make agent responses more informative

### Step 3: Convert Operations to LangChain Tools

Now, let's wrap our operations as LangChain tools:

```python
# ---------- LangChain Tools ----------
@tool
def save_invoice_image(bucket: str, key: str, image_base64: str) -> dict:
    """Save a base64 encoded invoice image to S3 bucket."""
    return _save_invoice_to_s3(bucket=bucket, key=key, image_base64=image_base64)

@tool
def extract_invoice_info(image_base64: str) -> dict:
    """Extract information from invoice image using OpenAI vision model."""
    return _extract_invoice_data(image_base64=image_base64)

@tool
def store_invoice_data(table_name: str, invoice_data: dict) -> dict:
    """Store extracted invoice data in DynamoDB table."""
    return _save_to_dynamodb(table_name=table_name, invoice_data=invoice_data)

@tool
def generate_invoice_summary(invoice_data: dict) -> dict:
    """Generate a human-readable summary of the invoice."""
    return _create_invoice_summary(invoice_data=invoice_data)

TOOLS = [save_invoice_image, extract_invoice_info, store_invoice_data, generate_invoice_summary]
```

- Wrapped each operation with the `@tool` decorator
- Added clear docstrings to help the LLM understand each tool's purpose
- Created a TOOLS list to hold all available tools

### Step 4: Define the Agent State and Core Components

Next, let's set up the agent's state and core components:

```python
# ---------- LangGraph agent setup ----------
class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

SYSTEM = SystemMessage(content=(
    "You are an invoice processing assistant. Your job is to:\n"
    "1. Save invoice images to S3\n"
    "2. Extract information from invoices using vision AI\n"
    "3. Store the extracted data in DynamoDB\n"
    "4. Provide a clear summary of the processed invoice\n"
    "Be systematic and process invoices step by step."
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
- Created a system message with clear instructions for invoice processing
- Set up the LLM with tools
- Created the agent_node function and tools_node

### Step 5: Build the LangGraph State Machine

Now, let's create the LangGraph state machine:

```python
def build_invoice_agent_graph():
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
- Set up the workflow flow
- Compiled the graph for execution

### Step 6: Create a User-Friendly Interface

Finally, let's create a simple function to run the agent:

```python
# ---------- Public entrypoint ----------
def process_invoice(image_base64: str, bucket: str, table_name: str) -> str:
    """
    Process an invoice image: save to S3, extract data, store in DynamoDB, and return summary.

    Args:
        image_base64: Base64 encoded invoice image
        bucket: S3 bucket name for storage
        table_name: DynamoDB table name for invoice data

    Returns:
        Processing summary as a string
    """
    graph = build_invoice_agent_graph()

    # Create the processing request
    prompt = f"""Process this invoice with the following steps:
    1. Save the image to S3 bucket '{bucket}' with key 'invoices/invoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png'
    2. Extract invoice information from the image
    3. Store the extracted data in DynamoDB table '{table_name}'
    4. Generate a summary of the processed invoice

    Image (base64): {image_base64[:100]}...[truncated]"""

    init_state = {"messages": [HumanMessage(content=prompt)]}
    out = graph.invoke(init_state)
    final_msg = out["messages"][-1]
    return getattr(final_msg, "content", str(final_msg))

# ---------- Demo ----------
if __name__ == "__main__":
    # Example usage (requires OPENAI_API_KEY + AWS creds)
    # Uncomment to test with a real base64 encoded invoice image

    # sample_image_base64 = "your_base64_encoded_invoice_here"
    # result = process_invoice(
    #     image_base64=sample_image_base64,
    #     bucket="my-invoice-bucket",
    #     table_name="InvoiceData"
    # )
    # print(result)

    print("Invoice agent ready. Provide a base64 encoded invoice image to process.")
  ```

    - Created a process_invoice function with clear parameters
    - Set up automatic key generation for S3 storage
    - Prepared example usage for testing

## Testing Your Invoice Agent

To test your invoice agent:

1. Setup virtualenv and install packages

  Run these commands at a CLI to setup a virtual environment and install packages:

  ```bash
  # create a virtual env folder
  python -m venv venv

  # activate the virtual env
  source venv/bin/activate

  # install packages
  pip install -U langgraph langchain-openai langchain-core pydantic boto3 awscli
  ```

2. Ensure your environment variables are set:
    ```bash
    export AWS_DEFAULT_REGION=us-west-2
    export AWS_ACCESS_KEY_ID=your_access_key
    export AWS_SECRET_ACCESS_KEY=your_secret_key
    export OPENAI_API_KEY=your_openai_key
    export OPENAI_MODEL=gpt-4o-mini
    ```

3. Create necessary AWS resources:
    ```bash
    # Create S3 bucket for invoices
    aws s3api create-bucket --create-bucket-configuration LocationConstraint=us-west-2 --bucket your-invoice-bucket

    # Create DynamoDB table
    aws dynamodb create-table \
      --table-name InvoiceData \
      --attribute-definitions AttributeName=invoice_id,AttributeType=S \
      --key-schema AttributeName=invoice_id,KeyType=HASH \
      --billing-mode PAY_PER_REQUEST
    ```

4. Run your agent:
    ```bash
    python invoice_agent.py
    ```

5. Verify the results:
    ```bash
    # Check S3 for saved invoices
    aws s3 ls s3://your-invoice-bucket/invoices/ --recursive

    # Check DynamoDB for invoice data
    aws dynamodb scan --table-name InvoiceData
    ```

## Summary

You've successfully built an intelligent invoice processing agent that can:
- Accept invoice images in base64 format
- Automatically save invoices to S3 for archival
- Extract key information using OpenAI's vision capabilities
- Store structured data in DynamoDB for easy querying
- Generate human-readable summaries of processed invoices

The architecture follows a clean pattern:
1. Core operations are defined as Python functions
2. Operations are wrapped as LangChain tools
3. A LangGraph state machine orchestrates the workflow
4. The agent processes invoices systematically through each step

This pattern can be extended to support additional features like:
- Multiple invoice formats and languages
- Validation of extracted data
- Integration with accounting systems
- Batch processing of multiple invoices
