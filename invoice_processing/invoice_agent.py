# invoice_agent.py
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

# ---------- logging setup ----------
logger = logging.getLogger("invoice_agent")
if not logger.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    logger.setLevel(level)

# ---------- AWS client helpers ----------
def s3_client():
    return boto3.client("s3")

def dynamodb_client():
    return boto3.resource("dynamodb")

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

def build_invoice_agent_graph():
    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()

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
