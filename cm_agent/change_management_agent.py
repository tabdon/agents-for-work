# change_management_agent.py
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, TypedDict, Annotated, Optional

import requests
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Set up basic logging
logger = logging.getLogger("change_management_agent")
if not logger.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")
    logger.setLevel(level)

# ---------- Basic GitHub operations ----------
def _get_commit_details(repo_owner: str, repo_name: str, commit_sha: str) -> Dict[str, Any]:
    """Retrieves details about a specific commit from GitHub."""
    try:
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{commit_sha}"

        headers = {}
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        commit_data = response.json()

        # Extract relevant information
        result = {
            "ok": True,
            "operation": "get_commit_details",
            "sha": commit_data.get("sha"),
            "author": {
                "name": commit_data.get("commit", {}).get("author", {}).get("name"),
                "email": commit_data.get("commit", {}).get("author", {}).get("email"),
                "date": commit_data.get("commit", {}).get("author", {}).get("date")
            },
            "committer": {
                "name": commit_data.get("commit", {}).get("committer", {}).get("name"),
                "email": commit_data.get("commit", {}).get("committer", {}).get("email"),
                "date": commit_data.get("commit", {}).get("committer", {}).get("date")
            },
            "message": commit_data.get("commit", {}).get("message"),
            "files": [
                {
                    "filename": file.get("filename"),
                    "status": file.get("status"),
                    "additions": file.get("additions"),
                    "deletions": file.get("deletions"),
                    "changes": file.get("changes"),
                    "blob_url": file.get("blob_url")
                }
                for file in commit_data.get("files", [])
            ],
            "stats": commit_data.get("stats"),
            "html_url": commit_data.get("html_url")
        }

        return result
    except Exception as e:
        logger.exception("get_commit_details error")
        return {"ok": False, "error": str(e)}

def _get_file_content(repo_owner: str, repo_name: str, file_path: str, ref: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves the content of a file from GitHub."""
    try:
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
        if ref:
            url += f"?ref={ref}"

        headers = {}
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        content_data = response.json()

        # GitHub API returns base64 encoded content
        import base64
        content = base64.b64decode(content_data.get("content", "")).decode("utf-8")

        return {
            "ok": True,
            "operation": "get_file_content",
            "path": file_path,
            "content": content,
            "sha": content_data.get("sha"),
            "size": content_data.get("size"),
            "url": content_data.get("html_url")
        }
    except Exception as e:
        logger.exception("get_file_content error")
        return {"ok": False, "error": str(e)}

def _get_pull_request(repo_owner: str, repo_name: str, pr_number: int) -> Dict[str, Any]:
    """Retrieves details about a pull request."""
    try:
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}"

        headers = {}
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        pr_data = response.json()

        return {
            "ok": True,
            "operation": "get_pull_request",
            "number": pr_data.get("number"),
            "title": pr_data.get("title"),
            "state": pr_data.get("state"),
            "user": {
                "login": pr_data.get("user", {}).get("login")
            },
            "body": pr_data.get("body"),
            "created_at": pr_data.get("created_at"),
            "updated_at": pr_data.get("updated_at"),
            "head": {
                "ref": pr_data.get("head", {}).get("ref"),
                "sha": pr_data.get("head", {}).get("sha")
            },
            "base": {
                "ref": pr_data.get("base", {}).get("ref"),
                "sha": pr_data.get("base", {}).get("sha")
            },
            "merged": pr_data.get("merged"),
            "mergeable": pr_data.get("mergeable"),
            "html_url": pr_data.get("html_url")
        }
    except Exception as e:
        logger.exception("get_pull_request error")
        return {"ok": False, "error": str(e)}

def _save_markdown_document(content: str, filename: Optional[str] = None) -> Dict[str, Any]:
    """Saves a markdown document to the file system."""
    try:
        # Generate a filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"change_management_{timestamp}.md"

        # Ensure filename has .md extension
        if not filename.endswith(".md"):
            filename += ".md"

        # Save the file
        with open(filename, "w", encoding="utf-8") as file:
            file.write(content)

        return {
            "ok": True,
            "operation": "save_markdown_document",
            "filename": filename,
            "size": len(content)
        }
    except Exception as e:
        logger.exception("save_markdown_document error")
        return {"ok": False, "error": str(e)}

# ---------- LangChain Tools ----------
@tool
def get_commit_details(repo_owner: str, repo_name: str, commit_sha: str) -> dict:
    """
    Retrieve detailed information about a specific commit from GitHub.

    Args:
        repo_owner: The owner of the repository (user or organization)
        repo_name: The name of the repository
        commit_sha: The SHA hash of the commit
    """
    return _get_commit_details(repo_owner=repo_owner, repo_name=repo_name, commit_sha=commit_sha)

@tool
def get_file_content(repo_owner: str, repo_name: str, file_path: str, ref: Optional[str] = None) -> dict:
    """
    Retrieve the content of a specific file from a GitHub repository.

    Args:
        repo_owner: The owner of the repository (user or organization)
        repo_name: The name of the repository
        file_path: The path to the file within the repository
        ref: Optional reference (branch, tag, or commit SHA)
    """
    return _get_file_content(repo_owner=repo_owner, repo_name=repo_name, file_path=file_path, ref=ref)

@tool
def get_pull_request(repo_owner: str, repo_name: str, pr_number: int) -> dict:
    """
    Retrieve information about a specific pull request.

    Args:
        repo_owner: The owner of the repository (user or organization)
        repo_name: The name of the repository
        pr_number: The pull request number
    """
    return _get_pull_request(repo_owner=repo_owner, repo_name=repo_name, pr_number=pr_number)

@tool
def save_markdown_document(content: str, filename: Optional[str] = None) -> dict:
    """
    Save a markdown document to the file system.

    Args:
        content: The markdown content to save
        filename: Optional filename (if not provided, a timestamp-based name will be used)
    """
    return _save_markdown_document(content=content, filename=filename)

TOOLS = [get_commit_details, get_file_content, get_pull_request, save_markdown_document]

# ---------- LangGraph agent loop ----------
class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

SYSTEM = SystemMessage(content=(
    "You are a Change Management Agent that processes GitHub commit notifications. "
    "When you receive a notification about a commit, your job is to:"
    "\n1. Retrieve details about the commit"
    "\n2. Analyze the changes made in the commit"
    "\n3. Create a comprehensive change management document in Markdown format"
    "\n4. Save the document for review"
    "\nThe change management document should include:"
    "\n- A summary of the changes"
    "\n- Technical details of what was modified"
    "\n- Potential impact assessment"
    "\n- Risk assessment"
    "\n- Rollback procedure"
    "\nBe thorough in your analysis but present the information in a clear, structured format."
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

def build_change_management_graph():
    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()

# ---------- Public entrypoint ----------
def handle_commit_notification(notification_data: Dict[str, Any]) -> str:
    """
    Process a GitHub commit notification.

    Args:
        notification_data: The notification data from GitHub webhook

    Returns:
        The path to the generated change management document
    """
    # Format the notification data into a human-readable message
    notification_message = format_notification_message(notification_data)

    # Run the agent with the notification message
    graph = build_change_management_graph()
    init_state = {"messages": [HumanMessage(content=notification_message)]}
    out = graph.invoke(init_state)
    final_msg = out["messages"][-1]

    # Extract the filename from the final message if available
    content = getattr(final_msg, "content", str(final_msg))
    filename_match = re.search(r"saved to ['\"]?([^'\"]+\.md)", content)
    filename = filename_match.group(1) if filename_match else "unknown.md"

    return filename

def format_notification_message(notification_data: Dict[str, Any]) -> str:
    """Format GitHub notification data into a readable message for the agent."""
    try:
        # Extract repository information
        repository = notification_data.get("repository", {})
        repo_name = repository.get("name", "unknown-repo")
        repo_owner = repository.get("owner", {}).get("name", repository.get("owner", {}).get("login", "unknown-owner"))

        # Extract commit information
        commit_sha = notification_data.get("after", "")
        commits = notification_data.get("commits", [])

        # Get the branch name
        ref = notification_data.get("ref", "")
        branch = ref.replace("refs/heads/", "") if ref.startswith("refs/heads/") else ref

        # Format commit messages for summary
        commit_summaries = []
        for commit in commits:
            commit_id = commit.get("id", "")[:7]  # Short SHA
            commit_msg = commit.get("message", "").split("\n")[0]  # First line only
            commit_author = commit.get("author", {}).get("name", "unknown")
            commit_summaries.append(f"- {commit_id} by {commit_author}: {commit_msg}")

        commit_summary_text = "\n".join(commit_summaries) if commit_summaries else "No commit details available"

        # Format the message
        message = f"""
GITHUB COMMIT NOTIFICATION
Repository: {repo_owner}/{repo_name}
Branch: {branch}
Latest Commit: {commit_sha}

Commits in this push:
{commit_summary_text}

Please create a change management document by:
1. Retrieving the details of the latest commit
2. Analyzing the changes made
3. Creating a comprehensive change management document
4. Saving the document for review
"""
        return message

    except Exception as e:
        logger.exception("Error formatting notification message")
        # Fallback to raw JSON if formatting fails
        return f"Please create a change management document for this GitHub commit notification:\n{json.dumps(notification_data, indent=2)}"

# ---------- Demo ----------
if __name__ == "__main__":
    # Example test notification (simulating GitHub webhook)
    test_notification = {
        "ref": "refs/heads/main",
        "before": "0000000000000000000000000000000000000000",
        "after": "1234567890abcdef1234567890abcdef12345678",
        "repository": {
            "name": "example-repo",
            "owner": {
                "name": "example-user",
                "login": "example-user"
            },
            "html_url": "https://github.com/example-user/example-repo"
        },
        "pusher": {
            "name": "Example User",
            "email": "user@example.com"
        },
        "commits": [
            {
                "id": "1234567890abcdef1234567890abcdef12345678",
                "message": "Update documentation and fix API endpoint",
                "timestamp": datetime.now().isoformat(),
                "author": {
                    "name": "Example User",
                    "email": "user@example.com"
                },
                "added": ["docs/api.md"],
                "removed": [],
                "modified": ["src/api/endpoints.js", "README.md"]
            }
        ]
    }

    # Run the agent with the test notification
    document_path = handle_commit_notification(test_notification)
    print(f"\nChange management document created: {document_path}")
