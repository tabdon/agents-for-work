# Building a Change Management Agent with LangGraph

## Introduction to Your Project

In this lesson, you'll build an intelligent change management agent that automatically processes GitHub commit notifications, analyzes changes, and creates detailed change management documentation. The agent will:

1. Receive notifications of commits made to GitHub branches
2. Pull commit details and analyze the changes
3. Generate a comprehensive change management document in Markdown format

This project combines the power of Large Language Models (LLMs) with GitHub integration through a structured LangGraph workflow to automate your change management documentation process.

## Prerequisites

Before starting, ensure you have:

1. **Python environment**: Python 3.8+ with pip
2. **GitHub access**:
   - GitHub personal access token with appropriate permissions
   - A repository to test with

3. **OpenAI API key**:
   - OPENAI_API_KEY
   - *Note: You will need to bring your own OpenAI key*

4. **Required packages**:
   ```
   pip install -U langgraph langchain-openai langchain-core pydantic requests
   ```

## Building the Change Management Agent Step by Step

### Step 1: Set Up Basic Project Structure

Let's start by creating the essential components for our agent:

```python
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
```

- Imported necessary libraries for GitHub interactions, LangGraph, and LangChain
- Set up logging for troubleshooting

### Step 2: Define the Core GitHub Operations

Now, let's implement the core GitHub operations our agent will use:

```python
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
```

- Implemented four core operations:
  - `get_commit_details`: Retrieves detailed information about a specific commit
  - `get_file_content`: Fetches the content of a file from a GitHub repository
  - `get_pull_request`: Gets information about a pull request
  - `save_markdown_document`: Saves the generated change management document
- Each function handles its own error cases and returns structured responses
- Added GitHub token authentication for API calls

### Step 3: Convert Operations to LangChain Tools

Now, let's wrap our operations as LangChain tools so they can be used by the LLM:

```python
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
```

- Defined the agent's state using TypedDict
- Created a detailed system message that gives the LLM clear instructions on its change management role
- Set up the LLM with our tools
- Created the agent_node function that processes messages and generates responses
- Set up the tools_node that will execute the chosen tools

### Step 5: Build the LangGraph State Machine

Now, let's create the LangGraph state machine that will orchestrate the agent's workflow:

```python
def build_change_management_graph():
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

### Step 6: Create Notification Processing and Interface Functions

Finally, let's create functions to process commit notifications and provide a user-friendly interface:

```python
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
```

- Created a `handle_commit_notification` function to process GitHub webhook notifications
- Added a `format_notification_message` function that converts raw notification data into a readable format
- Set up the workflow to run the agent with the formatted notification
- Added error handling to ensure the agent can still process notifications even if formatting fails

## Testing Your Change Management Agent

### Step 1: Set Up Your Environment

First, ensure your environment variables are set:

```bash
export GITHUB_TOKEN=your_github_token
export OPENAI_API_KEY=your_openai_key
export OPENAI_MODEL=gpt-4o-mini
```

### Step 2: Create a Test Script

Let's create a script to simulate a GitHub commit notification and test our agent. Save this as `test_change_management.py`:

```python
import json
from datetime import datetime
from change_management_agent import handle_commit_notification

# Replace these values with your actual repository information
REPO_OWNER = "your-username"
REPO_NAME = "your-repository"
COMMIT_SHA = "abcdef1234567890abcdef1234567890abcdef12"  # Replace with a real commit SHA
BRANCH = "main"

# Create a test notification (simulating GitHub webhook payload)
test_notification = {
    "ref": f"refs/heads/{BRANCH}",
    "before": "0000000000000000000000000000000000000000",
    "after": COMMIT_SHA,
    "repository": {
        "name": REPO_NAME,
        "owner": {
            "name": REPO_OWNER,
            "login": REPO_OWNER
        },
        "html_url": f"https://github.com/{REPO_OWNER}/{REPO_NAME}"
    },
    "pusher": {
        "name": "Test User",
        "email": "test@example.com"
    },
    "commits": [
        {
            "id": COMMIT_SHA,
            "message": "Update README with project documentation",
            "timestamp": datetime.now().isoformat(),
            "author": {
                "name": "Test User",
                "email": "test@example.com"
            },
            "added": ["docs/new_feature.md"],
            "removed": [],
            "modified": ["README.md"]
        }
    ]
}

# Run the change management agent
print("Running change management agent...")
document_path = handle_commit_notification(test_notification)
print(f"\nChange management document created: {document_path}")

# Print the document content
try:
    with open(document_path, "r") as file:
        print("\n=== DOCUMENT CONTENT ===\n")
        print(file.read())
except Exception as e:
    print(f"Error reading document: {e}")
```

### Step 3: Generate Test GitHub Events

For more realistic testing, you can use a script to generate GitHub webhook events. Save this as `generate_github_event.py`:

```python
import json
import requests
import argparse
from datetime import datetime

def get_commit_data(repo_owner, repo_name, commit_sha, github_token=None):
    """Fetch real commit data from GitHub API"""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{commit_sha}"

    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return response.json()

def create_push_event(repo_owner, repo_name, branch, commit_sha, github_token=None):
    """Create a GitHub push event payload with real commit data"""

    # Get the actual commit data
    commit_data = get_commit_data(repo_owner, repo_name, commit_sha, github_token)

    # Format commit for the payload
    formatted_commit = {
        "id": commit_data["sha"],
        "message": commit_data["commit"]["message"],
        "timestamp": commit_data["commit"]["author"]["date"],
        "author": {
            "name": commit_data["commit"]["author"]["name"],
            "email": commit_data["commit"]["author"]["email"]
        },
        "added": [],
        "removed": [],
        "modified": []
    }

    # Add file changes
    for file in commit_data.get("files", []):
        if file["status"] == "added":
            formatted_commit["added"].append(file["filename"])
        elif file["status"] == "removed":
            formatted_commit["removed"].append(file["filename"])
        elif file["status"] == "modified":
            formatted_commit["modified"].append(file["filename"])

    # Create the full event payload
    event = {
        "ref": f"refs/heads/{branch}",
        "before": commit_data.get("parents", [{}])[0].get("sha", "0000000000000000000000000000000000000000"),
        "after": commit_sha,
        "repository": {
            "name": repo_name,
            "owner": {
                "name": repo_owner,
                "login": repo_owner
            },
            "html_url": f"https://github.com/{repo_owner}/{repo_name}"
        },
        "pusher": {
            "name": commit_data["commit"]["author"]["name"],
            "email": commit_data["commit"]["author"]["email"]
        },
        "commits": [formatted_commit]
    }

    return event

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GitHub push event payload")
    parser.add_argument("--owner", required=True, help="Repository owner")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--branch", default="main", help="Branch name")
    parser.add_argument("--commit", required=True, help="Commit SHA")
    parser.add_argument("--token", help="GitHub token (optional)")
    parser.add_argument("--output", default="github_event.json", help="Output file name")

    args = parser.parse_args()

    try:
        event = create_push_event(
            args.owner,
            args.repo,
            args.branch,
            args.commit,
            args.token
        )

        with open(args.output, "w") as f:
            json.dump(event, f, indent=2)

        print(f"GitHub push event saved to {args.output}")

    except Exception as e:
        print(f"Error: {e}")
```

Run this script to generate a realistic GitHub event payload:

```bash
python generate_github_event.py --owner your-username --repo your-repository --commit abcdef1234567890 --token your_github_token
```

Then use the generated payload to test your agent:

```bash
python -c "import json; from change_management_agent import handle_commit_notification; with open('github_event.json') as f: event = json.load(f); handle_commit_notification(event)"
```

## Testing Your Agent

### Step 1: Set Up Test GitHub Webhook Payload

To test your agent with real GitHub data, you'll need to capture a webhook payload. Here's a script to generate a test webhook event for commits to a branch:

```python
# generate_test_logs.py
import requests
import json
import os
import argparse
from datetime import datetime

def create_test_webhook_payload(owner, repo, branch='main', token=None):
    """
    Create a test GitHub webhook payload using actual repository data.
    """
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'

    # Get the latest commit on the branch
    url = f'https://api.github.com/repos/{owner}/{repo}/commits/{branch}'
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    commit_data = response.json()
    commit_sha = commit_data['sha']

    # Create a webhook payload similar to what GitHub would send
    payload = {
        "ref": f"refs/heads/{branch}",
        "before": commit_data.get('parents', [{}])[0].get('sha', '0' * 40),
        "after": commit_sha,
        "repository": {
            "name": repo,
            "owner": {
                "name": owner,
                "login": owner
            },
            "html_url": f"https://github.com/{owner}/{repo}"
        },
        "pusher": {
            "name": commit_data['commit']['author']['name'],
            "email": commit_data['commit']['author']['email']
        },
        "commits": [
            {
                "id": commit_sha,
                "message": commit_data['commit']['message'],
                "timestamp": commit_data['commit']['author']['date'],
                "author": {
                    "name": commit_data['commit']['author']['name'],
                    "email": commit_data['commit']['author']['email']
                },
                "added": [],
                "removed": [],
                "modified": []
            }
        ]
    }

    # Try to get file changes (might not be available in all API responses)
    try:
        for file in commit_data.get('files', []):
            if file['status'] == 'added':
                payload['commits'][0]['added'].append(file['filename'])
            elif file['status'] == 'removed':
                payload['commits'][0]['removed'].append(file['filename'])
            elif file['status'] == 'modified':
                payload['commits'][0]['modified'].append(file['filename'])
    except:
        # If we can't get file details, use placeholder
        payload['commits'][0]['modified'] = ['README.md']

    return payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a test GitHub webhook payload')
    parser.add_argument('--owner', required=True, help='Repository owner')
    parser.add_argument('--repo', required=True, help='Repository name')
    parser.add_argument('--branch', default='main', help='Branch name')
    parser.add_argument('--token', help='GitHub token (optional)')
    parser.add_argument('--output', default='github_webhook.json', help='Output file')

    args = parser.parse_args()

    payload = create_test_webhook_payload(
        args.owner,
        args.repo,
        args.branch,
        args.token
    )

    with open(args.output, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"Test webhook payload saved to {args.output}")
    print("To test your agent, run:")
    print(f"python -c \"import json; from change_management_agent import handle_commit_notification; with open('{args.output}') as f: result = handle_commit_notification(json.load(f)); print(f'Document created: {result}')\"")
```

Run this script to generate a test webhook payload:

```bash
python generate_test_logs.py --owner your-username --repo your-repository --token your-github-token
```

### Step 2: Run Your Agent

Now you can run your agent with the generated webhook payload:

```bash
python -c "import json; from change_management_agent import handle_commit_notification; with open('github_webhook.json') as f: result = handle_commit_notification(json.load(f)); print(f'Document created: {result}')"
```

## Summary

You've successfully built an intelligent change management agent that can:

1. Process GitHub commit notifications from webhooks
2. Retrieve detailed information about commits
3. Analyze code changes and their potential impact
4. Generate comprehensive change management documentation
5. Save the documentation in Markdown format for review

This agent demonstrates a practical application of AI for DevOps, helping teams:
- Maintain better documentation of system changes
- Standardize change management processes
- Improve visibility into codebase modifications
- Create clear rollback procedures for each change
- Save developer time on repetitive documentation tasks

The architecture follows a clean pattern:
1. Core GitHub operations are defined as Python functions
2. Operations are wrapped as LangChain tools
3. A LangGraph state machine orchestrates the agent's workflow
4. GitHub webhook events trigger the agent's documentation process

You can extend this agent to handle additional features like:
- Integrating with JIRA or other project management tools
- Posting change summaries to Slack channels
- Implementing approval workflows for high-risk changes
- Supporting customized documentation templates for different types of changes
