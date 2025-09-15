# Building a Deep Research Agent with LangGraph

## Introduction to Your Project

In this lesson, you'll build a powerful deep research agent that can conduct comprehensive internet research on any topic. This agent will:

- Clarify the research query if needed
- Generate targeted search queries
- Fetch and process search results
- Extract content from relevant web pages
- Synthesize findings into a structured report

This project combines Large Language Models (LLMs), web search APIs, and content extraction tools through a structured LangGraph workflow.

## Prerequisites

Before starting, ensure you have:

1. **Python environment**: Python 3.10+ with pip

2. **OpenAI API key**:
   - Create an account at [platform.openai.com](https://platform.openai.com)
   - Get your API key at [platform.openai.com/settings/organization/api-keys](https://platform.openai.com/settings/organization/api-keys)
   - Note: You are responsible for all usage and security of your key. Consider creating a key just for this project, and deleting it when done.

3. **Tavily API key**:
   - Create an account at [tavily.com](https://tavily.com)
   - Get your API key (they have a generous free tier)

## Building the Deep Research Agent Step by Step

### Step 1: Set Up Your Environment

Let's start by setting up your Python environment and installing the required packages:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install langgraph langchain-openai tavily-python httpx beautifulsoup4 readability-lxml tiktoken rich

# Set environment variables (replace with your actual keys)
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."
```

### Step 2: Set Up Basic Project Structure

Create a file called `deep_research_agent.py` and start building the core components:

```python
import os, re, asyncio, sys, time
from typing import TypedDict, List, Dict, Any

# Packages for styling the console output
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.spinner import Spinner
from rich.markdown import Markdown

# Core packages required by the agent
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import httpx
from bs4 import BeautifulSoup
from readability import Document

console = Console()
```

This imports all the necessary libraries for our agent, including:
- Standard Python libraries for handling data and async operations
- Rich library for beautiful terminal output
- LangGraph for the agent workflow
- LangChain for the LLM interface
- Tavily for web search
- Tools for HTML processing and content extraction

### Step 3: Define the Research State Model

Create the state model that will track the agent's research process:

```python
# ---------- STATE ----------
class ResearchState(TypedDict):
    user_prompt: str
    clarifications: Dict[str, str]
    needs_clarification: bool
    subqueries: List[str]
    hits: List[Dict[str, Any]]     # {url,title,snippet,score}
    pages: List[Dict[str, Any]]    # {url,title,text}
    report_markdown: str
```

This state model allows us to track:
- The user's original research question
- Any clarifications needed and provided
- The subqueries generated for more targeted searches
- The search results (hits)
- The extracted content from web pages
- The final research report

### Step 4: Set Up the Models and Tools

Configure the LLM and search tools:

```python
# ---------- MODELS / TOOLS ----------
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
```

We're using:
- OpenAI's GPT-4o model with temperature=0 for maximum accuracy
- Tavily's search API for retrieving relevant web results

### Step 5: Implement Search and Content Extraction Functions

Now, let's create functions for searching the web and extracting content:

```python
async def web_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    res = tavily.search(query=query, max_results=k)
    hits = []
    for r in res.get("results", []):
        hits.append({
            "url": r.get("url"),
            "title": r.get("title"),
            "snippet": r.get("content") or r.get("snippet"),
            "score": r.get("score", 0),
        })
    return hits

async def fetch_page(url: str, timeout=15) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent":"Mozilla/5.0"}) as client:
            r = await client.get(url, follow_redirects=True)
            html = r.text
            doc = Document(html)
            title = doc.short_title()
            cleaned = doc.summary(html_partial=True)
            soup = BeautifulSoup(cleaned, "html.parser")
            text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
            return {"url": str(r.url), "title": title, "text": text[:120000]}
    except Exception as e:
        return {"url": url, "title": "", "text": f"[ERROR fetching: {e}]"}

def dedupe_hits(hits: List[Dict[str, Any]], limit=25) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for h in sorted(hits, key=lambda x: x.get("score",0), reverse=True):
        key = re.sub(r"#.*$", "", (h.get("url") or "").strip())
        if key and key not in seen:
            seen.add(key)
            unique.append(h)
        if len(unique) >= limit:
            break
    return unique
```

These functions handle:
- Performing web searches and extracting key information
- Fetching web pages and extracting readable content
- Deduplicating search results to avoid repetition

### Step 6: Define the Graph Nodes

Now, let's create the nodes for our LangGraph workflow. Each node represents a step in the research process:

```python
# ---------- NODES (with console updates) ----------
async def node_clarify(state: ResearchState) -> ResearchState:
    prompt = f"""You are a research assistant. The user asked:
\"\"\"{state['user_prompt']}\"\"\"

List any clarifying questions that would materially improve the research plan.
Return JSON with keys:
- needs: boolean
- questions: list of short questions (max 5)
If nothing is needed, needs=false and questions=[].
"""
    console.rule("[bold cyan]Clarify")
    with console.status("Thinking about what to clarify..."):
        msg = await llm.ainvoke(prompt)
    text = msg.content
    needs = "true" in text.lower()
    questions = re.findall(r"- (.+)", text) if needs else []

    if needs and questions:
        console.print(Panel("\n".join(f"• {q}" for q in questions), title="Clarifying Questions", style="yellow"))
    else:
        console.print("[green]No clarifications needed.[/green]")

    # Stash questions for CLI to ask
    return {**state, "needs_clarification": bool(needs and questions), "_questions": questions}  # type: ignore

async def node_plan(state: ResearchState) -> ResearchState:
    clar = "\n".join(f"- {k}: {v}" for k,v in state["clarifications"].items()) or "(none)"
    prompt = f"""Create a focused research plan as 3-8 web search queries.
User prompt: {state['user_prompt']}
Clarifications:
{clar}

Rules:
- Prefer specific entities, metrics, time ranges, and synonyms.
- Cover background, current state, risks/criticisms, and data points.
Return one query per line, no numbering.
"""
    console.rule("[bold cyan]Plan")
    with console.status("Drafting sub-queries..."):
        msg = await llm.ainvoke(prompt)
    queries = [q.strip("- ").strip() for q in msg.content.splitlines() if q.strip()]
    queries = queries[:8]

    tbl = Table(title="Sub-queries", show_header=True)
    tbl.add_column("#", width=3)
    tbl.add_column("Query")
    for i,q in enumerate(queries,1):
        tbl.add_row(str(i), q)
    console.print(tbl)
    return {**state, "subqueries": queries}

async def node_search(state: ResearchState) -> ResearchState:
    console.rule("[bold cyan]Search")
    hits: List[Dict[str, Any]] = []
    with console.status("Searching the web..."):
        for q in state["subqueries"]:
            console.print(f"• [bold]Searching[/bold]: {q}")
            try:
                res = await web_search(q, k=5)
                hits.extend(res)
            except Exception as e:
                console.print(f"[red]Search failed:[/red] {e}")
    hits = dedupe_hits(hits, limit=25)

    tbl = Table(title=f"Top {len(hits)} results (deduped)", show_header=True)
    tbl.add_column("#", width=3)
    tbl.add_column("Title", overflow="fold")
    tbl.add_column("URL", overflow="fold")
    for i,h in enumerate(hits,1):
        tbl.add_row(str(i), h.get("title") or "", h.get("url") or "")
    console.print(tbl)
    return {**state, "hits": hits}

async def node_fetch(state: ResearchState) -> ResearchState:
    console.rule("[bold cyan]Fetch")
    urls = [h["url"] for h in state["hits"] if h.get("url")][:12]

    pages: List[Dict[str, Any]] = []
    # Fetch sequentially so we can show progress in the terminal (keeps output tidy)
    for idx, u in enumerate(urls, 1):
        with console.status(f"Fetching [{idx}/{len(urls)}]: {u}"):
            p = await fetch_page(u)
        label = p.get("title") or p.get("url")
        if p.get("text","").startswith("[ERROR"):
            console.print(f"[red]✖[/red] {label}")
        else:
            console.print(f"[green]✓[/green] {label}")
        pages.append(p)

    console.print(f"Fetched {len(pages)} pages.")
    return {**state, "pages": pages}

async def node_synthesize(state: ResearchState) -> ResearchState:
    console.rule("[bold cyan]Synthesize (streaming)")
    # Build sources list & corpus
    sources = []
    for i, p in enumerate(state["pages"], start=1):
        title = p.get("title") or p.get("url")
        sources.append(f"[{i}] {title} — {p.get('url')}")
    sources_text = "\n".join(sources)

    corpus = []
    for i, p in enumerate(state["pages"], start=1):
        txt = (p.get("text") or "")[:6000]
        corpus.append(f"### Source {i}\nURL: {p.get('url')}\nTitle: {p.get('title')}\nText:\n{txt}\n")
    corpus_text = "\n".join(corpus)

    prompt = f"""Write a concise, well-structured research brief in Markdown addressing:
\"\"\"{state['user_prompt']}\"\"\".

Use sections:
- Executive Summary (5–8 bullet points)
- Key Findings
- Conflicting Views / Risks
- Data & Numbers
- Open Questions
- Next Actions
- Sources

Rules:
- Cite like [1], [2] inline after claims you derive from a source.
- Synthesize; do not just copy text.
- Prefer recent and authoritative sources.
- If sources conflict, say so.

Sources List:
{sources_text}

Corpus:
{corpus_text}
"""

    # STREAM THE FINAL REPORT
    report_chunks: List[str] = []
    console.print(Panel("Generating report… (tokens will stream below)", style="magenta"))
    with Live("", refresh_per_second=12, console=console) as live:
        async for chunk in llm.astream(prompt):
            piece = chunk.content or ""
            report_chunks.append(piece)
            # show last ~1200 chars to avoid laggy terminal
            tail = "".join(report_chunks)[-1200:]
            live.update(Markdown(tail if tail else "."))

    report = "".join(report_chunks)
    console.rule("[bold green]Report complete")
    return {**state, "report_markdown": report}
```

Each node handles a specific part of the research workflow:
- `node_clarify`: Determines if clarifying questions are needed
- `node_plan`: Generates targeted search queries
- `node_search`: Executes searches and collects results
- `node_fetch`: Extracts content from relevant web pages
- `node_synthesize`: Creates a structured research report

### Step 7: Build the LangGraph Workflow

Now, let's create the graph that defines how the nodes connect:

```python
# ---------- GRAPH ----------
builder = StateGraph(ResearchState)
builder.add_node("clarify", node_clarify)
builder.add_node("plan", node_plan)
builder.add_node("search", node_search)
builder.add_node("fetch", node_fetch)
builder.add_node("synthesize", node_synthesize)

builder.add_edge(START, "clarify")
def needs_more_info(state: ResearchState) -> str:
    return "plan" if not state["needs_clarification"] else "INTERRUPT"
builder.add_conditional_edges("clarify", needs_more_info, {
    "plan": "plan",
    "INTERRUPT": END,
})
builder.add_edge("plan", "search")
builder.add_edge("search", "fetch")
builder.add_edge("fetch", "synthesize")
builder.add_edge("synthesize", END)
graph = builder.compile()
```

This graph defines the workflow:
1. Start with clarification
2. If clarification is needed, interrupt and get user input
3. Generate a research plan
4. Execute searches
5. Fetch and extract content
6. Synthesize findings into a report

### Step 8: Create the Main Application Interface

Finally, let's create the main function to run the agent:

```python
# ---------- CLI ----------
async def main():
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Missing OPENAI_API_KEY[/red]"); sys.exit(1)
    if not os.getenv("TAVILY_API_KEY"):
        console.print("[yellow]Warning:[/yellow] Missing TAVILY_API_KEY (search will fail).")

    console.print(Panel("Deep Research (streaming) — LangGraph", style="cyan"))
    user_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not user_prompt:
        user_prompt = console.input("[bold]What do you want to research?[/bold] ")

    # 1st pass: clarify (might interrupt)
    state: ResearchState = await graph.ainvoke({
        "user_prompt": user_prompt,
        "clarifications": {},
        "needs_clarification": False,
        "subqueries": [],
        "hits": [],
        "pages": [],
        "report_markdown": ""
    })

    # If clarifications needed, ask in terminal, then resume at "plan"
    if state.get("needs_clarification"):
        console.print("[yellow]Please answer a few quick questions:[/yellow]")
        for q in state.get("_questions", []):  # type: ignore
            ans = console.input(f"• {q} ")
            state["clarifications"][q] = ans
        state["needs_clarification"] = False
        state = await graph.ainvoke(state, start_at="plan")

    # Print and optionally save
    report = state["report_markdown"]
    console.print(Panel("Final Report (Markdown)", style="green"))
    console.print(Markdown(report))

    # Save to file for convenience
    fname = "research_report.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(report)
    console.print(f"[dim]Saved to ./{fname}[/dim]")

if __name__ == "__main__":
    asyncio.run(main())
```

This main function:
- Validates environment variables
- Gets the research topic from the user
- Runs the research workflow
- Handles clarification questions if needed
- Displays and saves the final report

## Testing Your Deep Research Agent

To test your agent:

1. Ensure your environment variables are set:
   ```bash
   export OPENAI_API_KEY="your_openai_key"
   export TAVILY_API_KEY="your_tavily_key"
   ```

2. Run the agent:
   ```bash
   python deep_research_agent.py
   ```

3. Enter your research topic when prompted.

4. If clarifying questions are needed, answer them.

5. Watch as the agent generates a research plan, searches for information, extracts content, and synthesizes a report.

6. Review the final report, which is also saved as `research_report.md`.

## Example Research Topics

Try researching topics like:
- "What are the environmental impacts of electric vehicles?"
- "How is AI being used in healthcare today?"
- "What is the current state of quantum computing?"
- "What are the latest treatments for Alzheimer's disease?"
- "How has remote work affected productivity since 2020?"

## Summary

You've successfully built a powerful deep research agent that can:
- Clarify research questions when needed
- Generate targeted search queries
- Find relevant information across the web
- Extract and process content
- Synthesize findings into a comprehensive report

This agent demonstrates how LLMs and structured workflows can dramatically accelerate research tasks, turning hours of manual work into minutes of automated processing.

The architecture follows a clean pattern:
1. Define the state model to track research progress
2. Create functions for web search and content extraction
3. Build LangGraph nodes for each step of the research process
4. Connect the nodes into a workflow
5. Create a user-friendly interface

This pattern can be extended to support additional research capabilities, such as specialized data extraction, analysis of academic papers, or integration with other information sources.
