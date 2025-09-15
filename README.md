# Agents for Work

Collection of small, useful AI agents for common ops and workflow tasks. Each folder is a self‑contained agent with a runnable Python script and a hands‑on lesson.

This repo is designed to be shared publicly. Secrets are not committed; use environment variables or `.env` files locally.

## Contents

- `aws_ops_mcp_agent/` — LangGraph + MCP client that talks to AWS MCP servers.
- `cm_agent/` — Change‑management agent that reads commits/PRs and drafts docs.
- `deep_research_agent/` — Terminal research agent with rich TUI output.
- `ec2_ops_agent/` — EC2 list/info/start/stop/reboot with tool‑calling.
- `incident_response_agent/` — CloudWatch + SNS triage/notify workflow.
- `invoice_processing/` — Extract/structure invoice details from images.
- `s3_agent/` — S3 list/read/upload/delete with tool‑calling.
- `security_incident_agent/` — User lookup and failed‑login investigation flow.
- `slack_agent/` — Send notifications to Slack webhook (supports dry‑run).
- `langgraph_starter/` — Minimal FastAPI + WebSocket LangGraph starter with a tiny UI.

Each directory contains a `LESSON.md` that explains setup, env vars, and how to run the agent. The `langgraph_starter/` subproject has its own `README.md` and `requirements.txt` for a quick web demo.

## Quick Start

1) Clone and choose an agent:

```bash
git clone https://github.com/tabdon/agents-for-work
cd agents-for-work
```

2) Open one of the agent project folders:

```bash
cd aws_ops_mcp_agent/
```

Consult each folder's `LESSON.md` for the exact invocation and walkthrough.

## Development Notes

- Python 3.10+ recommended.
- Create a separate virtualenv per agent.
- Many agents use `langgraph` and `langchain-*` packages; install as needed.
- AWS examples require credentials with the least privilege necessary.

## Contributing

Issues and PRs are welcome. If you add a new agent, include:
- A short `LESSON.md` with setup, env vars, and how to run it.
- No hard‑coded secrets. Prefer `.env.example` for documentation.

## Supporting the Project

If you want to support this project please checkout my site, [Skillmix](https://skillmix.io).

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
