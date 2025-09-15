# Building an AWS Operations Agent with MCP

## Introduction to AI Agents and MCP

AI agents are software systems (often powered by LLMs) that can perform tasks by interacting with external data sources, tools, and services on our behalf. To enable such interactions, Anthropic developed the Model Context Protocol (MCP) – an open protocol that standardizes how AI models connect to external tools and data sources.

An MCP server is a lightweight program that exposes specific capabilities (like running commands or fetching data) through this protocol. AI applications (the agents or hosts with MCP clients) maintain a one-to-one connection with each MCP server, allowing the AI to request tool actions or data in a controlled, structured way.

![MCP Architecture](mcp-sketch.png)

*An overview of the MCP architecture: an AI host with an MCP client (e.g. an IDE or chat assistant) connects to various MCP servers, which in turn interface with local or remote data sources.*

### Why use MCP?

By adding MCP servers into the mix, we can greatly extend an AI agent's capabilities while keeping its core model unchanged. Some key benefits include:

1. **Up-to-date knowledge**: MCP servers can pull in fresh data or documentation that the base model might not know (for example, recent AWS service APIs). This ensures the agent isn't limited by the LLM's training cutoff.

2. **Improved accuracy**: The agent's responses become more accurate and relevant because it can retrieve real information instead of guessing. For specialized domains like AWS, providing context via MCP servers reduces hallucinations and aligns answers with best practices.

3. **Tool execution & automation**: MCP turns common workflows or operations into tools the agent can invoke. Instead of just telling you how to do something, an AI agent could actually perform tasks (e.g. provisioning infrastructure) through these tools.

4. **Standard integration**: MCP provides a unified, standardized interface for all tools. This means less custom code when connecting an AI to different systems – the agent talks to any MCP server in the same way, whether it's querying a database or calling an API.

## AWS API MCP Server

The AWS API MCP Server is a tool that bridges AI assistants with AWS services by translating high-level requests into AWS CLI commands under the hood. Essentially, it lets an AI agent create, update, and manage AWS resources as if it were running AWS CLI or Boto3 calls – but in a safe, controlled manner.

### Key Features

- **Comprehensive Coverage**: Supports nearly all AWS CLI commands with up-to-date coverage of AWS services
- **Command Validation**: Validates commands before running them to prevent invalid or dangerous operations
- **Built-in Tools**:
  - `call_aws`: Executes AWS CLI commands on behalf of the agent
  - `suggest_aws_commands`: Takes a natural language query and suggests the most likely AWS CLI commands
- **Security**: Uses your AWS credentials, operating with only the permissions you have provided

In summary, the AWS API MCP server acts as a smart AWS assistant for your AI. If the user asks "Can you show me all my EC2 instances?", the server will effectively run `aws ec2 describe-instances` and return the results to the agent in a readable format.

## Prerequisites

Before starting this project, ensure you have:

1. **Python environment**: Python 3.10+ with pip

2. **AWS credentials**:
   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - Default region configured

3. **Anthropic API key**:
   - Sign up at https://claude.ai/login
   - Create an API key at https://console.anthropic.com/settings/keys
   - Store your key securely and be prepared to set it as an environment variable

## Building the AWS Ops MCP Agent

### Step 1: Setup Your Environment

First, let's create a virtual environment and install the required packages:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
source venv/bin/activate

# Install packages
pip install langgraph langchain langchain-anthropic langchain-mcp-adapters awscli uv

# Set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here
```

### Step 2: Configure AWS CLI

If you haven't already configured the AWS CLI, you'll need to do so:

```bash
aws configure
AWS Access Key ID [None]: your_access_key
AWS Secret Access Key [None]: your_secret_key
Default region name [None]: us-west-2
Default output format [None]:
```

### Step 3: Understanding the Agent Code Structure

Our AWS Operations MCP Agent consists of several key components:

1. **Model Initialization**: Setting up Claude as our reasoning engine
2. **MCP Client**: Connecting to the AWS API MCP server
3. **Tool Integration**: Loading available tools from the MCP server
4. **Agent Creation**: Building a LangGraph REACT agent
5. **Interactive Chat**: Creating a user interface for interaction

### Step 4: The MCP Client Configuration

The MCP client is configured to launch the AWS API MCP server as a subprocess. It uses:

```python
mcp_client = MultiServerMCPClient(
    {
        "aws-api": {
            "command": "uvx",
            "args": ["awslabs.aws-api-mcp-server@latest"],
            "transport": "stdio",
            "env": {
                "AWS_REGION": os.getenv("AWS_REGION", "us-west-2"),
                "AWS_API_MCP_PROFILE_NAME": os.getenv("AWS_API_MCP_PROFILE_NAME", "default")
            }
        }
    }
)
```

This sets up a connection to the AWS API MCP server, which will be automatically installed and launched when needed.

### Step 5: Creating the Agent

The agent is created using LangGraph's `create_react_agent` function, which implements the ReAct (Reasoning and Acting) pattern:

```python
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt="""You are an AWS operations assistant...""",
    checkpointer=checkpointer
)
```

The prompt gives the agent its identity and instructions on how to help users manage AWS resources.

### Step 6: Running the Agent

Once the agent is created, we can run it in an interactive chat loop:

```python
async def interactive_aws_chat():
    agent = await create_aws_mcp_agent()
    config = {"configurable": {"thread_id": "interactive-session"}}

    while True:
        user_input = input("\n🧑 You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break

        # Process user input and get response
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config
        )

        # Print the assistant's response
        last_message = response["messages"][-1]
        print(f"🤖 Assistant: {last_message.content}")
```

This loop takes user input, passes it to the agent, and prints the response until the user decides to exit.

## Testing Your AWS Ops MCP Agent

To test your agent:

1. Make sure your environment is activated and AWS is configured
2. Run the agent:
   ```bash
   python aws_mcp_agent.py
   ```
3. Try some example commands:
   - "What EC2 instances are running?"
   - "Create an S3 bucket named my-unique-bucket-name"
   - "List my S3 buckets in us-west-2"
   - "What's my AWS account ID?"

The agent will process your natural language requests, determine the appropriate AWS CLI commands, execute them, and provide you with the results.

## Summary

You've built an intelligent AWS Operations assistant that can:

- Connect to AWS services using the Model Context Protocol
- Translate natural language requests into AWS CLI commands
- Execute those commands and return the results in a human-readable format
- Maintain context throughout a conversation

This pattern demonstrates how AI agents can be extended with specialized capabilities through MCP servers, making them more powerful and useful for specific domains like AWS infrastructure management.
