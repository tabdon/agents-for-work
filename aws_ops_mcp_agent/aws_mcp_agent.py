import os
from typing import Dict, Any
import asyncio
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

async def create_aws_mcp_agent():
    """Create a LangGraph agent with AWS MCP server integration."""

    # 1. Initialize the chat model (using Claude)
    model = init_chat_model(
        "anthropic:claude-3-7-sonnet-latest",
        temperature=0,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # 2. Set up MCP client connected to AWS MCP server subprocess
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

    # 3. Load available tools from the MCP server
    tools = await mcp_client.get_tools()

    # 4. Create checkpointer for conversation memory
    checkpointer = MemorySaver()

    # 5. Create the LangGraph agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="""You are an AWS operations assistant. You can help users manage their AWS resources through natural language commands.

When users ask about AWS resources:
- Use the available AWS tools to retrieve or modify resources
- Provide clear, helpful responses about what you found or did
- If an operation fails, explain what went wrong and suggest alternatives
- Always confirm destructive operations before proceeding

Be concise but informative in your responses.""",
        checkpointer=checkpointer
    )

    return agent

async def interactive_aws_chat():
    """Run an interactive chat session with the AWS agent."""

    agent = await create_aws_mcp_agent()
    config = {"configurable": {"thread_id": "interactive-session"}}

    print("🚀 Interactive AWS Assistant")
    print("Type 'quit' to exit, 'help' for examples")
    print("=" * 50)

    while True:
        user_input = input("\n🧑 You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break

        if user_input.lower() == 'help':
            print("""
📚 Example commands you can try:
- List my EC2 instances
- Show running instances in us-west-2
- Create an S3 bucket named my-bucket-name
- Delete S3 bucket my-old-bucket
- What's my AWS account ID?
- List my VPCs
            """)
            continue

        if not user_input:
            continue

        try:
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config
            )

            last_message = response["messages"][-1]
            print(f"🤖 Assistant: {last_message.content}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

# Run interactive mode
if __name__ == "__main__":
    asyncio.run(interactive_aws_chat())
