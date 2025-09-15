from __future__ import annotations
import os
from typing import Any, Dict, AsyncIterator, List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

class State(TypedDict):
    messages: List[BaseMessage]

# This function doesn't need to change since we'll handle streaming in the adapter
def llm_node(state: State) -> State:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    llm = ChatOpenAI(model=model, temperature=temperature)
    messages = state.get("messages", []) or [HumanMessage(content="Hello!")]
    ai = llm.invoke(messages)
    return {"messages": messages + [ai]}

_builder = StateGraph(State)
_builder.add_node("llm", llm_node)
_builder.add_edge(START, "llm")
_builder.add_edge("llm", END)
compiled_graph = _builder.compile()

class LangGraphAdapter:
    async def astream(self, payload: Dict[str, Any], config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        messages = payload.get("messages")
        prompt = payload.get("prompt")
        if not messages and prompt:
            messages = [HumanMessage(content=str(prompt))]
        messages = messages or [HumanMessage(content="Hello!")]

        # Create a streaming version of the LLM
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
        llm = ChatOpenAI(model=model, temperature=temperature, streaming=True)

        # Stream the tokens directly
        async for chunk in llm.astream(messages):
            if hasattr(chunk, 'content'):
                content = chunk.content
                if content:  # Only yield non-empty tokens
                    yield {"type": "token", "text": content}
