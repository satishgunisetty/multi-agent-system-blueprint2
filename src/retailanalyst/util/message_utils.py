# message_utils.py
from typing import List, Dict, Any
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    AnyMessage,
)


def convert_to_langchain_messages(
    input_items: List[Dict[str, Any]],
) -> List[AnyMessage]:
    """Convert ResponsesAgent input messages to LangChain messages."""
    messages = []
    for item in input_items:
        role = item.get("role")
        content = item.get("content")

        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "tool":
            messages.append(
                ToolMessage(content=content, tool_call_id=item.get("tool_call_id", ""))
            )

    return messages


def get_final_response_text(messages: List[AnyMessage]) -> str:
    """Extract the final assistant response text from message list."""
    for msg in reversed(messages or []):
        if isinstance(msg, AIMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""
