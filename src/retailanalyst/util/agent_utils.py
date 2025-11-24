import logging
import json
import re
from typing import Dict, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.retailanalyst.agent.state import State


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def safe_node_wrapper(node_func):
    """Ultra-comprehensive wrapper to handle ALL possible return types from nodes."""

    def convert_to_dict(result: Any, node_name: str) -> Dict[str, Any]:
        """Convert any object type to a valid LangGraph state dict."""

        log.info(f" {node_name} returned type: {type(result)}")
        log.info(f" {node_name} result repr: {repr(result)[:200]}...")

        # ✅ 1. Already a dict - validate it has proper structure
        if isinstance(result, dict):
            if "messages" in result and isinstance(result["messages"], list):
                # Ensure all messages are proper LangChain message objects
                validated_messages = []
                for msg in result["messages"]:
                    if isinstance(
                        msg, (AIMessage, HumanMessage, SystemMessage, ToolMessage)
                    ):
                        validated_messages.append(msg)
                    else:
                        # Convert non-message objects to AIMessage
                        content = str(msg) if msg is not None else "Empty message"
                        validated_messages.append(AIMessage(content=content))

                return {
                    "messages": validated_messages,
                    **{k: v for k, v in result.items() if k != "messages"},
                }
            else:
                # Dict but wrong structure - convert to proper format
                preserved_state = {k: v for k, v in result.items() if k != "messages"}

                return {
                    "messages": [
                        AIMessage(
                            content=f"Node returned dict: {json.dumps(result, default=str)}"
                        )
                    ],
                    **preserved_state,
                }

        # ✅ 2. LangChain Message objects
        elif isinstance(result, (AIMessage, HumanMessage, SystemMessage, ToolMessage)):
            return {"messages": [result]}

        # ✅ 3. List of messages
        elif isinstance(result, list):
            validated_messages = []
            for item in result:
                if isinstance(
                    item, (AIMessage, HumanMessage, SystemMessage, ToolMessage)
                ):
                    validated_messages.append(item)
                elif isinstance(item, dict) and "content" in item:
                    # Convert dict-like message to AIMessage
                    content = item.get("content", "")
                    validated_messages.append(AIMessage(content=str(content)))
                else:
                    # Convert any other list item to AIMessage
                    validated_messages.append(AIMessage(content=str(item)))

            if validated_messages:
                return {"messages": validated_messages}
            else:
                return {"messages": [AIMessage(content="Node returned empty list")]}

        # ✅ 4. String responses
        elif isinstance(result, str):
            return {"messages": [AIMessage(content=result)]}

        # ✅ 5. Numeric types
        elif isinstance(result, (int, float, bool)):
            return {"messages": [AIMessage(content=str(result))]}

        # ✅ 6. None or empty responses
        elif result is None:
            return {"messages": [AIMessage(content=f"Node {node_name} returned None")]}

        # ✅ 7. Objects with 'content' attribute (ChatDatabricks responses)
        elif hasattr(result, "content"):
            content = getattr(result, "content", "")
            tool_calls = getattr(result, "tool_calls", [])

            # Ensure content is string
            if not isinstance(content, str):
                content = str(content) if content is not None else ""

            # Ensure tool_calls is list
            if not isinstance(tool_calls, list):
                tool_calls = []

            # Create AIMessage with preserved attributes
            ai_message = AIMessage(content=content, tool_calls=tool_calls)

            # Add any additional attributes
            additional_kwargs = getattr(result, "additional_kwargs", {})
            if additional_kwargs and isinstance(additional_kwargs, dict):
                ai_message.additional_kwargs.update(additional_kwargs)

            return {"messages": [ai_message]}

        # ✅ 8. Objects with '__dict__' (any custom objects)
        elif hasattr(result, "__dict__"):
            try:
                obj_dict = result.__dict__
                content = json.dumps(obj_dict, default=str)
                return {"messages": [AIMessage(content=f"Object content: {content}")]}
            except Exception as e:
                return {
                    "messages": [
                        AIMessage(content=f"Object conversion failed: {str(e)}")
                    ]
                }

        # ✅ 9. Iterables (tuples, sets, etc.)
        elif hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
            try:
                items = list(result)
                content = f"Iterable with {len(items)} items: {str(items)[:500]}"
                return {"messages": [AIMessage(content=content)]}
            except Exception as e:
                return {
                    "messages": [
                        AIMessage(content=f"Iterable conversion failed: {str(e)}")
                    ]
                }

        # ✅ 10. Callable objects (functions, lambdas)
        elif callable(result):
            return {
                "messages": [
                    AIMessage(
                        content=f"Node {node_name} returned callable: {str(result)}"
                    )
                ]
            }

        # ✅ 11. Bytes or binary data
        elif isinstance(result, (bytes, bytearray)):
            try:
                decoded = result.decode("utf-8", errors="replace")
                return {
                    "messages": [AIMessage(content=f"Binary data: {decoded[:500]}")]
                }
            except Exception:
                return {
                    "messages": [
                        AIMessage(content=f"Binary data ({len(result)} bytes)")
                    ]
                }

        # ✅ 12. Last resort - convert anything to string
        else:
            try:
                content = str(result)
                return {
                    "messages": [
                        AIMessage(content=f"Unknown type converted: {content[:500]}")
                    ]
                }
            except Exception as e:
                return {
                    "messages": [
                        AIMessage(
                            content=f"Conversion failed for {type(result)}: {str(e)}"
                        )
                    ]
                }

    def wrapper(state: State):
        """The actual wrapper function that catches everything."""
        node_name = getattr(node_func, "__name__", "unknown_node")

        try:
            log.info(f" Executing {node_name}")

            # Execute the original node function
            result = node_func(state)

            # Convert result to proper dict format
            safe_result = convert_to_dict(result, node_name)

            log.info(
                f" {node_name} converted to safe dict with {len(safe_result.get('messages', []))} messages"
            )

            return safe_result

        except KeyboardInterrupt:
            # Don't catch keyboard interrupts
            raise

        except Exception as e:
            log.info(f" Exception in {node_name}: {type(e).__name__}: {str(e)}")

            # Return safe error response
            error_message = f"Error in {node_name}: {type(e).__name__}: {str(e)}"
            return {"messages": [AIMessage(content=error_message)]}

    return wrapper
