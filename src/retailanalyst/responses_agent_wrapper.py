# responses_agent_wrapper.py
from typing import Generator, Any, Dict
from uuid import uuid4
import mlflow
import logging
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from src.retailanalyst.util.message_utils import (
    convert_to_langchain_messages,
    get_final_response_text,
)
from src.retailanalyst.agent_graph import build_simple_multiagent_system

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class MultiAgentResponsesWrapper(ResponsesAgent):
    """ResponsesAgent wrapper for the multi-agent LangGraph system."""

    def __init__(self):
        self.graph = build_simple_multiagent_system()

    def _get_thread_id(self, payload: Dict[str, Any]) -> str:
        """Extract thread_id from request payload."""
        # Try context.conversation_id first, then params
        context = payload.get("context", {})
        custom_inputs = payload.get("custom_inputs", {})

        thread_id = (
            custom_inputs.get("thread_id") if isinstance(custom_inputs, dict) else None
        ) or context.get("conversation_id")

        if not thread_id:
            raise ValueError(
                "Missing thread identifier. Provide custom_inputs.thread_id or context.conversation_id"
            )

        return thread_id

    def _normalize_payload(self, request: Any) -> Dict[str, Any]:
        """Convert request to dict format."""
        if isinstance(request, dict):
            return request
        if hasattr(request, "model_dump"):
            return request.model_dump()

        # Fallback for typed requests
        payload = {"input": []}
        if hasattr(request, "input"):
            payload["input"] = [item.model_dump() for item in request.input]
        if hasattr(request, "context"):
            payload["context"] = request.context
        if hasattr(request, "params"):
            payload["params"] = request.params

        return payload

    def _invoke_graph(self, messages, thread_id: str) -> str:
        """Invoke the graph with minimal state - let LangGraph handle persistence."""
        log.info(f"_invoke_graph called with {len(messages)} messages")
        try:
            # ✅ FIXED: Only provide the new message, let LangGraph handle state persistence
            config = {"configurable": {"thread_id": thread_id}}

            # First, get the current persisted state (if any)
            try:
                current_state = self.graph.get_state(config)
                log.info(f"Retrieved current state: {current_state}")

                # If this is the first message in conversation, initialize state
                if not current_state.values:
                    log.info("Initializing new conversation state")
                    initial_state = {
                        "messages": messages,
                        "session_id": thread_id,
                        "awaiting_ticket_confirmation": False,
                        "awaiting_ticket_description": False,
                        "captured_issue_description": "",
                    }
                    log.info("About to call graph.invoke with initial state")
                    result = self.graph.invoke(initial_state, config=config)
                else:
                    # Continue existing conversation - only add new messages
                    log.info("Continuing existing conversation")
                    log.info("About to call graph.invoke with new messages")
                    result = self.graph.invoke({"messages": messages}, config=config)

            except Exception as state_error:
                log.warning(
                    f"Could not retrieve state: {state_error}, initializing fresh"
                )
                # Fallback: initialize fresh state
                initial_state = {
                    "messages": messages,
                    "session_id": thread_id,
                    "awaiting_ticket_confirmation": False,
                    "awaiting_ticket_description": False,
                    "captured_issue_description": "",
                }
                result = self.graph.invoke(initial_state, config=config)

            log.info("Graph invoke succeeded")
            return get_final_response_text(result.get("messages", []))

        except Exception as e:
            log.error(f"Graph invoke failed with: {e}")
            raise

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        log.info("Wrapper predict called")
        payload = self._normalize_payload(request)
        log.info("Payload normalized")

        input_messages = payload.get("input", [])
        thread_id = self._get_thread_id(payload)
        log.info(f"Thread ID: {thread_id}")

        lc_messages = convert_to_langchain_messages(input_messages)
        log.info(f"Converted messages: {len(lc_messages)}")

        try:
            response_text = self._invoke_graph(lc_messages, thread_id)
            log.info("Graph invocation successful")
        except Exception as e:
            log.info(f"Graph invocation failed: {e}")
            raise

        # Use MLflow's built-in helper method
        output_item = self.create_text_output_item(text=response_text, id=str(uuid4()))

        response = ResponsesAgentResponse(output=[output_item])
        log.info("ResponsesAgentResponse created successfully")
        log.info(f"Response type: {type(response)}")
        return response

    def predict_stream(
        self, request: Any
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming prediction method."""
        log.info("Wrapper predict_stream called")
        payload = self._normalize_payload(request)
        input_messages = payload.get("input", [])
        thread_id = self._get_thread_id(payload)

        # Convert and invoke
        lc_messages = convert_to_langchain_messages(input_messages)
        response_text = self._invoke_graph(lc_messages, thread_id)

        # Use MLflow's built-in helper method for streaming
        item_id = str(uuid4())
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=response_text, id=item_id),
        )


AGENT = MultiAgentResponsesWrapper()
mlflow.models.set_model(AGENT)
