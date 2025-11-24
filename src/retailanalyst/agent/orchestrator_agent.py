import os
import logging
import mlflow

from src.retailanalyst.agent.state import State
from databricks_langchain.chat_models import ChatDatabricks
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from src.retailanalyst.util.constants import LLM_ENDPOINT_NAME
from src.retailanalyst.util.agent_utils import (
    safe_node_wrapper,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

# Supervisor Agent - simple routing logic

# Initialize LLM
supervisor_llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, api_key=DATABRICKS_TOKEN)


@safe_node_wrapper
@mlflow.trace(name="supervisor_agent")
def supervisor_node(state: State):
    """Supervisor: routes user requests to agents, and formats agent outputs."""

    log.info(" Entering supervisor_node")

    log.info(f" State Schema: {state} ")
    last_msg = state["messages"][-1]
    log.info(f" supervisor last_msg type: {type(last_msg)}")

    # --- Case 1: Human user request ---
    if isinstance(last_msg, HumanMessage):
        log.info("supervisor processing HumanMessage/User message")
        user_message = last_msg.content


        routing_prompt = f"""
            You are the SUPERVISOR agent for a Retail Data Analysis system.

            Your job is to look at the user's request and decide which specialized
            agent should handle it. You must return ONLY the routing decision. Do NOT solve the query yourself.

            You have the following agents:

            1. SQL_AGENT
            - Use when the request involves:
            - Querying Databricks tables
            - Computing metrics, KPIs, trends, aggregations
            - Inventory, orders, customers, invoices, products
            - Any data stored in the database
            - SQL-based analysis or table lookups

            2. WEATHER_AGENT
            - Use when the request involves:
            - Weather details, forecasts, temperatures
            - Weather-driven analysis (heat waves, cold weather impacts)
            - Any external weather API information

            3. GENERAL_AGENT (fallback)
            - Use when the request:
            - Is conversational
            - Is about explanation, reasoning, or general tasks
            - Does not require data lookup or weather API calls

            ROUTING RULES:
            - If the task requires SQL, ALWAYS choose SQL_AGENT.
            - If the task requires weather details, ALWAYS choose WEATHER_AGENT.
            - If the task requires BOTH SQL and weather,
            choose SQL_AGENT first — it will report back — and you will then
            route to WEATHER_AGENT in a second step.
            - If the request is unclear, choose GENERAL_AGENT.

            Provide your answer in the following JSON format ONLY:

            RESPONSE FORMAT:
            Provide ONLY the routing decision: SQL_AGENT, WEATHER_AGENT, GREETING, or OUT_OF_SCOPE

            Now analyze the following request and choose the correct agent:

            REQUEST: "{user_message}"

            """
        
        try:
            response = supervisor_llm.invoke([SystemMessage(content=routing_prompt)])
            agent_name = (response.content or "").strip().upper()
            log.info(f"supervisor agent_name: {agent_name}")
        except Exception as e:
            return {
                **state,
                "messages": [AIMessage(content="Error routing request. Please retry.")],
            }
        
        # Routing decision
        if "SAP" in agent_name:
            return {**state, "messages": [AIMessage(content="ROUTE_TO: SAP_AGENT")]}
        elif "SERVICENOW" in agent_name:
            return {
                **state,
                "messages": [AIMessage(content="ROUTE_TO: SERVICENOW_AGENT")],
            }
        elif "GREETING" in agent_name:
            return {
                **state,
                "messages": [
                    AIMessage(
                        content="Hello! How can I help with POs, invoices, or ServiceNow tickets today?"
                    )
                ],
            }
        elif "OUT_OF_SCOPE" in agent_name:
            return {
                **state,
                "messages": [
                    AIMessage(
                        content="Sorry, I only handle Source-to-Pay processes like POs, invoices, and ServiceNow tickets."
                    )
                ],
            }
        else:
            return {
                **state,
                "messages": [
                    AIMessage(
                        content="I’m not sure which agent should handle this. Can you clarify?"
                    )
                ],
            }
    

    # --- Case 2: Agent response (AIMessage / ToolMessage) ---
    elif isinstance(last_msg, (AIMessage, ToolMessage)):
        log.info(f"supervisor handling: Tool Message / AIMessage - {type(last_msg)}")
        msg_content = getattr(last_msg, "content", "")
        msg_content = msg_content if isinstance(msg_content, str) else str(msg_content)
        log.info(f"supervisor Message contenct - {msg_content}")
        final_prompt = f"""
        You are the supervisor. The following agent responded:

        {msg_content}

        Reformat this into a clear, user-friendly response with markdown(HTML). If you have list of items represent it as a table. If the response is too long, summarize it.
        End with: "Would you like me to take any further action?"
        """

        try:
            response = supervisor_llm.invoke([SystemMessage(content=final_prompt)])
            log.info(f"supervisor formated response- {response}")
            out_text = response.content or "Error formatting response."
        except Exception:
            out_text = msg_content  # fallback

        return {
            **state,
            "messages": [AIMessage(content=out_text)],
        }

    # --- Default fallback ---
    return {"messages": []}