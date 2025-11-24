import logging
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.retailanalyst.util.postgre_connection_manager import TokenRotatingPostgresSaver
from src.retailanalyst.agent.state import State
from src.retailanalyst.agent.orchestrator_agent import supervisor_node
from src.retailanalyst.agent.sql_agent import sql_agent_node, sql_tools
# from src.multiagent.agent.snow_agent import servicenow_agent_node
# from src.multiagent.service.tools import sap_tools, snow_tools

############################################
#           Simple Routing Logic           #
############################################

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def route_to_agent(
    state: State,
) -> Literal["sql_agent", "supervisor"]:
    """Simple routing based on supervisor decision."""

    # Get the last message from supervisor
    last_message = state["messages"][-1]

    if hasattr(last_message, "content") and "ROUTE_TO:" in last_message.content:
        route_decision = last_message.content.replace("ROUTE_TO:", "").strip()

        if "SQL_AGENT" in route_decision:
            return "sql_agent"
        # elif "SERVICENOW_AGENT" in route_decision:
        #     return "servicenow_agent"

    return "supervisor"


############################################
#        LakeBase PostGre Connection       #
############################################

# Global instance following Databricks patterns
_token_rotating_saver = None


def get_robust_checkpointer():
    """Get production-ready checkpointer with token rotation."""
    global _token_rotating_saver

    if _token_rotating_saver is None:
        _token_rotating_saver = TokenRotatingPostgresSaver()

    return _token_rotating_saver.get_checkpointer()


############################################
#        Simple Multi-Agent Graph          #
############################################


def build_simple_multiagent_system():
    """Build a clean, simple multi-agent system."""

    # Create workflow
    state_graph = StateGraph(State)

    # --- Nodes ---
    state_graph.add_node("supervisor", supervisor_node)
    state_graph.add_node("sql_agent", sql_agent_node)
    # state_graph.add_node("servicenow_agent", servicenow_agent_node)

    # Tool nodes
    state_graph.add_node("sql_tools", ToolNode(tools=sql_tools))
    # state_graph.add_node("servicenow_tools", ToolNode(tools=snow_tools))

    # --- Edges ---
    # Start → Supervisor
    state_graph.add_edge(START, "supervisor")

    # Supervisor → Agent (based on routing)
    state_graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "sql_agent": "sql_agent",
            "servicenow_agent": "servicenow_agent",
            "supervisor": END,
        },
    )

    # SQL Agent → Tools OR return to Supervisor
    state_graph.add_conditional_edges(
        "sql_agent",
        tools_condition,
        {
            "tools": "sql_tools",  # goes to tools
            "__end__": "supervisor",  # no tool → direct to supervisor
        },
    )

    # ServiceNow Agent → Tools OR return to Supervisor
    # state_graph.add_conditional_edges(
    #     "servicenow_agent",
    #     tools_condition,
    #     {
    #         "tools": "servicenow_tools",
    #         "__end__": "supervisor",
    #     },
    # )

    state_graph.add_edge("sql_tools", "supervisor")
    # state_graph.add_edge("servicenow_tools", "supervisor")

    # Get the persistent checkpointer
    memory = get_robust_checkpointer()

    log.info("Compiling graph with resilient PostgresSaver")
    return state_graph.compile(checkpointer=memory)
