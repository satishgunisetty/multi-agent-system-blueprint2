import os
import logging
import mlflow

from langchain_core.messages import AIMessage, SystemMessage
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from databricks_langchain.chat_models import ChatDatabricks

from src.retailanalyst.util.constants import LLM_ENDPOINT_NAME, WAREHOUSE_ID
from src.retailanalyst.agent.state import State
from src.retailanalyst.util.agent_utils import safe_node_wrapper


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

sql_llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, api_key=DATABRICKS_TOKEN)


sql_database = SQLDatabase.from_databricks(
    catalog="agentic_ai_poc",
    schema="play_ground",
    include_tables=[],
    warehouse_id=WAREHOUSE_ID,
)

sql_tools = SQLDatabaseToolkit(db=sql_database, llm=sql_llm).get_tools()

sql_llm_with_tools = sql_llm.bind_tools(sql_tools)


@safe_node_wrapper
@mlflow.trace(name="sql_agent")
def sql_agent_node(state: State):
    """SQL specialist agent for PO and invoice operations."""
    log.info("Entering sql_agent_node")
    system_message = SystemMessage(
        content="""You are a helpful assistant for executing SQL queries on Databricks tables.
                    Use the provided SQL tools to translate natural language queries into secure, optimized SQL. If a query cannot be translated or executed, respond: 'Unable to process the SQL query'
                    While providing the answer give few and limited details."""
    )

    messages = [system_message] + state["messages"]
    log.info(f"sql_agent calling LLM with {len(messages)} messages")

    try:
        response = sql_llm_with_tools.invoke(messages)
        log.info(f"sql_agent got response type: {type(response)}")
        log.info(
            f"sql_agent response content type: {type(getattr(response, 'content', None))}"
        )
    except Exception as e:
        log.info(f"sql_agent LLM invoke failed: {e}")
        raise

    if isinstance(response, AIMessage):
        log.info("sql_agent response is already AIMessage")
        # Already correct type
        return {"messages": [response]}
    elif isinstance(response, str):
        log.info("sql_agent converting string to AIMessage")
        response = AIMessage(content=response)
    else:
        # Handle any other response type more carefully
        log.info(f"sql_agent converting {type(response)} to AIMessage")
        content = getattr(response, "content", "")
        if not isinstance(content, str):
            content = str(content) if content is not None else ""

        tool_calls = getattr(response, "tool_calls", [])
        # Ensure tool_calls is a list
        if not isinstance(tool_calls, list):
            tool_calls = []

        response = AIMessage(content=content, tool_calls=tool_calls)

    result = {"messages": [response]}
    log.info(
        f"sql_agent returning: {type(result)} with message type: {type(result['messages'][0])}"
    )
    return result
