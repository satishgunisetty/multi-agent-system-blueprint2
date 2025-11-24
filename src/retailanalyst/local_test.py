# Databricks notebook source
# MAGIC %pip install -U -qqqq backoff databricks-langchain langgraph==0.5.3 uv databricks-agents mlflow-skinny[databricks] psycopg[binary,pool] databricks-sql-connector langgraph-checkpoint-postgres
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Test locally before registering
from responses_agent_wrapper import AGENT

# Test with dict payload
result = AGENT.predict(
    {
        "input": [{"role": "user", "content": "Give me the details of PO 10005"}],
        "custom_inputs": {"thread_id": "test-session-1"},
    }
)

print(result)
