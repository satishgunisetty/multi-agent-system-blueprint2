# multi-agent-system-blueprint2
### Built with UV · LangGraph · Databricks · MLflow ResponsesAgent

This repository contains a **production-grade multi-agent AI application** built using **LangGraph**, **MLflow ResponsesAgent**, and **Databricks Mosaic AI**. It provides an extensible pattern for building enterprise-ready agentic systems, including:

- Multi-agent orchestration  
- Tool-calling agents for SAP, Snowflake, DB, and internal services  
- Persistent conversational state using Postgres  
- Unified Catalog model registration + Model Serving deployment  
- Fully reproducible environment using **UV** (ultra-fast Python package manager)

---

## 🚀 Features

### ✔ Multi-Agent Architecture  
Built using **LangGraph**, with:

- **Orchestrator Agent** – controls routing + reasoning  
- **SAP Agent / Snowflake Agent** – domain-specific tool callers  
- **Agent Helper modules**  
- **State Manager** – handles conversation/session-level state

---

### ✔ MLflow ResponsesAgent Integration  
The app is wrapped as an **MLflow `pyfunc` model** using the *ResponsesAgent* pattern:

- Enables model serving on Databricks  
- Supports chat-style inputs  
- Works with Model Serving end-to-end  

---

### ✔ Databricks Native Deployment  
The project integrates with:

- **Unity Catalog model registry**  
- **Model Serving endpoints**  
- **Databricks Agents (optional)**  
- **Databricks Secrets** for DB credentials + LLM endpoints  
- **Postgres-based LangGraph Checkpointing** using `langgraph-checkpoint-postgres`

---

### ✔ UV as Package Manager  
Lightning-fast environment resolution + full isolation from system Python.



