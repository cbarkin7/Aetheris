# AETHERIS — Esquema de Referencia del Proyecto

Eres un experto en el proyecto AETHERIS. Este documento contiene todos los requisitos
actualizados, decisiones de diseño y convenciones del proyecto. Úsalo como contexto
para cualquier tarea de desarrollo, corrección de errores o extensión del sistema.

---

## Identidad del Proyecto

**AETHERIS** es un Agente Cognitivo Autónomo desarrollado como TFM (Trabajo Fin de Máster).
Va más allá de un chatbot: orquesta RAG, búsqueda web en tiempo real, automatización de
Google Workspace con aprobación humana (HITL), memoria persistente entre sesiones y
observabilidad completa con LangSmith.

- **Entorno:** Python 3.12+
- **Estado actual:** Backend + Frontend funcionales. 132 tests pasando.
- **Ubicación:** `Aetheris/` (raíz del proyecto)

---

## Stack Tecnológico (versiones instaladas)

| Componente | Librería | Versión |
|---|---|---|
| Orquestación agente | `langgraph` | 1.1.3 |
| Abstracciones LLM | `langchain` | 1.2.12 |
| LLM principal | `langchain-openai` | 1.1.11 |
| LLM fallback | `langchain-aws` | 1.4.0 |
| MCP integration | `langchain-mcp-adapters` | 0.2.2 |
| Vector store | `langchain-chroma` | 1.1.0 |
| Backend API | `fastapi` | 0.135.3 |
| Servidor ASGI | `uvicorn` | 0.42.0 |
| Frontend | `streamlit` | 1.56.0 |
| Checkpointer async | `aiosqlite` | 0.22.1 |
| STT local | `faster-whisper` | opcional |
| Memoria conv. | `mem0ai` | opcional |

---

## Estructura de Ficheros

```
Aetheris/
├── .env / .env.example
├── requirements.txt
├── pyproject.toml
├── .claude/
│   └── commands/
│       └── py-multiagent-schema.md   ← este fichero
├── aetheris/
│   ├── config.py                     # Pydantic-Settings, todas las vars de entorno
│   ├── llm.py                        # Factoría LLM: OpenAI + fallback Bedrock
│   ├── agent/
│   │   ├── state.py                  # AgentState TypedDict
│   │   ├── graph.py                  # build_graph() → StateGraph compilado
│   │   ├── nodes.py                  # Todos los nodos del grafo
│   │   ├── edges.py                  # Funciones de enrutado condicional
│   │   └── prompts.py                # Prompts del sistema
│   ├── api/
│   │   ├── main.py                   # FastAPI app + lifespan async
│   │   ├── dependencies.py           # get_compiled_graph, get_app_settings
│   │   ├── schemas.py                # Pydantic schemas de request/response
│   │   ├── middleware.py             # CORS, request ID, errores
│   │   └── routers/
│   │       ├── chat.py               # POST /chat SSE, POST /chat/{id}/resume
│   │       ├── documents.py          # POST /upload, GET /, DELETE /{id}
│   │       ├── memory.py             # GET/PUT /memory/{user_id}
│   │       ├── health.py             # GET /health, /health/langsmith
│   │       └── speech.py             # POST /speech/transcribe (faster-whisper)
│   ├── rag/
│   │   ├── ingest.py                 # load→chunk→embed→store (cosine space)
│   │   ├── retriever.py              # retrieve() con score threshold
│   │   └── schemas.py                # IngestResult, RetrievalResult
│   ├── mcp/
│   │   ├── client.py                 # get_mcp_tools() — nueva API sin async with
│   │   ├── tavily_tools.py
│   │   └── google_tools.py
│   ├── memory/
│   │   ├── checkpointer.py           # create_async_checkpointer() — AsyncSqliteSaver
│   │   ├── long_term.py              # SQLite CRUD de memoria de usuario
│   │   ├── mem0_memory.py            # Integración mem0.ai (opcional)
│   │   └── schemas.py
│   ├── guardrails/
│   │   ├── input_guard.py            # Detección inyección EN+ES, redacción PII
│   │   ├── output_guard.py           # Redacción PII/secretos en salida
│   │   └── schemas.py                # GuardResult (Pydantic)
│   ├── observability/
│   │   └── tracing.py                # LangSmith init, get_recent_runs()
│   └── ui/
│       ├── app.py                    # Streamlit entry: st.navigation()
│       ├── pages/
│       │   ├── 01_chat.py            # Chat SSE + HITL modal + audio upload
│       │   ├── 02_documents.py       # Upload/list/delete documentos
│       │   └── 03_observability.py   # Health + trazas LangSmith
│       └── components/
│           ├── chat_message.py
│           ├── hitl_modal.py
│           └── document_card.py
├── tests/
│   ├── conftest.py                   # Fixtures globales + api_client
│   ├── unit/                         # 64 tests — sin servicios externos
│   ├── integration/                  # 46 tests — Chroma/SQLite reales en /tmp
│   └── e2e/                          # 15 tests — flujos completos mockeados
└── data/
    ├── chroma_db/                    # Vector store persistente (git-ignored)
    ├── sqlite/                       # checkpoints.db + memory.db (git-ignored)
    └── uploads/                      # Ficheros subidos temporalmente
```

---

## Flujo del Agente LangGraph

```
START
  └─► input_guardrail_node
        ├─[blocked]─────────────────────────────────────────► llm_node (rechaza)
        └─[ok]─► load_memory_node ─► manager_node
                                          ├─[rag]──────────► rag_node
                                          ├─[web_search]───► web_search_node
                                          ├─[google_action]► hitl_node (interrupt!)
                                          │                       ├─[approved]► google_action_node
                                          │                       └─[rejected]─► llm_node
                                          └─[plain_llm]───────────────────────► llm_node
                                                                                    └─► output_guardrail_node
                                                                                            └─► save_memory_node ─► END
```

**Encadenamiento multi-herramienta:** `manager_node` puede devolver un plan con varios pasos
(e.g. `["rag", "web_search"]`). `plan_dispatch_node` extrae el siguiente paso del plan
y redirige al nodo correspondiente, creando un ciclo controlado.

---

## AgentState

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    thread_id: str
    user_id: str
    intent: Literal["rag", "web_search", "google_action", "plain_llm", "unknown"]
    rag_context: list[dict]
    tool_calls_pending: list[dict]
    hitl_approved: bool | None
    user_memory: dict
    guardrail_passed: bool | None
    guardrail_violations: list[str]
    llm_provider: str              # "openai" | "bedrock" | "test"
    execution_plan: list[str]      # pasos pendientes del plan multi-herramienta
    error: str | None
```

---

## Decisiones de Diseño Críticas

### 1. AsyncSqliteSaver (NO SqliteSaver)
El grafo usa `astream_events` (async). El checkpointer **debe** ser `AsyncSqliteSaver`.
`SqliteSaver` síncrono lanza `"does not support async methods"` en producción.

```python
# CORRECTO — en aetheris/memory/checkpointer.py
async def create_async_checkpointer(db_path=None):
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    conn = await aiosqlite.connect(path)
    return AsyncSqliteSaver(conn)

# En tests de integración — SqliteSaver síncrono es válido con graph.invoke()
checkpointer = SqliteSaver(sqlite3.connect(str(tmp_path / "cp.db"), check_same_thread=False))
```

### 2. Filtro SSE de tokens por nodo
El `manager_node` también llama al LLM. Sus tokens internos (JSON del plan) **no deben**
enviarse al frontend. Filtrar por `langgraph_node`:

```python
# en aetheris/api/routers/chat.py
if kind == "on_chat_model_stream":
    node_name = event.get("metadata", {}).get("langgraph_node", "")
    if node_name not in ("llm_node",):
        continue
```

### 3. MCP — nueva API (langchain-mcp-adapters >= 0.1.0)
`MultiServerMCPClient` ya no se usa como context manager:

```python
# CORRECTO
client = MultiServerMCPClient(servers)
tools = await client.get_tools()

# INCORRECTO (versión antigua)
async with client as active_client:
    tools = await active_client.get_tools()
```

### 4. Chroma — espacio coseno obligatorio
Sin `collection_metadata={"hnsw:space": "cosine"}`, Chroma usa L2 por defecto y devuelve
scores negativos que son filtrados por el threshold (0.3), causando 0% de hit rate en RAG.

```python
Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=persist_dir,
    collection_metadata={"hnsw:space": "cosine"},
)
```

### 5. Guardrails — orden de patrones PII
Los patrones de redacción PII deben ejecutarse en este orden para evitar que `phone_intl`
destruya claves API o IBANs antes de ser detectados:
`email → iban → credit_card → ssn → dni_nie_es → phone_intl`

El guardrail de salida aplica detección de contenido inseguro (claves API, contraseñas)
**antes** de la redacción PII para el mismo motivo.

### 6. GenericFakeChatModel en tests
`FakeChatModel(responses=[...])` fue eliminado en versiones recientes de `langchain-core`.
Usar siempre `GenericFakeChatModel(messages=iter([...]))`.

### 7. Streamlit — set_page_config solo en app.py
`st.set_page_config()` solo puede llamarse una vez por sesión. Solo debe estar en `app.py`.
Las páginas hijas (`01_chat.py`, `02_documents.py`, `03_observability.py`) no deben llamarlo.

---

## API REST

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/v1/health` | Estado del sistema |
| GET | `/api/v1/health/langsmith` | Conectividad LangSmith |
| POST | `/api/v1/chat` | Chat con streaming SSE |
| POST | `/api/v1/chat/{thread_id}/resume` | Reanudar tras HITL |
| GET | `/api/v1/chat/{thread_id}/history` | Historial de conversación |
| POST | `/api/v1/documents/upload` | Subir e ingestar documento |
| GET | `/api/v1/documents` | Listar documentos indexados |
| DELETE | `/api/v1/documents/{id}` | Eliminar documento |
| GET | `/api/v1/memory/{user_id}` | Leer memoria de usuario |
| PUT | `/api/v1/memory/{user_id}` | Actualizar memoria de usuario |
| POST | `/api/v1/speech/transcribe` | Transcribir audio (faster-whisper) |

### Formato SSE del chat
```
data: {"type": "token", "content": "..."}
data: {"type": "hitl_required", "actions": [...]}
data: {"type": "guardrail_blocked", "violations": [...]}
data: {"type": "error", "message": "..."}
data: {"type": "done"}
```

---

## Comandos de Desarrollo

```bash
# Backend
uvicorn aetheris.api.main:app --reload --port 8000

# Frontend
streamlit run aetheris/ui/app.py --server.port 8501

# Tests
pytest tests/unit -v                    # 64 tests, sin servicios externos
pytest tests/integration -v             # 46 tests, Chroma/SQLite en /tmp
pytest tests/e2e -v                     # 15 tests, flujos mockeados
pytest tests/ -v                        # 132 tests totales

# Ingestión bulk de documentos
python scripts/ingest_documents.py <carpeta>

# Reset de memoria
python scripts/reset_memory.py
```

---

## Variables de Entorno Requeridas

```bash
# LLM principal
OPENAI_API_KEY=sk-...

# Fallback LLM (opcional)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=eu-west-1
BEDROCK_MODEL_ID=anthropic.claude-sonnet-4-5-20250929-v1:0

# Whisper STT (opcional)
WHISPER_MODEL_SIZE=small        # tiny|base|small|medium|large-v2|large-v3
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=aetheris

# MCP (opcional)
TAVILY_API_KEY=tvly-...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GOOGLE_REFRESH_TOKEN=...

# Mem0 (opcional, vacío = modo local Chroma)
MEM0_API_KEY=

# App
APP_ENV=development
LOG_LEVEL=INFO
SECRET_KEY=cambiar-en-produccion
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
RAG_SCORE_THRESHOLD=0.3
GUARDRAILS_ENABLED=true
```

---

## Convenciones de Código

- **Idioma de código:** inglés (variables, funciones, comentarios de código)
- **Idioma de UI/mensajes:** español
- **Logs:** español con structlog/logging estándar
- **Singletons:** patrón `_var: Type | None = None` con función `_get_var()`
- **Nodos del grafo:** reciben `AgentState`, devuelven `dict` parcial
- **Tests:** `pytest`, sin fixtures de clase para unit tests, `@pytest.mark.integration` / `@pytest.mark.e2e`
- **Mocks LLM:** `GenericFakeChatModel(messages=iter([...]))` de `langchain_core`
- **Checkpointer en tests:** `SqliteSaver(sqlite3.connect(...))` para tests síncronos; `AsyncMock` para `api_client`
