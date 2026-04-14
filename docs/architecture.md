# AETHERIS — Arquitectura del Sistema

## Visión General

```
Usuario (Interfaz Streamlit)
    │
    │ HTTP / SSE (puerto 8501 → 8000)
    ▼
Backend FastAPI (puerto 8000)
    │                          ┌─────────────────────────┐
    │ graph.astream_events()   │   LangSmith              │
    │ ◄────────────────────────┤   (trazas, costes,       │
    ▼                          │    latencia)             │
Grafo LangGraph (StateGraph)   └─────────────────────────┘
    │
    ├─────────────────────────────────────────────────────┐
    │                        │                            │
    ▼                        ▼                            ▼
Capa RAG               Capa de Herramientas MCP    Capa de Memoria
(Chroma DB)        (Tavily + Google mediante     (SQLite + Chroma)
                    langchain-mcp-adapters)
    │                        │                        │
    └────────────────────────┴────────────────────────┘
                             │
                             ▼
                      LLM Principal (gpt-4o-mini)
                             │ con_fallbacks()
                             ▼
                      Fallback (AWS Bedrock / Anthropic Claude)
```

---

## Flujo del Grafo LangGraph

```
INICIO
  │
  ▼
input_guardrail_node ──[bloqueado]──► llm_node (rechazo)
  │
  │[OK]
  ▼
load_memory_node  (SQLite KV + mem0)
  │
  ▼
manager_node
  │
  ├──[rag]──────────────────► rag_node ──────────────────────► llm_node
  ├──[web_search]───────────► web_search_node ────────────────► llm_node
  ├──[google_action]────────► hitl_node (¡interrupción!)
  │                               ├──[aprobado]─► google_action_node ─► llm_node
  │                               └──[rechazado]───────────────────────► llm_node
  └──[plain_llm]────────────────────────────────────────────────► llm_node
                                                                      │
                                                                      ▼
                                                           output_guardrail_node
                                                                      │
                                                                      ▼
                                                            save_memory_node
                                                                      │
                                                                      ▼
                                                                     FIN
```

---

## Detalles de Componentes

### Frontend (Streamlit, puerto 8501)

- **Página Chat**: Chat en streaming mediante SSE. Renderiza token a token. Muestra modal de aprobación HITL cuando se activa.
- **Página Documentos**: Subida de PDF/DOCX/TXT/MD. Lista y elimina documentos indexados.
- **Página Observabilidad**: Estado del sistema, conexión con LangSmith, lista de trazas recientes.

### Backend (FastAPI, puerto 8000)

- API REST asíncrona con streaming SSE para el chat.
- El ciclo de vida (`lifespan`) gestiona el arranque de servidores MCP, la compilación del grafo y la creación de directorios.
- CORS configurado para el origen de Streamlit.
- Middleware de ID de solicitud y manejo de errores.

### Agente (LangGraph StateGraph)

**AgentState** — campos clave:

| Campo | Tipo | Descripción |
|---|---|---|
| `messages` | `list[BaseMessage]` | Historial de conversación (reducer `add_messages`) |
| `thread_id` | `str` | ID de hilo para checkpointing de LangGraph |
| `user_id` | `str` | ID de usuario para memoria a largo plazo |
| `intent` | `Literal[...]` | Clasificación de intención: `rag`, `web_search`, `google_action`, `plain_llm` |
| `rag_context` | `list[dict]` | Fragmentos recuperados con puntuaciones |
| `tool_calls_pending` | `list[dict]` | Llamadas a herramientas MCP en espera de HITL |
| `hitl_approved` | `bool \| None` | Estado de aprobación HITL |
| `user_memory` | `dict` | Preferencias a largo plazo cargadas en la entrada |
| `guardrail_passed` | `bool \| None` | Resultado del guardrail de entrada |
| `guardrail_violations` | `list[str]` | Violaciones detectadas por guardrails |
| `llm_provider` | `str` | Proveedor LLM utilizado (`openai`, `bedrock`, `test`) |
| `execution_plan` | `list[str]` | Pasos pendientes del plan multi-herramienta |
| `error` | `str \| None` | Mensaje de error si ocurre alguno |

### Capa RAG

- **Cargadores**: PyMuPDF (PDF), python-docx (DOCX), TextLoader (TXT/MD)
- **Fragmentación**: `RecursiveCharacterTextSplitter` (1000 chars, 200 solapamiento)
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Almacenamiento**: Chroma con recuperación MMR (`k=5`, umbral de puntuación 0.3, métrica de distancia coseno obligatoria: `hnsw:space=cosine`)
- **IDs de documento**: Hash MD5 de la ruta del fichero (permite re-ingestión idempotente)
- **Objetivo de tasa de acierto**: ≥85%, validado en `tests/integration/test_rag_pipeline.py`

### Integración MCP

- `MultiServerMCPClient` (langchain-mcp-adapters) iniciado una vez en el ciclo de vida de FastAPI
- **Tavily**: `npx @modelcontextprotocol/server-tavily` (transporte stdio) — búsqueda web, noticias
- **Google**: `npx @googleapis/mcp-server-google` (transporte stdio) — Calendar, Gmail, Drive
- **Acciones destructivas** gateadas por HITL: `create_calendar_event`, `send_email`, `delete_file`

### Sistema de Memoria

| Capa | Almacén | Alcance | Tecnología |
|---|---|---|---|
| Corto plazo (sesión) | SQLite `checkpoints.db` | Por `thread_id` | LangGraph `AsyncSqliteSaver` (aiosqlite) |
| Corto plazo (conversacional) | mem0.ai | Por `user_id` + `session_id` | mem0 cloud o local |
| Largo plazo (preferencias) | SQLite `user_memory` | Por `user_id` entre sesiones | Tabla clave-valor |
| Largo plazo (hechos semánticos) | Chroma | Por `user_id`, búsqueda semántica | Colección `aetheris_long_term_facts` |

**Flujo de extracción**: `save_memory_node` llama al LLM con un prompt de extracción para identificar hechos a persistir (zona horaria, idioma, preferencias recurrentes). Almacena en los tres sistemas de forma paralela y tolerante a fallos.

### Guardrails de Seguridad (Bilingüe EN + ES)

**Guardrail de Entrada** (`input_guardrail_node`):
1. Comprobación de longitud máxima (configurable, por defecto 8000 chars)
2. Detección de inyección de prompts — patrones en inglés y español:
   - Ignorar/sobrescribir instrucciones (`ignore_instructions_en`, `ignore_instructions_es`)
   - Revelar prompt del sistema (`reveal_system_en`, `reveal_system_es`)
   - Secuestro de rol (`role_hijack_en`, `role_hijack_es`)
   - Inyección de código (`code_injection`)
3. Redacción de PII: email, teléfono, SSN, DNI/NIE español, tarjeta de crédito, IBAN

**Guardrail de Salida** (`output_guardrail_node`):
1. Truncado de respuestas muy largas (por defecto 16000 chars)
2. Detección de contenido inseguro: claves API filtradas, contraseñas, prompts internos (ejecutado **antes** de la redacción PII)
3. Redacción de PII en la respuesta (orden: email → IBAN → tarjeta → SSN → DNI/NIE → teléfono)

### Fallback LLM

```
Solicitud de inferencia
  │
  ▼
ChatOpenAI (gpt-4o-mini)  ──[error/timeout/cuota]──► ChatBedrockConverse (Claude)
  │
  ▼ [éxito]
Respuesta
```

Implementado mediante `llm.with_fallbacks([bedrock_llm])` de LangChain. El proveedor utilizado se registra en el estado como `llm_provider` y se traza en LangSmith.

### Observabilidad (LangSmith)

- Configurado mediante `LANGCHAIN_TRACING_V2=true` + `LANGSMITH_API_KEY`
- Todas las llamadas de LangChain/LangGraph se trazan automáticamente
- Metadatos de ejecución personalizados mediante `get_langsmith_callbacks()`
- Endpoint de salud: `GET /api/v1/health/langsmith`
- Panel interactivo: [https://smith.langchain.com](https://smith.langchain.com)

---

## Diagrama de Dependencias entre Módulos

```
aetheris/
├── config.py           ◄── todos los módulos
├── llm.py              ◄── agent/nodes.py
├── guardrails/         ◄── agent/nodes.py
│   ├── schemas.py
│   ├── input_guard.py
│   └── output_guard.py
├── agent/
│   ├── state.py
│   ├── prompts.py
│   ├── edges.py        ◄── graph.py
│   ├── nodes.py        ◄── graph.py
│   └── graph.py        ◄── api/main.py
├── rag/
│   ├── schemas.py
│   ├── ingest.py       ◄── api/routers/documents.py, scripts/
│   ├── retriever.py    ◄── agent/nodes.py
│   └── pipeline.py
├── mcp/
│   ├── client.py       ◄── api/main.py
│   ├── tavily_tools.py ◄── mcp/client.py
│   └── google_tools.py ◄── mcp/client.py
├── memory/
│   ├── checkpointer.py ◄── api/main.py
│   ├── long_term.py    ◄── agent/nodes.py
│   ├── mem0_memory.py  ◄── agent/nodes.py
│   └── schemas.py
├── observability/
│   └── tracing.py      ◄── api/main.py, ui/
└── api/
    ├── main.py
    ├── dependencies.py
    ├── schemas.py
    ├── middleware.py
    └── routers/
        ├── chat.py
        ├── documents.py
        ├── memory.py
        └── health.py
```
