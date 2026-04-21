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
(Chroma DB)        (tavily-mcp [5 tools] +        (SQLite + Chroma)
                    @cocal/google-calendar-mcp +
                    @gongrzhe/server-gmail-mcp
                    vía langchain-mcp-adapters)
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
  ├──[rag]──────────────────► rag_node ─────────────────────────► llm_node
  ├──[web_search]───────────► web_search_node (selector LLM) ────► llm_node
  ├──[google_action]────────► hitl_node
  │                               │
  │                               ├──[pending > 0]─► hitl_wait_node (interrupt_before)
  │                               │                       ├──[aprobado]─► google_action_node ─► llm_node
  │                               │                       └──[rechazado]────────────────────────► llm_node
  │                               └──[pending = 0]──────────────────────────────────────────────► llm_node
  └──[plain_llm]────────────────────────────────────────────────────────────────────────────────► llm_node
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

**Notas del flujo HITL:**
- `hitl_node` llama al LLM con herramientas para detectar qué acciones destructivas quiere ejecutar.
- Si hay acciones pendientes (`pending > 0`), enruta a `hitl_wait_node` donde el grafo pausa con `interrupt_before`.
- Si no hay acciones destructivas (`pending = 0`), el nodo NO añade el AIMessage con `tool_calls` al estado (evita el error 400 de OpenAI) y continúa directamente a `llm_node`.
- La reanudación se activa vía `POST /api/v1/chat/{thread_id}/resume` con `{"approved": true/false}`.

**Plan multi-herramienta:** `manager_node` puede devolver un plan de 2 pasos (ej. `["rag", "web_search"]`). `plan_dispatch_node` extrae el siguiente paso del plan y redirige al nodo correspondiente en un ciclo controlado por `execution_plan`.

---

## Detalles de Componentes

### Frontend (Streamlit, puerto 8501)

- **Página Chat** (`01_chat.py`): Chat en streaming mediante SSE. Renderiza token a token. Muestra modal de aprobación HITL cuando se activa. Soporte de entrada de audio (transcripción via `/api/v1/speech/transcribe`). Barra lateral con gestión de sesión: ID de usuario (`Admin-Aetheris` por defecto), ID de conversación (UUID autoasignado o recuperable), botón "Nueva conversación" y panel expandible "Retomar conversación".
- **Página Documentos** (`02_documents.py`): Subida de PDF/DOCX/TXT/MD. Lista y elimina documentos indexados.
- **Página Observabilidad** (`03_observability.py`): Estado del sistema, conexión con LangSmith, lista de trazas recientes.

### Backend (FastAPI, puerto 8000)

- API REST asíncrona con streaming SSE para el chat.
- El ciclo de vida (`lifespan`) gestiona el arranque de servidores MCP, la compilación del grafo y la creación de directorios.
- `thread_id` es opcional en `ChatRequest` — si no se envía, el backend genera un UUID y lo comunica al frontend via el evento SSE `conversation_id`.
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
| `tool_calls_pending` | `list[dict]` | Llamadas a herramientas MCP en espera de HITL (incluye `id`, `name`, `args`, `description`) |
| `hitl_approved` | `bool \| None` | Estado de aprobación HITL |
| `user_memory` | `dict` | Preferencias a largo plazo cargadas en la entrada |
| `guardrail_passed` | `bool \| None` | Resultado del guardrail de entrada |
| `guardrail_violations` | `list[str]` | Violaciones detectadas por guardrails |
| `llm_provider` | `str` | Proveedor LLM utilizado (`openai`, `bedrock`, `test`) |
| `execution_plan` | `list[str]` | Pasos pendientes del plan multi-herramienta |
| `error` | `str \| None` | Mensaje de error si ocurre alguno |

### Prompts

| Prompt | Nodo | Descripción |
|---|---|---|
| `SYSTEM_PROMPT` | `llm_node` | Prompt principal del asistente. Lista las 5 herramientas Tavily. |
| `MANAGER_PROMPT` | `manager_node` | Orquestador. Decide el plan de herramientas (`rag`, `web_search`, `google_action`, `plain_llm`). |
| `WEB_TOOL_SELECTOR_PROMPT` | `web_search_node` | Selecciona la herramienta Tavily correcta y construye sus argumentos. |
| `RAG_SYSTEM_PROMPT` | `llm_node` | Inyecta fragmentos RAG recuperados en el contexto. |
| `MEMORY_EXTRACTION_PROMPT` | `save_memory_node` | Extrae hechos memorables de la conversación. |
| `HITL_DESCRIPTION_PROMPT` | `hitl_node` | Genera descripción legible de la acción pendiente de aprobación. |

### Capa RAG

- **Cargadores**: PyMuPDF (PDF), python-docx (DOCX), TextLoader (TXT/MD)
- **Fragmentación**: `RecursiveCharacterTextSplitter` (1000 chars, 200 solapamiento)
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Almacenamiento**: Chroma con recuperación MMR (`k=5`, umbral de puntuación 0.3, métrica coseno: `hnsw:space=cosine` — obligatorio para evitar scores negativos)
- **IDs de documento**: Hash MD5 de la ruta del fichero (permite re-ingestión idempotente)
- **Objetivo de tasa de acierto**: ≥85%, validado en `tests/integration/test_rag_pipeline.py`

### Integración MCP

`MultiServerMCPClient` (langchain-mcp-adapters) arranca en el lifespan de FastAPI. Cada servidor se conecta de forma **independiente** — un fallo no cancela los demás.

**Importante:** Las herramientas MCP son **async-only** (`ainvoke`). Los nodos `web_search_node` y `google_action_node` son `async def` y usan `await tool.ainvoke(args)`.

#### Tavily (`tavily-mcp`)

| Herramienta | Campo requerido | Uso |
|---|---|---|
| `tavily_search` | `query` | Búsqueda general: noticias, hechos, eventos actuales |
| `tavily_research` | `input` | Investigación exhaustiva con múltiples fuentes |
| `tavily_extract` | `urls` (lista) | Extraer contenido de URLs específicas |
| `tavily_crawl` | `url`, `max_depth` | Rastrear un sitio completo desde una URL raíz |
| `tavily_map` | `url` | Mapear la estructura (listado de URLs) de un sitio |

El `web_search_node` usa `WEB_TOOL_SELECTOR_PROMPT` para que el LLM elija la herramienta correcta. Fallback automático a `tavily_search` si el selector falla.

#### Google Calendar (`@cocal/google-calendar-mcp`)

Autenticación via `GOOGLE_OAUTH_CREDENTIALS` (client secret) + `GOOGLE_CALENDAR_MCP_TOKEN_PATH` (token OAuth2).

**Acciones HITL gateadas** (requieren aprobación explícita del usuario):

```
create-event, create-events, update-event, delete-event, respond-to-event
```

**Acciones de lectura** (no requieren HITL): `list-calendars`, `list-events`, `search-events`, `get-event`, `get-freebusy`, `get-current-time`, `list-colors`.

#### Gmail (`@gongrzhe/server-gmail-autoauth-mcp`)

Autenticación via `GMAIL_OAUTH_PATH` (client secret) + `GMAIL_CREDENTIALS_PATH` (token OAuth2).

**Acciones HITL gateadas:** `send-email`, `reply-to-email`, `create-draft`.

#### Política de token Google

`ensure_google_token_files()` escribe los ficheros de token **solo si no existen o si les falta `access_token`** (token de arranque frío). Si el fichero ya contiene un token vivo generado por el servidor MCP (con `access_token` + `expiry_date`), no se sobreescribe — garantizando que los tokens refrescados por el servidor persisten entre reinicios del backend.

### Sistema de Memoria

| Capa | Almacén | Alcance | Tecnología |
|---|---|---|---|
| Corto plazo (sesión) | SQLite `checkpoints.db` | Por `thread_id` | LangGraph `AsyncSqliteSaver` (aiosqlite) |
| Corto plazo (conversacional) | mem0.ai | Por `user_id` + `session_id` | mem0 cloud o local |
| Largo plazo (preferencias) | SQLite `user_memory` | Por `user_id` entre sesiones | Tabla clave-valor |
| Largo plazo (hechos semánticos) | Chroma | Por `user_id`, búsqueda semántica | Colección `aetheris_long_term_facts` |

**Flujo de extracción**: `save_memory_node` llama al LLM con `MEMORY_EXTRACTION_PROMPT` para identificar hechos a persistir. Almacena en SQLite KV + Chroma semántico + mem0, de forma paralela y tolerante a fallos.

### Guardrails de Seguridad (Bilingüe EN + ES)

**Guardrail de Entrada** (`input_guardrail_node`):
1. Comprobación de longitud máxima (configurable, por defecto 8000 chars)
2. Detección de inyección de prompts — patrones en inglés y español:
   - Ignorar/sobrescribir instrucciones
   - Revelar prompt del sistema
   - Secuestro de rol
   - Inyección de código
3. Redacción de PII: email, teléfono, SSN, DNI/NIE español, tarjeta de crédito, IBAN

**Guardrail de Salida** (`output_guardrail_node`):
1. Truncado de respuestas muy largas (por defecto 16000 chars)
2. Detección de contenido inseguro: claves API, contraseñas, prompts internos (ejecutado **antes** de la redacción PII)
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

Implementado mediante `llm.with_fallbacks([bedrock_llm])`. El proveedor utilizado se registra en `llm_provider` y se traza en LangSmith.

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
│   ├── prompts.py      ◄── agent/nodes.py (6 prompts incluyendo WEB_TOOL_SELECTOR_PROMPT)
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
        ├── speech.py
        └── health.py
```

---

## Decisiones de Diseño Críticas

| Decisión | Motivo |
|---|---|
| `AsyncSqliteSaver` (no `SqliteSaver`) | El grafo usa `astream_events` (async). El checkpointer síncrono lanza `"does not support async methods"`. |
| `interrupt_before=["hitl_wait_node"]` | Solo interrumpe cuando hay acciones destructivas pendientes. `interrupt_after=["hitl_node"]` siempre pausaba, incluso sin acciones. |
| `ainvoke` en nodos MCP | `langchain-mcp-adapters` solo implementa `_arun`. `tool.invoke()` lanza `StructuredTool does not support sync invocation`. |
| `hnsw:space=cosine` en Chroma | Sin distancia coseno, Chroma usa L2 y devuelve scores negativos que el threshold 0.3 filtra todos → 0% hit rate en RAG. |
| No sobreescribir tokens Google | `ensure_google_token_files()` preserva tokens vivos (con `access_token`) para evitar invalidar el token refrescado por el servidor MCP. |
| `WEB_TOOL_SELECTOR_PROMPT` | Las 5 herramientas Tavily tienen APIs distintas (args `query` vs `input` vs `urls`). Un selector LLM garantiza el campo correcto por herramienta. |
| Filtro SSE por `langgraph_node` | `manager_node` también llama al LLM. Sus tokens internos (JSON del plan) no deben llegar al frontend. Solo se emiten tokens de `llm_node`. |
