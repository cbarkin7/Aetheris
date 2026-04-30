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
(Chroma DB)        (tavily-mcp [5 tools]          (SQLite + Chroma)
                    @cocal/google-calendar-mcp
                    Gmail MCP HTTP + Bearer
                    @modelcontextprotocol/server-gdrive
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
  │                               ├──[lectura, hitl_approved=True]──► google_action_node ─► llm_node
  │                               ├──[escritura, hitl_approved=None]─► hitl_wait_node (interrupt_before)
  │                               │                                         ├──[aprobado]─► google_action_node ─► llm_node
  │                               │                                         └──[rechazado]───────────────────────► llm_node
  │                               └──[sin acciones]──────────────────────────────────────────────────────────────► llm_node
  └──[plain_llm]────────────────────────────────────────────────────────────────────────────────────────────────► llm_node
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
- `hitl_node` llama al LLM con herramientas para detectar qué acciones quiere ejecutar.
- **Acciones de lectura** (`list-events`, `list_files`, `read_file`, etc.): `hitl_approved=True` → enrutado directo a `google_action_node` sin pausa.
- **Acciones destructivas** (`create-event`, `delete_file`, `send_email`, etc.): `hitl_approved=None` → enrutado a `hitl_wait_node` donde el grafo pausa con `interrupt_before`.
- Si no hay acciones pendientes, el nodo NO añade el AIMessage con `tool_calls` al estado (evita el error 400 de OpenAI) y continúa directamente a `llm_node`.
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
- Los clientes `MultiServerMCPClient` se guardan en `app.state.mcp_clients` para mantener los procesos `npx` vivos durante toda la vida de la aplicación. Si el cliente es GC'd, la conexión stdio se cierra y `tool.ainvoke()` falla aunque las tools parezcan cargadas.
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
| `hitl_approved` | `bool \| None` | Estado de aprobación HITL (`True` = auto-aprobado lectura / aprobado usuario; `None` = pendiente destructiva; `False` = rechazado) |
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

`MultiServerMCPClient` (langchain-mcp-adapters) arranca en el lifespan de FastAPI. Cada servidor se conecta de forma **independiente** — un fallo no cancela los demás. Los clientes se almacenan en `app.state.mcp_clients`.

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

Transporte: **stdio** (`npx -y @cocal/google-calendar-mcp`).  
Autenticación: `GOOGLE_OAUTH_CREDENTIALS` (client secret) + `GOOGLE_CALENDAR_MCP_TOKEN_PATH` (token OAuth2 en `data/google/.calendar-token.json`).

**Acciones HITL gateadas** (requieren aprobación explícita del usuario):
```
create-event, create-events, update-event, delete-event, respond-to-event
```

**Acciones de lectura** (auto-ejecutadas sin HITL): `list-calendars`, `list-events`, `search-events`, `get-event`, `get-freebusy`, `get-current-time`, `list-colors`.

#### Gmail (HTTP + Bearer)

Transporte: **HTTP** (`"transport": "http"`, `url: GMAIL_MCP_URL`).  
Autenticación: `Authorization: Bearer <access_token>` — token obtenido de `get_google_access_token()` (caché ~1 hora).  
Requiere un servidor Gmail MCP HTTP externo corriendo en `GMAIL_MCP_URL` (por defecto `http://localhost:30000/mcp`).

**Acciones HITL gateadas:** `send-email`, `reply-to-email`, `create-draft`.

**Acciones de lectura** (auto-ejecutadas): `list-emails`, `read-email`, `search-emails`.

#### Google Drive (`@modelcontextprotocol/server-gdrive`)

Transporte: **stdio** (`cmd /c npx -y @modelcontextprotocol/server-gdrive` en Windows; `npx -y ...` en Linux/Mac).  
Autenticación: `GOOGLE_OAUTH_CREDENTIALS` (client secret) + `GOOGLE_DRIVE_MCP_TOKEN_PATH` (token OAuth2 en `data/google/.drive-token.json`).

**Acciones HITL gateadas** (requieren aprobación):
```
delete-file, move-file, share-file, rename-file
```

**Acciones de lectura** (auto-ejecutadas sin HITL): `list_files`, `find_files`, `read_file`.

#### Política de token Google

`ensure_google_token_files()` y `ensure_google_drive_token_files()` escriben los ficheros de token al arrancar:

- Siempre actualizan `refresh_token` con el valor de `GOOGLE_REFRESH_TOKEN` de `.env`.
- Preservan `access_token` existente **solo si** `refresh_token` coincide con `.env` Y `expiry_date > 1` (token real, no cold-start).
- Nuevo token de arranque en frío: `access_token: ""`, `expiry_date: 1` → fuerza refresh inmediato.
- Si `refresh_token` cambia (rotación o revocación), siempre se escribe token limpio.

Ficheros generados en `data/google/`:
- `.calendar-token.json` — token Calendar
- `.gmail-token.json` — token Gmail (solo para referencia; el transporte HTTP usa Bearer dinámico)
- `.drive-token.json` — token Drive

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
3. Redacción de PII: email, teléfono, SSN (separadores obligatorios), DNI/NIE español, tarjeta de crédito, IBAN (case-insensitive)

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
├── config.py              ◄── todos los módulos
├── llm.py                 ◄── agent/nodes.py
├── guardrails/            ◄── agent/nodes.py
│   ├── schemas.py
│   ├── input_guard.py
│   └── output_guard.py
├── agent/
│   ├── state.py
│   ├── prompts.py         ◄── agent/nodes.py (6 prompts incluyendo WEB_TOOL_SELECTOR_PROMPT)
│   ├── edges.py           ◄── graph.py
│   ├── nodes.py           ◄── graph.py
│   └── graph.py           ◄── api/main.py
├── rag/
│   ├── schemas.py
│   ├── ingest.py          ◄── api/routers/documents.py, scripts/
│   ├── retriever.py       ◄── agent/nodes.py
│   └── pipeline.py
├── mcp_tools/
│   ├── client.py          ◄── api/main.py  (get_mcp_tools → devuelve (clients, tools))
│   ├── tavily_tools.py    ◄── mcp_tools/client.py
│   ├── google_tools.py    ◄── mcp_tools/client.py  (Calendar stdio + Gmail HTTP + Drive stdio)
│   ├── google_auth.py     ◄── mcp_tools/google_tools.py  (Bearer token OAuth2 para Gmail HTTP)
│   └── mcp_client.py      (cliente legacy — no usado en el flujo principal)
├── memory/
│   ├── checkpointer.py    ◄── api/main.py
│   ├── long_term.py       ◄── agent/nodes.py
│   ├── mem0_memory.py     ◄── agent/nodes.py
│   └── schemas.py
├── observability/
│   └── tracing.py         ◄── api/main.py, ui/
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
        └── health.py       (GET /health, /health/langsmith, /health/google)
```

---

## Decisiones de Diseño Críticas

| Decisión | Motivo |
|---|---|
| `AsyncSqliteSaver` (no `SqliteSaver`) | El grafo usa `astream_events` (async). El checkpointer síncrono lanza `"does not support async methods"`. |
| `interrupt_before=["hitl_wait_node"]` | Solo interrumpe cuando hay acciones destructivas pendientes. `hitl_node` enruta a `hitl_wait_node` únicamente en ese caso; las lecturas van directo a `google_action_node`. |
| Lecturas Google auto-aprobadas | `hitl_node` fija `hitl_approved=True` para herramientas no destructivas → `route_after_hitl_node` las envía directamente a `google_action_node` sin modal HITL. |
| `ainvoke` en nodos MCP | `langchain-mcp-adapters` solo implementa `_arun`. `tool.invoke()` lanza `StructuredTool does not support sync invocation`. |
| `hnsw:space=cosine` en Chroma | Sin distancia coseno, Chroma usa L2 y devuelve scores negativos que el threshold 0.3 filtra todos → 0% hit rate en RAG. |
| Token Google — siempre actualizar refresh_token | `ensure_google_token_files()` siempre sobreescribe `refresh_token` con el valor de `.env`. Preserva `access_token` solo si coincide y `expiry_date > 1`. Evita que tokens revocados persistan entre arranques. |
| `expiry_date: 1` en token cold-start | `google-auth-library` no refresca si no hay `expiry_date`. Valor `1` fuerza refresh inmediato en el primer uso del servidor MCP. |
| Gmail transport HTTP (no stdio) | `@gongrzhe/server-gmail-autoauth-mcp` (stdio) requería credenciales en fichero local. La migración a HTTP Bearer simplifica el despliegue y desacopla el servidor Gmail del backend de AETHERIS. |
| Drive `cmd /c npx` en Windows | En Windows, `npx` no es un ejecutable directo — debe invocarse a través de `cmd /c`. En Linux/Mac se usa `npx` directamente. |
| Clientes MCP en `app.state.mcp_clients` | Si los `MultiServerMCPClient` son GC'd, la conexión stdio al proceso `npx` se cierra y `tool.ainvoke()` falla aunque las tools parezcan cargadas. |
| `WEB_TOOL_SELECTOR_PROMPT` | Las 5 herramientas Tavily tienen APIs distintas (args `query` vs `input` vs `urls`). Un selector LLM garantiza el campo correcto por herramienta. |
| Filtro SSE por `langgraph_node` | `manager_node` también llama al LLM. Sus tokens internos (JSON del plan) no deben llegar al frontend. Solo se emiten tokens de `llm_node`. |
