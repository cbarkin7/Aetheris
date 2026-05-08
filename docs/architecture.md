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
                    gmail_mcp_server.py (Python stdio)
                    @piotr-agier/google-drive-mcp
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
  ├──[google_action]────────► google_planner_node
  │                               │
  │                               ├──[data_collection_required=True]────────────────────────────────► llm_node (pass-through)
  │                               │
  │                               └──[tool_calls_pending no vacío]──► hitl_node
  │                                       │
  │                                       ├──[lectura, hitl_approved=True]──► google_action_node
  │                                       │                                       │
  │                                       │                                       ├──[tool_calls_queue no vacío]──► hitl_node (siguiente acción)
  │                                       │                                       └──[cola vacía]──────────────────► google_planner_node
  │                                       │                                                                               │
  │                                       │                                                                   ┌───────────┴────────────┐
  │                                       │                                                           [PASO 0.A: terminales OK]   [replanning/datos]
  │                                       │                                                                   │                        │
  │                                       │                                                            llm_node (pass-through)   hitl_node / llm_node
  │                                       │
  │                                       └──[destructiva, hitl_approved=None]──► hitl_wait_node (interrupt_before)
  │                                                                                       │
  │                                                                               ┌───────┴──────────────────────┐
  │                                                                        [aprobado]                    [rechazado]
  │                                                                               │                              │
  │                                                                    google_action_node          ┌─────────────┴──────────────┐
  │                                                                                         [cola no vacía]             [cola vacía]
  │                                                                                                │                            │
  │                                                                                          hitl_node                   llm_node
  └──[plain_llm]────────────────────────────────────────────────────────────────────────────────────────────────────────► llm_node
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

**Notas del flujo HITL (uno a uno):**
- `google_planner_node` llama al LLM con herramientas filtradas por dominio para planificar todas las acciones. Puede devolver múltiples `tool_calls`.
- `hitl_node` saca **una sola acción** de la cola en cada iteración (`tool_calls_pending = [acción_actual]`; el resto queda en `tool_calls_queue`). Procesa UNA acción por iteración.
- **Acciones de lectura** (`list-events`, `listGoogleDocs`, `search`, etc.): `hitl_approved=True` → enrutado directo a `google_action_node` sin pausa.
- **Acciones destructivas** (`create-event`, `deleteItem`, `send_email`, etc.): `hitl_approved=None` → enrutado a `hitl_wait_node` donde el grafo pausa con `interrupt_before`.
- Cuando el usuario **rechaza** → `hitl_node` inyecta un `ToolMessage` sintético de rechazo para mantener el historial OpenAI válido, y saca la siguiente acción de la cola.
- Si `tool_calls_queue` no está vacía tras ejecutar una acción → `hitl_node` (no `google_planner_node`). Solo cuando la cola se vacía → `google_planner_node` (replanning o PASO 0.A).
- Si rechazado + cola vacía → `llm_node` genera resumen de lo ejecutado/rechazado.
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

#### Endpoints de Chat

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/v1/chat` | Iniciar o continuar sesión de chat (SSE streaming) |
| `POST` | `/api/v1/chat/{thread_id}/resume` | Reanudar grafo interrumpido tras aprobación/rechazo HITL |
| `GET` | `/api/v1/chat/{thread_id}/history` | Recuperar historial de conversación desde checkpoint LangGraph |
| `DELETE` | `/api/v1/chat/{thread_id}` | Eliminar conversación y checkpoints (tabla `conversations` + `checkpoints.db`) |
| `GET` | `/api/v1/chat/threads/{user_id}` | Listar las últimas conversaciones del usuario |

### Agente (LangGraph StateGraph)

**AgentState** — campos clave:

| Campo | Tipo | Descripción |
|---|---|---|
| `messages` | `list[BaseMessage]` | Historial de conversación (reducer `add_messages`) |
| `thread_id` | `str` | ID de hilo para checkpointing de LangGraph |
| `user_id` | `str` | ID de usuario para memoria a largo plazo |
| `intent` | `Literal[...]` | Clasificación de intención: `rag`, `web_search`, `google_action`, `plain_llm` |
| `rag_context` | `list[dict]` | Fragmentos recuperados con puntuaciones |
| `tool_calls_pending` | `list[dict]` | Acción actual en proceso HITL (siempre UNA: `{id, name, args, description, requires_approval}`) |
| `tool_calls_queue` | `list[dict]` | Cola de acciones pendientes tras la actual; `hitl_node` las procesa una a una hasta vaciar |
| `hitl_approved` | `bool \| None` | Estado de aprobación HITL (`True` = auto-aprobado lectura / aprobado usuario; `None` = pendiente destructiva; `False` = rechazado) |
| `action_results` | `list[dict]` | Resultados de cada acción ejecutada en `google_action_node`; emitidos como eventos SSE antes del resumen LLM |
| `user_memory` | `dict` | Preferencias a largo plazo cargadas en la entrada |
| `guardrail_passed` | `bool \| None` | Resultado del guardrail de entrada |
| `guardrail_violations` | `list[str]` | Violaciones detectadas por guardrails |
| `sanitized_user_input` | `str \| None` | Versión PII-redactada del último mensaje humano, usada **solo** para llamadas al LLM; el mensaje original se conserva sin modificar en `messages` |
| `pii_map` | `dict` | Mapa `{placeholder: valor_real}` generado por `input_guardrail_node`; consumido por `google_action_node` para restaurar emails/teléfonos reales en los args de las tools |
| `llm_provider` | `str` | Proveedor LLM utilizado (`openai`, `bedrock`, `test`) |
| `data_collection_required` | `bool` | `True` cuando `google_planner_node` generó texto (datos faltantes o tarea completa) en lugar de `tool_calls`; `llm_node` hace pass-through |
| `google_action_iterations` | `int` | Contador de iteraciones del bucle `google_action_node` (máx. 6); se resetea en `manager_node` |
| `execution_plan` | `list[str]` | Pasos pendientes del plan multi-herramienta |
| `error` | `str \| None` | Mensaje de error si ocurre alguno |

### Prompts

| Prompt | Nodo | Descripción |
|---|---|---|
| `SYSTEM_PROMPT` | `llm_node` | Prompt principal del asistente. Lista las 5 herramientas Tavily. |
| `MANAGER_PROMPT` | `manager_node` | Orquestador. Decide el plan de herramientas (`rag`, `web_search`, `google_action`, `plain_llm`). Incluye triggers de Gmail: "mensaje", "mensajes", "spam", "bandeja de entrada", "inbox", etc. |
| `WEB_TOOL_SELECTOR_PROMPT` | `web_search_node` | Selecciona la herramienta Tavily correcta y construye sus argumentos. |
| `RAG_SYSTEM_PROMPT` | `llm_node` | Inyecta fragmentos RAG recuperados en el contexto. |
| `MEMORY_EXTRACTION_PROMPT` | `save_memory_node` | Extrae hechos memorables de la conversación. |
| `GOOGLE_PLANNER_PROMPT` | `google_planner_node` | Planifica acciones Google Workspace con herramientas filtradas por dominio. Incluye fecha/hora actual. |

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

#### Gmail (Python nativo stdio)

Transporte: **stdio** — servidor Python propio `aetheris/mcp_tools/gmail_mcp_server.py` lanzado como subproceso por `gmail_server_config()` en `google_tools.py`.  
Autenticación: `google-auth` OAuth2 con `GMAIL_TOKEN_PATH` (`.gmail-token.json`) y `GMAIL_CLIENT_SECRET_PATH` (client secret JSON). El servidor refresca el token automáticamente con `google.oauth2.credentials.Credentials` + `Request(timeout=30)`.

**Optimizaciones del servidor:**
- `static_discovery=True` — usa el JSON de descubrimiento incluido en el paquete `google-api-python-client`, eliminando la petición HTTP a `discovery.googleapis.com` que causaba `WinError 10060` (timeout TCP) en Windows.
- Instancia de servicio cacheada (`_gmail_service_cache`) — se reutiliza entre tool calls mientras el token sea válido (margen de 60 s); evita reconstruir el cliente en cada invocación.
- Invalidación del caché en errores 401/`invalid_grant`/`expired` para forzar reconstrucción con token fresco.
- `Request(timeout=30)` en el refresco de token — falla en 30 s en lugar de bloquear indefinidamente.

**Herramientas expuestas:**

| Herramienta | Tipo HITL | Descripción |
|---|---|---|
| `list_emails` | Lectura (auto) | Lista emails recientes de la bandeja de entrada |
| `get_email` | Lectura (auto) | Obtiene contenido completo de un email por ID |
| `search_emails` | Lectura (auto) | Busca emails con sintaxis Gmail avanzada |
| `send_email` | Destructiva (HITL) | Envía un email nuevo |
| `create_draft` | Destructiva (HITL) | Crea un borrador sin enviar |
| `delete_email` | Destructiva (HITL) | Mueve un email a la papelera (reversible) |
| `reply_to_email` | Destructiva (HITL) | Responde a un email manteniendo el hilo |

**Acciones HITL gateadas:** `send_email`, `reply_to_email`, `create_draft`, `delete_email`, `batch_delete_emails`, `empty_trash` (y variantes con guión).

**Acciones de lectura** (auto-ejecutadas): `list_emails`, `get_email`, `search_emails`.

#### Google Drive (`@piotr-agier/google-drive-mcp`)

Transporte: **stdio** (`cmd /c npx -y @piotr-agier/google-drive-mcp` en Windows; `npx -y ...` en Linux/Mac).  
Autenticación: `GOOGLE_DRIVE_OAUTH_CREDENTIALS` (client secret) + `GOOGLE_DRIVE_MCP_TOKEN_PATH` (token OAuth2 en `data/google/.drive-token.json`, formato `authorized_user`).

**Acciones HITL gateadas** (requieren aprobación):
```
deleteItem, moveItem, renameItem, copyFile, uploadFile,
createFolder, createTextFile, updateTextFile,
createGoogleDoc, insertText, deleteRange, applyTextStyle, insertTable, addComment,
createGoogleSheet, updateGoogleSheet, appendSpreadsheetRows, formatGoogleSheetCells, addDataValidation,
createGoogleSlides, deleteGoogleSlide, formatGoogleSlidesText, setGoogleSlidesBackground
```

**Acciones de lectura** (auto-ejecutadas sin HITL): `listGoogleDocs`, `listGoogleSheets`, `listGoogleSlides`, `listFolder`, `search`, `readGoogleDoc`, `readGoogleSheet`, `getFileMetadata`.

**Flujo de borrado Drive** (regla invariante):
- `deleteItem` **siempre** requiere un Drive ID real (≥ 25 caracteres alfanuméricos, sin espacios ni puntos).
- Si el planner genera `deleteItem(fileId='nombre.txt')`, `_fix_delete_tools()` lo convierte automáticamente en `search(query="name='nombre.txt'", rawQuery=True)`.
- Flujo resultante: `search` (auto-aprobado) → si encuentra `fileId` → `deleteItem` (HITL) → si no encuentra → LLM informa al usuario.

#### Política de token Google

`ensure_google_token_files()` y `ensure_google_drive_token_files()` escriben los ficheros de token al arrancar:

- Siempre actualizan `refresh_token` con el valor de `GOOGLE_REFRESH_TOKEN` de `.env`.
- Preservan `access_token` existente **solo si** `refresh_token` coincide con `.env` Y `expiry_date > 1` (token real, no cold-start).
- Nuevo token de arranque en frío: `access_token: ""`, `expiry_date: 1` → fuerza refresh inmediato.
- Si `refresh_token` cambia (rotación o revocación), siempre se escribe token limpio.

Ficheros generados en `data/google/`:
- `.calendar-token.json` — token Calendar
- `.gmail-token.json` — token Gmail (access + refresh token para `gmail_mcp_server.py`; incluye `client_id` y `client_secret` para el refresco automático con `google-auth`)
- `.drive-token.json` — token Drive (formato `authorized_user`)

### `google_planner_node` — Planificador de Acciones Google

Responsabilidades del nodo:

1. **Filtrado por dominio**: `_categorize_google_tools()` separa las herramientas en tres grupos (calendar / gmail / drive). `_detect_relevant_domains()` determina qué dominios son pertinentes para la petición actual y el LLM recibe únicamente las herramientas del dominio relevante.

2. **Detección de dominios en dos fases**:
   - **Fase 1** — último mensaje humano (fuente de verdad): si el usuario menciona explícitamente palabras de un dominio, se usa solo ese dominio.
   - **Fase 2** — ToolMessages recientes (solo si la Fase 1 no detecta nada): útil para continuaciones multi-paso como "ahora elimínalo" donde el dominio activo se infiere del historial de herramientas.

3. **Correcciones deterministas post-LLM** (en orden de aplicación):
   - `_fix_folder_creation_tools`: `createGoogleDoc`/`createGoogleSheet` → `createFolder` cuando el usuario pide una carpeta y el contenido es vacío.
   - `_fix_delete_tools`: `deleteItem(fileId=<nombre>)` → `search(query="name='<nombre>'", rawQuery=True)` si el `fileId` no es un Drive ID real. `deleteGoogleSlide` sin `slideObjectId` → redirigido a `deleteItem` o `search` según corresponda.
   - `_fix_list_tools`: aplica correcciones para búsquedas en Drive (ver Reglas 0–3 más abajo).

4. **Filtro de herramientas en `google_planner_node`**: el filtro inicial de tools incluye el keyword `"draft"` para que `create_draft` de Gmail no sea excluida del conjunto disponible para el LLM.

5. **PASO 0.A determinístico**: antes de llamar al LLM, comprueba si ya hay `ToolMessages` terminales exitosos desde el último `HumanMessage` y sin fallos recientes. Si los hay → genera resumen directamente sin LLM y activa `data_collection_required=True`, evitando un bucle planner→hitl→action→planner innecesario.

   `_TERMINAL_TOOL_NAMES` incluye todas las acciones de Drive, Docs, Sheets, Slides, Calendar y Gmail que crean, modifican o eliminan recursos de forma definitiva. Las acciones de búsqueda/lectura **no** son terminales.

#### `_fix_list_tools` — Correcciones de búsqueda Drive

**Regla 0 — `rawQuery=True` para operadores Drive API (CRÍTICA):**  
Por defecto, el servidor Drive MCP (`@piotr-agier/google-drive-mcp`) envuelve cualquier query en `fullText contains '...' and trashed=false`. Esto convierte `name='X'` en una búsqueda por **contenido**, no por **nombre** — el archivo no se encuentra aunque exista.  
Para queries que contienen operadores propios de la Drive API (`name=`, `name contains`, `trashed=`, `in parents`), `_fix_list_tools` añade automáticamente `rawQuery=True`. Con esta flag la query llega directa a la Google Drive API (que añade solo `and trashed=false`), haciendo la búsqueda verdaderamente recursiva en todo el Drive del usuario.

**Regla 1 — `fullText` + `orderBy`:**  
La API de Google Drive no admite `orderBy` cuando la query contiene `fullText`. Se elimina `orderBy` para evitar el error "Sorting is not supported for queries with fullText terms". No aplica cuando el único `contains` es parte de `name contains 'X'`.

**Regla 2 — `mimeType`-only → `listFolder`:**  
Cuando el LLM genera `search(query="mimeType=...")` sin filtro de nombre, se sustituye por `listFolder()` que es la herramienta correcta para listados por tipo.

**Regla 3 — `name` + `mimeType` → solo `name`:**  
Cuando la query combina filtro de nombre y tipo MIME, se elimina el `mimeType` para simplificar y evitar que la API rechace la combinación.

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

**Comportamiento PII**: el nodo **no modifica `messages`**. Guarda la versión redactada en `sanitized_user_input` y el mapa inverso en `pii_map`. El mensaje original con datos reales se conserva intacto en el historial persistido en BD.

- `_apply_sanitized_input(messages, sanitized_input)`: helper que sustituye el último `HumanMessage` en memoria (sin tocar el estado) justo antes de enviarlo al LLM. Se usa en `manager_node`, `google_planner_node`, `web_search_node` y `llm_node`.
- `_restore_pii(args, pii_map)`: llamado por `google_action_node` antes de `tool.ainvoke()` para restaurar emails/teléfonos reales en los argumentos de las herramientas Google.

**Guardrail de Salida** (`output_guardrail_node`):
1. Truncado de respuestas muy largas (por defecto 16000 chars)
2. Detección de contenido inseguro: claves API, contraseñas, prompts internos (ejecutado **antes** de la redacción PII)
3. Redacción de PII en la respuesta (orden: email → IBAN → tarjeta → SSN → DNI/NIE → teléfono)

### SSE — Eventos emitidos

| Evento `type` | Cuándo se emite | Contenido |
|---|---|---|
| `conversation_id` | Inicio de cada request | `thread_id` activo |
| `token` | `on_chat_model_stream` desde `llm_node` | Fragmento de texto de la respuesta |
| `token` | `on_chain_end google_planner_node` cuando `data_collection_required=True` | Contenido del AIMessage generado por el planner (PASO 0.A o datos faltantes) |
| `hitl_required` | `on_chain_end hitl_node` con acción destructiva pendiente | Lista de acciones con `requires_approval=True`; el stream se detiene hasta que el usuario decida |
| `action_result` | `on_chain_end google_action_node` por cada acción exitosa | `{name, summary}` — feedback inmediato antes del resumen LLM |
| `action_error` | `on_chain_end google_action_node` por cada acción fallida | `{name, error}` |
| `guardrail_blocked` | `on_chain_end input_guardrail_node` con bloqueo | Lista de violaciones detectadas |
| `done` | Al completar el stream | Señal de fin |
| `error` | Excepción no capturada | Mensaje de error |

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
│   ├── prompts.py         ◄── agent/nodes.py (6 prompts incluyendo GOOGLE_PLANNER_PROMPT)
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
│   ├── google_tools.py    ◄── mcp_tools/client.py  (Calendar stdio + Gmail stdio + Drive stdio)
│   ├── google_auth.py     ◄── mcp_tools/google_tools.py  (access_token OAuth2 para pre-inyección en token files)
│   ├── gmail_mcp_server.py  (servidor MCP Gmail Python stdio — lanzado como subproceso)
│   └── mcp_client.py      ◄── mcp_tools/client.py  (get_google_env + get_project_root)
├── memory/
│   ├── checkpointer.py    ◄── api/main.py
│   ├── long_term.py       ◄── agent/nodes.py, api/routers/chat.py
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
| HITL uno a uno (`tool_calls_queue`) | El usuario aprueba o rechaza cada acción individualmente. `hitl_node` saca siempre UNA acción de `tool_calls_queue`; el resto espera en cola. Permite rechazo selectivo sin cancelar todo el plan. |
| `route_after_google` → `hitl_node` si cola no vacía | Cuando quedan acciones en `tool_calls_queue`, se envía directamente a `hitl_node` (no a `google_planner_node`). Evita replanning innecesario y mantiene el orden del plan original. |
| `google_planner_node` separado de `hitl_node` | Antes, `hitl_node` hacía la planificación LLM. La separación de responsabilidades permite que el planner aplique correcciones deterministas antes de mostrar al usuario, sin mezclarlas con la lógica de aprobación. |
| Lecturas Google auto-aprobadas | `hitl_node` fija `hitl_approved=True` para herramientas no destructivas → `route_after_hitl_node` las envía directamente a `google_action_node` sin modal HITL. |
| `sanitized_user_input` separado de `messages` | Los datos reales (emails, teléfonos) deben persistir en el historial del checkpoint y mostrarse correctamente al recargar conversaciones. La redacción PII solo aplica a lo que ve el LLM, nunca al historial almacenado. |
| `_fix_delete_tools` en lugar de prompt-only | Los prompts son ignorados en conversaciones largas con mucho contexto. La corrección determinista garantiza el flujo search→delete correcto independientemente del LLM. |
| PASO 0.A sin LLM | Evita el bucle planner→hitl→action→planner tras completar la tarea. El LLM del planner no distingue fiablemente "tarea completa" en contextos largos, lo que generaba replanning infinito. |
| `action_results` + SSE `action_result` | Feedback inmediato por acción ejecutada antes de que `llm_node` genere el resumen. El usuario ve el resultado de cada paso en tiempo real. |
| `on_chain_end google_planner_node` para PASO 0.A | Cuando `data_collection_required=True`, el AIMessage del planner no llega al frontend via `on_chat_model_stream` (porque `llm_node` hace pass-through). El evento `on_chain_end` del planner es el único punto donde se puede emitir ese contenido. |
| `ainvoke` en nodos MCP | `langchain-mcp-adapters` solo implementa `_arun`. `tool.invoke()` lanza `StructuredTool does not support sync invocation`. |
| `hnsw:space=cosine` en Chroma | Sin distancia coseno, Chroma usa L2 y devuelve scores negativos que el threshold 0.3 filtra todos → 0% hit rate en RAG. |
| Token Google — siempre actualizar refresh_token | `ensure_google_token_files()` siempre sobreescribe `refresh_token` con el valor de `.env`. Preserva `access_token` solo si coincide y `expiry_date > 1`. Evita que tokens revocados persistan entre arranques. |
| `expiry_date: 1` en token cold-start | `google-auth-library` no refresca si no hay `expiry_date`. Valor `1` fuerza refresh inmediato en el primer uso del servidor MCP. |
| Gmail transport stdio Python nativo | El anterior servidor npm HTTP (`@gongrzhe/server-gmail-autoauth-mcp` en `localhost:30000`) causaba `WinError 10060` (timeout TCP al descargar el discovery doc de Google) y añadía una dependencia npm externa que debía correr en paralelo. La migración a `gmail_mcp_server.py` (Python stdio, `google-auth` nativo, `static_discovery=True`) elimina la dependencia npm, resuelve el timeout, y unifica el transporte de los tres servicios Google en stdio. |
| `static_discovery=True` en `gmail_mcp_server.py` | Elimina la petición HTTP a `discovery.googleapis.com` que causaba `WinError 10060` (timeout TCP de ~25 s en Windows). El JSON de descubrimiento se lee del paquete `google-api-python-client` instalado. |
| `_fix_list_tools` Regla 0 — `rawQuery=True` | Sin `rawQuery=True`, el servidor Drive MCP envuelve `name='X'` en `fullText contains '...'`, convirtiendo la búsqueda por nombre en búsqueda por contenido. Con `rawQuery=True` la query llega directa a la API de Drive, permitiendo búsqueda recursiva en todo el Drive. |
| Drive `cmd /c npx` en Windows | En Windows, `npx` no es un ejecutable directo — debe invocarse a través de `cmd /c`. En Linux/Mac se usa `npx` directamente. |
| Clientes MCP en `app.state.mcp_clients` | Si los `MultiServerMCPClient` son GC'd, la conexión stdio al proceso `npx` se cierra y `tool.ainvoke()` falla aunque las tools parezcan cargadas. |
| `WEB_TOOL_SELECTOR_PROMPT` | Las 5 herramientas Tavily tienen APIs distintas (args `query` vs `input` vs `urls`). Un selector LLM garantiza el campo correcto por herramienta. |
| Filtro SSE por `langgraph_node` | `manager_node` también llama al LLM. Sus tokens internos (JSON del plan) no deben llegar al frontend. Solo se emiten tokens de `llm_node` (y del planner en caso PASO 0.A). |
