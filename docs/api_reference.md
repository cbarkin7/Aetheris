# AETHERIS — Referencia de la API

Base URL: `http://localhost:8000`

Documentación interactiva (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Chat

### `POST /api/v1/chat`

Inicia o continúa una conversación. Devuelve un stream SSE (Server-Sent Events) con los tokens generados token a token.

**Cuerpo de la solicitud:**
```json
{
  "message": "¿Qué dice mi informe Q4 sobre los ingresos?",
  "thread_id": "uuid-del-hilo",
  "user_id": "Admin-Aetheris",
  "stream": true
}
```

| Campo | Tipo | Obligatorio | Descripción |
|---|---|---|---|
| `message` | `string` | Sí | Mensaje del usuario (1–4096 chars) |
| `thread_id` | `string` | No | ID de hilo para checkpointing. Si se omite, el backend genera un UUID nuevo. |
| `user_id` | `string` | No | ID de usuario para memoria (por defecto: `"Admin-Aetheris"`) |
| `stream` | `boolean` | No | Activar streaming SSE (por defecto: `true`) |

**Eventos SSE de respuesta (en orden de aparición):**

| Tipo | Payload | Descripción |
|---|---|---|
| `conversation_id` | `{"type": "conversation_id", "thread_id": "uuid"}` | **Primer evento siempre.** Comunica el `thread_id` activo (generado o recibido). |
| `token` | `{"type": "token", "content": "texto"}` | Token de respuesta generado por el LLM. |
| `done` | `{"type": "done"}` | Respuesta completada correctamente. |
| `hitl_required` | `{"type": "hitl_required", "actions": [...]}` | Acción destructiva Google pendiente de aprobación del usuario. |
| `action_result` | `{"type": "action_result", "name": "...", "summary": "..."}` | Feedback inmediato de acción Google ejecutada correctamente (tras HITL aprobado). |
| `action_error` | `{"type": "action_error", "name": "...", "error": "..."}` | Feedback de acción Google fallida (tras HITL aprobado). |
| `guardrail_blocked` | `{"type": "guardrail_blocked", "violations": [...]}` | Mensaje bloqueado por guardrail de seguridad. |
| `error` | `{"type": "error", "message": "..."}` | Error durante el procesamiento del agente. |

**Ejemplo de evento `conversation_id`:**
```json
{"type": "conversation_id", "thread_id": "f40e6f4b-4df2-4f2c-9207-172380d4623d"}
```

**Ejemplo de evento `hitl_required`:**
```json
{
  "type": "hitl_required",
  "actions": [
    {
      "id": "call_abc123",
      "name": "deleteItem",
      "args": {"fileId": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"},
      "description": "Eliminar archivo 'informe_q4.pdf' de Drive.",
      "requires_approval": true
    }
  ]
}
```

> **Nota:** el campo `actions` contiene siempre **una sola acción** (la acción actual pendiente de aprobación). Si el plan original incluía varias acciones, las restantes se mantienen en la cola interna `tool_calls_queue` y se presentan de una en una tras cada respuesta HITL.

---

### `DELETE /api/v1/chat/{thread_id}`

Elimina una conversación y todos sus datos persistidos.

**Parámetros de ruta:**
- `thread_id` — ID del hilo a eliminar

**Qué elimina:**
- Registro en `memory.db` (tabla `conversations`)
- Todos los checkpoints en `checkpoints.db` (tablas `checkpoints`, `checkpoint_writes`, `checkpoint_blobs`)

**Respuesta (200 OK):**
```json
{"thread_id": "uuid", "memory_deleted": true, "checkpoints_deleted": true}
```

> Tras la eliminación, el `thread_id` queda inválido. Cualquier solicitud posterior que lo referencie devolverá `404 Not Found`.

---

### `GET /api/v1/chat/threads/{user_id}`

Devuelve las conversaciones recientes del usuario para mostrar en el historial lateral.

**Parámetros de ruta:**
- `user_id` — ID del usuario

**Query params:**

| Parámetro | Tipo | Obligatorio | Descripción |
|---|---|---|---|
| `limit` | `int` | No | Número máximo de conversaciones a devolver (por defecto: `30`) |

**Respuesta (200 OK):**
```json
{
  "user_id": "Admin-Aetheris",
  "conversations": [
    {"thread_id": "uuid", "title": "Elimina el archivo...", "created_at": "...", "updated_at": "..."}
  ]
}
```

---

### `POST /api/v1/chat/{thread_id}/resume`

Reanuda la ejecución del agente tras una interrupción HITL (Human-in-the-Loop).

**Parámetros de ruta:**
- `thread_id` — ID del hilo a reanudar (devuelto por el evento `conversation_id`)

**Cuerpo de la solicitud:**
```json
{
  "approved": true,
  "user_id": "Admin-Aetheris"
}
```

| Campo | Tipo | Obligatorio | Descripción |
|---|---|---|---|
| `approved` | `boolean` | Sí | `true` para aprobar la acción, `false` para rechazarla |
| `user_id` | `string` | No | ID de usuario (por defecto: `"Admin-Aetheris"`) |

**Respuesta:** Stream SSE con los eventos `action_result`/`action_error` por cada acción ejecutada, seguidos de `token` y `done` con el resumen generado por el LLM.

**Errores:**
- `404 Not Found` — No hay ninguna acción pendiente para el hilo indicado.

---

### `GET /api/v1/chat/{thread_id}/history`

Recupera el historial completo de mensajes de un hilo de conversación.

**Parámetros de ruta:**
- `thread_id` — ID del hilo

**Respuesta (200 OK):**
```json
{
  "thread_id": "f40e6f4b-4df2-4f2c-9207-172380d4623d",
  "messages": [
    {"role": "human", "content": "¿Qué es AETHERIS?", "timestamp": null},
    {"role": "ai",    "content": "AETHERIS es un agente cognitivo autónomo...", "timestamp": null}
  ]
}
```

---

## Documentos

### `POST /api/v1/documents/upload`

Sube e ingesta un documento en la base de conocimiento RAG.

**Cuerpo de la solicitud (form-data):**

| Campo | Tipo | Descripción |
|---|---|---|
| `file` | `File` | Fichero a ingestar (PDF, DOCX, TXT, MD) |
| `user_id` | `string` | ID de usuario propietario del documento |

**Respuesta (201 Created):**
```json
{
  "document_id": "md5hash",
  "filename": "informe_q4.pdf",
  "n_chunks": 42,
  "collection_name": "aetheris",
  "ingested_at": "2026-04-20T10:00:00"
}
```

**Errores:**
- `400 Bad Request` — Tipo de fichero no soportado
- `422 Unprocessable Entity` — Fichero corrupto o ilegible

---

### `GET /api/v1/documents`

Lista todos los documentos indexados en Chroma.

**Respuesta (200 OK):**
```json
[
  {
    "document_id": "md5hash",
    "filename": "informe_q4.pdf",
    "source": "data/uploads/informe_q4.pdf"
  }
]
```

---

### `DELETE /api/v1/documents/{document_id}`

Elimina un documento y todos sus fragmentos de Chroma.

**Parámetros de ruta:**
- `document_id` — ID del documento (hash MD5)

**Respuesta:**
- `204 No Content` — Eliminado correctamente
- `404 Not Found` — Documento no encontrado

---

## Memoria

### `GET /api/v1/memory/{user_id}`

Recupera la memoria a largo plazo (preferencias KV) de un usuario.

**Respuesta (200 OK):**
```json
{
  "user_id": "Admin-Aetheris",
  "preferences": {
    "language": "Spanish",
    "timezone": "Europe/Madrid",
    "report_format": "markdown"
  }
}
```

---

### `PUT /api/v1/memory/{user_id}`

Actualiza (upsert) las preferencias a largo plazo de un usuario.

**Cuerpo de la solicitud:**
```json
{
  "preferences": {
    "language": "Spanish",
    "timezone": "Europe/Madrid"
  }
}
```

**Respuesta (200 OK):**
```json
{
  "user_id": "Admin-Aetheris",
  "updated_keys": ["language", "timezone"]
}
```

---

## Audio / STT

### `POST /api/v1/speech/transcribe`

Transcribe un fichero de audio a texto mediante faster-whisper (STT local).

**Cuerpo de la solicitud (form-data):**

| Campo | Tipo | Descripción |
|---|---|---|
| `file` | `File` | Fichero de audio (mp3, wav, m4a, ogg, webm) |

**Respuesta (200 OK):**
```json
{
  "text": "Crea una reunión para el lunes a las diez de la mañana."
}
```

**Errores:**
- `400 Bad Request` — Formato de audio no soportado
- `503 Service Unavailable` — faster-whisper no está disponible (modelo no descargado)

---

## Estado del Sistema

### `GET /api/v1/health`

Comprueba el estado general del sistema.

**Respuesta (200 OK):**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "app_env": "development",
  "chroma_ok": true,
  "sqlite_ok": true
}
```

---

### `GET /api/v1/health/langsmith`

Comprueba la conectividad con LangSmith.

**Respuesta (200 OK):**
```json
{
  "langsmith_connected": true,
  "project_name": "aetheris",
  "error": null
}
```

**Respuesta cuando no conecta:**
```json
{
  "langsmith_connected": false,
  "project_name": "aetheris",
  "error": "Connection refused"
}
```

---

### `GET /api/v1/health/google`

Diagnóstico del estado de credenciales Google y herramientas MCP Google cargadas.

**Respuesta (200 OK):**
```json
{
  "status": "ok",
  "credentials": {
    "client_secret_file_exists": true,
    "refresh_token_set": true,
    "calendar_token_exists": true,
    "gmail_token_exists": true
  },
  "mcp": {
    "tools_loaded": 42,
    "google_tools": ["list-calendars", "create-event", "list_files", "..."],
    "clients_alive": 3
  }
}
```

| Campo `status` | Significado |
|---|---|
| `"ok"` | Credenciales configuradas y tools Google cargadas |
| `"partial"` | Credenciales OK pero ninguna tool Google disponible |
| `"error"` | Faltan credenciales (sin `client_secret_file` o sin `refresh_token`) |

---

## Resumen de Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/api/v1/chat` | Iniciar/continuar chat (stream SSE) |
| `POST` | `/api/v1/chat/{thread_id}/resume` | Reanudar tras aprobación HITL |
| `GET` | `/api/v1/chat/{thread_id}/history` | Historial de conversación |
| `DELETE` | `/api/v1/chat/{thread_id}` | Eliminar conversación e historial |
| `GET` | `/api/v1/chat/threads/{user_id}` | Listar conversaciones del usuario |
| `POST` | `/api/v1/documents/upload` | Subir e ingestar documento |
| `GET` | `/api/v1/documents` | Listar documentos indexados |
| `DELETE` | `/api/v1/documents/{document_id}` | Eliminar documento |
| `GET` | `/api/v1/memory/{user_id}` | Leer memoria del usuario |
| `PUT` | `/api/v1/memory/{user_id}` | Actualizar memoria del usuario |
| `POST` | `/api/v1/speech/transcribe` | Transcribir audio a texto (faster-whisper) |
| `GET` | `/api/v1/health` | Estado general del sistema |
| `GET` | `/api/v1/health/langsmith` | Conectividad con LangSmith |
| `GET` | `/api/v1/health/google` | Estado de credenciales y tools Google MCP |

---

## Códigos de Error Comunes

| Código | Descripción |
|---|---|
| `400` | Solicitud incorrecta (tipo de fichero no soportado, parámetros inválidos) |
| `404` | Recurso no encontrado (documento, hilo sin checkpoint HITL pendiente) |
| `422` | Entidad no procesable (validación Pydantic fallida) |
| `500` | Error interno del servidor (fallo del LLM, Chroma inaccesible) |
| `503` | Servicio no disponible (servidores MCP no iniciados, Whisper no disponible) |

---

## Autenticación

La API no requiere autenticación en el entorno de desarrollo. En producción se recomienda añadir un middleware de API key o JWT configurando `SECRET_KEY` en el fichero `.env`.
