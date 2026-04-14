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
  "user_id": "usuario123",
  "stream": true
}
```

| Campo | Tipo | Obligatorio | Descripción |
|---|---|---|---|
| `message` | `string` | Sí | Mensaje del usuario |
| `thread_id` | `string` | Sí | ID de hilo para checkpointing (usar UUID) |
| `user_id` | `string` | No | ID de usuario para memoria (por defecto: `"default"`) |
| `stream` | `boolean` | No | Activar streaming SSE (por defecto: `true`) |

**Eventos SSE de respuesta:**

| Tipo de evento | Datos | Descripción |
|---|---|---|
| `token` | `{"type": "token", "content": "texto"}` | Token de respuesta generado |
| `done` | `{"type": "done"}` | Respuesta completada |
| `hitl_required` | `{"type": "hitl_required", "actions": [...]}` | Acción Google pendiente de aprobación |
| `guardrail_blocked` | `{"type": "guardrail_blocked", "violations": [...]}` | Mensaje bloqueado por seguridad |
| `error` | `{"type": "error", "message": "..."}` | Error durante el procesamiento |

**Ejemplo de evento `hitl_required`:**
```json
{
  "type": "hitl_required",
  "actions": [
    {"name": "create_calendar_event", "args": {"title": "Reunión", "start": "2026-04-15T10:00"}}
  ]
}
```

---

### `POST /api/v1/chat/{thread_id}/resume`

Reanuda la ejecución del agente tras una interrupción HITL (Human-in-the-Loop).

**Parámetros de ruta:**
- `thread_id` — ID del hilo a reanudar

**Cuerpo de la solicitud:**
```json
{
  "approved": true,
  "user_id": "usuario123"
}
```

| Campo | Tipo | Obligatorio | Descripción |
|---|---|---|---|
| `approved` | `boolean` | Sí | `true` para aprobar la acción, `false` para rechazarla |
| `user_id` | `string` | No | ID de usuario (por defecto: `"default"`) |

**Respuesta:** Stream SSE idéntico al del endpoint `/chat`.

---

### `GET /api/v1/chat/{thread_id}/history`

Recupera el historial completo de mensajes de un hilo de conversación.

**Parámetros de ruta:**
- `thread_id` — ID del hilo

**Respuesta:**
```json
{
  "thread_id": "uuid-del-hilo",
  "messages": [
    {"role": "human", "content": "¿Qué es AETHERIS?"},
    {"role": "ai", "content": "AETHERIS es un agente cognitivo autónomo..."}
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
  "ingested_at": "2026-04-13T10:00:00"
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
    "n_chunks": 42,
    "collection_name": "aetheris",
    "ingested_at": "2026-04-13T10:00:00"
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
  "user_id": "usuario123",
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
  "user_id": "usuario123",
  "updated_keys": ["language", "timezone"]
}
```

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

## Resumen de Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/api/v1/chat` | Iniciar/continuar chat (stream SSE) |
| `POST` | `/api/v1/chat/{thread_id}/resume` | Reanudar tras aprobación HITL |
| `GET` | `/api/v1/chat/{thread_id}/history` | Historial de conversación |
| `POST` | `/api/v1/documents/upload` | Subir e ingestar documento |
| `GET` | `/api/v1/documents` | Listar documentos indexados |
| `DELETE` | `/api/v1/documents/{document_id}` | Eliminar documento |
| `GET` | `/api/v1/memory/{user_id}` | Leer memoria del usuario |
| `PUT` | `/api/v1/memory/{user_id}` | Actualizar memoria del usuario |
| `GET` | `/api/v1/health` | Estado general del sistema |
| `GET` | `/api/v1/health/langsmith` | Conectividad con LangSmith |

---

## Códigos de Error Comunes

| Código | Descripción |
|---|---|
| `400` | Solicitud incorrecta (tipo de fichero no soportado, parámetros inválidos) |
| `404` | Recurso no encontrado (documento, hilo) |
| `422` | Entidad no procesable (validación Pydantic fallida) |
| `500` | Error interno del servidor (fallo del LLM, Chroma inaccesible) |
| `503` | Servicio no disponible (servidores MCP no iniciados) |

---

## Autenticación

La API no requiere autenticación en el entorno de desarrollo. En producción se recomienda añadir un middleware de API key o JWT configurando `SECRET_KEY` en el fichero `.env`.
