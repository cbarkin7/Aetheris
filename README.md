# AETHERIS

**Agente Cognitivo Autónomo** — Trabajo Fin de Máster (TFM, Máster en Soluciones IAG)

AETHERIS evoluciona el concepto de chatbot hacia un agente que **piensa, recupera, busca y actúa**. Combina una base de conocimiento RAG, búsqueda web en tiempo real (Tavily), automatización de Google Workspace, confirmación Human-in-the-Loop, memoria persistente con mem0.ai y observabilidad completa con LangSmith — todo ello tras una interfaz Streamlit limpia y con seguridad mediante Guardrails bilingües (EN/ES).

---

## Arquitectura

```
Streamlit (puerto 8501) ──SSE──► FastAPI (puerto 8000) ──► Grafo LangGraph
                                                              │
                              ┌───────────────────────────────┤
                              │               │               │
                         RAG (Chroma)    Herramientas MCP  Memoria
                                       (Tavily/Google)    (mem0 + SQLite + Chroma)
```

Documentación completa de arquitectura: [`docs/architecture.md`](docs/architecture.md)

---

## Características

| Característica | Tecnología |
|---|---|
| RAG (documentos privados) | LangChain + Chroma, recuperación MMR, >85% tasa de acierto |
| Búsqueda web | Servidor MCP Tavily |
| Google Workspace | Calendar, Gmail, Drive mediante servidor MCP Google |
| Human-in-the-Loop | LangGraph `interrupt_before`, reanudación vía API |
| Guardrails de seguridad | Filtrado de entrada/salida bilingüe (EN+ES), detección de inyección de prompts, redacción de PII |
| Fallback LLM | OpenAI → AWS Bedrock (Anthropic Claude) automático |
| Memoria a corto plazo | mem0.ai (cloud o local con Chroma) |
| Memoria a largo plazo | SQLite (clave-valor) + Chroma (búsqueda semántica) |
| Observabilidad | LangSmith — trazas, costes, latencia |
| Streaming | FastAPI SSE → Streamlit renderizado token a token |

---

## Inicio rápido

### 1. Requisitos previos

- Python 3.12+
- Node.js 18+ (necesario para servidores MCP vía `npx`)
- Claves API: OpenAI, LangSmith, Tavily, Google OAuth
- (Opcional) Credenciales AWS para fallback Bedrock
- (Opcional) Clave API de mem0.ai para modo cloud

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar entorno

```bash
cp .env.example .env
# Edita .env y rellena todas las claves API
```

### 4. Ejecutar el backend

```bash
uvicorn aetheris.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Ejecutar el frontend

```bash
streamlit run aetheris/ui/app.py --server.port 8501
```

Abre [http://localhost:8501](http://localhost:8501) en tu navegador.

---

## Ingestar documentos

```bash
# Ingestar un solo fichero
python scripts/ingest_documents.py --file ./mi_informe.pdf

# Ingestar una carpeta completa
python scripts/ingest_documents.py --dir ./docs/
```

---

## Variables de entorno

| Variable | Obligatoria | Descripción |
|---|---|---|
| `OPENAI_API_KEY` | Sí | Clave API de OpenAI (LLM + embeddings) |
| `ANTHROPIC_API_KEY` | No | Opcional: para modelo `claude-haiku-4-5` |
| `AWS_ACCESS_KEY_ID` | No | Credenciales AWS para Bedrock (fallback) |
| `AWS_SECRET_ACCESS_KEY` | No | Credenciales AWS para Bedrock (fallback) |
| `AWS_REGION` | No | Región AWS (por defecto: `us-east-1`) |
| `BEDROCK_MODEL_ID` | No | ID del modelo en Bedrock |
| `LANGSMITH_API_KEY` | Sí | Observabilidad LangSmith |
| `LANGSMITH_PROJECT` | No | Nombre del proyecto (por defecto: `aetheris`) |
| `TAVILY_API_KEY` | No | Búsqueda web MCP Tavily |
| `MEM0_API_KEY` | No | mem0.ai cloud (dejar vacío para modo local) |
| `GOOGLE_CLIENT_ID` | No | OAuth de Google para Calendar/Gmail/Drive |
| `GUARDRAILS_ENABLED` | No | Activar guardrails de seguridad (por defecto: `true`) |
| `LLM_MODEL` | No | Nombre del modelo (por defecto: `gpt-4o-mini`) |

Consulta [`.env.example`](.env.example) para la lista completa.

---

## Flujo del agente

```
START → Guardrail entrada → [bloqueado → rechazo | OK → cargar memoria → clasificar intención]
    intención → {RAG | búsqueda web | acción Google (HITL) | LLM directo}
    → generar respuesta → Guardrail salida → guardar memoria → END
```

---

## Fallback LLM

AETHERIS implementa un sistema de fallback automático:

1. **OpenAI** (primario) — `gpt-4o-mini` por defecto
2. **AWS Bedrock** (fallback) — Anthropic Claude vía `ChatBedrockConverse`

Si OpenAI devuelve un error (timeout, cuota, fallo de API), el sistema redirige automáticamente la solicitud a Bedrock sin intervención del usuario. El proveedor utilizado se registra en LangSmith para trazabilidad.

---

## Sistema de memoria

| Capa | Almacén | Alcance | Tecnología |
|---|---|---|---|
| Corto plazo (sesión) | SQLite checkpoints | Por `thread_id` | LangGraph SqliteSaver |
| Corto plazo (conversacional) | mem0.ai | Por `user_id` + `session_id` | mem0 cloud o local |
| Largo plazo (preferencias) | SQLite user_memory | Por `user_id` entre sesiones | Tabla clave-valor |
| Largo plazo (hechos semánticos) | Chroma | Por `user_id`, búsqueda semántica | Colección `aetheris_long_term_facts` |

---

## Testing

```bash
# Todos los tests
pytest

# Solo tests unitarios (rápidos, sin E/S externa)
pytest tests/unit -v

# Tests de integración (Chroma/SQLite reales en /tmp)
pytest tests/integration -v

# Tests E2E (pila completa, APIs simuladas)
pytest tests/e2e -v

# Con cobertura
pytest --cov=aetheris --cov-report=term-missing
```

Consulta [`docs/test_documentation.md`](docs/test_documentation.md) para la documentación completa de tests.

---

## Estructura del proyecto

```
aetheris/
├── agent/          # StateGraph LangGraph, nodos, aristas, prompts
├── guardrails/     # Filtrado de seguridad entrada/salida (EN+ES)
├── rag/            # Ingesta, recuperación, cadena RAG
├── mcp/            # Cliente MCP (Tavily + Google)
├── memory/         # mem0 (corto plazo) + SQLite/Chroma (largo plazo)
├── observability/  # Helpers de trazado LangSmith
├── api/            # Backend FastAPI (routers, schemas, middleware)
├── ui/             # Frontend Streamlit (páginas, componentes)
└── llm.py          # Factoría LLM con fallback OpenAI → Bedrock
```

---

## Referencia de la API

| Método | Endpoint | Descripción |
|---|---|---|
| POST | `/api/v1/chat` | Iniciar/continuar chat (stream SSE) |
| POST | `/api/v1/chat/{id}/resume` | Reanudar tras aprobación HITL |
| GET | `/api/v1/chat/{id}/history` | Obtener historial de conversación |
| POST | `/api/v1/documents/upload` | Subir + ingestar documento |
| GET | `/api/v1/documents` | Listar documentos indexados |
| DELETE | `/api/v1/documents/{id}` | Eliminar documento |
| GET | `/api/v1/memory/{user_id}` | Obtener memoria del usuario |
| PUT | `/api/v1/memory/{user_id}` | Actualizar memoria del usuario |
| GET | `/api/v1/health` | Estado del sistema |
| GET | `/api/v1/health/langsmith` | Conectividad con LangSmith |

Documentación interactiva disponible en [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Licencia

Proyecto académico — Máster en Soluciones de Inteligencia Artificial Generativa (EBIS Business Techschool).
