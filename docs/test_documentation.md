# AETHERIS — Documentación de Tests

## Descripción General

La suite de tests está organizada en tres niveles siguiendo la pirámide de tests: unitarios, de integración y extremo a extremo (E2E). Todos los tests usan `pytest` con `pytest-asyncio` para soporte asíncrono.

---

## Niveles de Tests

### Tests Unitarios (`tests/unit/`)

Tests rápidos y aislados sin E/S externas. Todas las dependencias externas (LLMs, Chroma, SQLite, mem0) están simuladas mediante mocks.

| Fichero | Qué prueba |
|---|---|
| `test_config.py` | Carga de Settings, validación, comportamiento de caché |
| `test_agent_state.py` | Estructura del TypedDict AgentState, reducer `add_messages`, nuevos campos |
| `test_agent_nodes.py` | Funciones de nodo individuales con `GenericFakeChatModel`, incluyendo guardrail nodes |
| `test_agent_edges.py` | Funciones de enrutamiento puras para todas las intenciones, incluyendo `route_after_input_guardrail` |
| `test_rag_ingest.py` | Carga de documentos, fragmentación, preservación de metadatos |
| `test_rag_retriever.py` | Filtrado por umbral de puntuación, tipado de RetrievalResult |
| `test_memory.py` | CRUD de memoria a largo plazo con SQLite en memoria |
| `test_guardrails.py` | Guardrails de entrada y salida EN+ES: inyección, PII, redacción |
| `test_llm_factory.py` | Factoría LLM: construcción, fallback OpenAI→Bedrock, manejo de errores |

**Ejecutar:**
```bash
pytest tests/unit -v -m unit
```

---

### Tests de Integración (`tests/integration/`)

Tests que usan Chroma real (en `/tmp`) y SQLite real, pero simulan las APIs externas (OpenAI, Tavily, Google, mem0).

| Fichero | Qué prueba |
|---|---|
| `test_rag_pipeline.py` | Bucle completo ingestión → recuperación. **Valida ≥85% de tasa de acierto en 10 consultas.** |
| `test_agent_graph.py` | Compilación del grafo, invocación de un turno, persistencia de checkpoint |
| `test_api_chat.py` | Endpoints de salud, memoria y chat; filtrado SSE por nodo (`llm_node`); evento `guardrail_blocked` |
| `test_api_documents.py` | Endpoints de subida, listado y eliminación de documentos |
| `test_mcp_tools.py` | Configuración de servidores MCP, degradación sin claves, nueva API sin `async with` |

**Ejecutar:**
```bash
pytest tests/integration -v -m integration
```

---

### Tests E2E (`tests/e2e/`)

Tests de pila completa que ejercen flujos de trabajo automatizados completos. Las APIs externas (OpenAI, Tavily, Google) se simulan mediante `FakeChatModel` y `unittest.mock`.

| Fichero | Flujo probado |
|---|---|
| `test_market_research_flow.py` | Búsqueda web Tavily → recuperación RAG → HITL Google Calendar |
| `test_summarize_emails_flow.py` | Listar Gmail → resumen LLM → salida estructurada |
| `test_find_document_flow.py` | Recuperación RAG → respuesta con cita de fuente |

**Ejecutar:**
```bash
pytest tests/e2e -v -m e2e
```

---

## Ejecutar Todos los Tests

```bash
# Todos los tests
pytest

# Con informe de cobertura
pytest --cov=aetheris --cov-report=term-missing

# Nivel específico
pytest tests/unit -v
pytest tests/integration -v
pytest tests/e2e -v
```

---

## Fixtures Clave (`tests/conftest.py`)

| Fixture | Descripción |
|---|---|
| `mock_llm` | `GenericFakeChatModel` que devuelve `"This is a test response."` |
| `mock_llm_json` | `GenericFakeChatModel` que devuelve un JSON para extracción de memoria |
| `override_settings` | Redirige todas las rutas de datos a `tmp_path` (autouse) |
| `sample_txt_file` | Fichero de texto con contenido conocido sobre AETHERIS |
| `sample_md_file` | Fichero Markdown con contenido estructurado conocido |
| `api_client` | `FastAPI TestClient` con grafo simulado |
| `base_agent_state` | `AgentState` mínimo válido con todos los campos requeridos |

---

## Validación de Tasa de Acierto RAG

El test `test_rag_pipeline.py::test_rag_hit_rate_above_85_percent` es la validación principal del objetivo de precisión de recuperación >85%. El proceso es:

1. Ingesta 2 documentos con 10 hechos conocidos
2. Emite 10 consultas dirigidas (una por hecho)
3. Comprueba si la palabra clave esperada aparece en algún fragmento recuperado
4. Valida que ≥ 8 de 10 consultas tienen éxito (≥85%)

Este test usa `MockEmbeddings` — embeddings deterministas basados en hash — por lo que no se realizan llamadas reales a la API de OpenAI, pero la búsqueda de similitud de Chroma se ejecuta de verdad.

---

## Tests de Guardrails (`tests/unit/test_guardrails.py`)

Validan el comportamiento bilingüe (EN + ES) de los guardrails de seguridad:

**Guardrail de Entrada:**
- Detección de inyección en inglés: `ignore instructions`, `reveal system prompt`, `act as`
- Detección de inyección en español: `ignora las instrucciones`, `muestra el prompt del sistema`, `actúa como`
- Redacción de PII: email, teléfono, DNI/NIE español, tarjeta de crédito, IBAN
- Rechazo por longitud excesiva
- Pasaje limpio de texto sin riesgos

**Guardrail de Salida:**
- Redacción de claves API filtradas
- Redacción de contraseñas en EN y ES
- Truncado de respuestas muy largas
- Pasaje limpio de texto sin riesgos

---

## Tests de Factoría LLM (`tests/unit/test_llm_factory.py`)

Validan el sistema de fallback OpenAI → Bedrock:

- Construcción correcta del LLM de OpenAI cuando hay clave API
- Activación del fallback a Bedrock cuando está configurado
- Error claro cuando no hay ningún proveedor configurado
- Binding de herramientas MCP al LLM

---

## Notas

- Los tests que requieren claves API reales (`OPENAI_API_KEY`, `TAVILY_API_KEY`) deben ejecutarse en un entorno CI con secretos configurados, o saltarse localmente.
- El fixture `override_settings` limpia la caché LRU de `get_settings()` antes y después de cada test para garantizar el aislamiento.
- Los tests de subprocess MCP simulan `asyncio.create_subprocess_exec` para evitar lanzar procesos Node.js reales.
- Los mocks de `get_llm()` deben devolver una tupla `(fake_llm, "test")` ya que la función tiene múltiples valores de retorno.
- El campo `base_agent_state` incluye todos los campos del TypedDict: `guardrail_passed`, `guardrail_violations`, `llm_provider` y `execution_plan`.
- El checkpointer en tests síncronos usa `SqliteSaver(sqlite3.connect(..., check_same_thread=False))` directamente; en tests de API se usa `AsyncMock`.
- Usar siempre `GenericFakeChatModel(messages=iter([...]))` — `FakeChatModel(responses=[...])` fue eliminado en versiones recientes de `langchain-core`.
