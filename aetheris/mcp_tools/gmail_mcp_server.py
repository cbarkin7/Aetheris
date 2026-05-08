#!/usr/bin/env python3
"""
Servidor MCP de Gmail para AETHERIS -- transporte stdio.

Protocolo : MCP (Model Context Protocol) sobre stdin/stdout
Auth      : OAuth2 con google-auth (refresh automatico, sin flujo interactivo)
Ventajas  : sin dependencia npm, produccion-ready, refresh de token nativo

Variables de entorno requeridas:
  GMAIL_TOKEN_PATH          -> ruta a .gmail-token.json (access + refresh token)
  GMAIL_CLIENT_SECRET_PATH  -> ruta a client_secret_aetheris.json (client_id + secret)

Herramientas expuestas:
  list_emails      -> listar emails recientes de la bandeja de entrada
  get_email        -> obtener contenido completo de un email por ID
  search_emails    -> buscar emails con sintaxis Gmail (from:, subject:, is:unread...)
  send_email       -> enviar un email nuevo
  create_draft     -> crear un borrador sin enviar
  delete_email     -> mover un email a la papelera (reversible)
  reply_to_email   -> responder a un email manteniendo el hilo
"""
import asyncio
import base64
import json
import logging
import os
import sys
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# Redirigir logs a stderr para no contaminar el canal JSON-RPC de stdout
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[gmail-mcp] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Autenticacion Gmail -- servicio cacheado mientras el token sea valido
# ---------------------------------------------------------------------------

# Singleton del cliente Gmail: se reutiliza entre tool calls para evitar
# reconstruir el servicio (y su peticion HTTP de discovery) en cada invocacion.
_gmail_service_cache: object = None
_gmail_service_expiry: float = 0.0   # epoch segundos


def _build_gmail_service(force_refresh: bool = False):
    """
    Devuelve el cliente Gmail API reutilizando la instancia cacheada mientras
    el token sea valido. Reconstruye solo cuando el token expira o se fuerza.

    Optimizaciones:
    - static_discovery=True  -> usa el JSON de descubrimiento incluido en el
      paquete (sin peticion HTTP adicional en cada llamada). Elimina el
      WinError 10060 causado por el timeout al descargar el discovery doc.
    - Cache de la instancia   -> un unico build() por sesion del servidor.
    - Request(timeout=30)     -> el refresco de token falla en 30 s en lugar
      de esperar indefinidamente.
    """
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    global _gmail_service_cache, _gmail_service_expiry

    # Devolver el servicio cacheado si el token todavia es valido (margen 60 s)
    if not force_refresh and _gmail_service_cache is not None:
        if time.time() < _gmail_service_expiry - 60:
            return _gmail_service_cache

    SCOPES = [
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
    ]

    token_path = os.environ.get("GMAIL_TOKEN_PATH", "")
    secret_path = os.environ.get("GMAIL_CLIENT_SECRET_PATH", "")

    if not token_path or not Path(token_path).exists():
        raise FileNotFoundError(
            f"GMAIL_TOKEN_PATH no encontrado: '{token_path}'. "
            "Asegurate de que ensure_google_token_files() ha sido llamado."
        )

    with open(token_path, encoding="utf-8") as f:
        token_data = json.load(f)

    # Leer client_id y client_secret: primero del token, luego del secret file
    client_id = token_data.get("client_id", "")
    client_secret = token_data.get("client_secret", "")
    if (not client_id or not client_secret) and secret_path and Path(secret_path).exists():
        try:
            with open(secret_path, encoding="utf-8") as f:
                sec = json.load(f)
            inner = sec.get("installed") or sec.get("web") or {}
            client_id = client_id or inner.get("client_id", "")
            client_secret = client_secret or inner.get("client_secret", "")
        except Exception as e:
            logger.warning("No se pudo leer client_secret: %s", e)

    creds = Credentials(
        token=token_data.get("access_token") or None,
        refresh_token=token_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )

    if not creds.valid:
        if creds.expired and creds.refresh_token:
            logger.info("Access token caducado -- refrescando...")
            # Timeout explicito: falla en 30 s en lugar de esperar indefinidamente
            creds.refresh(Request(timeout=30))
            # Persistir el nuevo access_token
            token_data["access_token"] = creds.token
            if creds.expiry:
                token_data["expiry_date"] = int(creds.expiry.timestamp() * 1000)
            try:
                with open(token_path, "w", encoding="utf-8") as f:
                    json.dump(token_data, f, indent=2)
                logger.info("Token actualizado en %s", token_path)
            except Exception as e:
                logger.warning("No se pudo guardar el token actualizado: %s", e)
        else:
            raise RuntimeError(
                "Las credenciales Gmail no son validas y no se pueden refrescar. "
                "Ejecuta la reautorizacion OAuth2."
            )

    # static_discovery=True: usa el JSON incluido en el paquete.
    # Elimina la peticion HTTP de discovery que causaba WinError 10060
    # (timeout TCP de ~25 s al intentar contactar discovery.googleapis.com).
    service = build("gmail", "v1", credentials=creds, static_discovery=True)

    # Actualizar cache
    _gmail_service_cache = service
    _gmail_service_expiry = creds.expiry.timestamp() if creds.expiry else time.time() + 3540
    logger.info("Servicio Gmail construido | expira_en=%.0f s",
                _gmail_service_expiry - time.time())

    return service


# ---------------------------------------------------------------------------
# Helpers de parseo de mensajes
# ---------------------------------------------------------------------------

def _extract_headers(message: dict) -> dict:
    headers_list = message.get("payload", {}).get("headers", [])
    return {h["name"]: h["value"] for h in headers_list}


def _decode_body(payload: dict) -> str:
    """Extrae el cuerpo en texto plano de un payload Gmail (recursivo)."""
    # Cuerpo directo
    body_data = payload.get("body", {}).get("data", "")
    if body_data:
        return base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")

    # Partes multipart
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    # Fallback: primera parte con datos
    for part in payload.get("parts", []):
        data = part.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    return ""


def _format_message_summary(service, msg_id: str) -> str:
    """Devuelve una linea resumen de un mensaje (metadatos solo)."""
    full = service.users().messages().get(
        userId="me", id=msg_id, format="metadata",
        metadataHeaders=["Subject", "From", "Date"],
    ).execute()
    h = _extract_headers(full)
    return (
        f"ID: {msg_id}\n"
        f"  De     : {h.get('From', '')}\n"
        f"  Asunto : {h.get('Subject', '(sin asunto)')}\n"
        f"  Fecha  : {h.get('Date', '')}\n"
        f"  Snippet: {full.get('snippet', '')[:120]}"
    )


# ---------------------------------------------------------------------------
# Ejecucion sincrona de herramientas Gmail
# ---------------------------------------------------------------------------

def _execute_tool(service, name: str, args: dict) -> str:
    """Despacha la llamada a la herramienta Gmail correcta."""

    # -- list_emails -----------------------------------------------------------
    if name == "list_emails":
        max_results = min(int(args.get("maxResults", 10)), 50)
        label_ids = args.get("labelIds", ["INBOX"])
        query = args.get("query", "")

        params: dict = {"userId": "me", "maxResults": max_results}
        if label_ids:
            params["labelIds"] = label_ids
        if query:
            params["q"] = query

        result = service.users().messages().list(**params).execute()
        messages = result.get("messages", [])
        if not messages:
            return "No se encontraron emails."

        summaries = [_format_message_summary(service, m["id"]) for m in messages]
        return f"Se encontraron {len(messages)} emails:\n\n" + "\n\n".join(summaries)

    # -- get_email -------------------------------------------------------------
    elif name == "get_email":
        msg = service.users().messages().get(
            userId="me", id=args["messageId"], format="full",
        ).execute()
        h = _extract_headers(msg)
        body = _decode_body(msg.get("payload", {}))
        return json.dumps({
            "id": msg["id"],
            "threadId": msg.get("threadId", ""),
            "subject": h.get("Subject", "(sin asunto)"),
            "from": h.get("From", ""),
            "to": h.get("To", ""),
            "date": h.get("Date", ""),
            "snippet": msg.get("snippet", ""),
            "body": body[:3000],
            "labels": msg.get("labelIds", []),
        }, ensure_ascii=False, indent=2)

    # -- search_emails ---------------------------------------------------------
    elif name == "search_emails":
        max_results = min(int(args.get("maxResults", 10)), 50)
        result = service.users().messages().list(
            userId="me", q=args["query"], maxResults=max_results,
        ).execute()
        messages = result.get("messages", [])
        if not messages:
            return f"No se encontraron emails para la busqueda: \"{args['query']}\""

        summaries = [_format_message_summary(service, m["id"]) for m in messages]
        return f"{len(messages)} resultados para \"{args['query']}\":\n\n" + "\n\n".join(summaries)

    # -- send_email ------------------------------------------------------------
    elif name == "send_email":
        msg = MIMEMultipart("alternative")
        msg["To"] = args["to"]
        msg["Subject"] = args["subject"]
        if args.get("cc"):
            msg["Cc"] = args["cc"]
        msg.attach(MIMEText(args["body"], "plain", "utf-8"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        sent = service.users().messages().send(
            userId="me", body={"raw": raw},
        ).execute()
        return f"Email enviado correctamente a {args['to']}. ID: {sent['id']}"

    # -- create_draft ----------------------------------------------------------
    elif name == "create_draft":
        msg = MIMEMultipart("alternative")
        msg["To"] = args["to"]
        msg["Subject"] = args["subject"]
        msg.attach(MIMEText(args["body"], "plain", "utf-8"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        draft = service.users().drafts().create(
            userId="me", body={"message": {"raw": raw}},
        ).execute()
        return f"Borrador creado. ID: {draft['id']}"

    # -- delete_email ----------------------------------------------------------
    elif name == "delete_email":
        service.users().messages().trash(
            userId="me", id=args["messageId"],
        ).execute()
        return f"Email {args['messageId']} movido a la papelera."

    # -- reply_to_email --------------------------------------------------------
    elif name == "reply_to_email":
        original = service.users().messages().get(
            userId="me", id=args["messageId"], format="metadata",
            metadataHeaders=["Subject", "From", "Message-ID", "References"],
        ).execute()
        h = _extract_headers(original)

        msg = MIMEMultipart("alternative")
        msg["To"] = h.get("From", "")
        msg["Subject"] = ("Re: " + h.get("Subject", "")).replace("Re: Re: ", "Re: ")
        msg["In-Reply-To"] = h.get("Message-ID", "")
        msg["References"] = " ".join(
            filter(None, [h.get("References", ""), h.get("Message-ID", "")])
        )
        msg.attach(MIMEText(args["body"], "plain", "utf-8"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        sent = service.users().messages().send(
            userId="me",
            body={"raw": raw, "threadId": original.get("threadId", "")},
        ).execute()
        return f"Respuesta enviada al hilo del email {args['messageId']}. ID: {sent['id']}"

    else:
        return f"Error: herramienta '{name}' no reconocida."


# ---------------------------------------------------------------------------
# Servidor MCP principal
# ---------------------------------------------------------------------------

_TOOLS_SCHEMA = [
    {
        "name": "list_emails",
        "description": "Lista los emails mas recientes de la bandeja de entrada de Gmail.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "maxResults": {"type": "integer", "description": "Numero maximo de emails a devolver (defecto: 10, max: 50)", "default": 10},
                "labelIds": {"type": "array", "items": {"type": "string"}, "description": "Etiquetas a filtrar. Ej: ['INBOX'], ['UNREAD'], ['SENT']"},
                "query": {"type": "string", "description": "Filtro adicional en sintaxis Gmail. Ej: 'is:unread', 'from:pedro@empresa.com'"},
            },
        },
    },
    {
        "name": "get_email",
        "description": "Obtiene el contenido completo de un email por su ID (asunto, remitente, cuerpo).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "messageId": {"type": "string", "description": "ID del mensaje Gmail"},
            },
            "required": ["messageId"],
        },
    },
    {
        "name": "search_emails",
        "description": (
            "Busca emails usando la sintaxis de busqueda avanzada de Gmail. "
            "Ejemplos: 'from:pedro@empresa.com', 'subject:TFM', 'is:unread has:attachment', "
            "'after:2024/01/01 before:2024/12/31'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query de busqueda Gmail"},
                "maxResults": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "Envia un email desde la cuenta Gmail del usuario.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Destinatario (email o 'Nombre <email>')"},
                "subject": {"type": "string", "description": "Asunto del email"},
                "body": {"type": "string", "description": "Cuerpo del mensaje en texto plano"},
                "cc": {"type": "string", "description": "CC (opcional)"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "create_draft",
        "description": "Crea un borrador de email sin enviarlo.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "delete_email",
        "description": "Mueve un email a la papelera (accion reversible desde Gmail).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "messageId": {"type": "string", "description": "ID del mensaje a eliminar"},
            },
            "required": ["messageId"],
        },
    },
    {
        "name": "reply_to_email",
        "description": "Responde a un email existente manteniendo el hilo de conversacion.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "messageId": {"type": "string", "description": "ID del mensaje original al que responder"},
                "body": {"type": "string", "description": "Cuerpo de la respuesta"},
            },
            "required": ["messageId", "body"],
        },
    },
]


async def run_server() -> None:
    """Arranca el servidor MCP Gmail sobre stdio (JSON-RPC)."""
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    import mcp.types as types

    server = Server("gmail-aetheris")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in _TOOLS_SCHEMA
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        logger.info("Herramienta llamada: %s | args=%s", name, list(arguments.keys()))
        try:
            loop = asyncio.get_event_loop()
            service = await loop.run_in_executor(None, _build_gmail_service)
            result = await loop.run_in_executor(
                None, lambda: _execute_tool(service, name, arguments)
            )
            return [types.TextContent(type="text", text=result)]
        except Exception as exc:
            logger.error("Error en herramienta '%s': %s", name, exc)
            # Si el error puede ser de token expirado, invalidar cache para forzar
            # reconstruccion en la siguiente llamada
            error_str = str(exc).lower()
            if any(kw in error_str for kw in ("401", "invalid_grant", "expired", "unauthorized")):
                global _gmail_service_cache
                _gmail_service_cache = None
                logger.info("Cache de servicio invalidado por error de auth")
            return [types.TextContent(type="text", text=f"Error: {exc}")]

    logger.info("Servidor MCP Gmail arrancado (stdio)")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(run_server())
