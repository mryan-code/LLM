"""WebSocket server for live camera frames and microphone audio.

High-risk: this accepts a network stream of camera and microphone bytes and
forwards evaluation events. Raw media is not written to disk or the database.
"""

from __future__ import annotations

import asyncio
import json
import os
import ssl
import threading
from pathlib import Path
from typing import Any

from storm_zero_llm.agent import StormZeroAgent
from storm_zero_llm.realtime import RealtimeSession, configuration_error
from storm_zero_llm.realtime_models import describe_frame, transcribe_pcm
from storm_zero_llm.server import StormZeroRequestHandler
from storm_zero_llm.tts import KokoroTTSService

_active_connections: dict[int, Any] = {}
_active_lock = threading.Lock()


def _realtime_ssl_context(project_root: Path) -> ssl.SSLContext | None:
	"""TLS for the realtime socket, mirroring the API: certificates/<ENV>-key.pem and -cert.pem."""
	node_env = (os.environ.get("ENV") or "dev").strip() or "dev"
	key_path = Path(project_root) / "certificates" / f"{node_env}-key.pem"
	cert_path = Path(project_root) / "certificates" / f"{node_env}-cert.pem"
	if not key_path.is_file() or not cert_path.is_file():
		print(
			f"Realtime socket: {key_path.name}/{cert_path.name} not found, using ws://",
			flush=True,
		)
		return None
	context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
	context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
	return context


def start_realtime_server(agent: StormZeroAgent) -> threading.Thread | None:
    """Listen on LLM_REALTIME_PORT. A missing websockets package leaves the HTTP server running."""
    try:
        import websockets
    except ImportError:
        print(
            "Realtime socket was not started. Install the realtime extra: websockets and faster-whisper.",
            flush=True,
        )
        return None

    host = agent.config.llm_host
    port = agent.config.llm_realtime_port
    ssl_context = _realtime_ssl_context(agent.config.project_root)
    scheme = "wss" if ssl_context else "ws"
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    holder: dict[str, Any] = {}

    def runner() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_serve(agent, websockets, host, port, ready, holder, ssl_context))
        except Exception as exc:
            print(f"Realtime socket failed: {exc}", flush=True)
            ready.set()

    thread = threading.Thread(target=runner, name="llm-realtime", daemon=True)
    thread.start()
    ready.wait(timeout=5)
    if holder.get("server") is None:
        print("Realtime socket was not started.", flush=True)
        return None
    print(f"Realtime socket started on {scheme}://{host}:{port}/realtime", flush=True)
    return thread


async def _serve(agent: StormZeroAgent, websockets: Any, host: str, port: int, ready: threading.Event, holder: dict[str, Any], ssl_context: ssl.SSLContext | None) -> None:
    async def handler(websocket: Any) -> None:
        await handle_connection(agent, websocket)

    # No keepalive pings: this socket is loopback-only with constant frame traffic,
    # and the vision inference can starve the event loop long enough to trip a ping timeout.
    server = await websockets.serve(handler, host, port, ssl=ssl_context, ping_interval=None)
    holder["server"] = server
    ready.set()
    await server.wait_closed()


async def handle_connection(agent: StormZeroAgent, websocket: Any) -> None:
    """Serve one browser session forwarded by the API."""
    path = _socket_path(websocket)
    # Some websockets versions omit the path. This port only serves realtime, so an empty path is accepted.
    if path and path not in {"/realtime", "/realtime/"}:
        await websocket.close()
        return

    connection = _Connection(websocket)
    user_id: int | None = None
    try:
        async for message in websocket:
            if isinstance(message, str):
                user_id = await _handle_text(agent, connection, message, user_id)
                continue
            if isinstance(message, (bytes, bytearray)):
                await _handle_binary(connection, bytes(message))
                continue
    finally:
        if user_id is not None:
            with _active_lock:
                if _active_connections.get(user_id) is connection:
                    _active_connections.pop(user_id, None)


async def _handle_text(
    agent: StormZeroAgent,
    connection: "_Connection",
    message: str,
    user_id: int | None,
) -> int | None:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        await _send(connection.websocket, {"type": "error", "error": "invalid JSON"})
        return user_id
    if not isinstance(payload, dict):
        await _send(connection.websocket, {"type": "error", "error": "invalid JSON"})
        return user_id
    message_type = str(payload.get("type", ""))
    if message_type == "stop":
        await connection.websocket.close()
        return user_id
    if message_type != "start":
        await _send(connection.websocket, {"type": "error", "error": "session has not started"})
        return user_id

    error = configuration_error(agent.config)
    if error:
        await _send(connection.websocket, {"type": "error", "error": error})
        return user_id
    try:
        next_user_id = int(payload["user_id"])
    except (KeyError, TypeError, ValueError):
        await _send(connection.websocket, {"type": "error", "error": "user_id is required"})
        return user_id

    # A new start for the same user replaces the previous live session.
    previous = None
    with _active_lock:
        previous = _active_connections.get(next_user_id)
        _active_connections[next_user_id] = connection
    if previous is not None and previous is not connection:
        await previous.websocket.close()

    connection.session = RealtimeSession(
        user_id=next_user_id,
        vision_fn=lambda jpeg: describe_frame(jpeg, agent.config),
        transcribe_fn=lambda pcm: transcribe_pcm(pcm, agent.config),
        chat_fn=agent.chat,
        tts_fn=lambda text: _tts_payload(agent, next_user_id, text),
    )
    await _send(connection.websocket, {"type": "session_started", "user_id": next_user_id})
    return next_user_id


async def _handle_binary(connection: "_Connection", payload: bytes) -> None:
    if connection.session is None:
        await _send(connection.websocket, {"type": "error", "error": "session has not started"})
        return
    session = connection.session
    events = await asyncio.to_thread(session.handle_binary, payload)
    for event in events:
        await _send(connection.websocket, event)


def _tts_payload(agent: StormZeroAgent, user_id: int, text: str) -> dict[str, Any]:
    """Speak a reply with the same Kokoro payload shape the HTTP chat route uses."""
    voice = agent.get_avatar_voice(user_id) or "af_heart"
    try:
        result = KokoroTTSService(agent.config.tts_path, voice=voice).synthesize(text)
    except Exception as exc:
        return {"base64": None, "mime_type": None, "voice": voice, "error": str(exc)}
    return StormZeroRequestHandler._serialize_tts_result(result)


async def _send(websocket: Any, payload: dict[str, Any]) -> None:
    await websocket.send(json.dumps(payload))


def _socket_path(websocket: Any) -> str:
    request = getattr(websocket, "request", None)
    if request is not None and getattr(request, "path", None):
        return str(request.path)
    return str(getattr(websocket, "path", "") or "")


class _Connection:
    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket
        self.session: RealtimeSession | None = None
