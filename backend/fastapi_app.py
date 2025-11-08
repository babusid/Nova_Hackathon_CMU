from __future__ import annotations

import asyncio
from pathlib import Path

import modal
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

image = modal.Image.debian_slim().pip_install("fastapi[standard]")
static_path = Path(__file__).with_name("static_frontend").resolve()
image = image.add_local_dir(static_path, "/assets")

app = modal.App("example-fastapi-app", image=image)
web_app = FastAPI()

class EditorStateStore:
    """Keeps the latest editor snapshot in-memory."""

    def __init__(self) -> None:
        self._state: str = ""
        self._lock = asyncio.Lock()

    async def update(self, state: str) -> None:
        async with self._lock:
            self._state = state

    async def snapshot(self) -> str:
        async with self._lock:
            return self._state


editor_state_store = EditorStateStore()


class VoiceWebSocketSession:

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket: WebSocket = websocket
        self.inbound: asyncio.Queue = asyncio.Queue()
        self.outbound: asyncio.Queue = asyncio.Queue()
        self._shutdown: asyncio.Event = asyncio.Event()

    async def run(self) -> None:
        await self.websocket.accept()
        await self.websocket.send_json({"type": "connection_ack"})

        recv_task = asyncio.create_task(self.recv_loop())
        send_task = asyncio.create_task(self.send_loop())
        inference_task = asyncio.create_task(self.inference_loop())
        tasks = [recv_task, send_task, inference_task]

        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        finally:
            self._shutdown.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def recv_loop(self) -> None:
        try:
            while True:
                message = await self.websocket.receive()
                await self.inbound.put(message)
        except WebSocketDisconnect:
            pass
        finally:
            self._shutdown.set()

    async def send_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                try:
                    payload = await asyncio.wait_for(self.outbound.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                await self._send_payload(payload)
        except WebSocketDisconnect:
            pass
        finally:
            self._shutdown.set()

    async def inference_loop(self) -> None:
        # TODO: implement the speech->text->text->speech pipeline here
        # leverage the global editor_state_store snapshot method to access the
        # current editor code state
        await self._shutdown.wait()

    async def _send_payload(self, payload) -> None:
        if isinstance(payload, bytes):
            await self.websocket.send_bytes(payload)
        elif isinstance(payload, str):
            await self.websocket.send_text(payload)
        else:
            await self.websocket.send_json(payload)


@web_app.websocket("/voice_ws")
async def voice_ws(websocket: WebSocket):
    """Bi-directional audio/data channel for speech <-> text processing."""
    session = VoiceWebSocketSession(websocket=websocket)
    await session.run()


@web_app.post("/editor-state")
async def update_editor_state(payload: dict):
    """Overwrites the backend's view of the coderpad/editor contents."""
    document = payload.get("document")
    if document is None:
        raise HTTPException(status_code=400, detail="Missing 'document' field")
    await editor_state_store.update(document)
    return {"status": "updated"}


@app.function()
@modal.asgi_app()
def fastapi_app():
    web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app


if __name__ == "__main__":
    app.deploy("webapp")
