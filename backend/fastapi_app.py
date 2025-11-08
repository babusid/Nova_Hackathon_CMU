from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import modal
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

MODEL_CACHE_PATH = "/model-cache"
STATIC_SRC = Path(__file__).with_name("static_frontend").resolve()
MODEL_VOLUME = modal.Volume.from_name("voice-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "torch",
        "numpy",
        "moshi==0.1.0",
        "huggingface_hub==0.24.7",
        "hf_transfer==0.1.8",
        "sphn==0.1.4",
        "sentencepiece",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": MODEL_CACHE_PATH,
        }
    )
    .add_local_dir(STATIC_SRC, "/assets")
)

with image.imports():
    import numpy as np
    import sphn
    import torch
    from huggingface_hub import hf_hub_download
    from moshi.models import LMGen, loaders
    import sentencepiece


app = modal.App("voice-hackathon-app", image=image)
web_app = FastAPI()
logger = logging.getLogger("uvicorn.error")


class EditorStateStore:
    """Keeps the latest editor snapshot in-memory (currently unused in generation)."""

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


class MoshiEngine:
    """Loads Mimi/Moshi models and produces session state."""

    _instance: Optional["MoshiEngine"] = None

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_models()

    def _load_models(self) -> None:
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(8)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.sample_rate = self.mimi.sample_rate

        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        self.moshi = loaders.get_moshi_lm(moshi_weight, device=self.device)
        self.lm_gen = LMGen(
            self.moshi,
            temp=0.8,
            temp_text=0.8,
            top_k=250,
            top_k_text=25,
        )

        tokenizer_config = hf_hub_download(
            loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME
        )
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_config)

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self._warm_up()

    def _warm_up(self) -> None:
        for _ in range(4):
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=torch.float32, device=self.device
            )
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @classmethod
    def instance(cls) -> "MoshiEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset_models(self) -> None:
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()


class MoshiSession:
    """Handles a single websocket session using Mimi/Moshi streaming loops."""

    def __init__(self, websocket: WebSocket, engine: MoshiEngine) -> None:
        self.websocket = websocket
        self.engine = engine
        self.opus_stream_outbound = sphn.OpusStreamWriter(self.engine.sample_rate)
        self.opus_stream_inbound = sphn.OpusStreamReader(self.engine.sample_rate)
        self.tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        self.engine.reset_models()
        await self.websocket.accept()
        logger.info("voice_ws session started")

        try:
            self.tasks = [
                asyncio.create_task(self.recv_loop()),
                asyncio.create_task(self.inference_loop()),
                asyncio.create_task(self.send_loop()),
            ]
            await asyncio.gather(*self.tasks)
        except WebSocketDisconnect:
            logger.info("voice_ws disconnected")
        except Exception:
            logger.exception("voice_ws session crashed")
            await self.websocket.close(code=1011)
            raise
        finally:
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            try:
                await self.websocket.close(code=1000)
            except Exception:
                pass
            logger.info("voice_ws session ended")

    async def recv_loop(self) -> None:
        while True:
            data = await self.websocket.receive_bytes()
            if not data:
                continue
            self.opus_stream_inbound.append_bytes(data)

    async def send_loop(self) -> None:
        while True:
            await asyncio.sleep(0.001)
            msg = self.opus_stream_outbound.read_bytes()
            if not msg:
                continue
            await self.websocket.send_bytes(b"\x01" + msg)

    async def inference_loop(self) -> None:
        buffer_pcm: Optional[np.ndarray] = None
        while True:
            await asyncio.sleep(0.001)
            pcm = self.opus_stream_inbound.read_pcm()
            if pcm is None or pcm.shape[-1] == 0:
                continue

            if buffer_pcm is None:
                buffer_pcm = pcm
            else:
                buffer_pcm = np.concatenate((buffer_pcm, pcm), axis=-1)

            while buffer_pcm.shape[-1] >= self.engine.frame_size:
                chunk = buffer_pcm[:, : self.engine.frame_size]
                buffer_pcm = buffer_pcm[:, self.engine.frame_size :]

                torch_chunk = torch.from_numpy(chunk).to(self.engine.device)[None, None]
                codes = self.engine.mimi.encode(torch_chunk)

                for c in range(codes.shape[-1]):
                    tokens = self.engine.lm_gen.step(codes[:, :, c : c + 1])
                    if tokens is None:
                        continue

                    main_pcm = self.engine.mimi.decode(tokens[:, 1:])
                    main_pcm = main_pcm.cpu()
                    self.opus_stream_outbound.append_pcm(main_pcm[0, 0].numpy())

                    text_token = tokens[0, 0, 0].item()
                    if text_token not in (0, 3):
                        text = self.engine.text_tokenizer.id_to_piece(text_token)
                        text = text.replace("▁", " ")
                        await self.websocket.send_bytes(
                            b"\x02" + text.encode("utf-8", errors="ignore")
                        )


@web_app.websocket("/voice_ws")
async def voice_ws(websocket: WebSocket):
    engine = MoshiEngine.instance()
    session = MoshiSession(websocket, engine)
    await session.run()


@web_app.post("/editor-state")
async def update_editor_state(payload: dict):
    document = payload.get("document")
    if document is None:
        raise HTTPException(status_code=400, detail="Missing 'document' field")
    await editor_state_store.update(document)
    return {"status": "updated"}


@app.function(
    image=image,
    volumes={MODEL_CACHE_PATH: MODEL_VOLUME},
    gpu="A100",
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app


if __name__ == "__main__":
    app.deploy("webapp")
