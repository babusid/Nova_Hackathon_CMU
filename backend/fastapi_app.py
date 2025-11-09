from __future__ import annotations

import asyncio
import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import modal
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Modal image + app setup
# ---------------------------------------------------------------------------

MODEL_CACHE_PATH = "/model-cache"
STATIC_SRC = Path(__file__).with_name("static_frontend").resolve()
MODEL_VOLUME = modal.Volume.from_name("voice-model-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        f"nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12"
    )
    .apt_install("libcudnn9-cuda-12")  # cuDNN runtime
    .apt_install("libcudnn9-dev-cuda-12")  # cuDNN headers (needed by torch wheels)
    .uv_pip_install(
        "fastapi[standard]",
        "numpy",
        "torch",
        "torchaudio",
        "transformers",
        "accelerate",
        "sentencepiece",
        "huggingface_hub",
        "faster-whisper",
        "chatterbox-tts==0.1.1",
        "hf_transfer",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "HF_HUB_CACHE": MODEL_CACHE_PATH,
        }
    )
    .add_local_dir(STATIC_SRC, "/assets")
)

with image.imports():
    import numpy as np
    import torchaudio as ta
    from faster_whisper import WhisperModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from chatterbox.tts import ChatterboxTTS
    import torch


app = modal.App("voice-hackathon-app", image=image)
web_app = FastAPI()
logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Support classes
# ---------------------------------------------------------------------------


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


@dataclass
class ConversationTurn:
    role: str
    content: str


class ConversationTranscript:
    def __init__(self) -> None:
        self._turns: List[ConversationTurn] = []
        self._lock = asyncio.Lock()

    async def append(self, role: str, content: str) -> None:
        async with self._lock:
            self._turns.append(ConversationTurn(role, content.strip()))

    async def history(self) -> List[ConversationTurn]:
        async with self._lock:
            return list(self._turns)


# ---------------------------------------------------------------------------
# Model engines
# ---------------------------------------------------------------------------


class SpeechToTextEngine:
    _instance: Optional["SpeechToTextEngine"] = None

    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(
            "large-v3",
            device=device,
            compute_type=compute_type,
        )
        self.sample_rate = 16000

    @classmethod
    def instance(cls) -> "SpeechToTextEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def transcribe(self, audio_samples: np.ndarray) -> str:
        def _run():
            segments, _ = self.model.transcribe(
                audio_samples,
                beam_size=1,
                language="en",
            )
            return " ".join(segment.text.strip() for segment in segments).strip()

        return await asyncio.to_thread(_run)


class ConversationalBrain:
    _instance: Optional["ConversationalBrain"] = None
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

    def __init__(self) -> None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto",
        )

    @classmethod
    def instance(cls) -> "ConversationalBrain":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def generate_reply(
        self, turns: List[ConversationTurn], editor_state: str
    ) -> str:
        prompt = self._build_prompt(turns, editor_state)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        def _generate():
            return self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output = await asyncio.to_thread(_generate)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    def _build_prompt(self, turns: List[ConversationTurn], editor_state: str) -> str:
        history = "\n".join(
            f"{turn.role.title()}: {turn.content}" for turn in turns[-8:]
        )
        code_context = (
            f"\n\nCode workspace snapshot:\n{editor_state.strip()[:400]}"
            if editor_state.strip()
            else ""
        )
        return (
            "You are a thoughtful technical interviewer. "
            "Respond concisely, ask probing questions, and keep a collaborative tone."
            f"{code_context}\n\nConversation so far:\n{history}\nInterviewer:"
        )


class TextToSpeechEngine:
    _instance: Optional["TextToSpeechEngine"] = None

    def __init__(self) -> None:
        self.model = ChatterboxTTS.from_pretrained(device="cuda")

    @classmethod
    def instance(cls) -> "TextToSpeechEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""

        def _speak():
            wav = self.model.generate(text)
            buffer = io.BytesIO()
            ta.save(buffer, wav.cpu(), self.model.sr, format="wav")
            buffer.seek(0)
            return buffer.read()

        return await asyncio.to_thread(_speak)

try:
    global_stt = SpeechToTextEngine.instance()
    global_llm = ConversationalBrain.instance()
    global_tts = TextToSpeechEngine.instance()
except Exception as e:
    print("Error initializing models:", e)
    raise e

# ---------------------------------------------------------------------------
# Websocket session
# ---------------------------------------------------------------------------


class VoiceWebSocketSession:
    MIN_SAMPLES = 16000 * 2  # 2 seconds of audio

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.inbound: asyncio.Queue[bytes] = asyncio.Queue()
        self.outbound: asyncio.Queue[dict] = asyncio.Queue()
        self._shutdown = asyncio.Event()
        self.audio_buffer = bytearray()
        self.transcript = ConversationTranscript()

    async def run(self) -> None:
        await self.websocket.accept()
        await self.websocket.send_json({"type": "connection_ack"})
        print("run starts")

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
        await self.websocket.send_json({"type": "recv_loop_start"})
        print("recv loop starts")
        try:
            while True:
                message = await self.websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is None:
                    continue
                await self.inbound.put(data)
        except WebSocketDisconnect:
            logger.info("voice_ws disconnected by client")
        finally:
            self._shutdown.set()

    async def send_loop(self) -> None:
        await self.websocket.send_json({"type": "send_loop_start"})
        print("send loop starts")
        try:
            while not self._shutdown.is_set():
                try:
                    payload = await asyncio.wait_for(self.outbound.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                await self.websocket.send_json(payload)
        except WebSocketDisconnect:
            logger.info("voice_ws send loop ended")
        finally:
            self._shutdown.set()

    async def inference_loop(self) -> None:
        await self.websocket.send_json({"type": "inference_loop_start"})
        print("inference loop starts")

        print("calling get_stt")
        try:
            stt = get_stt()
        except Exception as e:
            # print the exception itself
            print(e)
            raise e
        print("stt get completed")

        print("calling get_llm")
        try:
            llm = get_llm()
        except Exception as e:
            # print the exception itself
            print(e)
            raise e
        print("llm get completed")

        print("calling get_tts")
        try:
            tts = get_tts()
        except Exception as e:
            # print the exception itself
            print(e)
            raise e
        print("tts get completed")

        try:
            while not self._shutdown.is_set():
                try:
                    chunk = await asyncio.wait_for(self.inbound.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                self.audio_buffer.extend(chunk)
                if len(self.audio_buffer) < self.MIN_SAMPLES * 2:
                    continue

                pcm16 = np.frombuffer(bytes(self.audio_buffer), dtype=np.int16).astype(
                    np.float32
                )
                self.audio_buffer.clear()
                pcm = pcm16 / 32768.0

                user_text = await stt.transcribe(pcm)
                if not user_text:
                    continue

                await self.transcript.append("user", user_text)
                editor_snapshot = await editor_state_store.snapshot()
                history = await self.transcript.history()
                reply = await llm.generate_reply(history, editor_snapshot)
                await self.transcript.append("interviewer", reply)

                audio_bytes = await tts.synthesize(reply)
                payload = {
                    "type": "interviewer_reply",
                    "text": reply,
                    "audio_base64": (
                        base64.b64encode(audio_bytes).decode("ascii")
                        if audio_bytes
                        else None
                    ),
                }
                await self.outbound.put(payload)
        except Exception:
            print("voice_ws inference loop crashed")
        finally:
            self._shutdown.set()


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------


@web_app.websocket("/voice_ws")
async def voice_ws(websocket: WebSocket):
    session = VoiceWebSocketSession(websocket)
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
    gpu="H200",
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app


if __name__ == "__main__":
    app.deploy("webapp")
