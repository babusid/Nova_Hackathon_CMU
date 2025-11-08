from __future__ import annotations

import asyncio
import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import modal
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

MODEL_CACHE_PATH = "/model-cache"
model_volume = modal.Volume.from_name("voice-model-cache", create_if_missing=True)
static_path = Path(__file__).with_name("static_frontend").resolve()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "torch",
        "torchaudio",
        "transformers",
        "accelerate",
        "sentencepiece",
        "huggingface_hub",
        "numpy",
        "soundfile",
        "TTS",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": MODEL_CACHE_PATH,
        }
    )
    .add_local_dir(static_path, "/assets")
)

with image.imports():
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from TTS.api import TTS as CoquiTTS

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


Speaker = Literal["user", "interviewer"]


@dataclass
class TranscriptEntry:
    speaker: Speaker
    text: str


class ConversationTranscript:
    """Tracks the rolling transcript as both entries and tagged text."""

    def __init__(self) -> None:
        self._entries: list[TranscriptEntry] = []
        self._log: str = ""
        self._lock = asyncio.Lock()

    async def append(self, speaker: Speaker, text: str) -> None:
        async with self._lock:
            sanitized = text.strip()
            self._entries.append(TranscriptEntry(speaker=speaker, text=sanitized))
            self._log += f"[{speaker}] {sanitized}\n"

    async def latest_user_message(self) -> Optional[str]:
        async with self._lock:
            for entry in reversed(self._entries):
                if entry.speaker == "user":
                    return entry.text
        return None

    async def dump(self) -> str:
        async with self._lock:
            return self._log

    async def to_string(self, num_turns: int | None = None) -> str:
        async with self._lock:
            entries = self._entries[-num_turns:] if num_turns else self._entries
            parts = []
            for entry in entries:
                tag = entry.speaker
                parts.append(f"<{tag}>{entry.text}</{tag}>")
            return " ".join(parts)


STT_MODEL_NAME = "openai/whisper-large-v3"
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
TTS_MODEL_NAME = "tts_models/en/vctk/vits"
STT_SAMPLE_RATE = 16000


def _pcm16le_to_float32(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    return audio / 32768.0


class SpeechToTextEngine:
    _instance: Optional["SpeechToTextEngine"] = None

    def __init__(self) -> None:
        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=STT_MODEL_NAME,
            device=device,
        )

    @classmethod
    def instance(cls) -> "SpeechToTextEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def transcribe(self, audio_bytes: bytes) -> str:
        waveform = _pcm16le_to_float32(audio_bytes)
        if waveform.size == 0:
            return ""
        inputs = {"array": waveform, "sampling_rate": STT_SAMPLE_RATE}
        result = await asyncio.to_thread(self._pipeline, inputs)
        return (result.get("text") or "").strip()


class ConversationalBrain:
    _instance: Optional["ConversationalBrain"] = None

    def __init__(self) -> None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME, use_fast=True, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    @classmethod
    def instance(cls) -> "ConversationalBrain":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def generate_reply(self, transcript: str, editor_state: str) -> str:
        prompt = build_interviewer_prompt(transcript, editor_state)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        def _generate():
            return self._model.generate(
                **inputs,
                max_new_tokens=220,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        output = await asyncio.to_thread(_generate)
        generated = output[0][inputs["input_ids"].shape[-1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()


class TextToSpeechEngine:
    _instance: Optional["TextToSpeechEngine"] = None

    def __init__(self) -> None:
        gpu = torch.cuda.is_available()
        self._tts = CoquiTTS(
            model_name=TTS_MODEL_NAME,
            progress_bar=False,
            gpu=gpu,
        )

    @classmethod
    def instance(cls) -> "TextToSpeechEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def synthesize(self, text: str) -> bytes:
        stripped = text.strip()
        if not stripped:
            return b""

        def _speak():
            return self._tts.tts(stripped)

        audio = await asyncio.to_thread(_speak)
        sample_rate = getattr(self._tts, "sample_rate", 22050)
        sample_rate = getattr(
            getattr(self._tts, "synthesizer", None), "output_sample_rate", 22050
        )
        buffer = io.BytesIO()
        waveform = np.asarray(audio, dtype=np.float32)
        sf.write(buffer, waveform, samplerate=sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()


def build_interviewer_prompt(transcript: str, editor_state: str) -> str:
    return f"""You are a thoughtful mock-interview coach.
Keep responses concise, inquisitive, and encouraging. Reference the code when helpful.

Conversation so far:
{transcript or "[interviewer] Begin when ready."}

Coderpad snapshot:
{editor_state or "(no code yet)"}

Reply as the interviewer in one short paragraph or two sentences."""


def get_speech_to_text_engine() -> SpeechToTextEngine:
    return SpeechToTextEngine.instance()


def get_conversational_brain() -> ConversationalBrain:
    return ConversationalBrain.instance()


def get_text_to_speech_engine() -> TextToSpeechEngine:
    return TextToSpeechEngine.instance()


class VoiceWebSocketSession:

    def __init__(
        self, websocket: WebSocket, transcript: ConversationTranscript
    ) -> None:
        self.websocket: WebSocket = websocket
        self.inbound: asyncio.Queue = asyncio.Queue()
        self.outbound: asyncio.Queue = asyncio.Queue()
        self._shutdown: asyncio.Event = asyncio.Event()
        self.transcript = transcript

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
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is None:
                    print("voice_ws: received non-binary frame; ignoring.")
                    continue
                await self.inbound.put(data)
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
        stt = get_speech_to_text_engine()
        llm = get_conversational_brain()
        tts = get_text_to_speech_engine()

        try:
            while not self._shutdown.is_set():
                try:
                    audio_chunk = await asyncio.wait_for(
                        self.inbound.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                user_text = await stt.transcribe(audio_chunk)
                if not user_text:
                    continue

                await self.transcript.append("user", user_text)
                editor_snapshot = await editor_state_store.snapshot()
                transcript_window = await self.transcript.to_string(num_turns=20)
                interviewer_text = await llm.generate_reply(
                    transcript_window, editor_snapshot
                )
                await self.transcript.append("interviewer", interviewer_text)

                audio_bytes = await tts.synthesize(interviewer_text)

                await self.outbound.put(
                    {
                        "type": "interviewer_reply",
                        "text": interviewer_text,
                        "audio_base64": (
                            base64.b64encode(audio_bytes).decode("ascii")
                            if audio_bytes
                            else None
                        ),
                    }
                )
        finally:
            self._shutdown.set()

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
    transcript = ConversationTranscript()
    session = VoiceWebSocketSession(websocket=websocket, transcript=transcript)
    await session.run()


@web_app.post("/editor-state")
async def update_editor_state(payload: dict):
    """Overwrites the backend's view of the coderpad/editor contents."""
    document = payload.get("document")
    if document is None:
        raise HTTPException(status_code=400, detail="Missing 'document' field")
    await editor_state_store.update(document)
    return {"status": "updated"}


@app.function(image=image, volumes={MODEL_CACHE_PATH: model_volume})
@modal.asgi_app()
def fastapi_app():
    web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app

if __name__ == "__main__":
    app.deploy("webapp")
