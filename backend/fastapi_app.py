# ---
# cmd: ["modal", "serve", "07_web_endpoints/fastapi_app.py"]
# ---

from typing import List, Optional
import base64
import os
from fastapi.staticfiles import StaticFiles
import httpx
import json  # <-- 1. ADDED for safe JSON parsing
import modal
from fastapi import FastAPI, Header, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# -------------------------------
# Modal setup
# -------------------------------
image = modal.Image.debian_slim().pip_install("fastapi[standard]")
static_path = Path(__file__).with_name("static_frontend").resolve()
image = image.add_local_dir(static_path, "/assets")

app = modal.App("Syntherview", image=image, secrets=[modal.Secret.from_name("openrouter-secret")])
web_app = FastAPI()

# web_app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, restrict this to your frontend's domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# -------------------------------
# Models
# -------------------------------
class Item(BaseModel):
    name: str

class PlanSection(BaseModel):
    id: str
    title: str
    description: str

class InterviewContext(BaseModel):
    plan: List[PlanSection]
    coding_question: str
    
# -------------------------------
# Routes# -------------------------------
@web_app.post("/generate_plan", response_model=InterviewContext)
async def generate_plan(
    resume: UploadFile = File(..., description="Candidate resume PDF"),
    job_description: str = Form(..., description="Job description"),
    prompt: Optional[str] = Form(
        "You are an expert technical interviewer. Review the resume and job description, "
        "then produce a structured interview plan with tips and one coding question."
    ),
):
    if resume.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Resume must be a PDF file")

    file_bytes = await resume.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Encode PDF to base64
    resume_b64 = base64.b64encode(file_bytes).decode("utf-8")

    # -------------------------------
    # Call OpenRouter API
    # -------------------------------
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OpenRouter API key")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://nova-hackathon-cmu.vercel.app", # <-- TODO: Change to your app's URL
        "X-Title": "Nova Interview Planner", # <-- TODO: Change to your app's name
    }

    # 2. UPDATED system prompt to ask for a valid JSON object
    system_prompt = (
        "You are an expert AI interviewer. Your task is to generate a JSON response. "
        "Your response must be a single JSON object with two top-level keys: 'plan' and 'coding_question'. "
        
        "1. The 'plan' key: Its value must be a list of 3-4 interview sections. Each section in the list "
        "must be an object with an 'id' (string, e.g., 'intro'), 'title' (string), and 'description' (string). "
        
        "2. The 'coding_question' key: Its value must be a string containing a simple, "
        "role-relevant Python coding question. Format this question as Python comments, "
        "including a function prototype. "
        "Example: '# Write a function to reverse a string.\\ndef reverse_string(s):\\n  # Your code here\\n  pass\\n' "
        
        "Base your plan and question on the user's prompt, the job description, and the candidate's resume (PDF). "
        
        "**Crucially, your entire response must be ONLY the JSON object, starting with `{` "
        "and ending with `}`. Do not include any other text, preamble, conversational "
        "phrasing, or markdown backticks.**"
    )
    # 3. UPDATED messages to use multimodal input
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"User Prompt: {prompt}\n\nJob Description:\n{job_description}"
                },
                {
                    "type": "image_url",
                    # This tells the model to treat the base64 string as a PDF
                    "image_url": {
                        "url": f"data:application/pdf;base64,{resume_b64}"
                    }
                }
            ],
        },
    ]

    payload = {
        # 4. UPDATED model to one that explicitly handles PDF media type
        "model": "anthropic/claude-3-haiku",
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 2048,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()  # Raise an exception for 4xx/5xx responses
        
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"HTTP request error: {e}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"OpenRouter error: {e.response.text}")

    result = response.json()
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    
    # 5. UPDATED parsing to be safe
    try:
        # ... (finding json_string is the same) ...
        start_index = content.find('{')
        end_index = content.rfind('}')
        
        if start_index == -1 or end_index == -1 or end_index < start_index:
            raise ValueError("Could not find valid JSON object delimiters { and } in response.")
        
        json_string = content[start_index : end_index + 1]
        data = json.loads(json_string)
        
        # Extract the list from the 'plan' key
        sections_data = data.get("plan", [])
        
        # --- 3. EXTRACT THE CODING QUESTION ---
        coding_question_str = data.get("coding_question", "# No coding question was generated.")
        # --- END ADD ---
        
        if not isinstance(sections_data, list):
            raise ValueError("Model did not return a list under the 'plan' key")

        plan = [PlanSection(**s) for s in sections_data]
        
        if not plan:
             raise ValueError("Model returned an empty plan")
        
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from model. Raw content (after extraction): {json_string}")
        raise HTTPException(status_code=500, detail="Invalid JSON structure returned from model.")
    except Exception as e:
        print(f"Failed to parse model response. Error: {e}. Raw content: {content}")
        raise HTTPException(status_code=500, detail=f"Invalid data returned from model: {e}")

    # --- 4. RETURN THE NEW OBJECT ---
    return InterviewContext(plan=plan, coding_question=coding_question_str)

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


# if __name__ == "__main__":
#     app.deploy("webapp")