# ---
# cmd: ["modal", "serve", "07_web_endpoints/fastapi_app.py"]
# ---

# # Deploy FastAPI app with Modal

# This example shows how you can deploy a [FastAPI](https://fastapi.tiangolo.com/) app with Modal.
# You can serve any app written in an ASGI-compatible web framework (like FastAPI) using this pattern or you can server WSGI-compatible frameworks like Flask with [`wsgi_app`](https://modal.com/docs/guide/webhooks#wsgi).

from typing import List, Optional

from fastapi.staticfiles import StaticFiles
import modal
from fastapi import FastAPI, Header, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "pydantic", "python-multipart")
app = modal.App("example-fastapi-app", image=image)
web_app = FastAPI()

# Allow requests from your frontend (e.g., http://localhost:3000)
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, "*" is fine. For prod, lock this down.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    name: str

class PlanSection(BaseModel):
    id: str
    title: str
    description: str

@web_app.get("/")
async def handle_root(user_agent: Optional[str] = Header(None)):
    print(f"GET /     - received user_agent={user_agent}")
    return "Hello World"


@web_app.post("/foo")
async def handle_foo(item: Item, user_agent: Optional[str] = Header(None)):
    print(f"POST /foo - received user_agent={user_agent}, item.name={item.name}")
    return item

@web_app.post("/generate_plan", response_model=List[PlanSection])
async def generate_plan(
    resume: UploadFile = File(..., description="Candidate resume PDF"),
    job_description: str = Form(..., description="Job description"),
    feedback: Optional[str] = Form(None, description="Optional feedback for revision"),
):
    # Validate content type
    if resume.content_type != ("application/pdf"):
        raise HTTPException(status_code=400, detail="Resume must be a PDF file")
    content = await resume.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    print(f"File: {resume.filename}, JD: {job_description[:50]}..., Feedback: {feedback}")

    # For now, don’t parse or forward the file.
    # Simply respond with a mock plan that matches the frontend's expected format.
    
    # Base plan
    plan = [
        PlanSection(
            id="plan-1",
            title="Introduction & Resume Deep Dive",
            description="We'll start with a brief intro and then dig into your resume, focusing on projects relevant to the job description."
        ),
        PlanSection(
            id="plan-2",
            title="Technical Challenge (Coding)",
            description="A practical coding problem to assess your problem-solving process, coding style, and testing approach."
        ),
        PlanSection(
            id="plan-3",
            title="Candidate Q&A and Wrap-up",
            description="Your turn to ask questions about the team, role, and Modal. We'll then discuss next steps."
        ),
    ]

    # If feedback was provided, modify the plan
    if feedback:
        plan.insert(0, PlanSection(
            id="plan-0-feedback",
            title="Revised: Feedback Incorporation",
            description=f"Adjusting plan based on your request: '{feedback}'"
        ))

    return plan

@app.function()
@modal.asgi_app()
def fastapi_app():
    web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app


# if __name__ == "__main__":
#     app.deploy("webapp")