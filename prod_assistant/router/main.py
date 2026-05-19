# router/main.py
# FastAPI web application that serves the chat UI and handles user messages.
# The /get endpoint creates an AgenticRAG instance and runs the full
# agentic workflow (retrieval + grading + generation) for each user message.

import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google.api_core.exceptions import ResourceExhausted
from groq import RateLimitError as GroqRateLimitError
# from prod_assistant.workflow.agentic_rag_workflow import AgenticRAG
from prod_assistant.workflow.normal_generation_workflow import AgenticRAG

RATE_LIMIT_EXCEPTIONS = (ResourceExhausted, GroqRateLimitError)


def _is_rate_limit(exc: BaseException) -> bool:
    """True if exc (or any wrapped cause/context) is a provider rate-limit error."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, RATE_LIMIT_EXCEPTIONS):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False

# Create the FastAPI application instance
app = FastAPI()

# Serve CSS/JS from the static/ directory (e.g. style.css)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load HTML templates from the templates/ directory (e.g. index.html)
templates = Jinja2Templates(directory="templates")

# Enable CORS for all origins (allows the frontend to call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- FastAPI Endpoints ----------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the chat UI homepage (index.html)."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get")
async def chat(msg: str = Form(...)):
    """Handle a chat message from the user.
    Creates a new AgenticRAG instance, runs the full workflow,
    and returns the assistant's response as a plain string."""
    try:
        rag_agent = AgenticRAG()
        answer = rag_agent.run(msg)
        return answer
    except Exception as e:
        if _is_rate_limit(e):
            return "I'm currently rate-limited by the AI provider. Please wait a moment and try again."
        return f"Sorry, something went wrong: {e}"