# server.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class CodeRequest(BaseModel):
    code: str
    language: str = "text"

# Response model
class QuestionsResponse(BaseModel):
    questions: List[str]

# Dummy question generation function
def generate_questions_from_code(code: str) -> List[str]:
    """
    Simple example function that generates questions.
    Replace with actual LLM / logic for production.
    """
    lines = code.split("\n")
    questions = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            questions.append(f"What does line {i+1} do: '{line}'?")
    return questions

@app.post("/api/code/questions", response_model=QuestionsResponse)
async def generate_questions(request: CodeRequest):
    code = request.code
    questions = generate_questions_from_code(code)
    return {"questions": questions}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
