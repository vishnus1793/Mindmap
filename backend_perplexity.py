# backend_perplexity.py
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import uvicorn

# Load API key (set in your shell: export PERPLEXITY_API_KEY="your_key_here")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not PERPLEXITY_API_KEY:
    raise ValueError("❌ Missing PERPLEXITY_API_KEY environment variable.")

# FastAPI app
app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to ["http://localhost:5173"] for Vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class PromptRequest(BaseModel):
    prompt: str

# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "message": "🚀 Backend running on port 9000"}

# Main route for generating mindmap text
@app.post("/api/mindmap/generate")
async def generate_mindmap(request: PromptRequest):
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",  # adjust model name if needed
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": request.prompt},
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        return {"response": data}
    except Exception as e:
        return {"error": str(e)}

# Run the server
if __name__ == "__main__":
    print("🚀 Starting backend on http://127.0.0.1:9000")
    uvicorn.run("backend_perplexity:app", host="0.0.0.0", port=9000, reload=True)
