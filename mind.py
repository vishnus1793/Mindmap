import os
import socket
import json
import uuid
import logging
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from pydantic import BaseModel

# --- Algorithmic Summarizers ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
scraped_data_dir = "scraped_data"
os.makedirs(scraped_data_dir, exist_ok=True)

# ---------- Utils ----------
def get_free_port(start_port=8000):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("0.0.0.0", port)) != 0:
                return port
            port += 1

def save_json(job_id, url, content, summary):
    safe_name = url.replace("://", "_").replace("/", "_")
    path = os.path.join(scraped_data_dir, f"{safe_name}.json")
    with open(path, "w") as f:
        json.dump({"id": job_id, "url": url, "content": content, "summary": summary}, f, indent=2)
    return path

# ---------- Scraper ----------
def scrape_website(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content tags
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.extract()

        text = " ".join(soup.stripped_strings)
        return text[:20000]  # keep first 20k chars for performance
    except Exception as e:
        logger.error(f"❌ Error scraping {url}: {e}")
        return ""

# ---------- Summarization ----------
def algorithmic_summary(text, sentences=5, method="lsa"):
    """Generate summary using LSA or TextRank"""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    if method == "lsa":
        summarizer = LsaSummarizer()
    else:
        summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences)
    return " ".join([str(s) for s in summary])

def hybrid_summary(text):
    """Combine LSA + TextRank for robustness"""
    lsa_summary = algorithmic_summary(text, method="lsa")
    textrank_summary = algorithmic_summary(text, method="textrank")
    return f"LSA: {lsa_summary}\n\nTextRank: {textrank_summary}"

# ---------- API Models ----------
class UpdateCellRequest(BaseModel):
    id: str
    content: str

cells_store = {}  # in-memory store

# ---------- Routes ----------
@app.post("/scrape")
async def scrape(request: Request):
    data = await request.json()
    url = data.get("url")
    job_id = str(uuid.uuid4())
    logger.info(f"🔎 Scraping {url} with job {job_id}")

    scraped_text = scrape_website(url)
    if not scraped_text:
        return {"error": f"Could not scrape {url}"}

    summary = hybrid_summary(scraped_text)
    save_json(job_id, url, scraped_text, summary)

    return {"job_id": job_id, "summary": summary}

@app.get("/get-text-cell")
async def get_text_cell(id: str):
    content = cells_store.get(id, "")
    return {"id": id, "content": content}

@app.post("/update-text-cell")
async def update_text_cell(req: UpdateCellRequest):
    cells_store[req.id] = req.content
    return {"status": "ok", "id": req.id, "content": req.content}

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    port = get_free_port(8000)
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run("mindmap:app", host="0.0.0.0", port=port, reload=True)
