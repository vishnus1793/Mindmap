#!/usr/bin/env python3
"""
Website Content Scraper and Mind Map Generator API Backend

FastAPI backend that provides endpoints for scraping website content,
generating summaries, and creating mind maps without external AI APIs.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Dict, List, Optional, Set
import requests
from bs4 import BeautifulSoup
import re
import json
import os
from dataclasses import dataclass, asdict
import uuid
import asyncio
from datetime import datetime
import logging
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Website Scraper & Mind Map API",
    description="API for scraping website content and generating mind maps",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo purposes (use a database in production)
scrape_jobs = {}

# Pydantic models for API requests/responses
class ScrapeRequest(BaseModel):
    url: HttpUrl
    summary_length: Optional[int] = 300

class ScrapeResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

class ContentSection(BaseModel):
    title: str
    content: str
    keywords: List[str] = []

class ScrapedContent(BaseModel):
    url: str
    title: str
    sections: List[ContentSection]
    full_text: str
    word_count: int

class SummaryData(BaseModel):
    summary: str
    key_concepts: List[str]
    method: str

class MindMapData(BaseModel):
    visual: str
    network: str
    hierarchical: str

class CompleteResult(BaseModel):
    scraped_content: ScrapedContent
    summary: SummaryData
    mind_maps: MindMapData

@dataclass
class JobData:
    job_id: str
    status: str
    progress: int
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None

class WebScraper:
    """Handles web scraping functionality"""
    
    def __init__(self, headers=None):
        self.session = requests.Session()
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session.headers.update(self.headers)
    
    def scrape_content(self, url: str) -> Dict[str, any]:
        """Scrape content from the specified URL"""
        try:
            response = self.session.get(str(url), timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No Title"
            
            # Extract main content
            content_selectors = [
                'main', 'article', '[role="main"]', '.content', 
                '.main-content', '#content', '.post-content'
            ]
            
            main_content = None
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.find('body')
            
            # Extract headings and paragraphs
            sections = []
            if main_content:
                current_section = {"title": title_text, "content": "", "keywords": []}
                
                for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
                    if element.name.startswith('h'):
                        if current_section["content"].strip():
                            sections.append(current_section)
                        current_section = {
                            "title": element.get_text().strip(),
                            "content": "",
                            "keywords": []
                        }
                    elif element.name == 'p':
                        text = element.get_text().strip()
                        if text:
                            current_section["content"] += text + " "
                
                if current_section["content"].strip():
                    sections.append(current_section)
            
            # Extract all text as fallback
            all_text = soup.get_text()
            clean_text = re.sub(r'\s+', ' ', all_text).strip()
            
            return {
                'url': str(url),
                'title': title_text,
                'sections': sections,
                'full_text': clean_text,
                'word_count': len(clean_text.split())
            }
            
        except requests.RequestException as e:
            raise Exception(f"Error scraping {url}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing content from {url}: {str(e)}")

class ContentSummarizer:
    """Handles content summarization using NLP techniques"""
    
    def __init__(self):
        """Initialize the summarizer with NLTK components"""
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Fallback if NLTK data not available
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
                'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 
                'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
                'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
                'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 
                'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 
                'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                'under', 'again', 'further', 'then', 'once'
            }
    
    def summarize_content(self, content: Dict[str, any], max_length: int = 500) -> Dict[str, any]:
        """Generate a summary of the scraped content using extractive methods"""
        full_text = content['full_text']
        title = content['title']
        
        # Extract key sentences for summary
        summary = self._extractive_summary(content, max_length)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(full_text, title)
        
        return {
            'summary': summary,
            'key_concepts': key_concepts,
            'method': 'nltk_extractive'
        }
    
    def _extractive_summary(self, content: Dict[str, any], max_length: int) -> str:
        """Create extractive summary using sentence scoring"""
        full_text = content['full_text']
        title = content['title']
        
        if not full_text or len(full_text.split()) < 10:
            return "Content too short to summarize effectively."
        
        # Tokenize into sentences
        try:
            sentences = sent_tokenize(full_text)
        except:
            # Fallback sentence splitting
            sentences = re.split(r'[.!?]+', full_text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 2:
            return full_text[:max_length * 5] + "..." if len(full_text) > max_length * 5 else full_text
        
        # Score sentences
        sentence_scores = {}
        
        # Word frequency scoring
        try:
            words = word_tokenize(full_text.lower())
        except:
            words = re.findall(r'\b\w+\b', full_text.lower())
        
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        word_freq = Counter(words)
        
        # Title word boost
        try:
            title_words = word_tokenize(title.lower())
        except:
            title_words = re.findall(r'\b\w+\b', title.lower())
        
        title_words = [word for word in title_words if word not in self.stop_words]
        
        for sentence in sentences:
            if len(sentence.split()) < 5:  # Skip very short sentences
                continue
                
            score = 0
            try:
                sentence_words = word_tokenize(sentence.lower())
            except:
                sentence_words = re.findall(r'\b\w+\b', sentence.lower())
            
            sentence_words = [word for word in sentence_words if word not in self.stop_words]
            
            # Frequency-based scoring
            for word in sentence_words:
                if word in word_freq:
                    score += word_freq[word]
            
            # Title relevance boost
            for word in sentence_words:
                if word in title_words:
                    score += 10
            
            # Position boost (first few sentences often important)
            sentence_index = sentences.index(sentence)
            if sentence_index < 3:
                score += 5
            
            # Length normalization
            sentence_scores[sentence] = score / len(sentence_words) if sentence_words else 0
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build summary maintaining original order
        summary_sentences = []
        summary_length = 0
        
        for sentence, score in top_sentences:
            sentence_words = len(sentence.split())
            if summary_length + sentence_words <= max_length:
                summary_sentences.append((sentence, sentences.index(sentence)))
                summary_length += sentence_words
            
            if summary_length >= max_length * 0.8:  # Allow some flexibility
                break
        
        # Sort by original order
        summary_sentences.sort(key=lambda x: x[1])
        summary = ' '.join([sentence for sentence, _ in summary_sentences])
        
        return summary if summary else sentences[0][:max_length * 5] + "..."
    
    def _extract_key_concepts(self, text: str, title: str) -> List[str]:
        """Extract key concepts using NLP techniques"""
        key_concepts = set()
        
        # Extract from title
        try:
            title_words = word_tokenize(title)
            title_pos = pos_tag(title_words)
        except:
            title_words = re.findall(r'\b[A-Z][a-zA-Z]+\b', title)
            title_pos = [(word, 'NN') for word in title_words]
        
        for word, pos in title_pos:
            if pos.startswith('NN') and len(word) > 2 and word.lower() not in self.stop_words:
                key_concepts.add(word.title())
        
        # Extract named entities and important nouns
        try:
            words = word_tokenize(text)
            pos_tags = pos_tag(words)
        except:
            # Fallback: extract capitalized words and common technical terms
            words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
            pos_tags = [(word, 'NN') for word in words]
        
        # Extract proper nouns and regular nouns
        for word, pos in pos_tags:
            if pos in ['NNP', 'NNPS', 'NN', 'NNS'] and len(word) > 2:
                if word.lower() not in self.stop_words:
                    key_concepts.add(word.title())
        
        # Extract technical terms and domain-specific words
        tech_patterns = [
            r'\b(?:API|SDK|framework|platform|service|application|system|database|server|network|cloud|data|security|algorithm|software|technology|development|infrastructure|architecture|protocol|interface|module|component|library|tool|solution)\b',
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w*(?:ing|tion|ment|ness|ship|able|ible)\b'  # Common suffixes
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 2 and match.lower() not in self.stop_words:
                    key_concepts.add(match.title())
        
        # Filter and rank concepts
        concept_freq = Counter()
        text_lower = text.lower()
        
        for concept in key_concepts:
            concept_freq[concept] = text_lower.count(concept.lower())
        
        # Return top concepts
        top_concepts = [concept for concept, freq in concept_freq.most_common(15)]
        return top_concepts[:12]

class MindMapGenerator:
    """Generates text-based mind maps from content summaries"""
    
    def create_mind_maps(self, title: str, summary: str, key_concepts: List[str]) -> Dict[str, str]:
        """Create all three types of mind maps and return as dictionary"""
        return {
            'visual': self._generate_text_mindmap(title, summary, key_concepts),
            'network': self._create_network_mind_map(title, key_concepts),
            'hierarchical': self._create_hierarchical_mindmap(title, key_concepts)
        }
    
    def _generate_text_mindmap(self, title: str, summary: str, key_concepts: List[str]) -> str:
        """Generate a text-based mind map representation"""
        lines = []
        
        title_box = f"╔{'═' * (len(title) + 4)}╗"
        title_content = f"║  {title.upper()}  ║"
        title_bottom = f"╚{'═' * (len(title) + 4)}╝"
        
        lines.extend([
            "",
            " " * (40 - len(title)//2) + title_box,
            " " * (40 - len(title)//2) + title_content,
            " " * (40 - len(title)//2) + title_bottom,
            ""
        ])
        
        if key_concepts:
            lines.append(" " * 40 + "│")
            lines.append(" " * 35 + "┌─────┴─────┐")
            lines.append(" " * 35 + "│           │")
            
            left_concepts = key_concepts[:len(key_concepts)//2]
            right_concepts = key_concepts[len(key_concepts)//2:]
            
            max_rows = max(len(left_concepts), len(right_concepts))
            
            for i in range(max_rows):
                left_text = ""
                right_text = ""
                
                if i < len(left_concepts):
                    concept = left_concepts[i]
                    left_text = f"├── 📌 {concept}"
                
                if i < len(right_concepts):
                    concept = right_concepts[i]
                    right_text = f"📌 {concept} ──┤"
                
                if left_text and right_text:
                    line = f"{left_text:<35}│{right_text:>35}"
                elif left_text:
                    line = f"{left_text:<35}│"
                elif right_text:
                    line = f"{'':<35}│{right_text:>35}"
                else:
                    line = f"{'':<35}│"
                
                lines.append(line)
        
        lines.extend([
            "",
            "┌" + "─" * 78 + "┐",
            "│" + " SUMMARY ".center(78) + "│",
            "├" + "─" * 78 + "┤"
        ])
        
        summary_words = summary.split()
        current_line = "│ "
        
        for word in summary_words:
            if len(current_line + word + " ") > 77:
                lines.append(current_line + " " * (78 - len(current_line)) + "│")
                current_line = "│ " + word + " "
            else:
                current_line += word + " "
        
        if current_line.strip() != "│":
            lines.append(current_line + " " * (78 - len(current_line)) + "│")
        
        lines.append("└" + "─" * 78 + "┘")
        
        return "\n".join(lines)
    
    def _create_network_mind_map(self, title: str, key_concepts: List[str]) -> str:
        """Create a network-style text mind map"""
        lines = []
        
        lines.extend([
            "",
            "╔" + "═" * 60 + "╗",
            f"║{'NETWORK MIND MAP'.center(60)}║",
            "╠" + "═" * 60 + "╣",
            f"║{title.center(60)}║",
            "╚" + "═" * 60 + "╝",
            ""
        ])
        
        lines.append("        🔵 " + title)
        lines.append("        │")
        lines.append("   ┌────┴────┐")
        
        for i, concept in enumerate(key_concepts[:8]):
            if i % 2 == 0:
                lines.append(f"   │         │")
                branch_line = f"   ├─ 🔸 {concept}"
            else:
                branch_line = f"   │         └─ 🔸 {concept}"
            
            lines.append(branch_line)
        
        if len(key_concepts) > 8:
            lines.append("   │")
            lines.append("   └─ 🔸 ... and more concepts")
        
        return "\n".join(lines)
    
    def _create_hierarchical_mindmap(self, title: str, key_concepts: List[str]) -> str:
        """Create a hierarchical tree-style mind map"""
        lines = []
        
        lines.extend([
            "",
            f"🌳 {title.upper()}",
            "│"
        ])
        
        for i, concept in enumerate(key_concepts):
            is_last = (i == len(key_concepts) - 1)
            
            if is_last:
                lines.append(f"└── 🌿 {concept}")
            else:
                lines.append(f"├── 🌿 {concept}")
        
        return "\n".join(lines)

# Ensure a folder exists to store JSON files
STORAGE_DIR = "scraped_data"
os.makedirs(STORAGE_DIR, exist_ok=True)

def sanitize_filename(url: str) -> str:
    """Convert URL into a safe filename"""
    # Replace all non-alphanumeric characters with underscore
    return re.sub(r'[^0-9a-zA-Z]+', '_', url)

async def process_scrape_job(job_id: str, url: str, summary_length: int):
    """Background task to process scraping job and save result locally"""
    try:
        job = scrape_jobs[job_id]
        job.status = "processing"
        job.progress = 10
        
        # Initialize components
        scraper = WebScraper()
        summarizer = ContentSummarizer()
        mind_map_gen = MindMapGenerator()
        
        # Scrape content
        logger.info(f"Scraping content from: {url}")
        job.progress = 30
        content = scraper.scrape_content(url)
        
        # Generate summary
        logger.info("Generating summary...")
        job.progress = 60
        summary_data = summarizer.summarize_content(content, summary_length)
        
        # Generate mind maps
        logger.info("Creating mind maps...")
        job.progress = 80
        mind_maps = mind_map_gen.create_mind_maps(
            content['title'],
            summary_data['summary'],
            summary_data['key_concepts']
        )
        
        # Prepare final result
        result = {
            'scraped_content': content,
            'summary': summary_data,
            'mind_maps': mind_maps
        }
        
        # Save result to local JSON file using sanitized URL as filename
        safe_filename = sanitize_filename(url)
        file_path = os.path.join(STORAGE_DIR, f"{safe_filename}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        job.result = result
        job.status = "completed"
        job.progress = 100
        job.completed_at = datetime.now().isoformat()
        logger.info(f"Job {job_id} completed successfully and saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.now().isoformat()

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Website Scraper & Mind Map Generator API",
        "version": "1.0.0",
        "endpoints": [
            "POST /scrape - Start scraping job",
            "GET /job/{job_id} - Get job status",
            "GET /jobs - List all jobs",
            "GET /health - Health check"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/scrape", response_model=ScrapeResponse)
async def create_scrape_job(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """Start a new scraping job"""
    job_id = str(uuid.uuid4())
    
    # Create job entry
    job = JobData(
        job_id=job_id,
        status="queued",
        progress=0,
        created_at=datetime.now().isoformat()
    )
    
    scrape_jobs[job_id] = job
    
    # Start background processing
    background_tasks.add_task(
        process_scrape_job, 
        job_id, 
        str(request.url), 
        request.summary_length
    )
    
    return ScrapeResponse(
        job_id=job_id,
        status="queued",
        message="Scraping job started successfully"
    )

@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a specific job"""
    if job_id not in scrape_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = scrape_jobs[job_id]
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at
    )

@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        "total": len(scrape_jobs),
        "jobs": [
            {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "created_at": job.created_at,
                "completed_at": job.completed_at
            }
            for job in scrape_jobs.values()
        ]
    }

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a specific job"""
    if job_id not in scrape_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del scrape_jobs[job_id]
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/job/{job_id}/result")
async def get_job_result(job_id: str):
    """Get the complete result of a completed job"""
    if job_id not in scrape_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = scrape_jobs[job_id]
    
    if job.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed. Current status: {job.status}"
        )
    
    return job.result

@app.get("/job/{job_id}/mindmaps")
async def get_job_mindmaps(job_id: str):
    """Get only the mind maps from a completed job"""
    if job_id not in scrape_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = scrape_jobs[job_id]
    
    if job.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed. Current status: {job.status}"
        )
    
    return job.result.get('mind_maps', {})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)