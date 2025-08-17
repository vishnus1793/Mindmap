#!/usr/bin/env python3
"""
Enhanced Website Content Scraper and Interactive Mind Map Generator API
FastAPI backend that provides endpoints for scraping website content,
generating summaries, and creating interactive mind maps with expandable subtopics.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Dict, List, Optional, Any
import requests
from bs4 import BeautifulSoup
import re
import json
import os
from dataclasses import dataclass
import uuid
from datetime import datetime
import logging
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Interactive Website Scraper & Mind Map API",
    description="API for scraping website content and generating interactive mind maps",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
scrape_jobs: Dict[str, "JobData"] = {}
mindmap_nodes: Dict[str, "InteractiveMindMap"] = {}  # Store expandable node data

# Ensure storage directory exists
STORAGE_DIR = "scraped_data"
os.makedirs(STORAGE_DIR, exist_ok=True)

# --------- Pydantic models ---------
class ScrapeRequest(BaseModel):
    url: HttpUrl
    summary_length: Optional[int] = 300
    extract_links: Optional[bool] = True
    max_depth: Optional[int] = 2

class MindMapNode(BaseModel):
    id: str
    title: str
    content: str
    url: Optional[str] = None
    parent_id: Optional[str] = None
    children: List[str] = []
    expanded: bool = False
    node_type: str = "concept"  # concept, link, section, keyword, root, link_category, related, subsection
    metadata: Dict[str, Any] = {}

class InteractiveMindMap(BaseModel):
    root_node: MindMapNode
    nodes: Dict[str, MindMapNode]
    visualization_data: Dict[str, Any]
    style_config: Dict[str, Any]

class ExpandNodeRequest(BaseModel):
    node_id: str
    expand_type: str = "related"  # related, links, subsections

class NodeExpansionResult(BaseModel):
    node_id: str
    new_nodes: List[MindMapNode]
    updated_visualization: Dict[str, Any]

class ScrapeResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

# --------- Dataclass for job tracking ---------
@dataclass
class JobData:
    job_id: str
    status: str
    progress: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None

# --------- Utils ---------
def sanitize_filename(url: str) -> str:
    """Convert URL into a safe filename"""
    return re.sub(r'[^0-9a-zA-Z]+', '_', url)

def strip_metadata_lines(text: str) -> str:
    """
    Remove lines like "**Source:** http...", "Source: http...", "**Word Count:** 2612", etc.
    Case-insensitive, tolerant of leading/trailing spaces and Markdown bold markers.
    """
    if not text:
        return text
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        l = line.strip()
        # Normalize markdown bold markers
        l_norm = re.sub(r'^\*+\s*', '', re.sub(r'\s*\*+$', '', l))
        if re.match(r'^(source|word\s*count)\s*[:：]', l_norm, flags=re.IGNORECASE):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def _is_noise_token(token: str) -> bool:
    """Heuristic to filter out noise tokens from concepts."""
    if not token:
        return True
    t = token.strip()
    if len(t) <= 2:  # too short like 'an', 'of', 'in'
        return True
    if all(ch in "-–—•·•.,:;!?'\"()[]{}|/\\&*" for ch in t):  # punctuation-like
        return True
    if t.isnumeric():
        return True
    junk = {"et", "al", "fig", "tbl", "www", "http", "https"}
    if t.lower() in junk:
        return True
    if re.fullmatch(r"[^\w]*", t):
        return True
    return False

# --------- Scraper ---------
class EnhancedWebScraper:
    """Enhanced web scraper with link extraction and content analysis"""
    def __init__(self, headers=None):
        self.session = requests.Session()
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session.headers.update(self.headers)

    def scrape_content(self, url: str, extract_links: bool = True) -> Dict[str, Any]:
        """Enhanced scraping with link extraction and content structure analysis"""
        try:
            response = self.session.get(str(url), timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            base_domain = urlparse(url).netloc

            # Remove script and style elements and common non-content areas
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

            # Extract structured sections with metadata
            sections: List[Dict[str, Any]] = []
            links: List[Dict[str, Any]] = []
            images: List[Dict[str, Any]] = []

            if main_content:
                current_section: Dict[str, Any] = {
                    "title": title_text,
                    "content": "",
                    "keywords": [],
                    "subsections": [],
                    "links": [],
                    "level": 0,
                }

                for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'a', 'img']):
                    if element.name and element.name.startswith('h'):
                        if current_section["content"].strip():
                            # sanitize before appending
                            current_section["content"] = strip_metadata_lines(current_section["content"])
                            sections.append(current_section)
                        level = int(element.name[1])
                        current_section = {
                            "title": element.get_text().strip(),
                            "content": "",
                            "keywords": [],
                            "subsections": [],
                            "links": [],
                            "level": level,
                        }
                    elif element.name == 'p':
                        text = element.get_text().strip()
                        if text:
                            current_section["content"] += text + " "
                    elif element.name in ['ul', 'ol']:
                        list_items = [li.get_text().strip() for li in element.find_all('li')]
                        current_section["subsections"].extend(list_items)
                    elif element.name == 'a' and extract_links:
                        href = element.get('href')
                        if href:
                            link_url = urljoin(str(url), href)
                            link_text = element.get_text().strip()
                            if link_text and self._is_valid_link(link_url, base_domain):
                                link_data = {
                                    "url": link_url,
                                    "text": link_text,
                                    "type": self._categorize_link(link_url, link_text),
                                }
                                links.append(link_data)
                                current_section["links"].append(link_data)
                    elif element.name == 'img':
                        img_src = element.get('src')
                        img_alt = element.get('alt', '')
                        if img_src:
                            img_url = urljoin(str(url), img_src)
                            images.append({
                                "url": img_url,
                                "alt": img_alt,
                                "caption": img_alt
                            })

                if current_section["content"].strip():
                    current_section["content"] = strip_metadata_lines(current_section["content"])
                    sections.append(current_section)

            all_text = soup.get_text()
            clean_text = re.sub(r'\s+', ' ', all_text).strip()
            clean_text = strip_metadata_lines(clean_text)

            return {
                'url': str(url),
                'title': title_text,
                'sections': sections,
                'full_text': clean_text,
                'word_count': len(clean_text.split()),
                'links': links,
                'images': images,
                'structure': self._analyze_content_structure(sections),
            }

        except requests.RequestException as e:
            raise Exception(f"Error scraping {url}: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing content from {url}: {str(e)}")

    def _is_valid_link(self, url: str, base_domain: str) -> bool:
        """Check if link is valid and relevant"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            if url.startswith('#') or url.startswith('javascript:'):
                return False
            return True
        except Exception:
            return False

    def _categorize_link(self, url: str, text: str) -> str:
        """Categorize links by type"""
        url_lower = url.lower()
        text_lower = text.lower()
        if any(ext in url_lower for ext in ['.pdf', '.doc', '.docx', '.ppt', '.xls', '.xlsx']):
            return 'document'
        elif any(domain in url_lower for domain in ['github.com', 'gitlab.com', 'bitbucket.org', 'bitbucket.com']):
            return 'code'
        elif any(domain in url_lower for domain in ['youtube.com', 'youtu.be', 'vimeo.com', 'twitch.tv']):
            return 'video'
        elif any(word in text_lower for word in ['documentation', 'docs', 'api', 'reference']):
            return 'documentation'
        elif any(word in text_lower for word in ['tutorial', 'guide', 'how-to', 'example']):
            return 'tutorial'
        else:
            return 'general'

    def _analyze_content_structure(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the hierarchical structure of content"""
        structure: Dict[str, Any] = {
            'total_sections': len(sections),
            'hierarchy': defaultdict(int),
            'topics': [],
            'relationships': [],
        }
        for section in sections:
            level = section.get('level', 0)
            structure['hierarchy'][f'h{level}'] += 1
            if section.get('title') and section['title'] != "No Title":
                topic = {
                    'title': section['title'],
                    'level': level,
                    'content_length': len(section.get('content', "")),
                    'has_subsections': len(section.get('subsections', [])) > 0,
                    'link_count': len(section.get('links', [])),
                }
                structure['topics'].append(topic)
        structure['hierarchy'] = dict(structure['hierarchy'])
        return structure

# --------- Mind map generator ---------
class InteractiveMindMapGenerator:
    """Enhanced mind map generator with interactive features"""
    def __init__(self):
        self.node_counter = 0

    def create_interactive_mindmap(self, content: Dict[str, Any], summary_data: Dict[str, Any]) -> InteractiveMindMap:
        """Create an interactive mind map with expandable nodes"""
        root_node_id = self._generate_node_id()

        # Root node content uses bullet summary; sanitize metadata lines
        summary_bullets = summary_data.get('summary_bullets', [])
        summary_text = "\n".join(f"- {strip_metadata_lines(b)}" for b in summary_bullets) if summary_bullets else strip_metadata_lines(summary_data.get('summary', "") or "")
        root_node = MindMapNode(
            id=root_node_id,
            title=content.get('title', 'Root'),
            content=(summary_text[:200] + "...") if len(summary_text) > 200 else summary_text,
            url=content.get('url'),
            node_type="root",
            metadata={
                'word_count': content.get('word_count', 0),
                'section_count': len(content.get('sections', [])),
                'link_count': len(content.get('links', [])),
                'expandable': False,
            },
        )

        nodes: Dict[str, MindMapNode] = {root_node_id: root_node}

        concept_nodes = self._create_concept_nodes(summary_data.get('key_concepts', []), root_node_id)
        nodes.update(concept_nodes)

        section_nodes = self._create_section_nodes(content.get('sections', []), root_node_id)
        nodes.update(section_nodes)

        link_nodes = self._create_link_nodes(content.get('links', []), root_node_id)
        nodes.update(link_nodes)

        root_node.children = list(concept_nodes.keys()) + list(section_nodes.keys()) + list(link_nodes.keys())

        viz_data = self._generate_visualization_data(nodes, root_node_id)
        style_config = self._get_style_config()

        return InteractiveMindMap(
            root_node=root_node,
            nodes=nodes,
            visualization_data=viz_data,
            style_config=style_config,
        )

    def _generate_node_id(self) -> str:
        self.node_counter += 1
        return f"node_{self.node_counter}_{uuid.uuid4().hex[:8]}"

    def _create_concept_nodes(self, concepts: List[str], parent_id: str) -> Dict[str, MindMapNode]:
        nodes: Dict[str, MindMapNode] = {}
        for concept in concepts[:8]:
            node_id = self._generate_node_id()
            clean_concept = strip_metadata_lines(concept)
            node = MindMapNode(
                id=node_id,
                title=clean_concept,
                content=f"Key concept: {clean_concept}",
                parent_id=parent_id,
                node_type="concept",
                metadata={'expandable': True, 'concept_type': 'key_term'},
            )
            nodes[node_id] = node
        return nodes

    def _create_section_nodes(self, sections: List[Dict[str, Any]], parent_id: str) -> Dict[str, MindMapNode]:
        nodes: Dict[str, MindMapNode] = {}
        for section in sections[:6]:
            title = section.get('title')
            if title and title != "No Title":
                node_id = self._generate_node_id()
                content_text = strip_metadata_lines(section.get('content', ''))
                content_preview = (content_text[:150] + "...") if len(content_text) > 150 else content_text
                node = MindMapNode(
                    id=node_id,
                    title=strip_metadata_lines(title),
                    content=content_preview,
                    parent_id=parent_id,
                    node_type="section",
                    metadata={
                        'expandable': True,
                        'level': section.get('level', 1),
                        'full_content': content_text,
                        'subsections': [strip_metadata_lines(s) for s in section.get('subsections', [])],
                        'links': section.get('links', []),
                    },
                )
                nodes[node_id] = node
        return nodes

    def _create_link_nodes(self, links: List[Dict[str, Any]], parent_id: str) -> Dict[str, MindMapNode]:
        nodes: Dict[str, MindMapNode] = {}
        link_types: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for link in links or []:
            link_types[link.get('type', 'general')].append(link)
        for link_type, type_links in link_types.items():
            if len(type_links) >= 2:
                category_id = self._generate_node_id()
                category_node = MindMapNode(
                    id=category_id,
                    title=f"{link_type.title()} Links",
                    content=f"{len(type_links)} {link_type} resources",
                    parent_id=parent_id,
                    node_type="link_category",
                    metadata={
                        'expandable': True,
                        'link_type': link_type,
                        'links': type_links[:5],
                    },
                )
                nodes[category_id] = category_node
        return nodes

    def _generate_visualization_data(self, nodes: Dict[str, MindMapNode], root_id: str) -> Dict[str, Any]:
        viz_nodes: List[Dict[str, Any]] = []
        viz_edges: List[Dict[str, Any]] = []
        for node in nodes.values():
            viz_node = {
                'id': node.id,
                'label': node.title,
                'type': node.node_type,
                'expandable': node.metadata.get('expandable', False),
                'expanded': node.expanded,
                'url': node.url,
                'size': self._calculate_node_size(node),
                'color': self._get_node_color(node.node_type),
                'shape': self._get_node_shape(node.node_type),
            }
            viz_nodes.append(viz_node)
        for node in nodes.values():
            for child_id in node.children:
                viz_edges.append({
                    'from': node.id,
                    'to': child_id,
                    'type': 'hierarchy',
                })
        return {'nodes': viz_nodes, 'edges': viz_edges, 'layout': 'hierarchical', 'center_node': root_id}

    def _calculate_node_size(self, node: MindMapNode) -> int:
        base_size = 30
        if node.node_type == "root":
            return base_size + 20
        elif node.node_type == "section":
            return base_size + 10
        elif node.node_type == "concept":
            return base_size + 5
        else:
            return base_size

    def _get_node_color(self, node_type: str) -> str:
        colors = {
            'root': '#FF6B6B',
            'concept': '#4ECDC4',
            'section': '#45B7D1',
            'link_category': '#96CEB4',
            'subsection': '#FFEAA7',
            'related': '#DDA0DD',
            'link': '#A8E6CF',
        }
        return colors.get(node_type, '#95A5A6')

    def _get_node_shape(self, node_type: str) -> str:
        shapes = {
            'root': 'circle',
            'concept': 'box',
            'section': 'ellipse',
            'link_category': 'diamond',
            'subsection': 'triangle',
            'related': 'star',
            'link': 'square',
        }
        return shapes.get(node_type, 'box')

    def _get_style_config(self) -> Dict[str, Any]:
        return {
            'layout': {'hierarchical': {'enabled': True, 'direction': 'UD', 'sortMethod': 'directed'}},
            'interaction': {'dragNodes': True, 'dragView': True, 'zoomView': True, 'selectConnectedEdges': False},
            'physics': {'enabled': True, 'hierarchicalRepulsion': {
                'nodeDistance': 120, 'centralGravity': 0.0, 'springLength': 100, 'springConstant': 0.01, 'damping': 0.09}},
            'nodes': {'borderWidth': 2, 'shadow': True, 'font': {'size': 14, 'color': '#333333'}},
            'edges': {'width': 2, 'shadow': True, 'smooth': {'type': 'continuous'}},
        }

# --------- Node expansion service ---------
class NodeExpansionService:
    def __init__(self, scraper: EnhancedWebScraper):
        self.scraper = scraper

    async def expand_node(self, node_id: str, expand_type: str, mindmap: InteractiveMindMap) -> NodeExpansionResult:
        if node_id not in mindmap.nodes:
            raise ValueError(f"Node {node_id} not found")
        node = mindmap.nodes[node_id]
        new_nodes: List[MindMapNode] = []
        if expand_type == "subsections" and node.node_type == "section":
            new_nodes = await self._expand_section_subsections(node, mindmap)
        elif expand_type == "links" and "links" in node.metadata:
            new_nodes = await self._expand_node_links(node, mindmap)
        elif expand_type == "related":
            new_nodes = await self._expand_related_concepts(node, mindmap)
        for new_node in new_nodes:
            mindmap.nodes[new_node.id] = new_node
            node.children.append(new_node.id)
        node.expanded = True
        updated_viz = self._update_visualization_for_expansion(mindmap, node_id, new_nodes)
        return NodeExpansionResult(node_id=node_id, new_nodes=new_nodes, updated_visualization=updated_viz)

    async def _expand_section_subsections(self, node: MindMapNode, mindmap: InteractiveMindMap) -> List[MindMapNode]:
        new_nodes: List[MindMapNode] = []
        subsections = node.metadata.get('subsections', []) or []
        for i, subsection in enumerate(subsections[:5]):
            sub_id = f"subsection_{node.id}_{i}"
            subsection_node = MindMapNode(
                id=sub_id,
                title=(subsection[:50] + "...") if len(subsection) > 50 else subsection,
                content=subsection,
                parent_id=node.id,
                node_type="subsection",
                metadata={'subsection_index': i},
            )
            new_nodes.append(subsection_node)
        return new_nodes

    async def _expand_node_links(self, node: MindMapNode, mindmap: InteractiveMindMap) -> List[MindMapNode]:
        new_nodes: List[MindMapNode] = []
        links = node.metadata.get('links', []) or []
        for i, link in enumerate(links[:3]):
            link_id = f"link_{node.id}_{i}"
            title_text = link.get('text', '') or link.get('url', '')
            link_node = MindMapNode(
                id=link_id,
                title=(title_text[:40] + "...") if len(title_text) > 40 else title_text,
                content=f"Link to: {title_text}",
                url=link.get('url'),
                parent_id=node.id,
                node_type="link",
                metadata={'link_data': link, 'clickable': True},
            )
            new_nodes.append(link_node)
        return new_nodes

    async def _expand_related_concepts(self, node: MindMapNode, mindmap: InteractiveMindMap) -> List[MindMapNode]:
        new_nodes: List[MindMapNode] = []
        content = node.metadata.get('full_content', node.content or "")
        try:
            words = word_tokenize(content.lower())
        except Exception:
            words = re.findall(r'\b\w+\b', content.lower())
        try:
            sw = set(stopwords.words('english'))
        except LookupError:
            sw = set()
        important_words = [w for w in words if len(w) > 4 and w not in sw]
        word_freq = Counter(important_words)
        for word, freq in word_freq.most_common(3):
            if freq > 1:
                rel_id = f"related_{node.id}_{word}"
                related_node = MindMapNode(
                    id=rel_id,
                    title=word.title(),
                    content=f"Related concept appearing {freq} times in {node.title}",
                    parent_id=node.id,
                    node_type="related",
                    metadata={'frequency': freq, 'source_node': node.id},
                )
                new_nodes.append(related_node)
        return new_nodes

    def _update_visualization_for_expansion(self, mindmap: InteractiveMindMap, expanded_node_id: str, new_nodes: List[MindMapNode]) -> Dict[str, Any]:
        viz_data = json.loads(json.dumps(mindmap.visualization_data))
        for n in new_nodes:
            viz_node = {
                'id': n.id,
                'label': n.title,
                'type': n.node_type,
                'expandable': n.metadata.get('expandable', False),
                'expanded': n.expanded,
                'url': n.url,
                'size': 25,
                'color': self._get_node_color(n.node_type),
                'shape': self._get_node_shape(n.node_type),
            }
            viz_data['nodes'].append(viz_node)
            viz_data['edges'].append({'from': expanded_node_id, 'to': n.id, 'type': 'expansion'})
        return viz_data

    def _get_node_color(self, node_type: str) -> str:
        colors = {
            'root': '#FF6B6B', 'concept': '#4ECDC4', 'section': '#45B7D1',
            'link_category': '#96CEB4', 'subsection': '#FFEAA7',
            'related': '#DDA0DD', 'link': '#A8E6CF',
        }
        return colors.get(node_type, '#95A5A6')

    def _get_node_shape(self, node_type: str) -> str:
        shapes = {
            'root': 'circle', 'concept': 'box', 'section': 'ellipse',
            'link_category': 'diamond', 'subsection': 'triangle',
            'related': 'star', 'link': 'square',
        }
        return shapes.get(node_type, 'box')

# --------- Summarizer (bullet points + strict concept filtering) ---------
class ContentSummarizer:
    """Handles content summarization using NLP techniques"""
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
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
        # Extra domain stop words and common fillers to filter out from concepts
        self.extra_stop = {
            'all', 'and', 'the', 'med', 'int', 'lin', 'inc', 'etc',
            'introduction', 'background', 'methods', 'results', 'discussion',
            'conclusion', 'overview', 'table', 'figure', 'supplement', 'appendix'
        }

    def summarize_content(self, content: Dict[str, Any], max_length: int = 500) -> Dict[str, Any]:
        """Generate a bullet-point summary and cleaned key concepts"""
        sentences, scores = self._rank_sentences(content)
        bullets = self._make_bullets(sentences, scores, max_length=max_length, max_bullets=8)
        # Sanitize bullets to remove metadata lines
        bullets = [strip_metadata_lines(b) for b in bullets if strip_metadata_lines(b).strip()]
        key_concepts = self._extract_key_concepts_clean(content)
        return {
            'summary': "\n".join(f"- {b}" for b in bullets),
            'summary_bullets': bullets,
            'key_concepts': key_concepts,
            'method': 'nltk_extractive_bullets',
        }

    def _rank_sentences(self, content: Dict[str, Any]):
        full_text = content.get('full_text', '')
        sections = content.get('sections', [])
        if not full_text or len(full_text.split()) < 10:
            return [full_text], {full_text: 1.0}

        important_text = ""
        for section in sections:
            if section.get('content') and section.get('level', 0) <= 3:
                important_text += section['content'] + " "
        if not important_text:
            important_text = full_text

        try:
            sentences = sent_tokenize(important_text)
        except Exception:
            sentences = re.split(r'[.!?]+', important_text)
        sentences = [s.strip() for s in sentences if s and s.strip()]
        scores = self._score_sentences_enhanced(sentences, content)
        return sentences, scores

    def _make_bullets(self, sentences: List[str], scores: Dict[str, float], max_length: int = 500, max_bullets: int = 8) -> List[str]:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        bullets: List[str] = []
        words_so_far = 0
        for sent, sc in ranked:
            s_clean = strip_metadata_lines(sent.strip())
            if len(s_clean) > 240:
                s_clean = re.sub(r'\s+', ' ', s_clean)[:240].rstrip() + "..."
            wcount = len(s_clean.split())
            if words_so_far + wcount > max_length and bullets:
                break
            bullets.append(s_clean)
            words_so_far += wcount
            if len(bullets) >= max_bullets:
                break
        order_map = {s: i for i, s in enumerate(sentences)}
        bullets.sort(key=lambda s: order_map.get(s, 1_000_000))
        return bullets

    def _score_sentences_enhanced(self, sentences: List[str], content: Dict[str, Any]) -> Dict[str, float]:
        sentence_scores: Dict[str, float] = {}
        title = content.get('title', '')
        sections = content.get('sections', [])
        links = content.get('links', [])
        important_words: List[str] = []
        for section in sections:
            if section.get('level', 0) <= 2:
                raw = section.get('content', '')
                try:
                    words = word_tokenize(raw.lower())
                except Exception:
                    words = re.findall(r'\b\w+\b', raw.lower())
                important_words.extend([w for w in words if w not in self.stop_words and len(w) > 2])
        word_freq = Counter(important_words)

        try:
            title_words = word_tokenize(title.lower())
        except Exception:
            title_words = re.findall(r'\b\w+\b', title.lower())
        title_words = [w for w in title_words if w not in self.stop_words]

        link_words: List[str] = []
        for link in links:
            try:
                ws = word_tokenize(link.get('text', '').lower())
            except Exception:
                ws = re.findall(r'\b\w+\b', link.get('text', '').lower())
            link_words.extend([w for w in ws if w not in self.stop_words])

        for sentence in sentences:
            if len(sentence.split()) < 5:
                continue
            score = 0.0
            try:
                sentence_words = word_tokenize(sentence.lower())
            except Exception:
                sentence_words = re.findall(r'\b\w+\b', sentence.lower())
            sentence_words = [w for w in sentence_words if w not in self.stop_words]
            for word in sentence_words:
                if word in word_freq:
                    score += float(word_freq[word])
            title_overlap = len(set(sentence_words) & set(title_words))
            score += title_overlap * 10.0
            link_overlap = len(set(sentence_words) & set(link_words))
            score += link_overlap * 5.0
            idx = sentences.index(sentence)
            if idx < 3:
                score += 8.0
            elif idx < max(1, int(len(sentences) * 0.1)):
                score += 5.0
            sentence_scores[sentence] = score / max(1, len(sentence_words))
        return sentence_scores

    def _normalize_concept(self, s: str) -> str:
        s = s.strip()
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'^[^\w]+|[^\w]+$', '', s)
        if s.isupper() and len(s) <= 6:
            return s
        return s.title()

    def _clean_tokens(self, tokens: List[str]) -> List[str]:
        out = []
        for t in tokens:
            t_clean = re.sub(r'^[^\w]+|[^\w]+$', '', t)
            if not t_clean:
                continue
            if t_clean.lower() in self.stop_words or t_clean.lower() in self.extra_stop:
                continue
            if _is_noise_token(t_clean):
                continue
            out.append(t_clean)
        return out

    def _extract_key_concepts_clean(self, content: Dict[str, Any]) -> List[str]:
        text = content.get('full_text', '') or ''
        title = content.get('title', '') or ''
        sections = content.get('sections', []) or []
        links = content.get('links', []) or []

        candidates: set = set()

        # Title
        try:
            title_words = word_tokenize(title)
            title_pos = pos_tag(title_words)
        except Exception:
            title_words = re.findall(r'\b[A-Za-z][A-Za-z0-9\-]+\b', title)
            title_pos = [(w, 'NN') for w in title_words]
        title_tokens = self._clean_tokens([w for w, p in title_pos if p.startswith('NN')])
        for w in title_tokens:
            candidates.add(self._normalize_concept(w))

        # Section titles
        for section in sections:
            st = section.get('title', '')
            if st and st != title:
                try:
                    words = word_tokenize(st)
                    pos_tags = pos_tag(words)
                except Exception:
                    words = re.findall(r'\b[A-Za-z][A-Za-z0-9\-]+\b', st)
                    pos_tags = [(w, 'NN') for w in words]
                section_tokens = self._clean_tokens([w for w, p in pos_tags if p.startswith('NN')])
                for w in section_tokens:
                    candidates.add(self._normalize_concept(w))

        # Link anchor texts
        for link in links:
            lt = link.get('text', '')
            try:
                words = word_tokenize(lt)
                pos_tags = pos_tag(words)
            except Exception:
                words = re.findall(r'\b[A-Za-z][A-Za-z0-9\-]+\b', lt)
                pos_tags = [(w, 'NN') for w in words]
            link_tokens = self._clean_tokens([w for w, p in pos_tags if p.startswith('NN')])
            for w in link_tokens:
                candidates.add(self._normalize_concept(w))

        # Main text NPs
        try:
            words = word_tokenize(text)
            pos_tags = pos_tag(words)
        except Exception:
            words = re.findall(r'\b[A-Za-z][A-Za-z0-9\-]+\b', text)
            pos_tags = [(w, 'NN') for w in words]
        main_tokens = self._clean_tokens([w for w, p in pos_tags if p in ['NNP', 'NNPS', 'NN', 'NNS']])
        for w in main_tokens:
            candidates.add(self._normalize_concept(w))

        # Technical patterns
        tech_patterns = [
            r'\b(?:API|SDK|framework|platform|service|application|system|database|server|network|cloud|data|security|algorithm|software|technology|development|infrastructure|architecture|protocol|interface|module|component|library|tool|solution|method|process|strategy|approach|technique|model|theory|concept|principle|standard|specification|implementation|integration|optimization|automation|analytics|intelligence|machine|learning|artificial|neural|blockchain|cryptocurrency|IoT|5G|AI|ML|DevOps|SaaS|PaaS|IaaS|REST|GraphQL|JSON|XML|HTTP|HTTPS|SSL|TLS|OAuth|JWT)\b',
            r'\b[A-Z]{2,}\b',
            r'\b\w+(?:ing|tion|ment|ness|ship|able|ible|ology|ography|ancy|ency)\b'
        ]
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                if m.lower() in self.stop_words or m.lower() in self.extra_stop:
                    continue
                if _is_noise_token(m):
                    continue
                candidates.add(self._normalize_concept(m))

        concept_freq = Counter()
        text_lower = text.lower()
        for concept in candidates:
            if not concept:
                continue
            base_freq = text_lower.count(concept.lower())
            title_boost = 5 if concept.lower() in title.lower() else 0
            for section in sections:
                if concept.lower() in (section.get('title', '') or '').lower():
                    title_boost += 4 * (4 - min(section.get('level', 1), 3))
            score = base_freq + title_boost
            if score > 0:
                concept_freq[concept] = score

        final_concepts = []
        for c, _ in concept_freq.most_common(20):
            c_low = c.lower()
            if c_low in self.stop_words or c_low in self.extra_stop:
                continue
            if _is_noise_token(c):
                continue
            if len(c) <= 3 and c.isalpha() and c_low not in {'h1', 'h2', 'h3'}:
                continue
            final_concepts.append(strip_metadata_lines(c))

        return final_concepts[:15]

# --------- Background job ---------
async def process_scrape_job(job_id: str, url: str, summary_length: int, extract_links: bool = True):
    try:
        job = scrape_jobs[job_id]
        job.status = "processing"
        job.progress = 10

        scraper = EnhancedWebScraper()
        summarizer = ContentSummarizer()
        mindmap_gen = InteractiveMindMapGenerator()

        logger.info(f"Scraping content from: {url}")
        job.progress = 30
        content = scraper.scrape_content(url, extract_links)

        logger.info("Generating enhanced summary...")
        job.progress = 50
        summary_data = summarizer.summarize_content(content, summary_length or 300)

        logger.info("Creating interactive mind map...")
        job.progress = 70
        interactive_mindmap = mindmap_gen.create_interactive_mindmap(content, summary_data)

        mindmap_nodes[job_id] = interactive_mindmap

        job.progress = 85
        text_mindmaps = {
            'visual': generate_text_mindmap(content['title'], summary_data['summary'], summary_data['key_concepts']),
            'network': generate_network_mindmap(content['title'], summary_data['key_concepts']),
            'hierarchical': generate_hierarchical_mindmap(content['title'], summary_data['key_concepts']),
        }

        result = {
            'scraped_content': content,
            'summary': summary_data,
            'mind_maps': text_mindmaps,
            'interactive_mindmap': {
                'root_node': interactive_mindmap.root_node.dict(),
                'nodes': {k: v.dict() for k, v in interactive_mindmap.nodes.items()},
                'visualization_data': interactive_mindmap.visualization_data,
                'style_config': interactive_mindmap.style_config,
            },
            'metadata': {
                'total_links': len(content.get('links', [])),
                'total_sections': len(content.get('sections', [])),
                'expandable_nodes': sum(1 for n in interactive_mindmap.nodes.values() if n.metadata.get('expandable', False)),
            },
        }

        safe_filename = sanitize_filename(url)
        file_path = os.path.join(STORAGE_DIR, f"{safe_filename}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        job.result = result
        job.status = "completed"
        job.progress = 100
        job.completed_at = datetime.now().isoformat()
        logger.info(f"Job {job_id} completed successfully with interactive features")
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {str(e)}")
        job = scrape_jobs.get(job_id)
        if job:
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.now().isoformat()

# --------- Text mindmap helpers ---------
def generate_text_mindmap(title: str, summary: str, concepts: List[str]) -> str:
    title = strip_metadata_lines(title)
    summary = strip_metadata_lines(summary)
    concepts = [strip_metadata_lines(c) for c in concepts]
    lines: List[str] = []
    top = f"╔{'═' * (len(title) + 4)}╗"
    mid = f"║ {title.upper()} ║"
    bot = f"╚{'═' * (len(title) + 4)}╝"
    pad = " " * max(0, (40 - len(title) // 2))
    lines.extend(["", pad + top, pad + mid, pad + bot, ""])
    if concepts:
        lines.append(" " * 40 + "│")
        lines.append(" " * 35 + "┌─────┴─────┐")
        for i, concept in enumerate(concepts[:8]):
            if i % 2 == 0:
                lines.append(f"├── 🔹 {concept:<30}│")
            else:
                lines.append(f"│{'':<35}├── 🔹 {concept}")
    return "\n".join(lines)

def generate_network_mindmap(title: str, concepts: List[str]) -> str:
    title = strip_metadata_lines(title)
    concepts = [strip_metadata_lines(c) for c in concepts]
    lines = [f"🌐 {title}", "│", "├── 📊 Key Concepts"]
    for concept in concepts[:6]:
        lines.append(f"│ ├── 🔸 {concept}")
    return "\n".join(lines)

def generate_hierarchical_mindmap(title: str, concepts: List[str]) -> str:
    title = strip_metadata_lines(title)
    concepts = [strip_metadata_lines(c) for c in concepts]
    lines = [f"🌳 {title}", "│"]
    for i, concept in enumerate(concepts):
        is_last = (i == len(concepts) - 1)
        if is_last:
            lines.append(f"└── 🍃 {concept}")
        else:
            lines.append(f"├── 🍃 {concept}")
    return "\n".join(lines)

# --------- Expansion service factory ---------
expansion_service: Optional[NodeExpansionService] = None
def get_expansion_service() -> NodeExpansionService:
    global expansion_service
    if expansion_service is None:
        expansion_service = NodeExpansionService(EnhancedWebScraper())
    return expansion_service

# --------- API Endpoints ---------
@app.get("/")
async def root():
    return {
        "message": "Interactive Website Scraper & Mind Map Generator API",
        "version": "2.0.0",
        "features": [
            "Enhanced web scraping with link extraction",
            "Interactive mind maps with expandable nodes",
            "Multiple visualization formats",
            "Real-time node expansion",
            "Hierarchical content analysis",
        ],
        "endpoints": [
            "POST /scrape - Start enhanced scraping job",
            "GET /job/{job_id} - Get job status",
            "GET /job/{job_id}/mindmap - Get interactive mindmap",
            "POST /expand-node - Expand mindmap node",
            "GET /job/{job_id}/export - Export mindmap data",
            "GET /jobs - List all jobs",
            "GET /health - Health check",
        ],
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "interactive_mindmaps": True,
            "node_expansion": True,
            "link_extraction": True,
            "multi_format_export": True,
        },
    }

@app.post("/scrape", response_model=ScrapeResponse)
async def create_enhanced_scrape_job(request: ScrapeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job = JobData(job_id=job_id, status="queued", progress=0, created_at=datetime.now().isoformat())
    scrape_jobs[job_id] = job
    background_tasks.add_task(
        process_scrape_job,
        job_id,
        str(request.url),
        request.summary_length or 300,
        bool(request.extract_links),
    )
    return ScrapeResponse(job_id=job_id, status="queued", message="Enhanced scraping job with interactive mindmap features started")

@app.get("/job/{job_id}/mindmap")
async def get_interactive_mindmap(job_id: str):
    if job_id not in scrape_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = scrape_jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job.status}")
    if job_id not in mindmap_nodes:
        raise HTTPException(status_code=404, detail="Interactive mindmap not found")
    mindmap = mindmap_nodes[job_id]
    return {
        "job_id": job_id,
        "mindmap": {
            "root_node": mindmap.root_node.dict(),
            "nodes": {k: v.dict() for k, v in mindmap.nodes.items()},
            "visualization_data": mindmap.visualization_data,
            "style_config": mindmap.style_config,
        },
    }

@app.post("/expand-node", response_model=NodeExpansionResult)
async def expand_mindmap_node(request: ExpandNodeRequest):
    target_job_id: Optional[str] = None
    for job_id, mindmap in mindmap_nodes.items():
        if request.node_id in mindmap.nodes:
            target_job_id = job_id
            break
    if not target_job_id:
        raise HTTPException(status_code=404, detail="Node not found in any mindmap")
    mindmap = mindmap_nodes[target_job_id]
    expansion_svc = get_expansion_service()
    try:
        result = await expansion_svc.expand_node(request.node_id, request.expand_type, mindmap)
        mindmap.visualization_data = result.updated_visualization
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expansion failed: {str(e)}")

@app.get("/job/{job_id}/export")
async def export_mindmap_data(job_id: str, format: str = "json"):
    if job_id not in scrape_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = scrape_jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job.status}")
    if format == "json":
        return job.result
    elif format == "nodes":
        if job_id in mindmap_nodes:
            mindmap = mindmap_nodes[job_id]
            return {
                "nodes": [node.dict() for node in mindmap.nodes.values()],
                "relationships": [
                    {"parent": node.id, "children": node.children}
                    for node in mindmap.nodes.values()
                    if node.children
                ],
            }
        else:
            raise HTTPException(status_code=404, detail="Interactive mindmap not found")
    elif format == "links":
        content = job.result.get('scraped_content', {}) if job.result else {}
        links = content.get('links', [])
        return {
            "extracted_links": links,
            "total_count": len(links),
            "categorized": {
                link_type: [l for l in links if l.get('type') == link_type]
                for link_type in set(l.get('type', 'general') for l in links)
            },
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")

@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in scrape_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = scrape_jobs[job_id]
    extra_metadata: Dict[str, Any] = {}
    if job.status == "completed" and job_id in mindmap_nodes:
        mindmap = mindmap_nodes[job_id]
        extra_metadata = {
            "total_nodes": len(mindmap.nodes),
            "expandable_nodes": sum(1 for node in mindmap.nodes.values() if node.metadata.get('expandable', False)),
            "node_types": list({node.node_type for node in mindmap.nodes.values()}),
        }
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        result={**(job.result or {}), "mindmap_metadata": extra_metadata} if job.result else None,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )

@app.get("/jobs")
async def list_jobs_enhanced():
    jobs_data: List[Dict[str, Any]] = []
    for job in scrape_jobs.values():
        job_info: Dict[str, Any] = {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }
        if job.status == "completed" and job.job_id in mindmap_nodes:
            mindmap = mindmap_nodes[job.job_id]
            job_info["mindmap_features"] = {
                "total_nodes": len(mindmap.nodes),
                "expandable_nodes": sum(1 for node in mindmap.nodes.values() if node.metadata.get('expandable', False)),
                "has_links": any(node.url for node in mindmap.nodes.values()),
            }
        jobs_data.append(job_info)
    return {"total": len(scrape_jobs), "jobs": jobs_data}

@app.delete("/job/{job_id}")
async def delete_job_enhanced(job_id: str):
    if job_id not in scrape_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    del scrape_jobs[job_id]
    if job_id in mindmap_nodes:
        del mindmap_nodes[job_id]
    return {"message": f"Job {job_id} and associated data deleted successfully"}

# --------- App startup: ensure NLTK resources exist ---------
@app.on_event("startup")
async def ensure_nltk_resources():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                logger.info(f"Downloading NLTK resource: {pkg}")
                nltk.download(pkg)
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {pkg}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
