#!/usr/bin/env python3
"""
Website Content Scraper and Mind Map Generator

This script scrapes content from a specified website, generates a summary,
and creates a mind map visualization of the key concepts.
"""

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import json
from typing import Dict, List, Tuple
import google.generativeai as genai
import os
from dataclasses import dataclass
import argparse
import sys

@dataclass
class ContentSection:
    """Represents a section of content with title and text"""
    title: str
    content: str
    keywords: List[str]

class WebScraper:
    """Handles web scraping functionality"""
    
    def __init__(self, headers=None):
        self.session = requests.Session()
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session.headers.update(self.headers)
    
    def scrape_content(self, url: str) -> Dict[str, any]:
        """
        Scrape content from the specified URL
        
        Args:
            url: The website URL to scrape
            
        Returns:
            Dictionary containing scraped content
        """
        try:
            response = self.session.get(url, timeout=10)
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
                headings = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                paragraphs = main_content.find_all('p')
                
                # Group content by sections
                current_section = {"title": title_text, "content": ""}
                
                for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
                    if element.name.startswith('h'):
                        if current_section["content"].strip():
                            sections.append(current_section)
                        current_section = {
                            "title": element.get_text().strip(),
                            "content": ""
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
                'url': url,
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
    """Handles content summarization using AI"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize summarizer
        
        Args:
            api_key: Google API key (can also be set via GOOGLE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or 'AIzaSyDwfaxEP-Ji9CA6eXwptj9wyjBuS6AhDLE'
        self.model = None
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Try different model names
                try:
                    self.model = genai.GenerativeModel('gemini-1.5-flash')
                except:
                    try:
                        self.model = genai.GenerativeModel('gemini-pro')
                    except:
                        self.model = genai.GenerativeModel('models/gemini-pro')
                
                print("✅ Google Gemini AI initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Google Gemini: {e}")
                self.model = None
    
    def summarize_content(self, content: Dict[str, any], max_length: int = 500) -> Dict[str, any]:
        """
        Generate a summary of the scraped content
        
        Args:
            content: Dictionary containing scraped content
            max_length: Maximum length of summary in words
            
        Returns:
            Dictionary containing summary and key points
        """
        full_text = content['full_text']
        
        # Fallback summarization if no API key or model failed to initialize
        if not self.api_key or not self.model:
            return self._extractive_summary(content, max_length)
        
        try:
            # Use Google Gemini for summarization
            prompt = f"""You are a helpful assistant that creates concise summaries and identifies key concepts. 
            Summarize the following content in {max_length} words or less, and identify 5-10 key concepts or topics.

            Title: {content['title']}
            
            Content: {full_text[:8000]}
            
            Please provide:
            1. A concise summary
            2. Key concepts (comma-separated list)
            """
            
            response = self.model.generate_content(prompt)
            summary_text = response.text
            
            # Extract key concepts using a separate call
            key_concepts = self._extract_key_concepts_ai(full_text)
            
            return {
                'summary': summary_text,
                'key_concepts': key_concepts,
                'method': 'ai_powered_gemini'
            }
            
        except Exception as e:
            print(f"Google AI summarization failed: {e}. Falling back to extractive method.")
            return self._extractive_summary(content, max_length)
    
    def _extractive_summary(self, content: Dict[str, any], max_length: int) -> Dict[str, any]:
        """Fallback extractive summarization method"""
        sections = content['sections']
        full_text = content['full_text']
        
        # Simple extractive approach: take first sentences from each section
        summary_parts = []
        key_concepts = set()
        
        # Extract from sections
        for section in sections[:5]:  # Limit to first 5 sections
            section_text = section['content'].strip()
            if section_text:
                # Take first sentence
                sentences = re.split(r'[.!?]+', section_text)
                if sentences:
                    first_sentence = sentences[0].strip()
                    if len(first_sentence) > 20:  # Avoid very short fragments
                        summary_parts.append(first_sentence)
                
                # Extract potential key concepts from both title and content
                title_words = re.findall(r'\b[A-Z][a-zA-Z]+\b', section['title'])
                content_words = re.findall(r'\b[A-Z][a-zA-Z]+\b', section_text)
                key_concepts.update(title_words[:2])
                key_concepts.update(content_words[:2])
        
        # Also extract concepts from the main title
        title_concepts = re.findall(r'\b[A-Z][a-zA-Z]+\b', content['title'])
        key_concepts.update(title_concepts)
        
        # Extract common nouns and technical terms
        tech_terms = re.findall(r'\b(?:cloud|platform|service|API|infrastructure|computing|data|security|network|database|server|application|software|technology|development|deployment|management|system)\w*\b', full_text.lower())
        key_concepts.update([term.title() for term in tech_terms[:5]])
        
        # If still no concepts, extract from full text
        if not key_concepts:
            important_words = re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', full_text)
            key_concepts.update(important_words[:8])
        
        summary = '. '.join(summary_parts) if summary_parts else "Content overview available in full text."
        
        # Trim to max length
        words = summary.split()
        if len(words) > max_length:
            summary = ' '.join(words[:max_length]) + '...'
        
        # Clean and limit key concepts
        filtered_concepts = []
        for concept in key_concepts:
            if len(concept) > 2 and concept.lower() not in ['the', 'and', 'for', 'are', 'with']:
                filtered_concepts.append(concept)
        
        return {
            'summary': summary,
            'key_concepts': list(set(filtered_concepts))[:10],  # Remove duplicates and limit
            'method': 'extractive'
        }
    
    def _extract_key_concepts_ai(self, text: str) -> List[str]:
        """Extract key concepts using Google Gemini AI"""
        try:
            prompt = f"""Extract 8-12 key concepts or topics from the following text. 
            Return only a comma-separated list of concepts, no other text.
            
            Text: {text[:6000]}
            """
            
            response = self.model.generate_content(prompt)
            concepts_text = response.text.strip()
            
            # Clean and parse the response
            concepts = [concept.strip() for concept in concepts_text.split(',')]
            # Filter out empty strings and limit to 12 concepts
            concepts = [c for c in concepts if c and len(c) > 2][:12]
            return concepts
            
        except Exception as e:
            print(f"Key concept extraction failed: {e}")
            return []

class MindMapGenerator:
    """Generates text-based mind maps from content summaries"""
    
    def __init__(self):
        pass
    
    def create_mind_map(self, title: str, summary: str, key_concepts: List[str], output_path: str = 'mindmap.txt'):
        """
        Create a text-based visual mind map
        
        Args:
            title: Central topic/title
            summary: Summary text
            key_concepts: List of key concepts
            output_path: Output file path for the mind map
        """
        # Create text-based mind map
        mind_map_text = self._generate_text_mindmap(title, summary, key_concepts)
        
        # Print to console
        print("\n" + "="*80)
        print("📋 TEXT MIND MAP")
        print("="*80)
        print(mind_map_text)
        print("="*80)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(mind_map_text)
        
        print(f"\n💾 Mind map saved as {output_path}")
    
    def _generate_text_mindmap(self, title: str, summary: str, key_concepts: List[str]) -> str:
        """Generate a text-based mind map representation"""
        lines = []
        
        # Title section
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
        
        # Concepts branching out
        if key_concepts:
            lines.append(" " * 40 + "│")
            lines.append(" " * 35 + "┌─────┴─────┐")
            lines.append(" " * 35 + "│           │")
            
            # Split concepts into left and right branches
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
                
                # Format the line
                if left_text and right_text:
                    line = f"{left_text:<35}│{right_text:>35}"
                elif left_text:
                    line = f"{left_text:<35}│"
                elif right_text:
                    line = f"{'':<35}│{right_text:>35}"
                else:
                    line = f"{'':<35}│"
                
                lines.append(line)
        
        # Add summary section
        lines.extend([
            "",
            "┌" + "─" * 78 + "┐",
            "│" + " SUMMARY ".center(78) + "│",
            "├" + "─" * 78 + "┤"
        ])
        
        # Wrap summary text
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
    
    def create_network_mind_map(self, title: str, key_concepts: List[str], output_path: str = 'network_mindmap.txt'):
        """Create a network-style text mind map"""
        lines = []
        
        # Header
        lines.extend([
            "",
            "╔" + "═" * 60 + "╗",
            f"║{'NETWORK MIND MAP'.center(60)}║",
            "╠" + "═" * 60 + "╣",
            f"║{title.center(60)}║",
            "╚" + "═" * 60 + "╝",
            ""
        ])
        
        # Network representation
        lines.append("        🔵 " + title)
        lines.append("        │")
        lines.append("   ┌────┴────┐")
        
        # Create branches for concepts
        for i, concept in enumerate(key_concepts[:8]):  # Limit to 8 for better formatting
            if i % 2 == 0:
                lines.append(f"   │         │")
                branch_line = f"   ├─ 🔸 {concept}"
            else:
                branch_line = f"   │         └─ 🔸 {concept}"
            
            lines.append(branch_line)
        
        # Add remaining concepts if any
        if len(key_concepts) > 8:
            lines.append("   │")
            lines.append("   └─ 🔸 ... and more concepts")
        
        network_text = "\n".join(lines)
        
        # Print to console
        print("\n" + "="*65)
        print("🕸️  NETWORK MIND MAP")
        print("="*65)
        print(network_text)
        print("="*65)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(network_text)
        
        print(f"\n💾 Network mind map saved as {output_path}")
    
    def create_hierarchical_mindmap(self, title: str, key_concepts: List[str], output_path: str = 'hierarchical_mindmap.txt'):
        """Create a hierarchical tree-style mind map"""
        lines = []
        
        # Root
        lines.extend([
            "",
            f"🌳 {title.upper()}",
            "│"
        ])
        
        # Main branches
        for i, concept in enumerate(key_concepts):
            is_last = (i == len(key_concepts) - 1)
            
            if is_last:
                lines.append(f"└── 🌿 {concept}")
            else:
                lines.append(f"├── 🌿 {concept}")
        
        tree_text = "\n".join(lines)
        
        # Print to console  
        print("\n" + "="*50)
        print("🌳 HIERARCHICAL MIND MAP")
        print("="*50)
        print(tree_text)
        print("="*50)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(tree_text)
        
        print(f"\n💾 Hierarchical mind map saved as {output_path}")

def main():
    """Main function to orchestrate the scraping and mind map generation"""
    parser = argparse.ArgumentParser(description="Scrape website content and generate mind maps")
    parser.add_argument("url", help="URL of the website to scrape")
    parser.add_argument("--api-key", default="AIzaSyDwfaxEP-Ji9CA6eXwptj9wyjBuS6AhDLE", 
                       help="Google API key (default provided)")
    parser.add_argument("--output", default="mindmap", help="Output filename prefix")
    parser.add_argument("--summary-length", type=int, default=300, help="Maximum summary length in words")
    
    args = parser.parse_args()
    
    try:
        print(f"🌐 Scraping content from: {args.url}")
        
        # Initialize components
        scraper = WebScraper()
        summarizer = ContentSummarizer(args.api_key)
        mind_map_gen = MindMapGenerator()
        
        # Scrape content
        content = scraper.scrape_content(args.url)
        print(f"✅ Successfully scraped {content['word_count']} words")
        print(f"📄 Title: {content['title']}")
        
        # Generate summary
        print("🤖 Generating summary...")
        summary_data = summarizer.summarize_content(content, args.summary_length)
        
        print(f"📝 Summary ({summary_data['method']}):")
        print(summary_data['summary'])
        print(f"\n🔑 Key concepts: {', '.join(summary_data['key_concepts'])}")
        
        # Generate mind maps
        print("🎨 Creating text-based mind maps...")
        mind_map_gen.create_mind_map(
            content['title'],
            summary_data['summary'],
            summary_data['key_concepts'],
            f"{args.output}_visual.txt"
        )
        
        mind_map_gen.create_network_mind_map(
            content['title'],
            summary_data['key_concepts'],
            f"{args.output}_network.txt"
        )
        
        mind_map_gen.create_hierarchical_mindmap(
            content['title'],
            summary_data['key_concepts'],
            f"{args.output}_hierarchical.txt"
        )
        
        # Save data as JSON
        output_data = {
            'url': args.url,
            'scraped_content': content,
            'summary': summary_data
        }
        
        with open(f"{args.output}_data.json", 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Data saved as {args.output}_data.json")
        print("✨ Process completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()