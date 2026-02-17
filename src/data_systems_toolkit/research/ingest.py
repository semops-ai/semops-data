"""
Ingest module: Fetch URLs/PDFs → Docling → chunks

Uses Docling service for document processing.
"""
import json
import hashlib
import requests
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from .config import config


@dataclass
class Source:
    """A research source (PDF or web page)."""
    url: str
    title: str
    source_type: str  # "pdf" or "web"
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    tags: list[str] = field(default_factory=list)

    @property
    def source_id(self) -> str:
        """Generate a unique ID from URL hash."""
        return hashlib.md5(self.url.encode()).hexdigest()[:12]


@dataclass
class Chunk:
    """A chunk of text from a source."""
    text: str
    source_id: str
    source_title: str
    source_url: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def fetch_pdf(url: str, cache_dir: Optional[Path] = None) -> Path:
    """Download PDF to local cache."""
    cache_dir = cache_dir or config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from URL hash
    filename = hashlib.md5(url.encode()).hexdigest()[:12] + ".pdf"
    local_path = cache_dir / filename

    if local_path.exists():
        print(f"Using cached PDF: {local_path}")
        return local_path

    print(f"Downloading PDF: {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    local_path.write_bytes(response.content)
    print(f"Saved to: {local_path}")
    return local_path


def process_with_docling(file_path: Path) -> dict:
    """Process a document with Docling service."""
    docling_url = f"{config.docling_url}/v1/convert/file"

    with open(file_path, "rb") as f:
        # Docling expects 'files' (plural) as an array
        files = [("files", (file_path.name, f, "application/pdf"))]
        data = {"to_formats": ["text"]}  # Request plain text output
        response = requests.post(docling_url, files=files, data=data, timeout=300)

    response.raise_for_status()
    return response.json()


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Split text into overlapping chunks."""
    chunk_size = chunk_size or config.chunk_size
    overlap = overlap or config.chunk_overlap

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end in last 20% of chunk
            search_start = end - int(chunk_size * 0.2)
            for sep in [". ", ".\n", "? ", "! "]:
                last_sep = text.rfind(sep, search_start, end)
                if last_sep > search_start:
                    end = last_sep + 1
                    break

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks


def ingest_pdf(source: Source) -> list[Chunk]:
    """Ingest a PDF source into chunks."""
    # Download PDF
    pdf_path = fetch_pdf(source.url)

    # Process with Docling
    result = process_with_docling(pdf_path)

    # Extract text (Docling returns structured output)
    # The exact structure depends on Docling version
    if "text" in result:
        full_text = result["text"]
    elif "content" in result:
        full_text = result["content"]
    elif "pages" in result:
        full_text = "\n\n".join(page.get("text", "") for page in result["pages"])
    else:
        # Fallback: stringify the result
        full_text = str(result)

    # Chunk the text
    text_chunks = chunk_text(full_text)

    # Create Chunk objects
    chunks = []
    for i, text in enumerate(text_chunks):
        chunk = Chunk(
            text=text,
            source_id=source.source_id,
            source_title=source.title,
            source_url=source.url,
            chunk_index=i,
            metadata={
                "authors": source.authors,
                "year": source.year,
                "tags": source.tags,
                "source_type": source.source_type,
            }
        )
        chunks.append(chunk)

    print(f"Created {len(chunks)} chunks from: {source.title}")
    return chunks


def ingest_url(source: Source) -> list[Chunk]:
    """Ingest a web page source into chunks."""
    # For web pages, we could use requests + BeautifulSoup
    # or Docling's URL endpoint if available
    print(f"Web ingestion not yet implemented for: {source.url}")
    return []


def ingest_sources(sources: list[Source]) -> list[Chunk]:
    """Ingest multiple sources."""
    all_chunks = []

    for source in sources:
        try:
            if source.source_type == "pdf":
                chunks = ingest_pdf(source)
            elif source.source_type == "web":
                chunks = ingest_url(source)
            else:
                print(f"Unknown source type: {source.source_type}")
                continue

            all_chunks.extend(chunks)

        except Exception as e:
            print(f"Error ingesting {source.title}: {e}")

    return all_chunks


def load_manifest(manifest_path: Optional[Path] = None) -> list[Source]:
    """Load sources from manifest.json."""
    manifest_path = manifest_path or (config.sources_dir / "manifest.json")

    if not manifest_path.exists():
        return []

    with open(manifest_path) as f:
        data = json.load(f)

    return [Source(**s) for s in data.get("sources", [])]


def save_manifest(sources: list[Source], manifest_path: Optional[Path] = None):
    """Save sources to manifest.json."""
    manifest_path = manifest_path or (config.sources_dir / "manifest.json")

    data = {
        "updated": datetime.now().isoformat(),
        "sources": [
            {
                "url": s.url,
                "title": s.title,
                "source_type": s.source_type,
                "authors": s.authors,
                "year": s.year,
                "tags": s.tags,
            }
            for s in sources
        ]
    }

    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved manifest with {len(sources)} sources to: {manifest_path}")
