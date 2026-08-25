"""Document processing and chunking utilities."""
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import markdown
from bs4 import BeautifulSoup
from loguru import logger

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

try:
    import markdownify
    MARKDOWNIFY_AVAILABLE = True
except ImportError:
    MARKDOWNIFY_AVAILABLE = False
    logger.debug("markdownify not available, HTML loading will use basic extraction")


@dataclass
class DocumentChunk:
    """Represents a chunk of document text with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str


class DocumentProcessor:
    """Process various document formats and chunk them for vectorization."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, markdown_only: bool = False, whole_document: bool = False, semantic_chunking: bool = False):
        """
        Initialize the document processor.

        Args:
            chunk_size: Maximum size of each chunk in characters (ignored if whole_document=True)
            chunk_overlap: Number of overlapping characters between chunks (ignored if whole_document=True)
            markdown_only: If True, only process markdown files (no MarkItDown conversion)
            whole_document: If True, embed entire documents without chunking
            semantic_chunking: If True, chunk by markdown headings instead of fixed character count
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.markdown_only = markdown_only
        self.whole_document = whole_document
        self.semantic_chunking = semantic_chunking

        # Only initialize MarkItDown if we need conversion
        if not markdown_only:
            if MarkItDown is None:
                raise ImportError(
                    "markitdown is required to convert non-markdown files. "
                    "Install the full local stack with: pip install -r requirements-full.txt"
                )
            self.markitdown = MarkItDown()
            logger.info(
                "Initialized MarkItDown converter for unified document processing")
        else:
            self.markitdown = None
            logger.info("Markdown-only mode: MarkItDown conversion disabled")

    def load_document(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """
        Load a document and extract its text content using MarkItDown.

        Args:
            file_path: Path to the document file

        Returns:
            Tuple of (markdown_text_content, metadata)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract metadata
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": file_path.suffix.lower(),
        }

        # Detect source organization from path or filename
        metadata["source_org"] = self._detect_source_org(str(file_path))

        # Detect region (Hong Kong or non-Hong Kong)
        metadata["region"] = self._detect_region(metadata["source_org"])

        # Route by file type
        # Check for HTML first (custom loader)
        if file_path.suffix.lower() in ['.html', '.htm']:
            return self._load_html(file_path), metadata

        # Check for PDF (custom loader attempt or MarkItDown)
        # MarkItDown handles PDF well, so we'll leave it to MarkItDown unless we need custom handling

        # Check for Markdown/Text (native loader)
        # Handle .md files explicitly since MarkItDown might not support them as input
        if file_path.suffix.lower() in ['.md', '.markdown', '.txt']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return text, metadata
            except UnicodeDecodeError:
                # Try latin-1 fallback
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                return text, metadata
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                return "", metadata

        # For other formats (PDF, PPTX, DOCX, etc.), use MarkItDown
        if self.markdown_only:
             # In markdown_only mode, we shouldn't reach here for non-markdown files
             # unless the caller made a mistake. But if we do, skip it.
             logger.warning(f"Skipping non-markdown file in markdown-only mode: {file_path.name}")
             return "", metadata

        try:
            # Use MarkItDown for unified conversion to Markdown
            logger.debug(
                f"Converting {file_path.name} to Markdown using MarkItDown...")
            result = self.markitdown.convert(str(file_path))

            # Extract text content
            text = result.text_content

            return text, metadata

        except Exception as e:
            logger.error(f"Error converting {file_path} with MarkItDown: {e}")
            # Fallback to simple text extraction
            logger.warning(
                f"Falling back to simple text extraction for {file_path.name}")
            try:
                # Basic text fallback for anything else
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(), metadata
            except Exception as e2:
                logger.error(f"Fallback extraction failed for {file_path}: {e2}")
                return "", metadata

    def _load_html(self, file_path: Path) -> str:
        """
        Load and extract content from an HTML file (e.g., Sickkids pages).

        Targets the main article content and converts it to Markdown.

        Args:
            file_path: Path to the HTML file

        Returns:
            Extracted text content as Markdown or plain text
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # Try to find the main content container (Sickkids uses #panel-container)
        main_content = (
            soup.find(id='panel-container')
            or soup.find(id='article-container')
            or soup.find('article')
            or soup.find('main')
            or soup.find('body')
        )

        if main_content is None:
            main_content = soup

        # Also grab key-points if present
        key_points = soup.find(id='key-points')
        overview = soup.find(id='article-overview')

        parts = []

        # Extract title
        title_el = soup.find(id='article_title') or soup.find('h1')
        if title_el:
            parts.append(f"# {title_el.get_text(strip=True)}")

        # Extract overview
        if overview:
            overview_text = overview.get_text(separator='\n', strip=True)
            if overview_text:
                parts.append(f"\n{overview_text}")

        # Extract key points
        if key_points:
            kp_text = key_points.get_text(separator='\n', strip=True)
            if kp_text:
                parts.append(f"\n## Key Points\n{kp_text}")

        # Convert main content
        if MARKDOWNIFY_AVAILABLE:
            content_md = markdownify.markdownify(
                str(main_content),
                heading_style="ATX",
                strip=['script', 'style', 'nav', 'select', 'button', 'dialog']
            )
            parts.append(content_md)
        else:
            # Fallback: extract text with separator
            content_text = main_content.get_text(separator='\n', strip=True)
            parts.append(content_text)

        # Extract review date if present
        review_date = soup.find(id='review-date')
        if review_date:
            parts.append(f"\n---\nLast updated: {review_date.get_text(strip=True)}")

        return '\n\n'.join(parts)

    def _markdown_to_text(self, markdown_text: str) -> str:
        """
        Convert markdown to plain text while preserving structure.

        Args:
            markdown_text: Markdown formatted text

        Returns:
            Plain text with preserved structure
        """
        try:
            # Convert markdown to HTML
            html = markdown.markdown(markdown_text)
            # Extract plain text using BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator='\n')
            return text
        except Exception as e:
            logger.warning(
                f"Error converting markdown to text: {e}, returning raw markdown")
            return markdown_text

    def _fallback_load(self, file_path: Path) -> str:
        """
        Fallback method for simple text extraction when MarkItDown fails.

        Args:
            file_path: Path to the file

        Returns:
            Extracted text content
        """
        try:
            # Try reading as plain text with different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, read as binary and decode with errors ignored
            with open(file_path, 'rb') as file:
                return file.read().decode('utf-8', errors='ignore')

        except Exception as e:
            logger.error(f"Fallback load failed for {file_path}: {e}")
            raise

    def _detect_source_org(self, file_path: str) -> str:
        """Detect source organization from file path or name."""
        path_lower = file_path.lower()

        if 'hkch' in path_lower or 'hkch' in path_lower:
            return 'HKCH'
        elif 'sickkids' in path_lower or 'sick_kids' in path_lower:
            return 'SickKids'
        elif 'hksir' in path_lower:
            return 'HKSIR'
        elif 'cirse' in path_lower:
            return 'CIRSE'
        elif 'sir' in path_lower and 'hksir' not in path_lower:
            return 'SIR'
        else:
            return 'Unknown'

    def _detect_region(self, source_org: str) -> str:
        """
        Detect region based on source organization.

        Args:
            source_org: Source organization identifier

        Returns:
            'Hong Kong' if source is HKCH or HKSIR, 'Non-Hong Kong' otherwise
        """
        hong_kong_orgs = ['HKCH', 'HKSIR']
        if source_org in hong_kong_orgs:
            return 'Hong Kong'
        else:
            return 'Non-Hong Kong'

    def _classify_procedure_category(self, text: str, file_path: str) -> str:
        """
        Classify document into procedure categories based on content.

        Categories:
        - Venous access (PICC, CVC, central line, etc.)
        - Angiogram related (angiography, angioplasty, etc.)
        - Embolization related (embolization, embolotherapy, etc.)
        - Biopsy related (biopsy, tissue sampling, etc.)
        - Pain injection relief related (pain injection, nerve block, etc.)

        Args:
            text: Document content text
            file_path: Path to the file (for filename-based detection)

        Returns:
            Procedure category string, or 'Other' if no match found
        """
        text_lower = text.lower()
        path_lower = file_path.lower()
        combined_text = f"{text_lower} {path_lower}"

        # Define keywords for each category
        category_keywords = {
            'venous_access': [
                'picc', 'peripherally inserted central catheter', 'central venous catheter',
                'cvc', 'central line', 'venous access', 'vascular access',
                'catheter insertion', 'catheter placement', 'catheter removal',
                'tunneled catheter', 'port-a-cath', 'portacath', 'implanted port',
                'hickman', 'broviac', 'vascular catheter'
            ],
            'angiogram_related': [
                'angiogram', 'angiography', 'angioplasty', 'angiographic',
                'vascular imaging', 'arteriography', 'venography',
                'selective angiography', 'digital subtraction angiography', 'dsa',
                'ct angiography', 'cta', 'mr angiography', 'mra',
                'balloon angioplasty', 'stent placement', 'stent insertion',
                'vascular stent', 'vascular intervention'
            ],
            'embolization_related': [
                'embolization', 'embolotherapy', 'embolize', 'embolic',
                'transarterial embolization', 'tae', 'selective embolization',
                'coil embolization', 'particle embolization', 'glue embolization',
                'vascular embolization', 'arterial embolization'
            ],
            'biopsy_related': [
                'biopsy', 'tissue sampling', 'needle biopsy', 'core biopsy',
                'fine needle aspiration', 'fna', 'trucut biopsy',
                'image guided biopsy', 'percutaneous biopsy', 'tissue diagnosis'
            ],
            'pain_injection_relief_related': [
                'pain injection', 'pain relief injection', 'nerve block',
                'local anesthetic', 'pain management injection', 'steroid injection',
                'therapeutic injection', 'pain control injection', 'analgesic injection',
                'transforaminal injection', 'facet injection', 'joint injection',
                'epidural injection', 'corticosteroid injection'
            ]
        }

        # Count matches for each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in combined_text:
                    # Give more weight to filename matches
                    if keyword in path_lower:
                        score += 3
                    else:
                        score += 1
            category_scores[category] = score

        # Find category with highest score
        max_score = max(category_scores.values()) if category_scores.values() else 0

        if max_score == 0:
            return 'Other'

        # Return category with highest score
        best_category = max(category_scores, key=category_scores.get)

        # Map internal names to user-friendly names
        category_mapping = {
            'venous_access': 'Venous Access',
            'angiogram_related': 'Angiogram Related',
            'embolization_related': 'Embolization Related',
            'biopsy_related': 'Biopsy Related',
            'pain_injection_relief_related': 'Pain Injection Relief Related'
        }

        return category_mapping.get(best_category, 'Other')

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Split text into chunks. Supports three modes:
        1. whole_document=True: entire doc as one chunk
        2. semantic_chunking=True: split by markdown headings
        3. Default: fixed character-count sliding window

        Args:
            text: Text content to chunk
            metadata: Document metadata to attach to each chunk

        Returns:
            List of DocumentChunk objects
        """
        # Clean the text
        text = self._clean_text(text)

        # If whole_document mode, return entire document as single chunk
        if self.whole_document:
            # Use document_id (if available) or filename for chunk ID base
            base_id = metadata.get('document_id', metadata.get('filename', 'doc'))
            chunk_id = f"{base_id}_whole"
            logger.info(f"Using whole document mode for {metadata['filename']} ({len(text)} chars)")
            return [DocumentChunk(
                content=text,
                metadata={**metadata, "chunk_index": 0, "chunk_size": len(text), "whole_document": True},
                chunk_id=chunk_id
            )]

        # Semantic chunking: split by markdown headings
        if self.semantic_chunking:
            return self._semantic_chunk(text, metadata)

        # Legacy: fixed character-count sliding window
        return self._sliding_window_chunk(text, metadata)

    def _semantic_chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Chunk text semantically by markdown headings (##, ###).
        Sections that exceed max_chunk_size are sub-split by paragraphs.
        Lists are kept as atomic units.

        Args:
            text: Cleaned text content
            metadata: Document metadata

        Returns:
            List of DocumentChunk objects
        """
        # Split by heading patterns (## or ###)
        # We keep the heading with its content
        heading_pattern = re.compile(r'(?=^#{1,3}\s)', re.MULTILINE)
        raw_sections = heading_pattern.split(text)

        # Filter empty sections
        raw_sections = [s.strip() for s in raw_sections if s.strip()]

        chunks = []
        chunk_index = 0

        for section in raw_sections:
            # Extract section title from the first line if it's a heading
            lines = section.split('\n', 1)
            section_title = ''
            if lines[0].startswith('#'):
                section_title = lines[0].lstrip('#').strip()

            # Use document_id (if available) or filename for chunk ID base
            base_id = metadata.get('document_id', metadata.get('filename', 'doc'))

            # If section is within size limit, keep it as one chunk
            if len(section) <= self.chunk_size:
                chunk_id = f"{base_id}_chunk_{chunk_index}"
                chunks.append(DocumentChunk(
                    content=section,
                    metadata={
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_size": len(section),
                        "section_title": section_title,
                        "chunking_method": "semantic",
                    },
                    chunk_id=chunk_id
                ))
                chunk_index += 1
            else:
                # Section too large: sub-split by paragraphs
                paragraphs = re.split(r'\n\n+', section)
                current_chunk = ""

                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue

                    # If adding this paragraph would exceed size, flush current chunk
                    if current_chunk and len(current_chunk) + len(para) + 2 > self.chunk_size:
                        chunk_id = f"{base_id}_chunk_{chunk_index}"
                        chunks.append(DocumentChunk(
                            content=current_chunk.strip(),
                            metadata={
                                **metadata,
                                "chunk_index": chunk_index,
                                "chunk_size": len(current_chunk.strip()),
                                "section_title": section_title,
                                "chunking_method": "semantic_paragraph",
                            },
                            chunk_id=chunk_id
                        ))
                        chunk_index += 1
                        current_chunk = ""

                    current_chunk += para + "\n\n"

                # Flush remaining content
                if current_chunk.strip():
                    chunk_id = f"{base_id}_chunk_{chunk_index}"
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        metadata={
                            **metadata,
                            "chunk_index": chunk_index,
                            "chunk_size": len(current_chunk.strip()),
                            "section_title": section_title,
                            "chunking_method": "semantic_paragraph",
                        },
                        chunk_id=chunk_id
                    ))
                    chunk_index += 1

        if not chunks:
            # Fallback: if no headings found, treat whole text as one chunk
            base_id = metadata.get('document_id', metadata.get('filename', 'doc'))
            chunk_id = f"{base_id}_chunk_0"
            chunks.append(DocumentChunk(
                content=text,
                metadata={**metadata, "chunk_index": 0, "chunk_size": len(text), "chunking_method": "semantic_fallback"},
                chunk_id=chunk_id
            ))

        avg_size = sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0
        max_size = max(len(c.content) for c in chunks) if chunks else 0
        logger.info(f"Semantic chunking: {len(chunks)} chunks from {metadata['filename']} "
                   f"(avg: {avg_size} chars, max: {max_size} chars)")
        return chunks

    def _sliding_window_chunk(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Legacy: split text into overlapping chunks with fixed character count.

        Args:
            text: Cleaned text content
            metadata: Document metadata

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        chunk_index = 0
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size

            # If this is not the last chunk, try to break at a sentence or word boundary
            if end < text_length:
                # Look for sentence endings within the last 20% of the chunk
                search_start = max(start, end - int(self.chunk_size * 0.2))
                sentence_end = max(
                    text.rfind('. ', search_start, end),
                    text.rfind('.\n', search_start, end),
                    text.rfind('! ', search_start, end),
                    text.rfind('? ', search_start, end),
                    text.rfind('。', search_start, end),  # Chinese period
                    text.rfind('！', search_start, end),  # Chinese exclamation
                    text.rfind('？', search_start, end),  # Chinese question mark
                )

                if sentence_end > search_start:
                    end = sentence_end + 1
                else:
                    # No sentence boundary found, look for word boundary
                    word_end = text.rfind(' ', search_start, end)
                    if word_end > search_start:
                        end = word_end

            # Extract chunk
            chunk_content = text[start:end].strip()

            # Only add non-empty chunks
            if chunk_content:
                base_id = metadata.get('document_id', metadata.get('filename', 'doc'))
                chunk_id = f"{base_id}_chunk_{chunk_index}"
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    metadata={**metadata, "chunk_index": chunk_index, "chunk_size": len(chunk_content)},
                    chunk_id=chunk_id
                ))
                chunk_index += 1

            # Move start position with overlap
            if end < text_length:
                start = end - self.chunk_overlap
            else:
                break

        avg_size = sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0
        max_size = max(len(c.content) for c in chunks) if chunks else 0

        logger.info(f"Created {len(chunks)} chunks from {metadata['filename']} "
                   f"(avg: {avg_size} chars, max: {max_size} chars)")

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text, including OCR artifacts."""
        # Remove form-style underlines (e.g., ______, ------)
        text = re.sub(r'[_]{3,}', '', text)
        text = re.sub(r'[-]{5,}', '', text)
        # Remove broken Unicode replacement chars (common OCR artifacts)
        text = re.sub(r'[\ufffd]+', '', text)  # Unicode replacement char
        text = re.sub(r'\?{3,}', '', text)  # Sequences of ??? from bad encoding
        # Remove excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)  # Collapse spaces/tabs but not newlines
        # Normalize newlines: collapse 3+ blank lines into 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()

    def process_directory(self, directory_path: str,
                          file_patterns: Optional[List[str]] = None) -> List[DocumentChunk]:
        """
        Process all documents in a directory using MarkItDown.

        Args:
            directory_path: Path to directory containing documents
            file_patterns: Optional list of file patterns to match (e.g., ['*.pdf', '*.docx'])

        Returns:
            List of all DocumentChunk objects from all documents
        """
        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        all_chunks = []

        # Markdown-only mode: only process markdown files
        if self.markdown_only:
            file_patterns = ['*.md', '*.markdown']
            logger.info("Markdown-only mode: Processing only .md and .markdown files")
        # Extended file patterns - MarkItDown supports many formats
        elif file_patterns is None:
            file_patterns = [
                '*.pdf', '*.docx', '*.doc', '*.pptx', '*.ppt',  # Office documents
                '*.xlsx', '*.xls', '*.csv',  # Spreadsheets
                '*.md', '*.markdown', '*.txt',  # Text formats
                '*.html', '*.htm',  # Web formats
                '*.jpg', '*.jpeg', '*.png',  # Images (with OCR if available)
                '*.mp3', '*.wav'  # Audio (with transcription if available)
            ]

        # Collect all matching files
        files = []
        for pattern in file_patterns:
            files.extend(directory.rglob(pattern))

        # Remove duplicates
        files = list(set(files))

        logger.info(f"Found {len(files)} files to process in {directory}")
        logger.info(
            f"Using MarkItDown for unified conversion to Markdown format")

        for file_path in files:
            try:
                logger.info(f"Processing: {file_path}")
                text, metadata = self.load_document(str(file_path))

                # Classify procedure category based on content
                # We need the text content for classification
                procedure_category = self._classify_procedure_category(text, str(file_path))
                metadata["procedure_category"] = procedure_category

                chunks = self.chunk_text(text, metadata)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
