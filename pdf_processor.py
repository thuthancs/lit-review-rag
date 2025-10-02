"""
PDF extraction and chunking utilities with enhanced features
Supports both PyPDF2 and pdfplumber for robust extraction
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

# Try importing both PDF libraries
try:
    import PyPDF2

    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Warning: PyPDF2 not installed. Install with: pip install PyPDF2")

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("Warning: pdfplumber not installed. Install with: pip install pdfplumber")

from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Structured metadata for a research paper"""

    title: str
    authors: str
    year: int
    abstract: str
    filename: str


class PDFProcessor:
    """Handles PDF extraction, metadata extraction, and chunking"""

    def __init__(self, chunk_size: int = 250, overlap_size: int = 50):
        """
        Initialize PDF processor

        Args:
            chunk_size: Number of words per chunk
            overlap_size: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def extract_text_pypdf2(self, pdf_path: str) -> Tuple[str, bool]:
        """
        Extract text using PyPDF2

        Returns:
            Tuple of (extracted_text, success_flag)
        """
        if not PYPDF2_AVAILABLE:
            return "", False

        try:
            text = ""
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    logger.warning(f"PDF is encrypted: {pdf_path}")
                    try:
                        pdf_reader.decrypt("")
                    except:
                        return "", False

                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        continue

            return text, len(text.strip()) > 0

        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return "", False

    def extract_text_pdfplumber(self, pdf_path: str) -> Tuple[str, bool]:
        """
        Extract text using pdfplumber (often better for complex PDFs)

        Returns:
            Tuple of (extracted_text, success_flag)
        """
        if not PDFPLUMBER_AVAILABLE:
            return "", False

        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        continue

            return text, len(text.strip()) > 0

        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
            return "", False

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF with fallback mechanism
        Tries pdfplumber first, then PyPDF2

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        # Try pdfplumber first (usually better)
        text, success = self.extract_text_pdfplumber(pdf_path)
        if success:
            logger.info(f"✓ Extracted text using pdfplumber: {pdf_path}")
            return text

        # Fall back to PyPDF2
        text, success = self.extract_text_pypdf2(pdf_path)
        if success:
            logger.info(f"✓ Extracted text using PyPDF2: {pdf_path}")
            return text

        # Both methods failed
        logger.error(f"✗ Failed to extract text from: {pdf_path}")
        raise ValueError(f"Could not extract text from PDF: {pdf_path}")

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove multiple spaces and normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove common PDF artifacts
        text = re.sub(r"\x00", "", text)  # Remove null characters
        text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII

        # Remove excessive punctuation
        text = re.sub(r"\.{3,}", "...", text)
        text = re.sub(r"-{2,}", "--", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def extract_metadata(self, text: str, filename: str) -> PaperMetadata:
        """
        Extract metadata from paper text
        Uses heuristics to identify title, authors, abstract

        Args:
            text: Full text of paper
            filename: PDF filename

        Returns:
            PaperMetadata object
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Extract title (usually first substantial line)
        title = "Unknown Title"
        for line in lines[:10]:  # Check first 10 lines
            if len(line.split()) > 3 and len(line) < 200:
                title = line
                break

        # Extract authors (look for common patterns)
        authors = "Unknown"
        author_patterns = [
            r"(?:by|authors?:?)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+){0,5})",
        ]

        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            for pattern in author_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    authors = match.group(1)
                    break
            if authors != "Unknown":
                break

        # Extract year
        year = 2024  # Default
        year_pattern = r"\b(19|20)\d{2}\b"
        year_matches = re.findall(
            year_pattern, text[:2000]
        )  # Search in first 2000 chars
        if year_matches:
            potential_years = [int(y) for y in year_matches if 1990 <= int(y) <= 2025]
            if potential_years:
                year = potential_years[0]

        # Extract abstract
        abstract = ""
        abstract_patterns = [
            r"abstract[:\s]+(.*?)(?:\n\n|introduction|keywords)",
            r"summary[:\s]+(.*?)(?:\n\n|introduction)",
        ]

        for pattern in abstract_patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                abstract = " ".join(abstract.split())[:500]  # Limit to 500 chars
                break

        return PaperMetadata(
            title=title[:200],  # Limit title length
            authors=authors[:200],
            year=year,
            abstract=abstract,
            filename=filename,
        )

    def chunk_text_sliding_window(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using sliding window

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []

        # Calculate step size (chunk_size - overlap)
        step_size = self.chunk_size - self.overlap_size

        # Sliding window chunking
        for i in range(0, len(words), step_size):
            chunk_words = words[i : i + self.chunk_size]

            # Only keep chunks with reasonable length
            if len(chunk_words) >= 20:  # Minimum 20 words
                chunk = " ".join(chunk_words)
                chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks (size={self.chunk_size}, overlap={self.overlap_size})"
        )
        return chunks

    def detect_section_type(
        self, chunk: str, chunk_index: int, total_chunks: int
    ) -> str:
        """
        Attempt to classify chunk by section type

        Args:
            chunk: Text chunk
            chunk_index: Index of chunk
            total_chunks: Total number of chunks

        Returns:
            Section type string
        """
        chunk_lower = chunk.lower()

        # Check for section headers
        if chunk_index == 0 or chunk_index < total_chunks * 0.1:
            if "abstract" in chunk_lower[:100]:
                return "abstract"
            return "introduction"

        if "method" in chunk_lower[:100] or "methodology" in chunk_lower[:100]:
            return "methods"

        if "result" in chunk_lower[:100] or "finding" in chunk_lower[:100]:
            return "results"

        if "discussion" in chunk_lower[:100] or "conclusion" in chunk_lower[:100]:
            return "discussion"

        if "reference" in chunk_lower[:100] or "bibliography" in chunk_lower[:100]:
            return "references"

        # Default classification based on position
        if chunk_index < total_chunks * 0.3:
            return "introduction"
        elif chunk_index < total_chunks * 0.7:
            return "body"
        else:
            return "conclusion"

    def process_paper(
        self, pdf_path: str, manual_metadata: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Complete pipeline: extract, chunk, and prepare for Weaviate

        Args:
            pdf_path: Path to PDF file
            manual_metadata: Optional dict with title, authors, year

        Returns:
            List of chunk objects ready for Weaviate insertion
        """
        try:
            # Extract text
            logger.info(f"Processing: {pdf_path}")
            raw_text = self.extract_text_from_pdf(pdf_path)

            if not raw_text or len(raw_text.strip()) < 100:
                raise ValueError(f"Insufficient text extracted from {pdf_path}")

            # Clean text
            cleaned_text = self.clean_text(raw_text)
            logger.info(f"Extracted {len(cleaned_text)} characters")

            # Extract or use manual metadata
            if manual_metadata:
                metadata = PaperMetadata(
                    title=manual_metadata.get("title", "Unknown"),
                    authors=manual_metadata.get("authors", "Unknown"),
                    year=int(manual_metadata.get("year", 2024)),
                    abstract=manual_metadata.get("abstract", ""),
                    filename=os.path.basename(pdf_path),
                )
            else:
                metadata = self.extract_metadata(
                    cleaned_text, os.path.basename(pdf_path)
                )

            logger.info(f"Metadata - Title: {metadata.title[:50]}...")
            logger.info(f"Metadata - Authors: {metadata.authors}")
            logger.info(f"Metadata - Year: {metadata.year}")

            # Chunk the text
            chunks = self.chunk_text_sliding_window(cleaned_text)

            # Prepare data objects for Weaviate
            chunk_objects = []
            for idx, chunk in enumerate(chunks):
                section_type = self.detect_section_type(chunk, idx, len(chunks))

                chunk_obj = {
                    "chunk": chunk,
                    "paper_title": metadata.title,
                    "authors": metadata.authors,
                    "year": metadata.year,
                    "chunk_index": idx,
                    "section_type": section_type,
                }
                chunk_objects.append(chunk_obj)

            logger.info(
                f"✓ Successfully processed {pdf_path}: {len(chunk_objects)} chunks"
            )
            return chunk_objects

        except Exception as e:
            logger.error(f"✗ Failed to process {pdf_path}: {e}")
            raise

    def process_directory(
        self, directory_path: str, metadata_map: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Process all PDFs in a directory

        Args:
            directory_path: Path to directory containing PDFs
            metadata_map: Optional dict mapping filenames to metadata dicts

        Returns:
            Dict mapping filenames to their chunk objects
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")

        pdf_files = list(directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")

        results = {}
        errors = []

        for pdf_file in pdf_files:
            filename = pdf_file.name
            try:
                # Get manual metadata if provided
                manual_meta = metadata_map.get(filename) if metadata_map else None

                # Process the paper
                chunks = self.process_paper(str(pdf_file), manual_meta)
                results[filename] = chunks

            except Exception as e:
                logger.error(f"✗ Error processing {filename}: {e}")
                errors.append((filename, str(e)))
                continue

        # Summary
        total_chunks = sum(len(chunks) for chunks in results.values())
        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY:")
        logger.info(f"  Successfully processed: {len(results)}/{len(pdf_files)} papers")
        logger.info(f"  Total chunks created: {total_chunks}")
        logger.info(f"  Failed: {len(errors)} papers")

        if errors:
            logger.info(f"\nErrors:")
            for filename, error in errors:
                logger.info(f"  - {filename}: {error}")

        logger.info(f"{'='*60}\n")

        return results


# Convenience functions for backward compatibility
def extract_text_from_pdf(pdf_path: str) -> str:
    """Convenience function for simple text extraction"""
    processor = PDFProcessor()
    return processor.extract_text_from_pdf(pdf_path)


def process_paper(pdf_path: str, paper_metadata: Dict[str, str] = None) -> List[Dict]:
    """Convenience function for processing a single paper"""
    processor = PDFProcessor(
        chunk_size=config.CHUNK_SIZE, overlap_size=config.OVERLAP_SIZE
    )
    return processor.process_paper(pdf_path, paper_metadata)


# Main execution for testing
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("PDF PROCESSOR TEST")
    print("=" * 60)

    # Check library availability
    print(f"\nLibrary Status:")
    print(f"  PyPDF2: {'✓ Available' if PYPDF2_AVAILABLE else '✗ Not installed'}")
    print(
        f"  pdfplumber: {'✓ Available' if PDFPLUMBER_AVAILABLE else '✗ Not installed'}"
    )

    if not (PYPDF2_AVAILABLE or PDFPLUMBER_AVAILABLE):
        print("\n⚠ Warning: No PDF libraries installed!")
        print("Install with: pip install PyPDF2 pdfplumber")
        sys.exit(1)

    # Test with single file
    print("\n" + "=" * 60)
    print("TEST 1: Single PDF Processing")
    print("=" * 60)

    test_pdf = "sample_paper.pdf"
    if os.path.exists(test_pdf):
        try:
            processor = PDFProcessor(chunk_size=250, overlap_size=50)

            # Test with manual metadata
            metadata = {
                "title": "Sample Research Paper on Machine Learning",
                "authors": "John Doe, Jane Smith",
                "year": "2023",
            }

            chunks = processor.process_paper(test_pdf, metadata)

            print(f"\n✓ Success! Generated {len(chunks)} chunks")
            print(f"\nSample chunk:")
            print(f"  Title: {chunks[0]['paper_title']}")
            print(f"  Authors: {chunks[0]['authors']}")
            print(f"  Year: {chunks[0]['year']}")
            print(f"  Section: {chunks[0]['section_type']}")
            print(f"  Text preview: {chunks[0]['chunk'][:200]}...")

        except Exception as e:
            print(f"\n✗ Error: {e}")
    else:
        print(f"  ℹ Test file '{test_pdf}' not found. Skipping single file test.")

    # Test with directory
    print("\n" + "=" * 60)
    print("TEST 2: Directory Processing")
    print("=" * 60)

    test_dir = "papers"
    if os.path.exists(test_dir):
        try:
            processor = PDFProcessor(chunk_size=250, overlap_size=50)

            # Optional: provide metadata for specific files
            metadata_map = {
                "paper1.pdf": {
                    "title": "First Paper Title",
                    "authors": "Author One",
                    "year": "2023",
                },
                # Add more mappings as needed
            }

            results = processor.process_directory(test_dir, metadata_map)

            print(f"\nProcessed {len(results)} papers successfully")

        except Exception as e:
            print(f"\n✗ Error: {e}")
    else:
        print(f"  ℹ Test directory '{test_dir}' not found. Skipping directory test.")
        print(f"  Create a '{test_dir}/' directory and add PDF files to test.")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
