"""Document ingestion for RAG."""
import hashlib
import mimetypes
from pathlib import Path
from typing import Generator

from ..config import Config, load_config
from .vectordb import VectorDB


class DocumentIngester:
    """Handles document ingestion into the vector database."""

    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx",
        ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".html", ".css", ".sh", ".bash", ".zsh",
        ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp",
        ".rb", ".php", ".sql", ".lua", ".vim",
    }

    def __init__(
        self,
        config: Config | None = None,
        project_root: Path | None = None,
        vectordb: VectorDB | None = None,
    ):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config = config or load_config(self.project_root)
        self.vectordb = vectordb or VectorDB(
            config=self.config,
            project_root=self.project_root,
        )
        self.chunk_size = self.config.rag.chunk_size
        self.chunk_overlap = self.config.rag.chunk_overlap

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence or paragraph boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sep in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
                        sent_break = text.rfind(sep, start, end)
                        if sent_break > start + self.chunk_size // 2:
                            end = sent_break + len(sep)
                            break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

    def _file_hash(self, content: str) -> str:
        """Generate hash of file content."""
        return hashlib.md5(content.encode()).hexdigest()

    def _read_file(self, path: Path) -> str | None:
        """Read file content with encoding detection."""
        encodings = ["utf-8", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                return path.read_text(encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue

        return None

    def ingest_file(self, path: Path) -> int:
        """Ingest a single file into the vector database."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        content = self._read_file(path)
        if content is None:
            raise ValueError(f"Could not read file: {path}")

        # Chunk the content
        chunks = self._chunk_text(content)

        # Prepare metadata
        file_hash = self._file_hash(content)
        metadatas = [
            {
                "source": str(path.absolute()),
                "filename": path.name,
                "file_hash": file_hash,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        # Generate IDs based on file path and chunk index
        ids = [
            f"{file_hash}_{i}"
            for i in range(len(chunks))
        ]

        # Add to vector database
        self.vectordb.add(chunks, metadatas=metadatas, ids=ids)

        return len(chunks)

    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = True,
        extensions: set[str] | None = None,
    ) -> Generator[tuple[Path, int], None, None]:
        """Ingest all supported files from a directory."""
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        extensions = extensions or self.SUPPORTED_EXTENSIONS

        pattern = "**/*" if recursive else "*"

        for path in directory.glob(pattern):
            if not path.is_file():
                continue

            if path.suffix.lower() not in extensions:
                continue

            # Skip hidden files and directories
            if any(part.startswith(".") for part in path.parts):
                continue

            try:
                chunk_count = self.ingest_file(path)
                yield path, chunk_count
            except (ValueError, UnicodeError) as e:
                # Skip files that can't be processed
                continue

    def ingest_text(
        self,
        text: str,
        source: str = "manual_input",
        metadata: dict | None = None,
    ) -> int:
        """Ingest raw text into the vector database."""
        chunks = self._chunk_text(text)

        text_hash = self._file_hash(text)
        metadatas = [
            {
                "source": source,
                "text_hash": text_hash,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {}),
            }
            for i in range(len(chunks))
        ]

        ids = [f"{text_hash}_{i}" for i in range(len(chunks))]

        self.vectordb.add(chunks, metadatas=metadatas, ids=ids)

        return len(chunks)
