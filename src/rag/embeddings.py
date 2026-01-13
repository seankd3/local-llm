"""Embedding model for RAG."""
from pathlib import Path

from sentence_transformers import SentenceTransformer

from ..config import Config, load_config


class EmbeddingModel:
    """Handles text embeddings for RAG retrieval."""

    def __init__(self, config: Config | None = None, project_root: Path | None = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config = config or load_config(self.project_root)
        self.model: SentenceTransformer | None = None

    def load(self) -> None:
        """Load the embedding model."""
        cache_dir = self.project_root / self.config.paths.base_model_cache / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(
            self.config.rag.embedding_model,
            cache_folder=str(cache_dir),
        )

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """Generate embeddings for text(s)."""
        if self.model is None:
            self.load()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query."""
        return self.embed(query)[0]

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self.model is None:
            self.load()
        return self.model.get_sentence_embedding_dimension()
