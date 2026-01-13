"""Vector database for RAG using ChromaDB."""
from pathlib import Path
from typing import Any
import uuid

import chromadb
from chromadb.config import Settings

from ..config import Config, load_config
from .embeddings import EmbeddingModel


class VectorDB:
    """ChromaDB-based vector database for knowledge storage."""

    def __init__(
        self,
        config: Config | None = None,
        project_root: Path | None = None,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config = config or load_config(self.project_root)
        self.embedding_model = embedding_model or EmbeddingModel(
            config=self.config,
            project_root=self.project_root,
        )
        self.client: chromadb.PersistentClient | None = None
        self.collection: chromadb.Collection | None = None

    def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        persist_dir = self.project_root / self.config.rag.persist_directory
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Ensure embedding model is loaded
        if self.embedding_model.model is None:
            self.embedding_model.load()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.rag.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add documents to the collection."""
        if self.collection is None:
            self.initialize()

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # Generate embeddings
        embeddings = self.embedding_model.embed(texts)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas or [{}] * len(texts),
        )

        return ids

    def query(
        self,
        query: str,
        n_results: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the collection for similar documents."""
        if self.collection is None:
            self.initialize()

        n_results = n_results or self.config.rag.top_k

        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return documents

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        if self.collection is None:
            self.initialize()
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """Get total document count."""
        if self.collection is None:
            self.initialize()
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from the collection."""
        if self.client is None:
            self.initialize()
        self.client.delete_collection(self.config.rag.collection_name)
        self.collection = self.client.create_collection(
            name=self.config.rag.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
