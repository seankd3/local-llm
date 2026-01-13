"""Main entry point for the local LLM."""
import argparse
import sys
from pathlib import Path

from rich.console import Console

from .config import load_config
from .inference import LLMInference
from .rag.vectordb import VectorDB
from .rag.ingest import DocumentIngester
from .learning.buffer import ExperienceBuffer
from .learning.dpo import DPOTrainer
from .learning.scheduler import TrainingScheduler
from .ui.terminal import TerminalUI


class LocalLLM:
    """Main application class that orchestrates all components."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.config = load_config(self.project_root)
        self.console = Console()

        # Components (lazy loaded)
        self._llm: LLMInference | None = None
        self._vectordb: VectorDB | None = None
        self._ingester: DocumentIngester | None = None
        self._buffer: ExperienceBuffer | None = None
        self._trainer: DPOTrainer | None = None
        self._scheduler: TrainingScheduler | None = None

    @property
    def llm(self) -> LLMInference:
        if self._llm is None:
            self._llm = LLMInference(self.config, self.project_root)
        return self._llm

    @property
    def vectordb(self) -> VectorDB:
        if self._vectordb is None:
            self._vectordb = VectorDB(self.config, self.project_root)
            self._vectordb.initialize()
        return self._vectordb

    @property
    def ingester(self) -> DocumentIngester:
        if self._ingester is None:
            self._ingester = DocumentIngester(
                self.config, self.project_root, self.vectordb
            )
        return self._ingester

    @property
    def buffer(self) -> ExperienceBuffer:
        if self._buffer is None:
            self._buffer = ExperienceBuffer(self.config, self.project_root)
        return self._buffer

    def load_model(self) -> None:
        """Load the LLM model."""
        with self.console.status("Loading model..."):
            self.llm.load_model()
            self.llm.initialize_lora()

            # Try to load latest adapter
            if self._trainer:
                adapter_path = self._trainer.get_latest_adapter_path()
                if adapter_path:
                    self.llm.load_adapter(adapter_path)
                    self.console.print(f"[dim]Loaded adapter: {adapter_path.name}[/dim]")

        trainable, total = self.llm.get_trainable_params()
        self.console.print(
            f"[dim]Model loaded. Trainable: {trainable:,} / {total:,} params[/dim]"
        )

    def generate(self, messages: list[dict], use_rag: bool = True) -> str:
        """Generate response with optional RAG context."""
        # Get the latest user message for RAG query
        user_message = messages[-1]["content"] if messages else ""

        context = None
        if use_rag and user_message:
            # Query vector database
            results = self.vectordb.query(user_message)
            if results:
                context = "\n\n---\n\n".join(
                    f"[Source: {r['metadata'].get('filename', 'unknown')}]\n{r['text']}"
                    for r in results
                )

        # Generate response
        system_prompt = (
            "You are a helpful, knowledgeable assistant. "
            "Provide clear, accurate, and concise responses. "
            "When using provided context, cite the source."
        )

        response = self.llm.chat(
            messages,
            system_prompt=system_prompt,
            context=context,
        )

        return response

    def train(self) -> dict:
        """Run DPO training."""
        if self._trainer is None:
            self._trainer = DPOTrainer(
                self.llm.model,
                self.llm.tokenizer,
                self.config,
                self.project_root,
            )

        result = self._trainer.train(self.buffer)

        # Hot-swap to new adapter
        adapter_path = self._trainer.get_latest_adapter_path()
        if adapter_path:
            self.llm.load_adapter(adapter_path)

        return result

    def ingest(self, path: str) -> int:
        """Ingest file or directory."""
        path_obj = Path(path).expanduser()

        if path_obj.is_file():
            return self.ingester.ingest_file(path_obj)
        elif path_obj.is_dir():
            total = 0
            for file_path, count in self.ingester.ingest_directory(path_obj):
                total += count
                self.console.print(f"[dim]  {file_path.name}: {count} chunks[/dim]")
            return total
        else:
            raise FileNotFoundError(f"Path not found: {path}")

    def status(self) -> dict:
        """Get system status."""
        return {
            "Model": {
                "name": self.config.model.name,
                "adapter": self.llm.current_adapter if self._llm else "not loaded",
            },
            "Knowledge Base": {
                "documents": self.vectordb.count() if self._vectordb else 0,
            },
            "Learning": self.buffer.stats(),
            "Scheduler": self._scheduler.status if self._scheduler else {"running": False},
        }

    def on_feedback(self, prompt: str, response: str, positive: bool) -> None:
        """Handle user feedback."""
        if positive:
            self.buffer.add_positive(prompt, response)
        else:
            self.buffer.add_negative(prompt, response)

    def on_correction(self, prompt: str, original: str, corrected: str) -> None:
        """Handle user correction."""
        self.buffer.add_correction(prompt, original, corrected)

    def reset_adapter(self) -> None:
        """Reset to base adapter."""
        self.llm.initialize_lora()

    def run_chat(self) -> None:
        """Run interactive chat."""
        self.load_model()

        # Initialize scheduler
        self._scheduler = TrainingScheduler(
            self.config,
            self.project_root,
            on_train=self.train,
        )
        self._scheduler.start(self.buffer)

        # Create UI
        ui = TerminalUI(
            on_feedback=self.on_feedback,
            on_correction=self.on_correction,
        )

        try:
            ui.run_chat_loop(
                generate_fn=self.generate,
                status_fn=self.status,
                ingest_fn=self.ingest,
                train_fn=self.train,
                reset_fn=self.reset_adapter,
            )
        finally:
            self._scheduler.stop()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Local continuously learning LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("path", help="File or directory to ingest")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run training")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show status")

    args = parser.parse_args()

    # Find project root
    project_root = Path(__file__).parent.parent

    app = LocalLLM(project_root)

    if args.command == "chat" or args.command is None:
        app.run_chat()

    elif args.command == "ingest":
        with Console().status("Ingesting..."):
            count = app.ingest(args.path)
        print(f"Ingested {count} chunks")

    elif args.command == "train":
        app.load_model()
        result = app.train()
        print(f"Training complete. Loss: {result.get('train_loss', 'N/A'):.4f}")

    elif args.command == "status":
        # Initialize components for status
        status = app.status()
        console = Console()
        for section, data in status.items():
            console.print(f"\n[bold cyan]{section}[/bold cyan]")
            if isinstance(data, dict):
                for k, v in data.items():
                    console.print(f"  {k}: {v}")
            else:
                console.print(f"  {data}")


if __name__ == "__main__":
    main()
