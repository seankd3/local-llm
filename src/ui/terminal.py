"""Rich terminal UI for the LLM."""
import sys
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table
from rich.text import Text


class TerminalUI:
    """Terminal-based chat interface using Rich."""

    def __init__(
        self,
        on_message: Callable[[str], str] | None = None,
        on_feedback: Callable[[str, str, bool], None] | None = None,
        on_correction: Callable[[str, str, str], None] | None = None,
    ):
        self.console = Console()
        self.on_message = on_message
        self.on_feedback = on_feedback
        self.on_correction = on_correction

        self._conversation: list[dict] = []
        self._last_response: str = ""
        self._last_prompt: str = ""

    def print_welcome(self) -> None:
        """Print welcome message."""
        self.console.print()
        self.console.print(
            Panel(
                "[bold cyan]Local LLM[/bold cyan] - Continuously Learning Assistant\n\n"
                "[dim]Commands:[/dim]\n"
                "  [green]/help[/green]    - Show all commands\n"
                "  [green]/status[/green]  - Show system status\n"
                "  [green]/ingest[/green]  - Add documents to knowledge base\n"
                "  [green]/train[/green]   - Force training run\n"
                "  [green]/quit[/green]    - Exit\n\n"
                "[dim]Feedback:[/dim]\n"
                "  [green]y[/green] - Thumbs up (after response)\n"
                "  [green]n[/green] - Thumbs down\n"
                "  [green]c[/green] - Correct last response",
                title="Welcome",
                border_style="cyan",
            )
        )
        self.console.print()

    def print_help(self) -> None:
        """Print help message."""
        table = Table(title="Commands", show_header=True, header_style="bold cyan")
        table.add_column("Command", style="green")
        table.add_column("Description")

        commands = [
            ("/help", "Show this help message"),
            ("/status", "Show training status and buffer stats"),
            ("/ingest <path>", "Ingest file or directory into knowledge base"),
            ("/train", "Force a training run with current buffer"),
            ("/reset", "Reset to base adapter (discard learning)"),
            ("/clear", "Clear conversation history"),
            ("/quit, /exit", "Exit the application"),
            ("y", "Thumbs up last response (after AI responds)"),
            ("n", "Thumbs down last response"),
            ("c", "Correct/edit last response"),
        ]

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        self.console.print(table)

    def print_status(self, status: dict) -> None:
        """Print system status."""
        table = Table(title="System Status", show_header=False)
        table.add_column("Key", style="cyan")
        table.add_column("Value")

        for key, value in status.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    table.add_row(f"  {k}", str(v))
            else:
                table.add_row(key, str(value))

        self.console.print(table)

    def print_user(self, message: str) -> None:
        """Print user message."""
        self.console.print(f"\n[bold blue]You:[/bold blue] {message}")

    def print_assistant(self, message: str, streaming: bool = False) -> None:
        """Print assistant message."""
        if streaming:
            # For streaming, we'll use Live display
            self.console.print("[bold green]Assistant:[/bold green]")
            self.console.print(Markdown(message))
        else:
            self.console.print(f"\n[bold green]Assistant:[/bold green]")
            self.console.print(Markdown(message))

    def print_assistant_streaming(self, token_generator) -> str:
        """Print assistant message with streaming."""
        self.console.print(f"\n[bold green]Assistant:[/bold green]")

        full_response = ""
        with Live(Text(""), console=self.console, refresh_per_second=10) as live:
            for token in token_generator:
                full_response += token
                # Render as markdown for better formatting
                live.update(Markdown(full_response))

        return full_response

    def print_error(self, message: str) -> None:
        """Print error message."""
        self.console.print(f"[bold red]Error:[/bold red] {message}")

    def print_info(self, message: str) -> None:
        """Print info message."""
        self.console.print(f"[dim]{message}[/dim]")

    def print_success(self, message: str) -> None:
        """Print success message."""
        self.console.print(f"[bold green]{message}[/bold green]")

    def get_input(self, prompt: str = "You") -> str:
        """Get user input."""
        try:
            return Prompt.ask(f"\n[bold blue]{prompt}[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            return "/quit"

    def get_correction(self) -> str | None:
        """Get corrected response from user."""
        self.console.print("\n[yellow]Enter corrected response (empty to cancel):[/yellow]")
        try:
            lines = []
            self.console.print("[dim](Enter blank line to finish)[/dim]")
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)

            correction = "\n".join(lines)
            return correction if correction.strip() else None
        except (KeyboardInterrupt, EOFError):
            return None

    def show_loading(self, message: str = "Thinking...") -> Status:
        """Show loading indicator."""
        return self.console.status(message, spinner="dots")

    def run_chat_loop(
        self,
        generate_fn: Callable[[list[dict]], str],
        status_fn: Callable[[], dict] | None = None,
        ingest_fn: Callable[[str], int] | None = None,
        train_fn: Callable[[], dict] | None = None,
        reset_fn: Callable[[], None] | None = None,
    ) -> None:
        """Run the main chat loop."""
        self.print_welcome()

        while True:
            user_input = self.get_input()

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.split()[0].lower()
                args = user_input[len(cmd):].strip()

                if cmd in ("/quit", "/exit", "/q"):
                    self.print_info("Goodbye!")
                    break

                elif cmd == "/help":
                    self.print_help()

                elif cmd == "/status":
                    if status_fn:
                        self.print_status(status_fn())
                    else:
                        self.print_error("Status not available")

                elif cmd == "/ingest":
                    if not args:
                        self.print_error("Usage: /ingest <path>")
                    elif ingest_fn:
                        with self.show_loading("Ingesting documents..."):
                            try:
                                count = ingest_fn(args)
                                self.print_success(f"Ingested {count} chunks")
                            except Exception as e:
                                self.print_error(str(e))
                    else:
                        self.print_error("Ingestion not available")

                elif cmd == "/train":
                    if train_fn:
                        with self.show_loading("Training..."):
                            try:
                                result = train_fn()
                                self.print_success(
                                    f"Training complete. Loss: {result.get('train_loss', 'N/A'):.4f}"
                                )
                            except Exception as e:
                                self.print_error(str(e))
                    else:
                        self.print_error("Training not available")

                elif cmd == "/reset":
                    if reset_fn:
                        reset_fn()
                        self.print_success("Reset to base adapter")
                    else:
                        self.print_error("Reset not available")

                elif cmd == "/clear":
                    self._conversation = []
                    self.print_success("Conversation cleared")

                else:
                    self.print_error(f"Unknown command: {cmd}")

                continue

            # Handle feedback shortcuts
            if user_input.lower() in ("y", "yes") and self._last_response:
                if self.on_feedback:
                    self.on_feedback(self._last_prompt, self._last_response, True)
                    self.print_info("Thanks for the feedback!")
                continue

            if user_input.lower() in ("n", "no") and self._last_response:
                if self.on_feedback:
                    self.on_feedback(self._last_prompt, self._last_response, False)
                    self.print_info("Got it. Use 'c' to provide a correction.")
                continue

            if user_input.lower() == "c" and self._last_response:
                correction = self.get_correction()
                if correction and self.on_correction:
                    self.on_correction(self._last_prompt, self._last_response, correction)
                    self.print_success("Correction recorded!")
                continue

            # Regular message
            self.print_user(user_input)
            self._last_prompt = user_input

            # Add to conversation
            self._conversation.append({"role": "user", "content": user_input})

            # Generate response
            with self.show_loading():
                try:
                    response = generate_fn(self._conversation)
                except Exception as e:
                    self.print_error(str(e))
                    self._conversation.pop()
                    continue

            self._last_response = response
            self._conversation.append({"role": "assistant", "content": response})

            self.print_assistant(response)
            self.print_info("[y]es/[n]o/[c]orrect")
