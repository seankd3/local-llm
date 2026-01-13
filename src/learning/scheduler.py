"""Training scheduler for continuous learning."""
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread, Event
from typing import Callable

from ..config import Config, load_config
from .buffer import ExperienceBuffer


class TrainingScheduler:
    """Manages automatic training triggers."""

    def __init__(
        self,
        config: Config | None = None,
        project_root: Path | None = None,
        on_train: Callable[[], None] | None = None,
    ):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config = config or load_config(self.project_root)
        self.on_train = on_train

        self._stop_event = Event()
        self._thread: Thread | None = None
        self._last_training: datetime | None = None
        self._last_buffer_size: int = 0

        # Training intervals
        self.min_interval = timedelta(minutes=30)  # Don't train more often than this
        self.max_interval = timedelta(hours=4)  # Train at least this often if buffer grows

    def start(self, buffer: ExperienceBuffer) -> None:
        """Start the background scheduler."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = Thread(
            target=self._scheduler_loop,
            args=(buffer,),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background scheduler."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _scheduler_loop(self, buffer: ExperienceBuffer) -> None:
        """Background loop that checks training conditions."""
        while not self._stop_event.is_set():
            if self._should_train(buffer):
                self._trigger_training(buffer)

            # Check every minute
            self._stop_event.wait(60)

    def _should_train(self, buffer: ExperienceBuffer) -> bool:
        """Check if training should be triggered."""
        if not buffer.ready_for_training:
            return False

        now = datetime.now()

        # Respect minimum interval
        if self._last_training:
            if now - self._last_training < self.min_interval:
                return False

        stats = buffer.stats()
        current_size = stats["usable_for_dpo"]

        # Trigger if buffer has grown significantly
        growth = current_size - self._last_buffer_size
        growth_threshold = self.config.training.min_buffer_size

        if growth >= growth_threshold:
            return True

        # Trigger if max interval exceeded and buffer isn't empty
        if self._last_training:
            if now - self._last_training > self.max_interval and current_size > 0:
                return True

        return False

    def _trigger_training(self, buffer: ExperienceBuffer) -> None:
        """Execute training callback."""
        if self.on_train is None:
            return

        self._last_training = datetime.now()
        self._last_buffer_size = buffer.stats()["usable_for_dpo"]

        try:
            self.on_train()
        except Exception as e:
            # Log error but don't crash scheduler
            print(f"Training error: {e}")

    def force_train(self) -> None:
        """Manually trigger training."""
        if self.on_train:
            self._last_training = datetime.now()
            self.on_train()

    @property
    def status(self) -> dict:
        """Get scheduler status."""
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "last_training": self._last_training.isoformat() if self._last_training else None,
            "last_buffer_size": self._last_buffer_size,
        }
