"""Experience buffer for continuous learning."""
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

from ..config import Config, load_config


@dataclass
class Experience:
    """A single learning experience."""
    prompt: str
    chosen: str  # Good response (thumbs up or original if corrected)
    rejected: str | None = None  # Bad response (if correction provided)
    timestamp: str = ""
    feedback_type: str = "positive"  # positive, negative, correction

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ExperienceBuffer:
    """Stores experiences for DPO training."""

    def __init__(self, config: Config | None = None, project_root: Path | None = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config = config or load_config(self.project_root)
        self.buffer_path = self.project_root / self.config.paths.experiences
        self.buffer_path.mkdir(parents=True, exist_ok=True)

        self._experiences: list[Experience] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing experiences from disk."""
        buffer_file = self.buffer_path / "experiences.jsonl"
        if not buffer_file.exists():
            return

        with open(buffer_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self._experiences.append(Experience(**data))

    def _save(self) -> None:
        """Save all experiences to disk."""
        buffer_file = self.buffer_path / "experiences.jsonl"
        with open(buffer_file, "w") as f:
            for exp in self._experiences:
                f.write(json.dumps(asdict(exp)) + "\n")

    def add_positive(self, prompt: str, response: str) -> None:
        """Add a positive experience (thumbs up)."""
        exp = Experience(
            prompt=prompt,
            chosen=response,
            feedback_type="positive",
        )
        self._experiences.append(exp)
        self._enforce_max_size()
        self._save()

    def add_correction(self, prompt: str, original: str, corrected: str) -> None:
        """Add a correction experience (user edited response)."""
        exp = Experience(
            prompt=prompt,
            chosen=corrected,
            rejected=original,
            feedback_type="correction",
        )
        self._experiences.append(exp)
        self._enforce_max_size()
        self._save()

    def add_negative(self, prompt: str, response: str) -> None:
        """Add a negative experience (thumbs down)."""
        # For negatives without correction, we store for potential use
        # but don't have a "chosen" response yet
        exp = Experience(
            prompt=prompt,
            chosen="",  # Empty - needs correction to be useful for DPO
            rejected=response,
            feedback_type="negative",
        )
        self._experiences.append(exp)
        self._enforce_max_size()
        self._save()

    def _enforce_max_size(self) -> None:
        """Remove oldest experiences if buffer exceeds max size."""
        max_size = self.config.training.max_buffer_size
        if len(self._experiences) > max_size:
            # Keep most recent
            self._experiences = self._experiences[-max_size:]

    def get_dpo_pairs(self) -> list[dict]:
        """Get experiences formatted for DPO training."""
        pairs = []
        for exp in self._experiences:
            # Only include experiences with both chosen and rejected
            if exp.chosen and exp.rejected:
                pairs.append({
                    "prompt": exp.prompt,
                    "chosen": exp.chosen,
                    "rejected": exp.rejected,
                })
            elif exp.chosen and exp.feedback_type == "positive":
                # For positive-only, we can create a synthetic pair
                # by using the original as both (regularization)
                pairs.append({
                    "prompt": exp.prompt,
                    "chosen": exp.chosen,
                    "rejected": "",  # Will be handled in training
                })
        return pairs

    def get_sft_examples(self) -> list[dict]:
        """Get experiences formatted for SFT (supervised fine-tuning)."""
        examples = []
        for exp in self._experiences:
            if exp.chosen:
                examples.append({
                    "prompt": exp.prompt,
                    "completion": exp.chosen,
                })
        return examples

    def sample(self, n: int) -> list[Experience]:
        """Sample n random experiences."""
        return random.sample(self._experiences, min(n, len(self._experiences)))

    def __len__(self) -> int:
        return len(self._experiences)

    def __iter__(self) -> Iterator[Experience]:
        return iter(self._experiences)

    @property
    def ready_for_training(self) -> bool:
        """Check if buffer has enough experiences for training."""
        min_size = self.config.training.min_buffer_size
        # Count usable pairs (those with corrections)
        usable = sum(1 for e in self._experiences if e.chosen and e.rejected)
        return usable >= min_size

    def clear(self) -> None:
        """Clear all experiences."""
        self._experiences = []
        self._save()

    def stats(self) -> dict:
        """Get buffer statistics."""
        positive = sum(1 for e in self._experiences if e.feedback_type == "positive")
        negative = sum(1 for e in self._experiences if e.feedback_type == "negative")
        corrections = sum(1 for e in self._experiences if e.feedback_type == "correction")
        usable_pairs = sum(1 for e in self._experiences if e.chosen and e.rejected)

        return {
            "total": len(self._experiences),
            "positive": positive,
            "negative": negative,
            "corrections": corrections,
            "usable_for_dpo": usable_pairs,
            "ready_for_training": self.ready_for_training,
        }
