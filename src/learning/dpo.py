"""DPO (Direct Preference Optimization) training."""
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset
from peft import PeftModel
from trl import DPOConfig, DPOTrainer as TRLDPOTrainer
from transformers import AutoTokenizer

from ..config import Config, load_config
from .buffer import ExperienceBuffer


class DPOTrainer:
    """Handles DPO training from experience buffer."""

    def __init__(
        self,
        model: PeftModel,
        tokenizer: AutoTokenizer,
        config: Config | None = None,
        project_root: Path | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config = config or load_config(self.project_root)
        self.training_history: list[dict] = []

    def _prepare_dataset(self, buffer: ExperienceBuffer) -> Dataset:
        """Prepare dataset from experience buffer."""
        pairs = buffer.get_dpo_pairs()

        # Filter out pairs without rejected responses
        valid_pairs = [p for p in pairs if p["rejected"]]

        if not valid_pairs:
            raise ValueError("No valid DPO pairs in buffer")

        # Format for DPO trainer
        formatted = []
        for pair in valid_pairs:
            formatted.append({
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            })

        return Dataset.from_list(formatted)

    def train(
        self,
        buffer: ExperienceBuffer,
        output_dir: str | None = None,
    ) -> dict:
        """Run DPO training on buffered experiences."""
        if not buffer.ready_for_training:
            raise ValueError(
                f"Buffer not ready. Need {self.config.training.min_buffer_size} "
                f"usable pairs, have {buffer.stats()['usable_for_dpo']}"
            )

        # Prepare output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = str(
                self.project_root
                / self.config.paths.adapters
                / f"dpo_{timestamp}"
            )

        # Prepare dataset
        dataset = self._prepare_dataset(buffer)

        # Training config
        training_config = DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            max_steps=min(self.config.training.max_steps, len(dataset) * 3),
            warmup_steps=self.config.training.warmup_steps,
            beta=self.config.training.dpo_beta,
            save_steps=self.config.training.save_steps,
            logging_steps=10,
            remove_unused_columns=False,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            optim="adamw_8bit",
        )

        # Initialize trainer
        trainer = TRLDPOTrainer(
            model=self.model,
            args=training_config,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        # Train
        train_result = trainer.train()

        # Save final adapter
        trainer.save_model()

        # Record history
        result = {
            "timestamp": datetime.now().isoformat(),
            "num_examples": len(dataset),
            "train_loss": train_result.training_loss,
            "output_dir": output_dir,
        }
        self.training_history.append(result)

        return result

    def get_latest_adapter_path(self) -> Path | None:
        """Get path to most recently trained adapter."""
        adapters_dir = self.project_root / self.config.paths.adapters

        if not adapters_dir.exists():
            return None

        # Find most recent dpo_ directory
        dpo_dirs = sorted(adapters_dir.glob("dpo_*"), reverse=True)

        if not dpo_dirs:
            return None

        return dpo_dirs[0]
