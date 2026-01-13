"""Configuration management."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    device_map: str = "auto"


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 100
    warmup_steps: int = 10
    min_buffer_size: int = 10
    max_buffer_size: int = 1000
    dpo_beta: float = 0.1
    save_steps: int = 50


@dataclass
class RAGConfig:
    collection_name: str = "knowledge_base"
    persist_directory: str = "data/chromadb"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class PathsConfig:
    experiences: str = "data/experiences"
    documents: str = "data/documents"
    adapters: str = "models/adapters"
    base_model_cache: str = "models/base"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(
            model=ModelConfig(**data.get("model", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            training=TrainingConfig(**data.get("training", {})),
            rag=RAGConfig(**data.get("rag", {})),
            paths=PathsConfig(**data.get("paths", {})),
        )

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "training": self.training.__dict__,
            "rag": self.rag.__dict__,
            "paths": self.paths.__dict__,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


def load_config(project_root: Path | None = None) -> Config:
    """Load configuration from project root."""
    if project_root is None:
        project_root = Path(__file__).parent.parent

    config_path = project_root / "config.yaml"
    return Config.from_yaml(config_path)
