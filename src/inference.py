"""LLM inference with LoRA adapter support."""
import os
from pathlib import Path
from typing import Generator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from threading import Thread

from .config import Config, load_config


class LLMInference:
    """Handles model loading and inference with LoRA adapters."""

    def __init__(self, config: Config | None = None, project_root: Path | None = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.config = config or load_config(self.project_root)
        self.model = None
        self.tokenizer = None
        self.current_adapter: str | None = None
        self._setup_cache_dir()

    def _setup_cache_dir(self) -> None:
        """Set HuggingFace cache to project directory."""
        cache_dir = self.project_root / self.config.paths.base_model_cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)

    def load_model(self) -> None:
        """Load base model with 4-bit quantization."""
        model_cfg = self.config.model

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_cfg.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, model_cfg.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name,
            quantization_config=bnb_config,
            device_map=model_cfg.device_map,
            trust_remote_code=True,
        )

        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)

    def initialize_lora(self) -> None:
        """Initialize LoRA adapter on the model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        lora_cfg = self.config.lora
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.lora_dropout,
            bias=lora_cfg.bias,
            task_type=lora_cfg.task_type,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.current_adapter = "default"

    def load_adapter(self, adapter_path: Path | str) -> None:
        """Load a saved LoRA adapter."""
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        if isinstance(self.model, PeftModel):
            # Add new adapter
            adapter_name = adapter_path.name
            self.model.load_adapter(str(adapter_path), adapter_name)
            self.model.set_adapter(adapter_name)
            self.current_adapter = adapter_name
        else:
            # First time loading adapter
            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter_path),
            )
            self.current_adapter = adapter_path.name

    def save_adapter(self, name: str | None = None) -> Path:
        """Save current LoRA adapter."""
        if not isinstance(self.model, PeftModel):
            raise RuntimeError("No LoRA adapter to save.")

        adapter_dir = self.project_root / self.config.paths.adapters
        adapter_dir.mkdir(parents=True, exist_ok=True)

        save_name = name or self.current_adapter or "adapter"
        save_path = adapter_dir / save_name
        self.model.save_pretrained(str(save_path))
        return save_path

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """Generate response from prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        if stream:
            return self._generate_stream(inputs, max_new_tokens, temperature, top_p)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def _generate_stream(
        self,
        inputs: dict,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generator[str, None, None]:
        """Stream tokens as they're generated."""
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            yield token

        thread.join()

    def chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        context: str | None = None,
        **kwargs,
    ) -> str | Generator[str, None, None]:
        """Generate chat response with conversation history."""
        # Build conversation
        conversation = []

        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})

        # Add RAG context if provided
        if context:
            context_msg = f"Relevant context:\n{context}\n\nUse this context to inform your response if relevant."
            conversation.append({"role": "system", "content": context_msg})

        conversation.extend(messages)

        # Format with chat template
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        return self.generate(prompt, **kwargs)

    def get_trainable_params(self) -> tuple[int, int]:
        """Get trainable and total parameter counts."""
        if not isinstance(self.model, PeftModel):
            total = sum(p.numel() for p in self.model.parameters())
            return 0, total

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total
