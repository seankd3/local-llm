# Local Continuously Learning LLM

A local LLM that learns from your feedback over time using DPO (Direct Preference Optimization).

## Features

- **Continuous Learning**: Learns from thumbs up/down feedback and corrections
- **RAG Knowledge Base**: Ingest documents to expand the LLM's knowledge
- **Local & Private**: Everything runs on your machine, no data sent to cloud
- **Memory Efficient**: Uses 4-bit quantization and LoRA to fit in 8GB VRAM

## Quick Start

```bash
# Activate environment and start chat
./run.sh

# Or use specific commands
./run.sh chat      # Start interactive chat
./run.sh ingest ~/Documents  # Add documents to knowledge base
./run.sh status    # Show system status
./run.sh train     # Force training run
```

## Chat Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/status` | Show training status and buffer stats |
| `/ingest <path>` | Ingest file or directory |
| `/train` | Force a training run |
| `/reset` | Reset to base adapter |
| `/clear` | Clear conversation history |
| `/quit` | Exit |

## Feedback

After each AI response, you can provide feedback:

- `y` - Thumbs up (response was good)
- `n` - Thumbs down (response was bad)
- `c` - Correct (provide an edited version)

Corrections are the most valuable for learning, as they provide both positive and negative examples.

## How Learning Works

1. **Feedback Collection**: Your feedback is stored in an experience buffer
2. **DPO Training**: When enough corrections accumulate (default: 10), training can be triggered
3. **Adapter Update**: LoRA adapter weights are updated, model improves
4. **Hot Swap**: New adapter is loaded without restart

The scheduler automatically triggers training when:
- Buffer has enough usable correction pairs (≥10)
- At least 30 minutes since last training
- Or buffer has grown significantly

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"  # Base model
  load_in_4bit: true                  # Quantization

lora:
  r: 16                               # LoRA rank
  lora_alpha: 32                      # LoRA alpha

training:
  min_buffer_size: 10                 # Min corrections before training
  learning_rate: 0.0002               # DPO learning rate
```

## Architecture

```
├── src/
│   ├── main.py          # Entry point
│   ├── inference.py     # Model loading & generation
│   ├── rag/             # RAG system (ChromaDB)
│   ├── learning/        # DPO training & experience buffer
│   └── ui/              # Terminal interface
├── data/
│   ├── chromadb/        # Vector database
│   └── experiences/     # Training examples
└── models/
    └── adapters/        # Saved LoRA checkpoints
```

## Requirements

- NVIDIA GPU with 8GB+ VRAM
- Python 3.11+
- ~10GB disk space for model + dependencies
