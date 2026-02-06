<p align="center">
  <img src="https://img.shields.io/badge/🍎_Apple_Silicon-Optimized-black?style=for-the-badge" alt="Apple Silicon"/>
  <img src="https://img.shields.io/badge/MLX-Native-orange?style=for-the-badge" alt="MLX Native"/>
  <img src="https://img.shields.io/badge/GRPO-Training-blue?style=for-the-badge" alt="GRPO"/>
</p>

<h1 align="center">🧠 MLX Guided GRPO</h1>

<p align="center">
  <strong>Train reasoning models on your Mac. No cloud needed.</strong>
</p>

<p align="center">
  The first production-ready GRPO training framework for Apple Silicon.<br/>
  Fine-tune LLMs to <em>think step-by-step</em> using your M1/M2/M3/M4 Mac.
</p>

<p align="center">
  <a href="https://github.com/adeelahmad/mlx-guided-grpo/stargazers"><img src="https://img.shields.io/github/stars/adeelahmad/mlx-guided-grpo?style=social" alt="Stars"/></a>
  <a href="https://github.com/adeelahmad/mlx-guided-grpo/network/members"><img src="https://img.shields.io/github/forks/adeelahmad/mlx-guided-grpo?style=social" alt="Forks"/></a>
  <a href="https://github.com/adeelahmad/mlx-guided-grpo/issues"><img src="https://img.shields.io/github/issues/adeelahmad/mlx-guided-grpo" alt="Issues"/></a>
  <a href="https://github.com/adeelahmad/mlx-guided-grpo/blob/main/LICENSE"><img src="https://img.shields.io/github/license/adeelahmad/mlx-guided-grpo" alt="License"/></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-why-guided-grpo">Why Guided GRPO</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-examples">Examples</a> •
  <a href="#-documentation">Docs</a>
</p>

---

## 🎯 Train Your Own Reasoning Model in 5 Minutes

```bash
# Install
pip install mlx-guided-grpo

# Train (yes, it's this simple)
mlx-grpo --model mlx-community/Qwen2.5-3B-Instruct-4bit \
         --data ./your_data.jsonl \
         --train --train-type lora \
         --curriculum-enabled
```

**That's it.** Your Mac is now training a reasoning model with curriculum learning.

---

## 🤔 Why Guided GRPO?

<table>
<tr>
<td width="50%">

### The Problem

Training reasoning models (like DeepSeek-R1, o1) requires:
- ❌ Expensive cloud GPUs ($$$)
- ❌ Complex distributed setups
- ❌ NVIDIA-only frameworks
- ❌ Weeks of engineering

**Most developers can't train reasoning models.**

</td>
<td width="50%">

### The Solution

MLX Guided GRPO gives you:
- ✅ **Train on your Mac** - M1/M2/M3/M4
- ✅ **One command** - No config hell
- ✅ **Curriculum learning** - Progressive difficulty
- ✅ **Production ready** - Crash recovery, logging

**Train reasoning models on consumer hardware.**

</td>
</tr>
</table>

---

## ✨ Features

<table>
<tr>
<td>

### 🎓 Curriculum Learning
Gradually reduce scaffolding so models learn to think independently. Start with 100% guidance, end with 0%.

</td>
<td>

### 🔄 Two-Phase Generation
Automatic recovery for incomplete `<think>` outputs. Never lose a training sample.

</td>
</tr>
<tr>
<td>

### 🎯 Smart Token Masking
Only train on tokens the model generated. Scaffolded tokens are properly masked from loss.

</td>
<td>

### ⚡ Apple Silicon Native
Built on MLX for maximum Metal GPU utilization. 2-3x faster than PyTorch on Mac.

</td>
</tr>
<tr>
<td>

### 🧠 Conditional Gradient Scaling
Train different layers for thinking vs answering. Fine-grained control over what the model learns.

</td>
<td>

### 💾 Crash Recovery
Automatic checkpointing and resume. Metal GPU crashes? Training continues.

</td>
</tr>
</table>

### Full Feature List

- **Training**: GRPO, DR-GRPO, BNPO loss variants
- **Adapters**: LoRA, DoRA, Full fine-tuning
- **Type System**: Extensible type-aware rewards for tool calling, MCQ, and general Q&A ([docs](TYPE_SYSTEM.md))
- **Memory**: Gradient checkpointing, cache management
- **Rewards**: Type-dispatched rewards, custom reward functions
- **Logging**: WandB integration, rollout logging
- **Monitoring**: Threshold-based early stopping

---

## 📊 Benchmarks

| Model | Hardware | Tokens/sec | Memory |
|-------|----------|------------|--------|
| Qwen2.5-3B-4bit | M3 Max 64GB | ~150 | 12GB |
| Qwen2.5-7B-4bit | M3 Max 64GB | ~80 | 24GB |
| Llama-3.2-3B-4bit | M2 Pro 32GB | ~120 | 10GB |

*GRPO training with group_size=4, batch_size=2*

---

## 🚀 Installation

### From PyPI (Recommended)

```bash
pip install mlx-guided-grpo
```

### From Source

```bash
git clone https://github.com/adeelahmad/mlx-guided-grpo.git
cd mlx-guided-grpo
pip install -e ".[all]"
```

### Requirements

- macOS 13.5+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- 16GB+ RAM recommended

---

## 🏃 Quick Start

### 1. Prepare Your Data

Create a JSONL file with prompts and reasoning traces:

```json
{"prompt": "What is 15 * 7?", "answer": "<think>\nI need to multiply 15 by 7.\n15 * 7 = 105\n</think>\n\n\\boxed{105}"}
{"prompt": "Solve: 2x + 5 = 13", "answer": "<think>\nSubtract 5 from both sides:\n2x = 8\nDivide by 2:\nx = 4\n</think>\n\n\\boxed{4}"}
```

### 2. Train Your Model

```bash
mlx-grpo \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --data ./math_data.jsonl \
    --train \
    --train-type lora \
    --iters 1000 \
    --batch-size 2 \
    --group-size 4 \
    --curriculum-enabled \
    --adapter-path ./my-reasoning-model
```

### 3. Use Your Model

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-4bit",
                        adapter_path="./my-reasoning-model")

prompt = "What is 23 * 17?"
response = generate(model, tokenizer, prompt=prompt, max_tokens=500)
print(response)
# <think>
# I need to multiply 23 by 17...
# </think>
# \boxed{391}
```

---

## 📖 Examples

### Basic GRPO Training

```bash
mlx-grpo \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --data ./data \
    --train --train-type lora \
    --group-size 4 \
    --learning-rate 1e-5
```

### Curriculum Learning (Recommended for Reasoning)

```bash
mlx-grpo \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --data ./reasoning_data \
    --train --train-type lora \
    --curriculum-enabled \
    --curriculum-start-ratio 1.0 \
    --curriculum-end-ratio 0.0 \
    --curriculum-warmup-iters 100 \
    --curriculum-taper-iters 500 \
    --enforce-thinking
```

### With WandB Logging

```bash
mlx-grpo \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --data ./data \
    --train --train-type lora \
    --wandb my-experiment \
    --log-rollouts \
    --log-rollouts-to-wandb
```

### Advanced: Dual-Gradient Mode (CGS)

```bash
mlx-grpo \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --data ./data \
    --train --train-type lora \
    --thinking-layers "0-15" \
    --answer-layers "16-31" \
    --thinking-gradient-weight 0.5 \
    --answer-gradient-weight 1.0
```

---

## 🔧 Key Concepts

### Curriculum Learning

Progressive scaffolding teaches models to reason independently:

```
Iteration 0-100:   [████████████] 100% scaffolding (model learns format)
Iteration 100-400: [████████░░░░]  66% scaffolding (gradual reduction)
Iteration 400-700: [████░░░░░░░░]  33% scaffolding (increasing independence)
Iteration 700+:    [░░░░░░░░░░░░]   0% scaffolding (full independence)
```

### Smart Token Masking

Only train on what the model actually generated:

```
[PROMPT] [SCAFFOLD PREFIX] [MODEL GENERATION]
   ↓           ↓                  ↓
 masked      masked         LOSS COMPUTED
```

This prevents the model from getting "free credit" for scaffolded tokens.

### Two-Phase Generation

Automatic recovery for incomplete structured outputs:

```
Phase 1: Model generates → "<think>Let me solve this... 2+2="
         (Incomplete! Missing </think>)

Phase 2: Inject "</think>\n\boxed{" → Continue generation → "4}"
         (Complete! Injected tokens masked from loss)
```

---

## 📚 Documentation

| Topic | Link |
|-------|------|
| Full CLI Reference | [docs/cli.md](docs/cli.md) |
| Training Arguments | [docs/arguments.md](docs/arguments.md) |
| Custom Rewards | [docs/rewards.md](docs/rewards.md) |
| Type System | [TYPE_SYSTEM.md](TYPE_SYSTEM.md) |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| API Reference | [docs/api.md](docs/api.md) |

---

## 🆚 Comparison

| Feature | MLX Guided GRPO | TRL (HuggingFace) | OpenRLHF |
|---------|-----------------|-------------------|----------|
| Apple Silicon Native | ✅ | ❌ | ❌ |
| Curriculum Learning | ✅ | ❌ | ❌ |
| Scaffold Token Masking | ✅ | ❌ | ❌ |
| Two-Phase Generation | ✅ | ❌ | ❌ |
| Single GPU Training | ✅ | ✅ | ⚠️ |
| Consumer Hardware | ✅ | ⚠️ | ❌ |
| One-Command Training | ✅ | ❌ | ❌ |

---

## 🛠️ Troubleshooting

<details>
<summary><strong>Out of Memory?</strong></summary>

```bash
# Reduce memory usage
mlx-grpo ... \
    --grad-checkpoint \
    --batch-size 1 \
    --group-size 2 \
    --max-completion-length 256
```

</details>

<details>
<summary><strong>Metal GPU Crash?</strong></summary>

Training auto-saves checkpoints. Just resume:

```bash
mlx-grpo ... --resume
```

</details>

<details>
<summary><strong>Slow Training?</strong></summary>

```bash
# Use quantized model
--model mlx-community/Qwen2.5-3B-Instruct-4bit

# Reduce group size
--group-size 2
```

</details>

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Setup development environment
git clone https://github.com/adeelahmad/mlx-guided-grpo.git
cd mlx-guided-grpo
pip install -e ".[dev]"

# Run formatting
black mlx_grpo/
isort mlx_grpo/
```

---

## 📜 Citation

If you use MLX Guided GRPO in your research, please cite:

```bibtex
@software{mlx_guided_grpo,
  author = {Ahmad, Adeel},
  title = {MLX Guided GRPO: Reasoning Model Training for Apple Silicon},
  year = {2024},
  url = {https://github.com/adeelahmad/mlx-guided-grpo}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-examples) - MLX language model utilities
- [DeepSeek](https://github.com/deepseek-ai) - GRPO algorithm
- [Qwen](https://github.com/QwenLM) - Excellent base models

---

<p align="center">
  <strong>Built with ❤️ for the Mac ML community</strong>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/adeelahmadch">LinkedIn</a> •
  <a href="https://github.com/adeelahmad">GitHub</a> •
  <a href="mailto:adeel@adeelahmad.net">Contact</a>
</p>

<p align="center">
  <sub>If this project helps you, please ⭐ star the repo!</sub>
</p>
