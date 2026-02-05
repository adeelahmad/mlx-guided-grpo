# Contributing to MLX Guided GRPO

Thank you for your interest in contributing to MLX Guided GRPO! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We're all here to learn and improve the project together.

## Getting Started

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- Git

### Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/mlx-guided-grpo.git
cd mlx-guided-grpo
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install in development mode with dev dependencies:
```bash
pip install -e ".[dev,wandb,sklearn]"
```

4. Install pre-commit hooks (optional but recommended):
```bash
pip install pre-commit
pre-commit install
```

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Making Changes

1. Create a new branch from `main`:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following the code style guidelines below.

3. Run formatting and checks:
```bash
black mlx_grpo/
isort mlx_grpo/
mypy mlx_grpo/
```

4. Test your changes:
```bash
# Quick validation
python -m mlx_grpo.train \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --data ./test_data \
    --train --train-type lora \
    --iters 2 --batch-size 1
```

5. Commit your changes:
```bash
git add .
git commit -m "feat: description of your change"
```

6. Push and create a pull request.

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `style:` - Code style (formatting, etc.)
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `test:` - Adding tests
- `chore:` - Maintenance tasks

## Code Style Guidelines

### Python Style

- **Formatting**: Use `black` with line length 100
- **Imports**: Use `isort` with black profile
- **Type hints**: Required for all functions
- **Docstrings**: Google style

Example:
```python
def compute_loss(
    model: nn.Module,
    inputs: mx.array,
    targets: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, dict[str, float]]:
    """Compute the training loss.

    Args:
        model: The model to compute loss for.
        inputs: Input token IDs of shape [batch, seq_len].
        targets: Target token IDs of shape [batch, seq_len].
        mask: Optional mask of shape [batch, seq_len].

    Returns:
        Tuple of (loss, metrics) where loss is a scalar and
        metrics is a dict of metric names to values.

    Raises:
        ValueError: If inputs and targets have different shapes.
    """
    ...
```

### Architecture Guidelines

We follow SOLID principles:

1. **Single Responsibility**: Each module/class should have one purpose
2. **Open/Closed**: Extend via configuration, not modification
3. **Liskov Substitution**: Subtypes must be substitutable
4. **Interface Segregation**: Prefer focused interfaces
5. **Dependency Inversion**: Depend on abstractions

### Memory Management

Always use the safe evaluation pattern:
```python
from mlx_grpo.trainer.grpo import safe_eval, safe_clear

# Evaluate arrays with crash tracing
safe_eval(loss, gradients, checkpoint="backward_pass")

# Clear cache with tracking
safe_clear(checkpoint="after_step")
```

## Adding New Features

### Adding a Reward Function

1. Create or edit a file in `mlx_grpo/trainer/`:
```python
from mlx_grpo.trainer.rewards import reward

@reward("my_new_reward", default=False)
def my_new_reward(prompts, completions, answers, types=None):
    """Short description.

    Args:
        prompts: List of prompt strings.
        completions: List of completion strings.
        answers: List of expected answers.
        types: Optional list of sample types.

    Returns:
        List of float scores in range [0, 1].
    """
    scores = []
    for completion, answer in zip(completions, answers):
        # Your logic here
        score = 1.0 if answer in completion else 0.0
        scores.append(score)
    return scores
```

2. Test it:
```bash
python -c "from mlx_grpo.trainer.rewards import get_reward; print(get_reward('my_new_reward'))"
```

### Adding CLI Arguments

1. Add to `train.py` in `build_parser()`:
```python
parser.add_argument(
    "--my-new-arg",
    type=float,
    default=0.5,
    help="Description of the argument",
)
```

2. Add to `CONFIG_DEFAULTS` dict in `train.py`.

3. Add to `GRPOTrainingArgs` dataclass in `grpo/config.py`.

4. Pass through in `train_model()` function.

### Adding Training Features

1. Identify the appropriate module:
   - Loss changes: `grpo/loss.py`
   - Generation: `grpo/generation.py`
   - Curriculum: `grpo/curriculum.py`
   - Layer control: `grpo/layers.py`

2. Follow the existing patterns in that module.

3. Export from `grpo/__init__.py` if needed.

4. Update `grpo_trainer.py` to use the new feature.

## Testing

### Manual Testing

```bash
# Basic smoke test
python -m mlx_grpo.train \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --data ./test_data \
    --train --train-type lora \
    --iters 2 --batch-size 1

# Test with curriculum learning
python -m mlx_grpo.train \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --data ./test_data \
    --train --train-type lora \
    --curriculum-enabled \
    --iters 2 --batch-size 1
```

### Import Testing

```python
# Test all imports work
from mlx_grpo.trainer import train_grpo, GRPOTrainingArgs
from mlx_grpo.trainer.rewards import list_rewards, get_reward
from mlx_grpo.trainer.grpo import (
    generate_grpo,
    grpo_loss,
    build_curriculum_prefix,
)
print("All imports successful!")
```

## Documentation

- Update README.md for user-facing changes
- Update CHANGELOG.md for all notable changes
- Add/update docstrings for new functions
- Update CLAUDE.md if architecture changes significantly

## Pull Request Process

1. Ensure all checks pass (formatting, type hints)
2. Update documentation as needed
3. Add entry to CHANGELOG.md under "Unreleased"
4. Request review from maintainers
5. Address feedback and iterate
6. Squash and merge when approved

## Questions?

Feel free to open an issue for questions or discussions about proposed changes.
