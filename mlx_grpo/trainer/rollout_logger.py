"""
Rollout Logger Module
=====================

Comprehensive logging for GRPO training rollouts:
- Per-generation JSONL logging (one line per completion)
- Per-iteration CSV logging (aggregated metrics)
- WandB integration (tables + artifacts)

Usage:
    from .rollout_logger import RolloutLogger, RolloutLoggerConfig

    logger = RolloutLogger(config=RolloutLoggerConfig(enabled=True))
    logger.log_rollout(iteration=100, prompts=..., completions=..., ...)
    logger.log_iteration(iteration=100, metrics=..., loss=...)
    logger.close()
"""

import csv
import json
import logging
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class RolloutLoggerConfig:
    """Configuration for rollout logging."""

    enabled: bool = True
    output_dir: Optional[str] = None
    log_every_n_steps: int = 1
    jsonl_filename: str = "rollouts.jsonl"
    csv_filename: str = "metrics.csv"
    log_to_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_run_id: Optional[str] = None  # For resume - use same run ID
    resume_from_iteration: int = 0  # Non-zero if resuming
    max_completion_preview_chars: int = 500
    log_full_completions: bool = True


@dataclass
class RolloutEntry:
    """Single rollout entry (one completion)."""

    iteration: int
    update: int
    group_index: int
    prompt_index: int
    prompt_full: str
    prompt_text: str
    completion: str
    answer_expected: str
    rewards: Dict[str, float]
    total_reward: float
    advantage: float
    cross_sampled: bool = False
    cross_sample_source_idx: Optional[int] = None
    two_phase_recovered: bool = False  # True if completion was recovered via two-phase generation
    scaffold_ratio: float = (
        0.0  # Curriculum scaffold level (0.0 = no scaffold, 1.0 = full scaffold)
    )
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    answer_tokens: int = 0
    timestamp: str = ""
    type_info: Optional[str] = None
    reward_details: Optional[Dict[str, Any]] = (
        None  # Detailed reward breakdown (e.g., exam reward components)
    )


@dataclass
class IterationMetrics:
    """Aggregated metrics for one iteration."""

    iteration: int
    update: int
    timestamp: str
    loss: float
    learning_rate: float
    memory_gb: float
    tokens_per_sec: float
    iterations_per_sec: float
    reward_mean: float
    reward_max: float
    reward_min: float
    reward_std: float
    kl_mean: float
    kl_max: float = 0.0
    kl_min: float = 0.0
    kl_std: float = 0.0
    reward_functions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cross_sampled_count: int = 0
    cross_sampled_ratio: float = 0.0
    cross_sampled_reward_mean: float = 0.0
    non_cross_sampled_reward_mean: float = 0.0
    avg_completion_tokens: float = 0.0
    avg_thinking_tokens: float = 0.0
    avg_answer_tokens: float = 0.0
    max_completion_tokens: int = 0
    min_completion_tokens: int = 0
    hit_max_tokens_ratio: float = 0.0
    clip_ratio_low: float = 0.0
    clip_ratio_high: float = 0.0
    clip_ratio_total: float = 0.0


def extract_thinking_length(text: str) -> int:
    """Extract word count of thinking section."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return len(match.group(1).split())
    return 0


def extract_answer_length(text: str) -> int:
    """Extract word count of answer section."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return len(match.group(1).split())
    return 0


class RolloutLogger:
    """Comprehensive logger for GRPO training rollouts."""

    def __init__(
        self,
        config: RolloutLoggerConfig,
        adapter_file: Optional[str] = None,
    ):
        self.config = config
        self.adapter_file = adapter_file

        if config.output_dir:
            self.output_dir = Path(config.output_dir)
        elif adapter_file:
            self.output_dir = Path(adapter_file).parent
        else:
            self.output_dir = Path(".")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_file = None
        self._csv_file = None
        self._csv_writer = None
        self._csv_header_written = False
        self._wandb_run = None
        self._wandb_table = None
        self._total_rollouts = 0
        self._total_iterations = 0
        self._rollout_buffer = []

        # Curriculum progress tracking
        self._curriculum_history: Dict[float, List[Tuple[int, float]]] = (
            {}
        )  # scaffold_ratio -> [(iter, reward), ...]
        self._curriculum_file = None

        if config.enabled:
            self._init_files()
            if config.log_to_wandb:
                self._init_wandb()

    def _init_files(self):
        """Initialize file handles."""
        jsonl_path = self.output_dir / self.config.jsonl_filename
        csv_path = self.output_dir / self.config.csv_filename
        curriculum_path = self.output_dir / "curriculum_progress.jsonl"

        self._jsonl_file = open(jsonl_path, "a", encoding="utf-8")
        self._csv_file = open(csv_path, "a", newline="", encoding="utf-8")
        self._curriculum_file = open(curriculum_path, "a", encoding="utf-8")

        if csv_path.stat().st_size == 0:
            self._csv_header_written = False
        else:
            self._csv_header_written = True

        logger.info(f"Rollout logger initialized: {jsonl_path}, {csv_path}, {curriculum_path}")

    def _init_wandb(self):
        """Initialize WandB logging with resume support."""
        try:
            import wandb

            if wandb.run is None:
                # Check if there's a run to resume
                run_id = self.config.wandb_run_id  # From config if passed
                resume_mode = "allow"

                if not run_id and self.output_dir:
                    wandb_id_file = self.output_dir / "wandb_run_id.txt"
                    if wandb_id_file.exists():
                        run_id = wandb_id_file.read_text().strip()

                if run_id:
                    resume_mode = "must"  # Must resume if we have an ID
                    logger.info(f"Resuming WandB run: {run_id}")

                wandb.init(
                    project=self.config.wandb_project or "grpo-training",
                    name=self.config.wandb_run_name,
                    resume=resume_mode,
                    id=run_id,
                )

                # Save run ID for future resumes
                if self.output_dir and wandb.run:
                    wandb_id_file = self.output_dir / "wandb_run_id.txt"
                    wandb_id_file.write_text(wandb.run.id)

            self._wandb_run = wandb.run
            self._wandb_run_id = wandb.run.id if wandb.run else None
            self._wandb_table = wandb.Table(
                columns=[
                    "iteration",
                    "update",
                    "group_index",
                    "prompt_preview",
                    "completion_preview",
                    "total_reward",
                    "advantage",
                    "cross_sampled",
                    "completion_tokens",
                    "thinking_tokens",
                ]
            )

            if self.config.resume_from_iteration > 0:
                logger.info(
                    f"WandB rollout logging resumed from iteration {self.config.resume_from_iteration}"
                )
            else:
                logger.info("WandB rollout logging initialized")

        except ImportError:
            logger.warning("WandB not installed, skipping WandB logging")
            self.config.log_to_wandb = False
        except Exception as e:
            logger.warning(f"WandB initialization failed: {e}")
            self.config.log_to_wandb = False

    def get_wandb_run_id(self) -> Optional[str]:
        """Get the WandB run ID for resuming later."""
        if hasattr(self, "_wandb_run_id") and self._wandb_run_id:
            return self._wandb_run_id
        if self._wandb_run:
            return self._wandb_run.id
        # Try to read from file
        if self.output_dir:
            wandb_id_file = self.output_dir / "wandb_run_id.txt"
            if wandb_id_file.exists():
                return wandb_id_file.read_text().strip()
        return None

    def log_rollout(
        self,
        iteration: int,
        update: int,
        prompts: List[str],
        prompt_texts: List[str],
        completions: List[str],
        answers: List[str],
        rewards_per_func: Dict[str, List[float]],
        total_rewards: List[float],
        advantages: List[float],
        prompt_tokens: List[int],
        completion_tokens: List[int],
        batch_indices: List[int],
        type_info: Optional[List] = None,
        cross_sample_metadata: Optional[List] = None,
        group_size: int = 4,
        two_phase_recovered: Optional[List[bool]] = None,
        scaffold_levels: Optional[List[float]] = None,
        reward_details: Optional[
            List[Optional[Dict]]
        ] = None,  # Per-completion reward breakdown (e.g., exam details)
    ):
        """Log rollouts for one step."""
        if not self.config.enabled:
            return

        if iteration % self.config.log_every_n_steps != 0:
            return

        timestamp = datetime.now().isoformat()

        # Track rewards by scaffold level for curriculum progress
        scaffold_rewards: Dict[float, List[float]] = {}

        for i in range(len(completions)):
            rewards_dict = {}
            for func_name, func_rewards in rewards_per_func.items():
                # Skip non-list entries (like exam_details which is a list of dicts)
                if func_name == "exam_details":
                    continue
                if not isinstance(func_rewards, (list, tuple)):
                    continue
                if i < len(func_rewards):
                    try:
                        rewards_dict[func_name] = float(func_rewards[i])
                    except (TypeError, ValueError):
                        # Skip entries that can't be converted to float
                        continue

            thinking_len = extract_thinking_length(completions[i])
            answer_len = extract_answer_length(completions[i])

            cross_sampled = False
            cross_source = None
            if cross_sample_metadata and i < len(cross_sample_metadata):
                meta = cross_sample_metadata[i]
                if meta:
                    cross_sampled = True
                    cross_source = getattr(meta, "paired_with_idx", None)

            # Check if this completion used two-phase recovery
            recovered = False
            if two_phase_recovered and i < len(two_phase_recovered):
                recovered = two_phase_recovered[i]

            # Get scaffold ratio for this completion
            scaffold_ratio = 0.0
            if scaffold_levels and i < len(scaffold_levels):
                scaffold_ratio = scaffold_levels[i]

            # Track for curriculum progress
            reward_val = float(total_rewards[i]) if i < len(total_rewards) else 0.0
            if scaffold_ratio not in scaffold_rewards:
                scaffold_rewards[scaffold_ratio] = []
            scaffold_rewards[scaffold_ratio].append(reward_val)

            # Get reward details for this completion (e.g., exam breakdown)
            completion_reward_details = None
            if reward_details and i < len(reward_details):
                completion_reward_details = reward_details[i]

            entry = RolloutEntry(
                iteration=iteration,
                update=update,
                group_index=i % group_size,
                prompt_index=batch_indices[i] if i < len(batch_indices) else i,
                prompt_full=prompts[i] if i < len(prompts) else "",
                prompt_text=prompt_texts[i] if i < len(prompt_texts) else "",
                completion=(
                    completions[i] if self.config.log_full_completions else completions[i][:500]
                ),
                answer_expected=answers[i] if i < len(answers) else "",
                rewards=rewards_dict,
                total_reward=reward_val,
                advantage=float(advantages[i]) if i < len(advantages) else 0.0,
                cross_sampled=cross_sampled,
                cross_sample_source_idx=cross_source,
                two_phase_recovered=recovered,
                scaffold_ratio=scaffold_ratio,
                prompt_tokens=prompt_tokens[i] if i < len(prompt_tokens) else 0,
                completion_tokens=completion_tokens[i] if i < len(completion_tokens) else 0,
                thinking_tokens=thinking_len,
                answer_tokens=answer_len,
                timestamp=timestamp,
                type_info=type_info[i] if type_info and i < len(type_info) else None,
                reward_details=completion_reward_details,
            )

            self._write_jsonl(entry)

            if self.config.log_to_wandb:
                self._rollout_buffer.append(entry)

            self._total_rollouts += 1

        # Log curriculum progress (per-scaffold-level rewards)
        self._log_curriculum_progress(iteration, scaffold_rewards, timestamp)

        if self.config.log_to_wandb and len(self._rollout_buffer) >= 100:
            self._flush_wandb_buffer()

    def _log_curriculum_progress(
        self, iteration: int, scaffold_rewards: Dict[float, List[float]], timestamp: str
    ):
        """Log curriculum progress - per-scaffold-level reward averages."""
        if not scaffold_rewards:
            return

        progress_entry = {
            "iteration": iteration,
            "timestamp": timestamp,
            "scaffold_levels": {},
        }

        for scaffold_ratio, rewards in sorted(scaffold_rewards.items()):
            if rewards:
                mean_reward = sum(rewards) / len(rewards)
                std_reward = 0.0
                if len(rewards) > 1:
                    std_reward = statistics.stdev(rewards)

                # Round scaffold ratio for cleaner keys
                ratio_key = f"{scaffold_ratio:.2f}"
                progress_entry["scaffold_levels"][ratio_key] = {
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "count": len(rewards),
                }

                # Track historical progress
                if scaffold_ratio not in self._curriculum_history:
                    self._curriculum_history[scaffold_ratio] = []
                self._curriculum_history[scaffold_ratio].append((iteration, mean_reward))

        # Calculate key curriculum metrics
        if len(scaffold_rewards) > 1:
            sorted_ratios = sorted(scaffold_rewards.keys())

            # No scaffold (0.0) vs full scaffold (1.0) gap
            if 0.0 in scaffold_rewards and 1.0 in scaffold_rewards:
                no_scaffold_mean = sum(scaffold_rewards[0.0]) / len(scaffold_rewards[0.0])
                full_scaffold_mean = sum(scaffold_rewards[1.0]) / len(scaffold_rewards[1.0])
                progress_entry["scaffold_gap"] = full_scaffold_mean - no_scaffold_mean
                progress_entry["learning_progress"] = (
                    no_scaffold_mean  # Key metric: how well does model do without help?
                )

            # Lowest scaffold reward (shows if model is learning)
            min_ratio = sorted_ratios[0]
            progress_entry["min_scaffold_ratio"] = min_ratio
            progress_entry["min_scaffold_reward"] = sum(scaffold_rewards[min_ratio]) / len(
                scaffold_rewards[min_ratio]
            )

        # Write to curriculum progress file
        if self._curriculum_file:
            self._curriculum_file.write(json.dumps(progress_entry) + "\n")
            self._curriculum_file.flush()

        # Log to wandb if available
        if self.config.log_to_wandb and self._wandb_run:
            try:
                import wandb

                wandb_dict = {"iteration": iteration}
                for ratio_key, stats in progress_entry.get("scaffold_levels", {}).items():
                    wandb_dict[f"curriculum/scaffold_{ratio_key}/mean"] = stats["mean_reward"]
                    wandb_dict[f"curriculum/scaffold_{ratio_key}/std"] = stats["std_reward"]
                if "scaffold_gap" in progress_entry:
                    wandb_dict["curriculum/scaffold_gap"] = progress_entry["scaffold_gap"]
                if "learning_progress" in progress_entry:
                    wandb_dict["curriculum/learning_progress"] = progress_entry["learning_progress"]
                wandb.log(wandb_dict, step=iteration)
            except Exception as e:
                logger.warning(f"Failed to log curriculum to WandB: {e}")

    def _write_jsonl(self, entry: RolloutEntry):
        """Write single entry to JSONL."""
        if self._jsonl_file:
            data = asdict(entry)
            self._jsonl_file.write(json.dumps(data) + "\n")
            self._jsonl_file.flush()

    def _flush_wandb_buffer(self):
        """Flush rollout buffer to WandB table."""
        if not self._wandb_table or not self._rollout_buffer:
            return

        try:
            for entry in self._rollout_buffer:
                self._wandb_table.add_data(
                    entry.iteration,
                    entry.update,
                    entry.group_index,
                    entry.prompt_text[:200],
                    entry.completion[: self.config.max_completion_preview_chars],
                    entry.total_reward,
                    entry.advantage,
                    entry.cross_sampled,
                    entry.completion_tokens,
                    entry.thinking_tokens,
                )
            self._rollout_buffer.clear()
        except Exception as e:
            logger.warning(f"Failed to flush WandB buffer: {e}")

    def log_iteration(
        self,
        iteration: int,
        update: int,
        loss: float,
        learning_rate: float,
        metrics: Dict[str, float],
        reward_funcs: Optional[List] = None,
        tokens_per_sec: float = 0.0,
        iterations_per_sec: float = 0.0,
        memory_gb: float = 0.0,
        rollout_rewards: Optional[List[float]] = None,
        cross_sample_flags: Optional[List[bool]] = None,
    ):
        """Log aggregated metrics for one iteration."""
        if not self.config.enabled:
            return

        timestamp = datetime.now().isoformat()

        reward_mean = metrics.get("total_rewards_mean", 0.0)
        reward_std = metrics.get("total_rewards_std", 0.0)

        reward_max = reward_mean + 2 * reward_std
        reward_min = reward_mean - 2 * reward_std
        if rollout_rewards:
            reward_max = max(rollout_rewards)
            reward_min = min(rollout_rewards)
            if len(rollout_rewards) > 1:
                reward_std = statistics.stdev(rollout_rewards)

        kl_mean = metrics.get("kl", 0.0)

        reward_functions = {}
        if reward_funcs:
            for func in reward_funcs:
                func_name = func.__name__
                reward_functions[func_name] = {
                    "mean": metrics.get(f"{func_name}_mean", 0.0),
                    "std": metrics.get(f"{func_name}_std", 0.0),
                    "coverage": metrics.get(f"{func_name}_coverage", 0.0),
                }

        cross_sampled_count = 0
        cross_sampled_reward_mean = 0.0
        non_cross_sampled_reward_mean = 0.0

        if cross_sample_flags and rollout_rewards:
            cross_rewards = []
            non_cross_rewards = []
            for flag, reward in zip(cross_sample_flags, rollout_rewards):
                if flag:
                    cross_rewards.append(reward)
                else:
                    non_cross_rewards.append(reward)

            cross_sampled_count = len(cross_rewards)
            if cross_rewards:
                cross_sampled_reward_mean = statistics.mean(cross_rewards)
            if non_cross_rewards:
                non_cross_sampled_reward_mean = statistics.mean(non_cross_rewards)

        cross_sampled_ratio = cross_sampled_count / max(1, len(cross_sample_flags or []))

        iter_metrics = IterationMetrics(
            iteration=iteration,
            update=update,
            timestamp=timestamp,
            loss=loss,
            learning_rate=learning_rate,
            memory_gb=memory_gb,
            tokens_per_sec=tokens_per_sec,
            iterations_per_sec=iterations_per_sec,
            reward_mean=reward_mean,
            reward_max=reward_max,
            reward_min=reward_min,
            reward_std=reward_std,
            kl_mean=kl_mean,
            reward_functions=reward_functions,
            cross_sampled_count=cross_sampled_count,
            cross_sampled_ratio=cross_sampled_ratio,
            cross_sampled_reward_mean=cross_sampled_reward_mean,
            non_cross_sampled_reward_mean=non_cross_sampled_reward_mean,
            avg_completion_tokens=metrics.get("average_generated_tokens", 0.0),
            max_completion_tokens=int(metrics.get("max_generated_tokens", 0)),
            min_completion_tokens=int(metrics.get("min_generated_tokens", 0)),
            hit_max_tokens_ratio=metrics.get("hit_max_tokens_ratio", 0.0),
            clip_ratio_low=metrics.get("clip_ratio_low", 0.0),
            clip_ratio_high=metrics.get("clip_ratio_high", 0.0),
            clip_ratio_total=metrics.get("clip_ratio_total", 0.0),
        )

        self._write_csv(iter_metrics)

        if self.config.log_to_wandb:
            self._log_wandb_metrics(iter_metrics)

        self._total_iterations += 1

    def _write_csv(self, metrics: IterationMetrics):
        """Write iteration metrics to CSV."""
        if not self._csv_file:
            return

        row = {
            "iteration": metrics.iteration,
            "update": metrics.update,
            "timestamp": metrics.timestamp,
            "loss": metrics.loss,
            "learning_rate": metrics.learning_rate,
            "memory_gb": metrics.memory_gb,
            "tokens_per_sec": metrics.tokens_per_sec,
            "iterations_per_sec": metrics.iterations_per_sec,
            "reward_mean": metrics.reward_mean,
            "reward_max": metrics.reward_max,
            "reward_min": metrics.reward_min,
            "reward_std": metrics.reward_std,
            "kl_mean": metrics.kl_mean,
            "cross_sampled_count": metrics.cross_sampled_count,
            "cross_sampled_ratio": metrics.cross_sampled_ratio,
            "cross_sampled_reward_mean": metrics.cross_sampled_reward_mean,
            "non_cross_sampled_reward_mean": metrics.non_cross_sampled_reward_mean,
            "avg_completion_tokens": metrics.avg_completion_tokens,
            "max_completion_tokens": metrics.max_completion_tokens,
            "min_completion_tokens": metrics.min_completion_tokens,
            "hit_max_tokens_ratio": metrics.hit_max_tokens_ratio,
            "clip_ratio_low": metrics.clip_ratio_low,
            "clip_ratio_high": metrics.clip_ratio_high,
            "clip_ratio_total": metrics.clip_ratio_total,
        }

        for func_name, func_stats in metrics.reward_functions.items():
            clean_name = func_name.replace("_reward_func", "")
            row[f"{clean_name}_mean"] = func_stats.get("mean", 0.0)
            row[f"{clean_name}_std"] = func_stats.get("std", 0.0)
            row[f"{clean_name}_coverage"] = func_stats.get("coverage", 0.0)

        if not self._csv_header_written:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
            self._csv_header_written = True
        elif self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _log_wandb_metrics(self, metrics: IterationMetrics):
        """Log metrics to WandB."""
        if not self._wandb_run:
            return

        try:
            import wandb

            log_dict = {
                "iteration": metrics.iteration,
                "loss": metrics.loss,
                "learning_rate": metrics.learning_rate,
                "memory_gb": metrics.memory_gb,
                "reward/mean": metrics.reward_mean,
                "reward/max": metrics.reward_max,
                "reward/min": metrics.reward_min,
                "reward/std": metrics.reward_std,
                "kl/mean": metrics.kl_mean,
                "cross_sampling/count": metrics.cross_sampled_count,
                "cross_sampling/ratio": metrics.cross_sampled_ratio,
                "cross_sampling/reward_mean": metrics.cross_sampled_reward_mean,
                "cross_sampling/non_cross_reward_mean": metrics.non_cross_sampled_reward_mean,
                "generation/avg_tokens": metrics.avg_completion_tokens,
                "generation/max_tokens": metrics.max_completion_tokens,
                "generation/hit_limit_ratio": metrics.hit_max_tokens_ratio,
                "clipping/low": metrics.clip_ratio_low,
                "clipping/high": metrics.clip_ratio_high,
                "speed/tokens_per_sec": metrics.tokens_per_sec,
                "speed/iterations_per_sec": metrics.iterations_per_sec,
            }

            for func_name, func_stats in metrics.reward_functions.items():
                clean_name = func_name.replace("_reward_func", "")
                log_dict[f"rewards/{clean_name}/mean"] = func_stats.get("mean", 0.0)
                log_dict[f"rewards/{clean_name}/std"] = func_stats.get("std", 0.0)

            wandb.log(log_dict, step=metrics.iteration)

        except Exception as e:
            logger.warning(f"Failed to log to WandB: {e}")

    def close(self):
        """Close all file handles and finalize logging."""
        if not self.config.enabled:
            return

        if self.config.log_to_wandb:
            self._flush_wandb_buffer()

            try:
                import wandb

                if self._wandb_table and self._wandb_run:
                    wandb.log({"rollouts_table": self._wandb_table})

                    artifact = wandb.Artifact(
                        name=f"training_logs_{self._wandb_run.id}",
                        type="training_logs",
                    )

                    jsonl_path = self.output_dir / self.config.jsonl_filename
                    csv_path = self.output_dir / self.config.csv_filename

                    if jsonl_path.exists():
                        artifact.add_file(str(jsonl_path))
                    if csv_path.exists():
                        artifact.add_file(str(csv_path))

                    wandb.log_artifact(artifact)

            except Exception as e:
                logger.warning(f"Failed to finalize WandB logging: {e}")

        if self._jsonl_file:
            self._jsonl_file.close()
            self._jsonl_file = None

        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

        # Write curriculum summary and close file
        if self._curriculum_file:
            self._write_curriculum_summary()
            self._curriculum_file.close()
            self._curriculum_file = None

        logger.info(
            f"Rollout logger closed. Total rollouts: {self._total_rollouts}, "
            f"Total iterations: {self._total_iterations}"
        )

    def _write_curriculum_summary(self):
        """Write final curriculum learning summary."""
        if not self._curriculum_history:
            return

        summary = {
            "type": "summary",
            "timestamp": datetime.now().isoformat(),
            "scaffold_levels": {},
        }

        for scaffold_ratio, history in sorted(self._curriculum_history.items()):
            if len(history) < 2:
                continue

            iterations = [h[0] for h in history]
            rewards = [h[1] for h in history]

            # Calculate trends
            first_half = rewards[: len(rewards) // 2] if len(rewards) >= 4 else rewards[:1]
            second_half = rewards[len(rewards) // 2 :] if len(rewards) >= 4 else rewards[-1:]

            first_mean = sum(first_half) / len(first_half) if first_half else 0
            second_mean = sum(second_half) / len(second_half) if second_half else 0
            improvement = second_mean - first_mean

            ratio_key = f"{scaffold_ratio:.2f}"
            summary["scaffold_levels"][ratio_key] = {
                "first_reward": rewards[0],
                "last_reward": rewards[-1],
                "first_half_mean": first_mean,
                "second_half_mean": second_mean,
                "improvement": improvement,
                "total_samples": len(history),
                "iterations_range": [min(iterations), max(iterations)],
            }

        # Key insight: Is the model learning to reason without scaffold?
        if 0.0 in self._curriculum_history and len(self._curriculum_history[0.0]) >= 2:
            no_scaffold_history = self._curriculum_history[0.0]
            rewards = [h[1] for h in no_scaffold_history]
            first_half = rewards[: len(rewards) // 2]
            second_half = rewards[len(rewards) // 2 :]

            summary["learning_assessment"] = {
                "no_scaffold_initial": sum(first_half) / len(first_half) if first_half else 0,
                "no_scaffold_final": sum(second_half) / len(second_half) if second_half else 0,
                "no_scaffold_trend": (
                    "improving"
                    if sum(second_half) / len(second_half) > sum(first_half) / len(first_half)
                    else "stable_or_declining"
                ),
            }

        if self._curriculum_file:
            self._curriculum_file.write(json.dumps(summary) + "\n")
            self._curriculum_file.flush()

        # Print summary to console
        logger.info("=" * 60)
        logger.info("CURRICULUM LEARNING SUMMARY")
        logger.info("=" * 60)
        for ratio_key, stats in summary.get("scaffold_levels", {}).items():
            logger.info(
                f"Scaffold {ratio_key}: {stats['first_reward']:.3f} -> {stats['last_reward']:.3f} (Δ={stats['improvement']:+.3f})"
            )
        if "learning_assessment" in summary:
            assessment = summary["learning_assessment"]
            logger.info(f"No-scaffold trend: {assessment['no_scaffold_trend'].upper()}")
            logger.info(f"  Initial: {assessment['no_scaffold_initial']:.3f}")
            logger.info(f"  Final:   {assessment['no_scaffold_final']:.3f}")
        logger.info("=" * 60)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
