"""
GRPO Training Monitor with Threshold Alerts
============================================

Real-time monitoring for GRPO training with colored console output
and automatic threshold checking for early stopping decisions.

Monitoring Thresholds (configurable):
┌─────────────┬───────────┬─────────────┬────────────────┐
│ Metric      │ Good      │ Warning     │ Stop & Rollback│
├─────────────┼───────────┼─────────────┼────────────────┤
│ KL mean     │ < 0.025   │ 0.025-0.04  │ > 0.04         │
│ Correctness │ > 0.45    │ 0.30-0.45   │ < 0.30         │
│ Reward      │ > 0.50    │ 0.40-0.50   │ < 0.40         │
└─────────────┴───────────┴─────────────┴────────────────┘

Usage:
    from mlx_grpo.trainer.training_monitor import TrainingMonitor

    monitor = TrainingMonitor()

    # In training loop after computing metrics:
    should_stop = monitor.log_step(
        step=it,
        metrics=avg_metrics,  # dict from grpo_trainer
        loss=train_loss,
    )

    if should_stop:
        print("Early stopping triggered!")
        break

Integration with grpo_trainer.py:
    Add after line ~1008 in the reporting block:

    should_stop = monitor.log_step(step=it, metrics=avg_metrics, loss=train_loss)
    if should_stop:
        tqdm.write(monitor.get_stop_reason())
        break
"""

import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# Use existing Colors from visuals.py or define inline
try:
    from ..visuals import Colors
except ImportError:

    class Colors:
        HEADER = "\033[95m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        MAGENTA = "\033[35m"
        WHITE = "\033[97m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        UNDERLINE = "\033[4m"
        RESET = "\033[0m"
        BG_BLACK = "\033[40m"
        BG_BLUE = "\033[44m"
        BG_GREEN = "\033[42m"
        BG_YELLOW = "\033[43m"
        BG_RED = "\033[41m"


class Status(Enum):
    """Status levels for metrics."""

    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# ============================================================================
# THRESHOLD CONFIGURATION
# ============================================================================


@dataclass
class ThresholdConfig:
    """Configuration for a single metric's thresholds."""

    name: str
    good_threshold: float
    warning_threshold: float
    critical_threshold: float
    higher_is_better: bool = True
    unit: str = ""

    def check(self, value: float) -> Status:
        """Check value against thresholds."""
        if value is None:
            return Status.UNKNOWN

        if self.higher_is_better:
            if value > self.good_threshold:
                return Status.GOOD
            elif value >= self.warning_threshold:
                return Status.WARNING
            else:
                return Status.CRITICAL
        else:
            if value < self.good_threshold:
                return Status.GOOD
            elif value <= self.warning_threshold:
                return Status.WARNING
            else:
                return Status.CRITICAL


@dataclass
class MonitorConfig:
    """Complete monitoring configuration."""

    # Default thresholds matching the spec
    kl_mean: ThresholdConfig = field(
        default_factory=lambda: ThresholdConfig(
            name="KL Divergence",
            good_threshold=0.025,
            warning_threshold=0.04,
            critical_threshold=0.04,
            higher_is_better=False,
            unit="",
        )
    )

    correctness: ThresholdConfig = field(
        default_factory=lambda: ThresholdConfig(
            name="Correctness",
            good_threshold=0.45,
            warning_threshold=0.30,
            critical_threshold=0.30,
            higher_is_better=True,
            unit="",
        )
    )

    reward: ThresholdConfig = field(
        default_factory=lambda: ThresholdConfig(
            name="Reward",
            good_threshold=0.50,
            warning_threshold=0.40,
            critical_threshold=0.40,
            higher_is_better=True,
            unit="",
        )
    )

    # Behavior
    stop_on_critical: bool = True
    critical_count_threshold: int = 3  # Stop after N consecutive criticals
    rolling_window: int = 10

    # Display
    show_sparklines: bool = True
    show_trends: bool = True
    compact_mode: bool = False


# ============================================================================
# SPARKLINE GENERATION
# ============================================================================


def sparkline(values: List[float], width: int = 10) -> str:
    """Generate a sparkline from values."""
    if not values:
        return " " * width

    values = list(values)[-width:]
    if len(values) < width:
        values = [values[0]] * (width - len(values)) + values

    chars = "▁▂▃▄▅▆▇█"

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val

    if range_val == 0:
        return chars[4] * len(values)

    result = ""
    for v in values:
        idx = int((v - min_val) / range_val * 7)
        idx = max(0, min(7, idx))
        result += chars[idx]

    return result


def trend_arrow(values: List[float], window: int = 5) -> str:
    """Get trend arrow based on recent values."""
    if len(values) < 2:
        return "→"

    recent = list(values)[-window:]
    if len(recent) < 2:
        return "→"

    avg_first = sum(recent[: len(recent) // 2]) / max(1, len(recent) // 2)
    avg_second = sum(recent[len(recent) // 2 :]) / max(1, len(recent) - len(recent) // 2)

    diff = avg_second - avg_first
    threshold = 0.01

    if diff > threshold:
        return "↑"
    elif diff < -threshold:
        return "↓"
    else:
        return "→"


# ============================================================================
# TRAINING MONITOR
# ============================================================================


class TrainingMonitor:
    """
    Real-time training monitor with colored console output.

    Integrates with grpo_trainer.py metrics dict.
    """

    def __init__(
        self,
        config: Optional[MonitorConfig] = None,
        correctness_key: str = "hierarchical_rewards_mean",  # Key in metrics dict
        reward_key: str = "total_rewards_mean",
        kl_key: str = "kl",
    ):
        self.config = config or MonitorConfig()

        # Metric keys mapping
        self.correctness_key = correctness_key
        self.reward_key = reward_key
        self.kl_key = kl_key

        # History tracking
        self.history: Dict[str, deque] = {
            "kl": deque(maxlen=1000),
            "correctness": deque(maxlen=1000),
            "reward": deque(maxlen=1000),
            "loss": deque(maxlen=1000),
        }

        # Consecutive critical counter
        self.consecutive_criticals: Dict[str, int] = {
            "kl": 0,
            "correctness": 0,
            "reward": 0,
        }

        # Best values tracking
        self.best_values: Dict[str, float] = {
            "kl": float("inf"),
            "correctness": 0.0,
            "reward": 0.0,
            "loss": float("inf"),
        }

        # Stop reason
        self._stop_reason: Optional[str] = None
        self._should_stop: bool = False

        # Stats
        self.start_time = time.time()
        self.total_criticals = 0
        self.total_warnings = 0

    def _get_status_color(self, status: Status) -> str:
        """Get color for status."""
        return {
            Status.GOOD: Colors.GREEN,
            Status.WARNING: Colors.YELLOW,
            Status.CRITICAL: Colors.RED,
            Status.UNKNOWN: Colors.DIM,
        }.get(status, Colors.RESET)

    def _get_status_icon(self, status: Status) -> str:
        """Get icon for status."""
        return {
            Status.GOOD: f"{Colors.GREEN}✓{Colors.RESET}",
            Status.WARNING: f"{Colors.YELLOW}⚠{Colors.RESET}",
            Status.CRITICAL: f"{Colors.RED}✗{Colors.RESET}",
            Status.UNKNOWN: f"{Colors.DIM}?{Colors.RESET}",
        }.get(status, "?")

    def _format_value(self, value: float, threshold_config: ThresholdConfig, status: Status) -> str:
        """Format a value with color based on status."""
        color = self._get_status_color(status)
        return f"{color}{value:.4f}{Colors.RESET}"

    def _update_best(self, key: str, value: float, higher_is_better: bool) -> bool:
        """Update best value, return True if new best."""
        if higher_is_better:
            if value > self.best_values[key]:
                self.best_values[key] = value
                return True
        else:
            if value < self.best_values[key]:
                self.best_values[key] = value
                return True
        return False

    def _check_stop_condition(self, key: str, status: Status) -> bool:
        """Check if we should stop based on consecutive criticals."""
        if status == Status.CRITICAL:
            self.consecutive_criticals[key] += 1
            self.total_criticals += 1
            if self.consecutive_criticals[key] >= self.config.critical_count_threshold:
                self._stop_reason = (
                    f"{Colors.RED}{Colors.BOLD}STOP CONDITION:{Colors.RESET} "
                    f"{key.upper()} has been critical for "
                    f"{self.consecutive_criticals[key]} consecutive steps"
                )
                return True
        else:
            self.consecutive_criticals[key] = 0

        if status == Status.WARNING:
            self.total_warnings += 1

        return False

    def log_step(
        self,
        step: int,
        metrics: Dict[str, Any],
        loss: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Log a training step and check thresholds.

        Args:
            step: Current iteration number
            metrics: Dict of metrics from grpo_trainer (avg_metrics)
            loss: Training loss value
            extra_metrics: Additional metrics to display

        Returns:
            True if training should stop, False otherwise
        """
        # Extract values from metrics dict
        kl_value = metrics.get(self.kl_key, None)
        correctness_value = metrics.get(self.correctness_key, None)
        reward_value = metrics.get(self.reward_key, None)

        # Convert mx.array to float if needed
        if hasattr(kl_value, "item"):
            kl_value = float(kl_value.item())
        if hasattr(correctness_value, "item"):
            correctness_value = float(correctness_value.item())
        if hasattr(reward_value, "item"):
            reward_value = float(reward_value.item())
        if loss is not None and hasattr(loss, "item"):
            loss = float(loss.item())

        # Update history
        if kl_value is not None:
            self.history["kl"].append(kl_value)
        if correctness_value is not None:
            self.history["correctness"].append(correctness_value)
        if reward_value is not None:
            self.history["reward"].append(reward_value)
        if loss is not None:
            self.history["loss"].append(loss)

        # Check statuses
        kl_status = self.config.kl_mean.check(kl_value) if kl_value is not None else Status.UNKNOWN
        corr_status = (
            self.config.correctness.check(correctness_value)
            if correctness_value is not None
            else Status.UNKNOWN
        )
        rew_status = (
            self.config.reward.check(reward_value) if reward_value is not None else Status.UNKNOWN
        )

        # Update bests
        if kl_value is not None:
            self._update_best("kl", kl_value, higher_is_better=False)
        if correctness_value is not None:
            self._update_best("correctness", correctness_value, higher_is_better=True)
        if reward_value is not None:
            self._update_best("reward", reward_value, higher_is_better=True)
        if loss is not None:
            self._update_best("loss", loss, higher_is_better=False)

        # Check stop conditions
        should_stop = False
        if self.config.stop_on_critical:
            if kl_value is not None and self._check_stop_condition("kl", kl_status):
                should_stop = True
            if correctness_value is not None and self._check_stop_condition(
                "correctness", corr_status
            ):
                should_stop = True
            if reward_value is not None and self._check_stop_condition("reward", rew_status):
                should_stop = True

        self._should_stop = should_stop

        # Print status line
        self._print_status_line(
            step=step,
            kl_value=kl_value,
            kl_status=kl_status,
            correctness_value=correctness_value,
            corr_status=corr_status,
            reward_value=reward_value,
            rew_status=rew_status,
            loss=loss,
        )

        return should_stop

    def _print_status_line(
        self,
        step: int,
        kl_value: Optional[float],
        kl_status: Status,
        correctness_value: Optional[float],
        corr_status: Status,
        reward_value: Optional[float],
        rew_status: Status,
        loss: Optional[float],
    ):
        """Print a colored status line."""

        # Build the status line
        parts = []

        # Step info
        elapsed = time.time() - self.start_time
        parts.append(f"{Colors.DIM}[{elapsed/60:.1f}m]{Colors.RESET}")

        # KL
        if kl_value is not None:
            icon = self._get_status_icon(kl_status)
            color = self._get_status_color(kl_status)
            trend = trend_arrow(self.history["kl"]) if self.config.show_trends else ""
            spark = sparkline(self.history["kl"], 8) if self.config.show_sparklines else ""
            parts.append(
                f"KL:{icon}{color}{kl_value:.4f}{Colors.RESET}{trend} {Colors.DIM}{spark}{Colors.RESET}"
            )

        # Correctness
        if correctness_value is not None:
            icon = self._get_status_icon(corr_status)
            color = self._get_status_color(corr_status)
            trend = trend_arrow(self.history["correctness"]) if self.config.show_trends else ""
            spark = sparkline(self.history["correctness"], 8) if self.config.show_sparklines else ""
            parts.append(
                f"Corr:{icon}{color}{correctness_value:.3f}{Colors.RESET}{trend} {Colors.DIM}{spark}{Colors.RESET}"
            )

        # Reward
        if reward_value is not None:
            icon = self._get_status_icon(rew_status)
            color = self._get_status_color(rew_status)
            trend = trend_arrow(self.history["reward"]) if self.config.show_trends else ""
            spark = sparkline(self.history["reward"], 8) if self.config.show_sparklines else ""
            parts.append(
                f"Rew:{icon}{color}{reward_value:.3f}{Colors.RESET}{trend} {Colors.DIM}{spark}{Colors.RESET}"
            )

        # Loss
        if loss is not None:
            loss_trend = trend_arrow(self.history["loss"]) if self.config.show_trends else ""
            parts.append(f"Loss:{Colors.CYAN}{loss:.4f}{Colors.RESET}{loss_trend}")

        # Critical/Warning counts
        if self.total_criticals > 0 or self.total_warnings > 0:
            parts.append(
                f"{Colors.RED}C:{self.total_criticals}{Colors.RESET} "
                f"{Colors.YELLOW}W:{self.total_warnings}{Colors.RESET}"
            )

        # Print
        line = " │ ".join(parts)
        print(f"\r{line}", end="\n")

    def get_stop_reason(self) -> str:
        """Get the reason for stopping."""
        return self._stop_reason or "No stop condition triggered"

    def get_summary(self) -> str:
        """Get a summary of the training run as a string."""
        elapsed = time.time() - self.start_time
        lines = []

        lines.append(f"\n{'═' * 70}")
        lines.append("TRAINING MONITOR SUMMARY")
        lines.append(f"{'═' * 70}")

        lines.append(f"\nDuration: {elapsed/60:.1f} minutes")
        lines.append(f"Total Warnings: {self.total_warnings}")
        lines.append(f"Total Criticals: {self.total_criticals}")

        lines.append(f"\nBest Values:")
        lines.append(f"  • KL:          {self.best_values['kl']:.6f}")
        lines.append(f"  • Correctness: {self.best_values['correctness']:.4f}")
        lines.append(f"  • Reward:      {self.best_values['reward']:.4f}")
        lines.append(f"  • Loss:        {self.best_values['loss']:.4f}")

        if self.history["kl"]:
            lines.append(f"\nFinal Values:")
            lines.append(f"  • KL:          {self.history['kl'][-1]:.6f}")
        if self.history["correctness"]:
            lines.append(f"  • Correctness: {self.history['correctness'][-1]:.4f}")
        if self.history["reward"]:
            lines.append(f"  • Reward:      {self.history['reward'][-1]:.4f}")
        if self.history["loss"]:
            lines.append(f"  • Loss:        {self.history['loss'][-1]:.4f}")

        if self._should_stop:
            lines.append(f"\n⚠ EARLY STOP TRIGGERED")
            lines.append(f"  {self._stop_reason}")

        lines.append(f"\n{'═' * 70}\n")

        return "\n".join(lines)

    def print_summary(self):
        """Print a summary of the training run."""
        elapsed = time.time() - self.start_time

        print(f"\n{Colors.CYAN}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}TRAINING MONITOR SUMMARY{Colors.RESET}")
        print(f"{Colors.CYAN}{'═' * 70}{Colors.RESET}")

        print(f"\n{Colors.BOLD}Duration:{Colors.RESET} {elapsed/60:.1f} minutes")
        print(
            f"{Colors.BOLD}Total Warnings:{Colors.RESET} {Colors.YELLOW}{self.total_warnings}{Colors.RESET}"
        )
        print(
            f"{Colors.BOLD}Total Criticals:{Colors.RESET} {Colors.RED}{self.total_criticals}{Colors.RESET}"
        )

        print(f"\n{Colors.BOLD}Best Values:{Colors.RESET}")
        print(f"  • KL:          {Colors.GREEN}{self.best_values['kl']:.6f}{Colors.RESET}")
        print(f"  • Correctness: {Colors.GREEN}{self.best_values['correctness']:.4f}{Colors.RESET}")
        print(f"  • Reward:      {Colors.GREEN}{self.best_values['reward']:.4f}{Colors.RESET}")
        print(f"  • Loss:        {Colors.GREEN}{self.best_values['loss']:.4f}{Colors.RESET}")

        # Final values
        if self.history["kl"]:
            print(f"\n{Colors.BOLD}Final Values:{Colors.RESET}")
            print(f"  • KL:          {self.history['kl'][-1]:.6f}")
        if self.history["correctness"]:
            print(f"  • Correctness: {self.history['correctness'][-1]:.4f}")
        if self.history["reward"]:
            print(f"  • Reward:      {self.history['reward'][-1]:.4f}")
        if self.history["loss"]:
            print(f"  • Loss:        {self.history['loss'][-1]:.4f}")

        if self._should_stop:
            print(f"\n{Colors.RED}{Colors.BOLD}⚠ EARLY STOP TRIGGERED{Colors.RESET}")
            print(f"  {self._stop_reason}")

        print(f"\n{Colors.CYAN}{'═' * 70}{Colors.RESET}\n")

    def print_thresholds(self):
        """Print the current threshold configuration."""
        print(f"\n{Colors.CYAN}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}MONITORING THRESHOLDS{Colors.RESET}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.RESET}")
        print(f"{'Metric':<15} {'Good':<15} {'Warning':<15} {'Stop & Rollback':<15}")
        print(f"{Colors.CYAN}{'─' * 70}{Colors.RESET}")

        # KL
        kl = self.config.kl_mean
        print(
            f"{Colors.BOLD}KL mean{Colors.RESET:<8} "
            f"{Colors.GREEN}< {kl.good_threshold}{Colors.RESET:<14} "
            f"{Colors.YELLOW}{kl.good_threshold}-{kl.warning_threshold}{Colors.RESET:<14} "
            f"{Colors.RED}> {kl.warning_threshold}{Colors.RESET}"
        )

        # Correctness
        corr = self.config.correctness
        print(
            f"{Colors.BOLD}Correctness{Colors.RESET:<4} "
            f"{Colors.GREEN}> {corr.good_threshold}{Colors.RESET:<14} "
            f"{Colors.YELLOW}{corr.warning_threshold}-{corr.good_threshold}{Colors.RESET:<14} "
            f"{Colors.RED}< {corr.warning_threshold}{Colors.RESET}"
        )

        # Reward
        rew = self.config.reward
        print(
            f"{Colors.BOLD}Reward{Colors.RESET:<9} "
            f"{Colors.GREEN}> {rew.good_threshold}{Colors.RESET:<14} "
            f"{Colors.YELLOW}{rew.warning_threshold}-{rew.good_threshold}{Colors.RESET:<14} "
            f"{Colors.RED}< {rew.warning_threshold}{Colors.RESET}"
        )

        print(f"{Colors.CYAN}{'═' * 70}{Colors.RESET}\n")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_monitor(
    kl_good: float = 0.025,
    kl_warning: float = 0.04,
    corr_good: float = 0.45,
    corr_warning: float = 0.30,
    reward_good: float = 0.50,
    reward_warning: float = 0.40,
    stop_on_critical: bool = True,
    critical_count: int = 3,
) -> TrainingMonitor:
    """
    Create a training monitor with custom thresholds.

    Example:
        monitor = create_monitor(
            kl_warning=0.05,  # More lenient KL
            corr_warning=0.25,  # More lenient correctness
        )
    """
    config = MonitorConfig(
        kl_mean=ThresholdConfig(
            name="KL Divergence",
            good_threshold=kl_good,
            warning_threshold=kl_warning,
            critical_threshold=kl_warning,
            higher_is_better=False,
        ),
        correctness=ThresholdConfig(
            name="Correctness",
            good_threshold=corr_good,
            warning_threshold=corr_warning,
            critical_threshold=corr_warning,
            higher_is_better=True,
        ),
        reward=ThresholdConfig(
            name="Reward",
            good_threshold=reward_good,
            warning_threshold=reward_warning,
            critical_threshold=reward_warning,
            higher_is_better=True,
        ),
        stop_on_critical=stop_on_critical,
        critical_count_threshold=critical_count,
    )
    return TrainingMonitor(config=config)


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    import random

    print(f"\n{Colors.BOLD}{Colors.MAGENTA}Training Monitor Demo{Colors.RESET}\n")

    # Create monitor
    monitor = TrainingMonitor()
    monitor.print_thresholds()

    print(f"{Colors.BOLD}Simulating training run...{Colors.RESET}\n")

    # Simulate training
    for step in range(1, 21):
        # Simulate metrics with some variance
        metrics = {
            "kl": 0.015 + random.gauss(0, 0.008) + step * 0.001,  # Slowly increasing KL
            "hierarchical_rewards_mean": 0.35
            + random.gauss(0, 0.05)
            + step * 0.01,  # Improving correctness
            "total_rewards_mean": 0.42 + random.gauss(0, 0.05) + step * 0.008,  # Improving reward
        }
        loss = 2.5 - step * 0.05 + random.gauss(0, 0.1)

        should_stop = monitor.log_step(
            step=step,
            metrics=metrics,
            loss=loss,
        )

        if should_stop:
            print(f"\n{Colors.RED}Early stopping at step {step}!{Colors.RESET}")
            print(monitor.get_stop_reason())
            break

        time.sleep(0.1)  # Simulate training time

    monitor.print_summary()
