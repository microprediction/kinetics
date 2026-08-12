"""Configuration for the watermark-races experiment.

Everything needed to reproduce a run lives in one WatermarkConfig plus the seed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
RESULTS_DIR = PACKAGE_DIR / "results"
FIGURES_DIR = PACKAGE_DIR / "figures"


@dataclass(frozen=True)
class WatermarkConfig:
    # models
    model_name: str = "Qwen/Qwen2.5-1.5B"
    fallback_model_name: str = "Qwen/Qwen2.5-0.5B"  # if GPU memory is insufficient
    proxy_model_name: str = "Qwen/Qwen2.5-0.5B"

    # watermark (Kirchenbauer-style green list)
    secret_key: int = 15485863  # known by construction; NOT anyone's production key
    gamma: float = 0.25  # green fraction
    deltas: tuple[float, ...] = (1.0, 2.0, 3.0)
    context_width: int = 1  # previous tokens hashed into the green list

    # sampling
    temperature: float = 1.0
    extra_temperatures: tuple[float, ...] = (0.7, 1.3)  # later phase
    top_p: float = 1.0
    top_k: int | None = None

    # corpus
    n_prompts: int = 500  # prefer 1000 if computation permits
    max_new_tokens: int = 256
    seed: int = 20260811

    # storage: per-token top-M raw logits kept for detector-side race models
    top_m_store: int = 1000
    race_top_m_grid: tuple[int, ...] = (100, 500, 1000, 5000)

    # evaluation
    eval_lengths: tuple[int, ...] = (16, 32, 64, 128, 256)
    n_bootstrap: int = 1000
    train_fraction: float = 0.4  # w(H) and Thurstone tuning use this split only
    val_fraction: float = 0.2  # sigma / ability-map tuning
    # remaining 0.4 is the held-out test set for all final comparisons


DEFAULT_CONFIG = WatermarkConfig()

# ---------------------------------------------------------------------------
# Prompts: >= 500 diverse prompts built from category templates. Deterministic.
# ---------------------------------------------------------------------------

_TOPICS = [
    "photosynthesis", "the French Revolution", "binary search trees", "plate tectonics",
    "the Roman aqueducts", "supply and demand", "neural networks", "the water cycle",
    "Bayesian inference", "the printing press", "antibiotic resistance", "black holes",
    "the Silk Road", "compound interest", "natural selection", "semiconductors",
    "the Great Depression", "ocean currents", "public key cryptography", "the immune system",
    "renewable energy storage", "the Industrial Revolution", "protein folding",
    "urban planning", "the Antarctic Treaty", "inflation targeting", "coral reefs",
    "quantum tunneling", "medieval guilds", "recommendation systems", "soil erosion",
    "the Marshall Plan", "game theory", "volcanic eruptions", "the telegraph",
    "herd immunity", "chess openings", "glacier formation", "the Ottoman Empire",
    "database indexing", "photolithography", "the gold standard", "bird migration",
    "reinforcement learning", "the Suez Canal", "fermentation", "radar",
    "the Hanseatic League", "error-correcting codes", "monsoon seasons",
]

_TEMPLATES = [
    # (category, template)
    ("factual", "Explain {} to a curious high-school student."),
    ("factual", "What are the three most important things to understand about {}?"),
    ("creative", "Write a short story in which {} plays a decisive role."),
    ("creative", "Compose a scene where two strangers argue about {} on a train."),
    ("code", "Describe how you would implement a small program to simulate or model {}."),
    ("technical", "Write a technical overview of {} for an engineering newsletter."),
    ("email", "Draft a friendly email to a colleague summarizing a talk about {}."),
    ("summary", "Summarize the current state of knowledge about {} in one paragraph, then elaborate."),
    ("history", "Trace the historical development of {} and its consequences."),
    ("science", "Describe an experiment that could deepen our understanding of {}."),
    ("dialogue", "Write a casual conversation between friends who just learned about {}."),
    ("argument", "Make the strongest argument you can that {} is underappreciated, then consider objections."),
]


def build_prompts(n: int | None = None) -> list[str]:
    """Deterministic list of diverse prompts; len >= 500 by default."""
    prompts = [t.format(topic) for topic, t in itertools.product(_TOPICS, [tpl for _, tpl in _TEMPLATES])]
    if n is not None:
        if n > len(prompts):
            raise ValueError(f"only {len(prompts)} prompts available, {n} requested")
        prompts = prompts[:n]
    return prompts


def prompt_categories() -> list[str]:
    """Category label for each prompt returned by build_prompts()."""
    return [cat for _, cat in itertools.product(_TOPICS, [(tpl, cat) for cat, tpl in _TEMPLATES])]
