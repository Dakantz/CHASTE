from dataclasses import dataclass, field


@dataclass
class BeamSearchConfig:
    top_k: int = 3
    max_depth: int = 4
    end_token: str | None = "\n"
    skip_tokens: int = 1
    oversample: int = 8129
    k_progress: list[int] | None = field(
        default_factory=lambda: [4, 2, -1]
    )  # k for depth, -1 means to take top until end_token
    length_cost: float = 0
