from dataclasses import dataclass, replace


@dataclass
class TrainingConfig:
    steps: int = 1000
    batch_size: int = 1
    seq_len: int = 1024
    grad_accum: int = 8
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    precision: str = "auto"
    scheduler: str = "cosine"
    checkpoint_every: int = 500
    seed: int = 42

    @classmethod
    def for_model(cls, preset: str) -> "TrainingConfig":
        values = {
            "300m": dict(batch_size=2, seq_len=1024, grad_accum=8, learning_rate=3e-4),
            "1b": dict(batch_size=1, seq_len=2048, grad_accum=16, learning_rate=2e-4),
            "3b": dict(batch_size=1, seq_len=2048, grad_accum=32, learning_rate=1.5e-4),
            "7b": dict(batch_size=1, seq_len=2048, grad_accum=64, learning_rate=1e-4),
            "14b": dict(batch_size=1, seq_len=2048, grad_accum=128, learning_rate=8e-5),
            "38b": dict(batch_size=1, seq_len=2048, grad_accum=256, learning_rate=5e-5),
        }
        return cls(**values.get(preset, {}))

    def with_overrides(self, **kwargs) -> "TrainingConfig":
        return replace(self, **kwargs)

    def validate(self) -> None:
        if min(self.steps, self.batch_size, self.seq_len, self.grad_accum) < 1:
            raise ValueError("steps, batch size, sequence length and grad accumulation must be positive")
        if self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("learning rate and gradient clip must be positive")
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("precision must be auto, fp32, fp16 or bf16")
        if self.scheduler not in {"constant", "cosine"}:
            raise ValueError("scheduler must be constant or cosine")
