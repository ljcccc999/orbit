"""Orbit: an approachable, trainable language-model research project."""

from .config import OrbitConfig

__all__ = ["OrbitConfig", "OrbitForCausalLM"]


def __getattr__(name: str):
    # Do not import PyTorch into the small idle API controller. It is loaded
    # only when a model is trained or used.
    if name == "OrbitForCausalLM":
        from .model import OrbitForCausalLM
        return OrbitForCausalLM
    raise AttributeError(name)
