import torch

from orbit import OrbitConfig, OrbitForCausalLM


def test_local_forward_backward():
    torch.manual_seed(0)
    cfg = OrbitConfig.tiny().with_overrides(vocab_size=128, max_seq_len=32)
    model = OrbitForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    result = model(ids, targets=ids.roll(-1, dims=1))
    assert result["logits"].shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.backbone.embedding.weight.grad is not None


def test_one_billion_config_without_allocation():
    cfg = OrbitConfig.one_billion()
    cfg.validate()
    assert 900_000_000 < cfg.estimate_parameters() < 1_100_000_000
