import torch

from src.training.dpo_video import dpo_loss


def test_dpo_loss_prefers_positive_implicit_margin():
    better, *_ = dpo_loss(
        torch.tensor([2.0]), torch.tensor([0.0]),
        torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0,
    )
    worse, *_ = dpo_loss(
        torch.tensor([0.0]), torch.tensor([2.0]),
        torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0,
    )
    assert better < worse
