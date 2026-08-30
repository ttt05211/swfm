import torch
import torch.nn as nn

from real_motion.models.anchor_cfm import AnchorWindowCFM
from real_motion.occfm_io import OccFMVAEAdapter


class ConstantTransition(nn.Module):
    def forward(self, batch):
        # Anchor=0, target=1 and rescale_factor=10 -> exact velocity is 10.
        return {"predicted_latent": torch.full_like(batch["noised_sequence"], 10.0)}


def test_flow_loss_returns_exact_differentiable_endpoint():
    model = AnchorWindowCFM(ConstantTransition(), rescale_factor=10.0, sample_steps=2)
    hist = torch.zeros(1, 2, 1, 2, 2)
    anchor = torch.zeros(1, 1, 1, 2, 2)
    target = torch.ones_like(anchor)
    loss, info = model.flow_loss(
        hist,
        target,
        anchor,
        t_override=0.5,
        source_noise=torch.zeros_like(anchor),
        return_endpoint=True,
    )
    assert torch.allclose(loss, torch.zeros_like(loss))
    assert torch.allclose(info["predicted_endpoint"], target)
    assert tuple(info["sampled_t"].shape) == (1, 1, 1, 1, 1)


class FakeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip = False
        self.height_num = 1
        self.cate = 1
        self.class_embeds = nn.Embedding(3, 1)
        with torch.no_grad():
            self.class_embeds.weight[:, 0] = torch.tensor([1.0, 2.0, 3.0])


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip = False

    def forward(self, batch):
        # [N,C,H,W] -> one decoded category channel.
        return {"decoded_map": batch["sampled_features"][:, :1]}


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = FakeEmbedding()
        self.decoder = FakeDecoder()


def test_sparse_decoder_projection_keeps_gradient_to_latent_with_frozen_vae():
    vae_model = FakeVAE()
    vae_model.requires_grad_(False)
    adapter = OccFMVAEAdapter(vae_model)

    latent = torch.arange(4.0).reshape(1, 1, 1, 2, 2).requires_grad_(True)
    logits = adapter.decode_logits_at_flat_indices(
        latent,
        [torch.tensor([0, 3], dtype=torch.long)],
    )[0]
    assert tuple(logits.shape) == (2, 3)
    loss = logits.sum()
    loss.backward()
    assert latent.grad is not None
    # Only the two selected decoded voxels should receive gradient in this fake 1x1 decoder.
    grad = latent.grad.reshape(-1)
    assert grad[0] != 0
    assert grad[3] != 0
    assert grad[1] == 0
    assert grad[2] == 0
    assert all(p.grad is None for p in vae_model.parameters())
