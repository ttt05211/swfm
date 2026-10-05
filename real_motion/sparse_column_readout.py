"""Weight-preserving, INFERENCE-only memory-length diagnostic for epoch19.

This is NOT the shared encoder and NOT an equivalence optimization. Encoder,
positions, source queries, gates and population remain unchanged. A failed
zero-shot result is not evidence that a retrained sparse reader cannot work.
"""
from contextlib import contextmanager
from types import MethodType
import torch


def sparse_token_ids(frames, device=None):
    # Preserve the ORIGINAL seven-cell relative coordinates and encoder padding.
    spatial = torch.tensor([0, 3, 6], device=device)
    xy = (spatial[:, None]*7+spatial[None, :]).flatten()
    return (torch.arange(frames, device=device)[:, None]*49+xy).flatten()


def safe_memory(memory, invalid):
    """A sparse subset can be empty even when the full patch was not empty."""
    empty = invalid.all(1)
    memory = memory.clone(); invalid = invalid.clone()
    memory[:, 0] = torch.where(empty[:, None], 0., memory[:, 0])
    invalid[:, 0] &= ~empty
    return memory, invalid


@contextmanager
def token_probe(model):
    if model.training: raise RuntimeError('zero-shot token probe requires eval')
    if model.config.patch != 7: raise ValueError('probe requires original seven-cell patches')
    if hasattr(model, 'column_execution_session'):
        raise RuntimeError('token probe cannot change an active execution/graph session')
    original = model.decode_history
    had_local = 'decode_history' in model.__dict__
    local = model.__dict__.get('decode_history')
    def decode(self, memory, invalid, base, fallback, context, kind, classes, *, query_extra=None):
        if self.training: raise RuntimeError('probe is not a training architecture')
        ids = sparse_token_ids(self.history_frames, memory.device)
        memory, invalid = safe_memory(memory.index_select(1, ids), invalid.index_select(1, ids))
        q = (self.query(torch.cat((context.float(), self.semantic(base.long()).flatten(1),
            self.semantic(fallback.long()).flatten(1)), 1))+self.kind(kind.long())
            +self.classes(classes.long())).unsqueeze(1)
        if query_extra is not None: q = q+query_extra[:, None].to(q.dtype)
        for block in self.decoder: q = block(q, memory, invalid)
        return self.norm(q[:, 0])
    model.decode_history = MethodType(decode, model)
    try: yield model
    finally:
        if had_local: model.decode_history = local
        else: del model.decode_history
