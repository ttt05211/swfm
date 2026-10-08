"""Bounded, byte-verified static CCR CUDA graphs. Execution, not model code.

The chunk size, candidate order, BF16 arithmetic and six readouts are unchanged.
Only pure static chunks are captured; mixed/dynamic chunks retain the reference
path. Buffers contain one live invocation, never persistent learned features.
"""
from collections import OrderedDict, defaultdict
import time

import numpy as np
import torch

from .ccr_frozen_b import tensor


class SurfaceExecution:
    def __init__(self, head, device, *, graphs=True, max_graphs=4, chunk=8192):
        self.head, self.device = head, torch.device(device)
        if not 1 <= max_graphs <= 4 or chunk != 8192:
            raise ValueError('bounded graphs and unchanged reference chunk=8192 required')
        if head.training or any(p.requires_grad for p in head.parameters()):
            raise ValueError('read-only eval head required')
        self.versions = tuple(p._version for p in head.parameters())
        self.graphs = bool(graphs and self.device.type == 'cuda')
        self.max_graphs, self.chunk = max_graphs, chunk
        self.cache = OrderedDict()
        self.rejected = set()
        self.counts = defaultdict(int)
        self.seconds = defaultdict(float)
        self.failures = []

    def _call(self, name, fn):
        tick = time.perf_counter()
        value = fn()
        self.seconds[name] += time.perf_counter() - tick
        return value

    def _forward(self, values, live):
        features, labels, actors, classes, context, base, fallback, legal = values
        encoded = self.head.encode(features, labels, actors, classes, live)
        logits = self.head.decode(encoded, actors, context, base, fallback, legal, live)
        score = torch.zeros_like(logits, dtype=torch.float32)
        score[..., 0] = logits[..., 0].float().sigmoid()
        return score

    def _capture(self, key, values, live):
        # All static actors select source row zero in the unchanged encoder.
        # Keep EXACT row-zero values/zero-add arithmetic, not an algebraic rewrite.
        has_source = bool(len(live['history_source_context']))
        local = {'_surface_inference_role': True,
                 'history_source_context': live['history_source_context'][:1],
                 '_ccr_source': live['_ccr_source'][:1]}
        buffers = tuple(v.clone() for v in values)
        source = local['_ccr_source'].clone()
        local['_ccr_source'] = source
        while len(self.cache) >= self.max_graphs:
            torch.cuda.synchronize(self.device)
            self.cache.popitem(last=False)
            self.counts['evictions'] += 1
        expected = self._forward(values, live).cpu().numpy()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._forward(buffers, local)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            score = self._forward(buffers, local)
        graph.replay()
        actual = score.cpu().numpy()
        if not np.array_equal(expected, actual):
            raise RuntimeError('static CUDA graph probability bytes differ from eager')
        entry = dict(graph=graph, values=buffers, source=source, output=score,
                     has_source=has_source)
        self.cache[key] = entry
        self.counts['captures_verified'] += 1
        return entry

    @torch.no_grad()
    def __call__(self, head, evidence, plan, output, device):
        if head is not self.head or torch.device(device) != self.device:
            raise ValueError('execution session belongs to one head/device')
        if (head.training or any(p.requires_grad for p in head.parameters())
                or tuple(p._version for p in head.parameters()) != self.versions):
            raise RuntimeError('head changed; recreate execution session, never reuse stale graphs')
        result = []
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == 'cuda'):
            live = self._call('source_projection_host', lambda: head.project_sources(output))
            for start in range(0, len(evidence), self.chunk):
                sl = slice(start, start + self.chunk)
                actors = evidence.actor[sl]
                batch = head.inference_batch(live, actors)
                values = self._call('upload_host', lambda: tuple(tensor(v, self.device) for v in (
                    evidence.features[sl], evidence.labels[sl], actors, evidence.classes[sl],
                    plan.context[sl], plan.base[sl], plan.fallback[sl], plan.legal[sl])))
                role = batch['_surface_inference_role']
                key = (len(actors), bool(len(live['history_source_context'])))
                entry = None
                if self.graphs and role is True and key not in self.rejected:
                    entry = self.cache.get(key)
                    if entry is None:
                        try:
                            entry = self._call('capture_and_parity', lambda: self._capture(key, values, batch))
                        except (RuntimeError, torch.OutOfMemoryError) as exc:
                            # Optional backend rejection is visible; predictions fall
                            # back to unchanged eager, never a reduced support/batch.
                            self.rejected.add(key)
                            self.failures.append(dict(shape=key, error=type(exc).__name__ + ': ' + str(exc)))
                            self.counts['capture_rejections'] += 1
                    else:
                        self.cache.move_to_end(key)
                if entry is not None:
                    def replay():
                        for destination, value in zip(entry['values'], values):
                            destination.copy_(value)
                        entry['source'].copy_(live['_ccr_source'][:1])
                        entry['graph'].replay()
                        return entry['output']
                    score = self._call('static_graph_replay_host', replay)
                    self.counts['graph_replays'] += 1
                else:
                    score = self._call('eager_forward_host', lambda: self._forward(values, batch))
                    self.counts['eager_chunks'] += 1
                # Same blocking, bounded readback as reference: no asynchronous
                # buffer reuse races, padded batches or altered output precision.
                result.append(self._call('readback_host', lambda: score.cpu().numpy().copy()))
                self.counts['points'] += len(actors)
        return np.concatenate(result) if result else np.empty((0, 6, 2), np.float32)

    def stats(self):
        return dict(graphs_enabled=self.graphs, counts=dict(self.counts),
                    host_seconds=dict(self.seconds), resident_graphs=len(self.cache),
                    max_graphs=self.max_graphs, failures=list(self.failures),
                    scope='host stages, NOT CUDA active utilization; capture/parity separately recorded')

    def close(self):
        if self.cache:
            torch.cuda.synchronize(self.device)
        self.cache.clear()
