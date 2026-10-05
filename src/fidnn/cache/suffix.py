"""Run only the part of the fault instance's graph that an injection can change."""

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import fx

from fidnn.cache.graph import Resume
from fidnn.cache.store import ProbeCache
from fidnn.taps.hooks import observe
from fidnn.taps.registry import TapInfo


class _Clean(Exception):
    """Raised at a cut point whose value equals the clean one: everything after it is clean."""

    def __init__(self, node: str):
        super().__init__(node)
        self.node = node  # `fx.Interpreter` rewrites `args` with its own context


class _Suffix(fx.Interpreter):
    """See `fx.Interpreter.run(initial_env=…)`: nodes before the resume point are never executed."""

    def __init__(self, gm: fx.GraphModule, cache: ProbeCache, idx: np.ndarray, checks: set[str],
                 tap_of: dict[str, str], sat: dict[str, float], features: dict[str, torch.Tensor]):
        super().__init__(gm)
        self.cache, self.idx, self.checks = cache, idx, checks
        self.tap_of, self.sat, self.features = tap_of, sat, features

    def run_node(self, n: fx.Node):
        out = super().run_node(n)
        if n.op == "call_module" and n.target in self.tap_of:
            tap_id = self.tap_of[n.target]
            self.features[tap_id] = observe(out, self.sat.get(tap_id, float("inf")))
        if n.name in self.checks and _equal(out, self.cache.tensor(n.name, self.idx)):
            raise _Clean(n.name)
        return out


def _equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.is_quantized:
        return (b.is_quantized and a.q_scale() == b.q_scale()
                and a.q_zero_point() == b.q_zero_point() and torch.equal(a.int_repr(), b.int_repr()))
    return torch.equal(a, b)


class SuffixRunner:
    """Fault-side logits and tap features for one injection, resumed from the clean cache.

    `gm` is the **fault** instance (injected in place); `cache` came from the clean instance.
    """

    def __init__(self, gm: fx.GraphModule, cache: ProbeCache, taps: Sequence[TapInfo],
                 sat: dict[str, float]):
        self.gm, self.cache, self.sat = gm, cache, sat
        self.resume = Resume(gm)
        pos = self.resume.pos
        tap_nodes = {n.target: n.name for n in gm.graph.nodes if n.op == "call_module"}
        self.tap_of = {t.path: t.tap_id for t in taps}
        self.tap_pos = {t.tap_id: pos[tap_nodes[t.path]] for t in taps}
        self.tap_ids = sorted(self.tap_pos, key=self.tap_pos.__getitem__)
        if self.tap_ids != list(cache.tap_ids):
            raise ValueError(f"cache taps {cache.tap_ids} differ from {self.tap_ids}")
        self.before = {name: [m for m in gm.graph.nodes
                              if m.op not in ("get_attr", "output") and pos[m.name] <= pos[name]]
                       for name in cache.cuts}
        self.stats = {"runs": 0, "no_flip": 0, "early_exit": 0}

    def _clean(self, idx: np.ndarray, taps: Iterable[str]) -> dict[str, torch.Tensor]:
        return {t: self.cache.tap_features(t, idx) for t in taps}

    @torch.no_grad()
    def run(self, layers: Iterable[str], idx: np.ndarray) -> tuple[np.ndarray, dict]:
        """(logits, {tap_id: (n, 26) features}) for probes `idx` with the injection applied.

        A forward that raises propagates (CRASH); `self.partial` then holds the taps seen so far.
        """
        self.stats["runs"] += 1
        layers = list(layers)
        span = self.resume.span(layers)
        if span is None:  # e.g. stuck-at where every bit already held the stuck value
            self.stats["no_flip"] += 1
            return self.cache.logits[idx], self._clean(idx, self.tap_ids)
        pos = self.resume.pos
        start = self.resume.start(layers)
        feats = self._clean(idx, [t for t in self.tap_ids if self.tap_pos[t] <= pos[start]])
        self.partial = feats
        env = {m: None for m in self.before[start]}
        env[next(m for m in self.before[start] if m.name == start)] = self.cache.tensor(start, idx)
        # a cut point can only prove the rest clean once every faulted module has run
        checks = {c for c in self.cache.cuts if pos[c] > span[1]}
        interp = _Suffix(self.gm, self.cache, idx, checks, self.tap_of, self.sat, feats)
        try:
            logits = interp.run(initial_env=env, enable_io_processing=False).float().numpy()
        except _Clean as clean:
            self.stats["early_exit"] += 1
            feats.update(self._clean(idx, [t for t in self.tap_ids
                                           if self.tap_pos[t] > pos[clean.node]]))
            logits = self.cache.logits[idx]
        return logits, {t: feats[t] for t in self.tap_ids}
