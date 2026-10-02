"""FX graph cut points: where a forward pass can be resumed from one cached tensor."""

from collections.abc import Iterable

from torch import fx, nn


def traced(model: nn.Module) -> fx.GraphModule:
    """INT8 models already are FX graphs; FP32 ones are traced, sharing the same submodules."""
    return model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)


def _compute_nodes(gm: fx.GraphModule) -> list[fx.Node]:
    """Every node that carries a per-sample value; `get_attr` constants are re-read on resume."""
    return [n for n in gm.graph.nodes if n.op not in ("get_attr", "output")]


def cut_points(gm: fx.GraphModule) -> list[str]:
    """Nodes whose output is the only live value: nothing after them reads an earlier node."""
    nodes = _compute_nodes(gm)
    pos = {n: i for i, n in enumerate(nodes)}
    cuts, reach = [], -1
    for i, n in enumerate(nodes):
        if reach <= i and n.users:
            cuts.append(n.name)
        reach = max(reach, *(pos.get(u, len(nodes)) for u in n.users), i)
    return cuts


def stateful_modules(gm: fx.GraphModule) -> list[str]:
    """`call_module` targets that hold weights, biases or buffers, i.e. anything injectable."""
    return [n.target for n in gm.graph.nodes
            if n.op == "call_module" and gm.get_submodule(n.target).state_dict()]


class Resume:
    """Positions of cut points and module calls, for picking where an injection resumes."""

    def __init__(self, gm: fx.GraphModule):
        nodes = _compute_nodes(gm)
        self.pos = {n.name: i for i, n in enumerate(nodes)}
        self.module_pos: dict[str, int] = {}
        for n in nodes:
            if n.op == "call_module":
                self.module_pos.setdefault(n.target, self.pos[n.name])
        self.cuts = cut_points(gm)

    def span(self, layers: Iterable[str]) -> tuple[int, int] | None:
        """Graph positions of the first and last faulted module call; None if nothing is faulted."""
        layers = list(layers)
        if not layers:
            return None
        missing = [layer for layer in layers if layer not in self.module_pos]
        if missing:
            raise KeyError(f"faulted modules not called in the graph: {missing}")
        at = [self.module_pos[layer] for layer in layers]
        return min(at), max(at)

    def start(self, layers: Iterable[str]) -> str | None:
        """Latest cut point before the first faulted module call; None if nothing is faulted."""
        span = self.span(layers)
        return None if span is None else [c for c in self.cuts if self.pos[c] < span[0]][-1]

    def starts(self, layers: Iterable[str]) -> list[str]:
        """The cut points to cache so that any of `layers` can be resumed from, in graph order."""
        return sorted({self.start([layer]) for layer in layers}, key=self.pos.__getitem__)
