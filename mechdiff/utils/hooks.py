from contextlib import contextmanager


def get_blocks(model):
    """Return the list of transformer blocks (Llama architecture)."""
    return model.model.layers


class LayerActCache:
    """Caches the output (residual) after a given block index during forward."""

    def __init__(self, block):
        self.acts = []
        self._hook = block.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        # output typically includes residual stream at this point
        self.acts.append(output.detach())

    def close(self):
        self._hook.remove()


@contextmanager
def cache_layer(model, layer_idx: int):
    """Context manager that collects forward outputs for the specified block."""
    block = get_blocks(model)[layer_idx]
    cache = LayerActCache(block)
    try:
        yield cache
    finally:
        cache.close()


