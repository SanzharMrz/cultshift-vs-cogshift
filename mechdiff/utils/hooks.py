from contextlib import contextmanager


def get_blocks(model):
    """Return the list of transformer blocks (Llama architecture)."""
    return model.model.layers


class LayerActCache:
    """Caches a tensor at a given block according to hook_type."""

    def __init__(self, block, hook_type: str = "resid_post"):
        self.acts = []
        self._handles = []
        hook_type = hook_type.lower()
        if hook_type == "resid_post":
            self._handles.append(block.register_forward_hook(self._on_forward))
        elif hook_type == "resid_pre":
            self._handles.append(block.register_forward_pre_hook(self._on_pre))
        elif hook_type == "attn_out":
            attn = getattr(block, "self_attn", None)
            if attn is None:
                raise ValueError("Block has no self_attn for attn_out hook")
            self._handles.append(attn.register_forward_hook(self._on_forward))
        elif hook_type == "mlp_out":
            mlp = getattr(block, "mlp", None)
            if mlp is None:
                raise ValueError("Block has no mlp for mlp_out hook")
            self._handles.append(mlp.register_forward_hook(self._on_forward))
        else:
            raise ValueError(f"Unknown hook_type {hook_type}")

    def _on_forward(self, module, inputs, output):
        # Handle modules that return tuples or output objects
        out = output
        if isinstance(out, (tuple, list)) and len(out) > 0:
            out = out[0]
        # Some HF modules return objects with .last_hidden_state
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        try:
            self.acts.append(out.detach())
        except Exception:
            # Best-effort conversion
            self.acts.append(torch.as_tensor(out).detach())

    def _on_pre(self, module, inputs):
        if isinstance(inputs, tuple) and len(inputs) > 0:
            self.acts.append(inputs[0].detach())
        else:
            # best effort
            try:
                self.acts.append(inputs.detach())
            except Exception:
                pass

    def close(self):
        for h in self._handles:
            h.remove()


@contextmanager
def cache_layer(model, layer_idx: int):
    """Backward-compat: cache block output (resid_post)."""
    block = get_blocks(model)[layer_idx]
    cache = LayerActCache(block, hook_type="resid_post")
    try:
        yield cache
    finally:
        cache.close()


@contextmanager
def cache_layer_with_hook(model, layer_idx: int, hook_type: str):
    """Cache tensor at a specific site within the block."""
    block = get_blocks(model)[layer_idx]
    cache = LayerActCache(block, hook_type=hook_type)
    try:
        yield cache
    finally:
        cache.close()


